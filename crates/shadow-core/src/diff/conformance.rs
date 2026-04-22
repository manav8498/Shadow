//! Axis 9: schema / format conformance rate.
//!
//! For each response whose text content is claimed to be JSON (via a simple
//! heuristic: starts with `{` or `[`), check whether it actually parses.
//! Rate = fraction of responses where the parse succeeded.

use crate::agentlog::Record;
use crate::diff::axes::{Axis, AxisStat, Severity};
use crate::diff::bootstrap::paired_ci;

fn response_text(r: &Record) -> String {
    let Some(arr) = r.payload.get("content").and_then(|c| c.as_array()) else {
        return String::new();
    };
    arr.iter()
        .filter_map(|p| {
            if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                p.get("text")
                    .and_then(|t| t.as_str())
                    .map(ToString::to_string)
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn conforms(r: &Record) -> Option<f64> {
    let text = response_text(r);
    let trimmed = text.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        Some(
            if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
                1.0
            } else {
                0.0
            },
        )
    } else {
        None // no JSON intent → not counted
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Compute the schema-conformance axis.
pub fn compute(pairs: &[(&Record, &Record)], seed: Option<u64>) -> AxisStat {
    let mut b = Vec::new();
    let mut c = Vec::new();
    for (br, cr) in pairs {
        if let (Some(bv), Some(cv)) = (conforms(br), conforms(cr)) {
            b.push(bv);
            c.push(cv);
        }
    }
    if b.is_empty() {
        return AxisStat::empty(Axis::Conformance);
    }
    let bm = mean(&b);
    let cm = mean(&c);
    let delta = cm - bm;
    let ci = paired_ci(&b, &c, |bs, cs| mean(cs) - mean(bs), 0, seed);
    AxisStat {
        axis: Axis::Conformance,
        baseline_median: bm,
        candidate_median: cm,
        delta,
        ci95_low: ci.low,
        ci95_high: ci.high,
        severity: Severity::classify(delta, bm.max(0.01), ci.low, ci.high),
        n: b.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::Kind;
    use serde_json::json;

    fn response(text: &str) -> Record {
        Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "latency_ms": 0,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        )
    }

    #[test]
    fn valid_json_counts_as_conforming() {
        assert_eq!(conforms(&response(r#"{"a": 1}"#)), Some(1.0));
    }

    #[test]
    fn invalid_json_counts_as_non_conforming() {
        assert_eq!(conforms(&response("{a: 1")), Some(0.0));
    }

    #[test]
    fn non_json_text_is_excluded() {
        assert_eq!(conforms(&response("hello world")), None);
    }

    #[test]
    fn regression_from_all_valid_to_half_valid_is_detected() {
        let baseline: Vec<Record> = (0..10).map(|_| response(r#"{"ok": true}"#)).collect();
        let candidate: Vec<Record> = (0..10)
            .map(|i| {
                if i < 5 {
                    response(r#"{"ok": true}"#)
                } else {
                    response("{broken")
                }
            })
            .collect();
        let pairs: Vec<(&Record, &Record)> = baseline.iter().zip(candidate.iter()).collect();
        let stat = compute(&pairs, Some(1));
        assert!((stat.baseline_median - 1.0).abs() < 1e-9);
        assert!((stat.candidate_median - 0.5).abs() < 1e-9);
    }
}
