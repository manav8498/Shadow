//! Axis 9: schema / format conformance rate.
//!
//! Intent-gated on the baseline side: if the baseline response's text
//! looks like JSON (starts with `{` or `[` after trim), then this pair
//! is counted and BOTH sides are scored on whether their text parses as
//! JSON. This correctly flags the "baseline produced JSON, candidate
//! regressed to prose" failure mode — the one you most want to catch.
//!
//! Pairs where baseline has no JSON intent are excluded (we don't penalise
//! the candidate for gratuitously adding JSON output when nobody asked).

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

/// True if `text` (after trim) parses as JSON.
fn is_json_parseable(text: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(text.trim()).is_ok()
}

/// True if `text` (after trim) starts with `{` or `[` — our heuristic for
/// "this response intends to be JSON."
fn has_json_intent(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.starts_with('{') || trimmed.starts_with('[')
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
        let baseline_text = response_text(br);
        if !has_json_intent(&baseline_text) {
            // Baseline wasn't trying to produce JSON — this pair isn't
            // part of the conformance population.
            continue;
        }
        // Baseline has JSON intent. Both sides are scored on whether
        // their text is parseable JSON. A candidate that returned prose
        // gets 0 (conformance regression), not excluded.
        let b_score = f64::from(is_json_parseable(&baseline_text));
        let candidate_text = response_text(cr);
        let c_score = f64::from(is_json_parseable(&candidate_text));
        b.push(b_score);
        c.push(c_score);
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
        // Use absolute-scale severity: a rate axis whose baseline is 1.0
        // and candidate is 0.5 is a 50% regression, not "within noise."
        severity: Severity::classify_rate(delta, ci.low, ci.high),
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
    fn baseline_json_intent_candidate_prose_flags_severe() {
        // Headline fix: the case where baseline produces valid JSON and
        // candidate regresses to prose MUST surface as a severe drop.
        let baseline = response(r#"[{"a": 1}]"#);
        let candidate = response("Here are your results: ...");
        let pairs = [(&baseline, &candidate); 3];
        let stat = compute(&pairs, Some(1));
        assert!((stat.baseline_median - 1.0).abs() < 1e-9);
        assert!((stat.candidate_median - 0.0).abs() < 1e-9);
        assert_eq!(stat.severity, Severity::Severe);
        assert_eq!(stat.n, 3);
    }

    #[test]
    fn both_sides_valid_json_is_no_regression() {
        let r = response(r#"{"a": 1}"#);
        let pairs = [(&r, &r); 5];
        let stat = compute(&pairs, Some(1));
        assert_eq!(stat.severity, Severity::None);
    }

    #[test]
    fn baseline_without_json_intent_is_excluded_from_population() {
        // Both responses are prose — no JSON intent on the baseline side,
        // so the pair doesn't count toward the conformance axis.
        let baseline = response("hello");
        let candidate = response("world");
        let pairs = [(&baseline, &candidate); 3];
        let stat = compute(&pairs, Some(1));
        assert_eq!(stat.n, 0);
    }

    #[test]
    fn baseline_json_candidate_broken_json_is_counted() {
        // Candidate attempted JSON but produced invalid output.
        let baseline = response(r#"{"ok": true}"#);
        let candidate = response("{broken");
        let pairs = [(&baseline, &candidate); 4];
        let stat = compute(&pairs, Some(1));
        assert!((stat.baseline_median - 1.0).abs() < 1e-9);
        assert!((stat.candidate_median - 0.0).abs() < 1e-9);
        assert_eq!(stat.severity, Severity::Severe);
    }

    #[test]
    fn partial_regression_is_moderate() {
        // Baseline always JSON, candidate half-and-half.
        let baseline = response(r#"{"ok": true}"#);
        let good = response(r#"{"ok": true}"#);
        let bad = response("plain text response");
        let pairs = vec![(&baseline, &good), (&baseline, &good), (&baseline, &bad)];
        let stat = compute(&pairs, Some(1));
        assert!((stat.candidate_median - (2.0 / 3.0)).abs() < 1e-9);
        assert!(matches!(
            stat.severity,
            Severity::Moderate | Severity::Severe
        ));
    }
}
