//! Axis 7: reasoning depth — thinking tokens + self-correction markers.
//!
//! "Thinking" tokens come from the response `usage.thinking_tokens` field.
//! Self-correction markers are conservative: the count of content parts
//! whose `type == "thinking"`. We sum these per response.

use crate::agentlog::Record;
use crate::diff::axes::{Axis, AxisStat, Severity};
use crate::diff::bootstrap::{median, paired_ci};

fn reasoning_score(r: &Record) -> Option<f64> {
    let thinking_tokens = r
        .payload
        .get("usage")
        .and_then(|u| u.get("thinking_tokens"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let thinking_parts = r
        .payload
        .get("content")
        .and_then(|c| c.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("thinking"))
                .count() as f64
        })
        .unwrap_or(0.0);
    Some(thinking_tokens + thinking_parts)
}

/// Compute the reasoning-depth axis.
pub fn compute(pairs: &[(&Record, &Record)], seed: Option<u64>) -> AxisStat {
    let mut b = Vec::with_capacity(pairs.len());
    let mut c = Vec::with_capacity(pairs.len());
    for (br, cr) in pairs {
        if let (Some(bv), Some(cv)) = (reasoning_score(br), reasoning_score(cr)) {
            b.push(bv);
            c.push(cv);
        }
    }
    if b.is_empty() {
        return AxisStat::empty(Axis::Reasoning);
    }
    let bm = median(&b);
    let cm = median(&c);
    let delta = cm - bm;
    let ci = paired_ci(&b, &c, |bs, cs| median(cs) - median(bs), 0, seed);
    AxisStat {
        axis: Axis::Reasoning,
        baseline_median: bm,
        candidate_median: cm,
        delta,
        ci95_low: ci.low,
        ci95_high: ci.high,
        severity: Severity::classify(delta, bm, ci.low, ci.high),
        n: b.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::Kind;
    use serde_json::json;

    fn response(thinking_tokens: u64, with_thinking_part: bool) -> Record {
        let content = if with_thinking_part {
            json!([{"type": "thinking", "text": "..."}, {"type": "text", "text": "done"}])
        } else {
            json!([{"type": "text", "text": "done"}])
        };
        Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": content,
                "stop_reason": "end_turn",
                "latency_ms": 0,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": thinking_tokens},
            }),
            "2026-04-21T10:00:00Z",
            None,
        )
    }

    #[test]
    fn thinking_tokens_and_parts_are_summed() {
        let baseline: Vec<Record> = (0..10).map(|_| response(0, false)).collect();
        let candidate: Vec<Record> = (0..10).map(|_| response(100, true)).collect();
        let pairs: Vec<(&Record, &Record)> = baseline.iter().zip(candidate.iter()).collect();
        let stat = compute(&pairs, Some(1));
        assert_eq!(stat.baseline_median, 0.0);
        assert_eq!(stat.candidate_median, 101.0);
    }
}
