//! Axis 4: verbosity (output-token count) from
//! `chat_response.usage.output_tokens`.

use crate::agentlog::Record;
use crate::diff::axes::{Axis, AxisStat};
use crate::diff::bootstrap::{median, paired_ci};

fn output_tokens(r: &Record) -> Option<f64> {
    r.payload
        .get("usage")
        .and_then(|u| u.get("output_tokens"))
        .and_then(|v| v.as_f64())
}

/// Compute the verbosity axis from paired response records.
pub fn compute(pairs: &[(&Record, &Record)], seed: Option<u64>) -> AxisStat {
    let mut b = Vec::with_capacity(pairs.len());
    let mut c = Vec::with_capacity(pairs.len());
    for (br, cr) in pairs {
        if let (Some(bv), Some(cv)) = (output_tokens(br), output_tokens(cr)) {
            b.push(bv);
            c.push(cv);
        }
    }
    if b.is_empty() {
        return AxisStat::empty(Axis::Verbosity);
    }
    let bm = median(&b);
    let cm = median(&c);
    let delta = cm - bm;
    let ci = paired_ci(&b, &c, |bs, cs| median(cs) - median(bs), 0, seed);
    AxisStat::new_value(Axis::Verbosity, bm, cm, delta, ci.low, ci.high, b.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::Kind;
    use serde_json::json;

    fn response(output: u64) -> Record {
        Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": [],
                "stop_reason": "end_turn",
                "latency_ms": 0,
                "usage": {"input_tokens": 1, "output_tokens": output, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        )
    }

    #[test]
    fn candidate_half_as_verbose_is_moderate_or_severe() {
        use crate::diff::axes::Severity;
        let baseline: Vec<Record> = (0..20).map(|i| response(100 + i)).collect();
        let candidate: Vec<Record> = (0..20).map(|i| response(50 + i)).collect();
        let pairs: Vec<(&Record, &Record)> = baseline.iter().zip(candidate.iter()).collect();
        let stat = compute(&pairs, Some(7));
        assert!(stat.delta < 0.0);
        assert!(matches!(
            stat.severity,
            Severity::Moderate | Severity::Severe
        ));
    }
}
