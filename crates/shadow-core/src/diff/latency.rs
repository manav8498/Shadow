//! Axis 5: end-to-end latency (SPEC §4.2 `chat_response.latency_ms`).

use crate::agentlog::Record;
use crate::diff::axes::{Axis, AxisStat};
use crate::diff::bootstrap::{median, paired_ci};

/// Extract `payload.latency_ms` from a chat_response, or `None` if missing.
fn latency_ms(r: &Record) -> Option<f64> {
    r.payload.get("latency_ms").and_then(|v| v.as_f64())
}

/// Compute the latency axis from paired response records.
pub fn compute(pairs: &[(&Record, &Record)], seed: Option<u64>) -> AxisStat {
    let mut baseline_vals = Vec::with_capacity(pairs.len());
    let mut candidate_vals = Vec::with_capacity(pairs.len());
    for (b, c) in pairs {
        if let (Some(bv), Some(cv)) = (latency_ms(b), latency_ms(c)) {
            baseline_vals.push(bv);
            candidate_vals.push(cv);
        }
    }
    if baseline_vals.is_empty() {
        return AxisStat::empty(Axis::Latency);
    }
    let baseline_median = median(&baseline_vals);
    let candidate_median = median(&candidate_vals);
    let delta = candidate_median - baseline_median;
    let ci = paired_ci(
        &baseline_vals,
        &candidate_vals,
        |b, c| median(c) - median(b),
        0,
        seed,
    );
    AxisStat::new_value(
        Axis::Latency,
        baseline_median,
        candidate_median,
        delta,
        ci.low,
        ci.high,
        baseline_vals.len(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::Kind;
    use serde_json::json;

    fn response(latency: f64) -> Record {
        Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": [],
                "stop_reason": "end_turn",
                "latency_ms": latency,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        )
    }

    use crate::diff::axes::Severity;

    #[test]
    fn equal_latency_has_zero_delta_and_no_severity() {
        let rs: Vec<Record> = (0..20).map(|i| response(100.0 + i as f64)).collect();
        let pairs: Vec<(&Record, &Record)> = rs.iter().zip(rs.iter()).collect();
        let stat = compute(&pairs, Some(1));
        assert_eq!(stat.axis, Axis::Latency);
        assert_eq!(stat.severity, Severity::None);
        assert!(stat.delta.abs() < 1e-6);
    }

    #[test]
    fn candidate_2x_slower_is_moderate_or_severe() {
        let baseline: Vec<Record> = (0..20).map(|i| response(100.0 + i as f64)).collect();
        let candidate: Vec<Record> = (0..20).map(|i| response(200.0 + 2.0 * i as f64)).collect();
        let pairs: Vec<(&Record, &Record)> = baseline.iter().zip(candidate.iter()).collect();
        let stat = compute(&pairs, Some(1));
        assert!(stat.delta > 90.0);
        assert!(matches!(
            stat.severity,
            Severity::Moderate | Severity::Severe
        ));
    }

    #[test]
    fn missing_latency_is_skipped() {
        let without_latency = Record::new(
            Kind::ChatResponse,
            json!({"model": "x"}),
            "2026-04-21T10:00:00Z",
            None,
        );
        let with_latency = response(100.0);
        let pairs = [(&without_latency, &with_latency)];
        let stat = compute(&pairs, Some(1));
        assert_eq!(stat.n, 0);
    }
}
