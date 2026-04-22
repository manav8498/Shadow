//! Axis 6: cost (input+output tokens × per-model pricing).
//!
//! Pricing is a user-editable `HashMap<String, (input_per_token, output_per_token)>`
//! keyed on `chat_response.payload.model`. Unknown models contribute 0 cost —
//! we'd rather surface "0" than burn the report with phantom infinities.

use std::collections::HashMap;

use crate::agentlog::Record;
use crate::diff::axes::{Axis, AxisStat, Severity};
use crate::diff::bootstrap::{median, paired_ci};

/// Price per token in USD: `(input, output)`. USD so callers can plug
/// their own currency without a type change.
pub type Pricing = HashMap<String, (f64, f64)>;

fn cost_of(r: &Record, pricing: &Pricing) -> Option<f64> {
    let model = r.payload.get("model")?.as_str()?;
    let usage = r.payload.get("usage")?;
    let input = usage.get("input_tokens")?.as_f64()?;
    let output = usage.get("output_tokens")?.as_f64()?;
    let (pi, po) = pricing.get(model).copied().unwrap_or((0.0, 0.0));
    Some(input * pi + output * po)
}

/// Compute the cost axis.
pub fn compute(pairs: &[(&Record, &Record)], pricing: &Pricing, seed: Option<u64>) -> AxisStat {
    let mut b = Vec::with_capacity(pairs.len());
    let mut c = Vec::with_capacity(pairs.len());
    for (br, cr) in pairs {
        if let (Some(bv), Some(cv)) = (cost_of(br, pricing), cost_of(cr, pricing)) {
            b.push(bv);
            c.push(cv);
        }
    }
    if b.is_empty() {
        return AxisStat::empty(Axis::Cost);
    }
    let bm = median(&b);
    let cm = median(&c);
    let delta = cm - bm;
    let ci = paired_ci(&b, &c, |bs, cs| median(cs) - median(bs), 0, seed);
    AxisStat {
        axis: Axis::Cost,
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

    fn response(model: &str, input: u64, output: u64) -> Record {
        Record::new(
            Kind::ChatResponse,
            json!({
                "model": model,
                "content": [],
                "stop_reason": "end_turn",
                "latency_ms": 0,
                "usage": {"input_tokens": input, "output_tokens": output, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        )
    }

    #[test]
    fn pricing_lookup_drives_cost() {
        let mut pricing = Pricing::new();
        pricing.insert("opus".to_string(), (0.000015, 0.000075));
        pricing.insert("haiku".to_string(), (0.0000008, 0.000004));
        let baseline: Vec<Record> = (0..10).map(|_| response("opus", 1000, 500)).collect();
        // Candidate switches to haiku — should be much cheaper.
        let candidate: Vec<Record> = (0..10).map(|_| response("haiku", 1000, 500)).collect();
        let pairs: Vec<(&Record, &Record)> = baseline.iter().zip(candidate.iter()).collect();
        let stat = compute(&pairs, &pricing, Some(1));
        assert!(stat.delta < 0.0);
        assert_eq!(stat.severity, Severity::Severe);
    }

    #[test]
    fn unknown_model_costs_zero() {
        let pricing = Pricing::new();
        let r = response("mystery", 1000, 500);
        let pairs = [(&r, &r)];
        let stat = compute(&pairs, &pricing, Some(1));
        assert_eq!(stat.baseline_median, 0.0);
    }
}
