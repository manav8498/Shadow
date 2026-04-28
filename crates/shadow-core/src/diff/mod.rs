//! Nine-axis behavioral differ, bootstrap CI, and report renderers.
//!
//! See README.md "The nine axes" for the list and SPEC §Replay for what
//! "diverges" means in this context.
//!
//! Usage:
//! ```no_run
//! # use shadow_core::agentlog::Record;
//! # use shadow_core::diff;
//! # fn demo(baseline: Vec<Record>, candidate: Vec<Record>) {
//! let pricing = diff::cost::Pricing::new();
//! let report = diff::compute_report(&baseline, &candidate, &pricing, Some(42));
//! println!("{}", report.to_terminal());
//! # }
//! ```

use crate::agentlog::{Kind, Record};

pub mod alignment;
pub mod axes;
pub mod bootstrap;
pub mod conformance;
pub mod cost;
pub mod drill_down;
pub mod embedder;
pub mod judge;
pub mod latency;
pub mod reasoning;
pub mod recommendations;
pub mod report;
pub mod safety;
pub mod semantic;
pub mod trajectory;
pub mod verbosity;

pub use alignment::{DivergenceKind, FirstDivergence};
pub use axes::{Axis, AxisStat, Severity};
pub use bootstrap::{paired_ci, CiResult};
pub use drill_down::{PairAxisScore, PairDrilldown};
pub use recommendations::{ActionKind, Recommendation, RecommendationSeverity};
pub use report::DiffReport;

/// Extract (baseline_response, candidate_response) pairs by pairing the
/// i-th `chat_response` in `baseline` with the i-th in `candidate`.
///
/// If the counts differ (e.g. candidate had backend errors), truncate to
/// the shorter of the two. Callers that need divergence-on-count should
/// consult the `replay_summary` record directly.
pub fn extract_response_pairs<'a>(
    baseline: &'a [Record],
    candidate: &'a [Record],
) -> Vec<(&'a Record, &'a Record)> {
    let b: Vec<&Record> = baseline
        .iter()
        .filter(|r| r.kind == Kind::ChatResponse)
        .collect();
    let c: Vec<&Record> = candidate
        .iter()
        .filter(|r| r.kind == Kind::ChatResponse)
        .collect();
    b.into_iter().zip(c).collect()
}

/// Compute a [`DiffReport`] from a baseline and candidate trace.
///
/// The Judge axis is set to `empty(Axis::Judge)` because no Judge is
/// supplied here; the Python layer plugs in a Judge via `compute_report_with_judge`.
pub fn compute_report(
    baseline: &[Record],
    candidate: &[Record],
    pricing: &cost::Pricing,
    seed: Option<u64>,
) -> DiffReport {
    let pairs = extract_response_pairs(baseline, candidate);
    let rows = vec![
        semantic::compute(&pairs, seed),
        trajectory::compute(&pairs, seed),
        safety::compute(&pairs, seed),
        verbosity::compute(&pairs, seed),
        latency::compute(&pairs, seed),
        cost::compute(&pairs, pricing, seed),
        reasoning::compute(&pairs, seed),
        AxisStat::empty(Axis::Judge),
        conformance::compute(&pairs, seed),
    ];
    let first_divergence = alignment::detect(baseline, candidate);
    let divergences = alignment::detect_top_k(baseline, candidate, alignment::DEFAULT_K);
    let drill_down = drill_down::compute(&pairs, pricing, drill_down::DEFAULT_K);
    let mut report = DiffReport {
        rows,
        baseline_trace_id: baseline.first().map(|r| r.id.clone()).unwrap_or_default(),
        candidate_trace_id: candidate.first().map(|r| r.id.clone()).unwrap_or_default(),
        pair_count: pairs.len(),
        first_divergence,
        divergences,
        drill_down,
        recommendations: Vec::new(),
    };
    // Recommendations are derived from the rest of the report, so fill
    // the field last. Keeps the function ordering natural and avoids
    // passing the half-built report around.
    report.recommendations = recommendations::generate(&report);
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::Kind;
    use serde_json::json;

    fn make_trace(responses: Vec<(u64, &str)>) -> Vec<Record> {
        let meta = Record::new(
            Kind::Metadata,
            json!({"sdk": {"name": "shadow"}}),
            "2026-04-21T10:00:00Z",
            None,
        );
        let mut out = vec![meta];
        for (i, (latency, text)) in responses.iter().enumerate() {
            let req = Record::new(
                Kind::ChatRequest,
                json!({"model": "x", "messages": [{"role": "user", "content": format!("q{i}")}], "params": {}}),
                format!("2026-04-21T10:00:{:02}.000Z", i),
                out.last().map(|r| r.id.clone()),
            );
            let resp = Record::new(
                Kind::ChatResponse,
                json!({
                    "model": "x",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "latency_ms": latency,
                    "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0},
                }),
                format!("2026-04-21T10:00:{:02}.500Z", i),
                Some(req.id.clone()),
            );
            out.push(req);
            out.push(resp);
        }
        out
    }

    #[test]
    fn compute_report_shapes_to_nine_axes() {
        let baseline = make_trace(vec![(100, "yes"), (110, "ok"), (90, "sure")]);
        let candidate = make_trace(vec![(200, "yes"), (220, "ok"), (180, "sure")]);
        let pricing = cost::Pricing::new();
        let report = compute_report(&baseline, &candidate, &pricing, Some(42));
        assert_eq!(report.rows.len(), 9);
        assert_eq!(report.pair_count, 3);
        // Latency axis should show a delta.
        let latency_row = report
            .rows
            .iter()
            .find(|r| r.axis == Axis::Latency)
            .unwrap();
        assert!(latency_row.delta > 0.0);
    }

    #[test]
    fn extract_response_pairs_truncates_to_shorter() {
        let b = make_trace(vec![(1, "a"), (2, "b"), (3, "c")]);
        let c = make_trace(vec![(1, "a"), (2, "b")]);
        let pairs = extract_response_pairs(&b, &c);
        assert_eq!(pairs.len(), 2);
    }
}
