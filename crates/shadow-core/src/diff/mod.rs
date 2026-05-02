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

/// Stable trace identifier for a `.agentlog` record stream.
///
/// Two-step resolution:
///
/// 1. **Envelope `meta.trace_id`** — preferred. The Python SDK's
///    `Session` mints a unique 128-bit hex `trace_id` per instance and
///    stamps it on every record's envelope `meta`. Envelope meta is
///    deliberately *not* part of the content hash (SPEC §6), so this
///    stays unique even when two sessions emit byte-identical
///    metadata payloads.
///
/// 2. **First record's content id** — fallback. Used for traces that
///    don't carry envelope-level `meta.trace_id` (third-party
///    OpenTelemetry imports, hand-constructed fixtures, traces from
///    SDK versions older than v1.x). The `id` field of the first
///    record is the SHA-256 of its canonical payload, so this is
///    stable for any given input but can collide across runs whose
///    metadata payloads happen to match exactly — which is the case
///    `meta.trace_id` exists to prevent.
///
/// Returns `String::new()` for an empty record list.
fn trace_id_for(records: &[Record]) -> String {
    records
        .iter()
        .find_map(|r| {
            r.meta
                .as_ref()
                .and_then(|m| m.get("trace_id"))
                .and_then(|v| v.as_str())
                .map(str::to_string)
        })
        .or_else(|| records.first().map(|r| r.id.clone()))
        .unwrap_or_default()
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
        baseline_trace_id: trace_id_for(baseline),
        candidate_trace_id: trace_id_for(candidate),
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

    #[test]
    fn trace_ids_use_envelope_meta_to_avoid_payload_collisions() {
        // Two traces with byte-identical metadata payloads (no tags
        // distinguish them) but different envelope-level meta.trace_id.
        // Before the fix, the diff report used `Record.id` (the content
        // hash of the payload) for `baseline_trace_id` and
        // `candidate_trace_id`, which collided whenever the metadata
        // payload was the same — i.e. on every default-tagless run pair.
        //
        // After the fix: the report prefers `meta.trace_id` from the
        // envelope, which the Python SDK Session mints uniquely per
        // instance. Envelope meta is not part of the content hash, so it
        // stays unique even when payloads match exactly.
        fn stamp_meta(mut rec: Record, trace_id: &str) -> Record {
            let mut m = serde_json::Map::new();
            m.insert("trace_id".into(), json!(trace_id));
            rec.meta = Some(m);
            rec
        }
        let b = make_trace(vec![(1, "hello")])
            .into_iter()
            .map(|r| stamp_meta(r, "trace-aaaa"))
            .collect::<Vec<_>>();
        let c = make_trace(vec![(2, "hello")])
            .into_iter()
            .map(|r| stamp_meta(r, "trace-bbbb"))
            .collect::<Vec<_>>();

        // Sanity: the metadata payload (Record.id of the first record)
        // is identical across baseline and candidate — this is the
        // collision case the bug report cited.
        assert_eq!(b[0].id, c[0].id);

        let pricing = cost::Pricing::new();
        let report = compute_report(&b, &c, &pricing, Some(42));

        assert_eq!(report.baseline_trace_id, "trace-aaaa");
        assert_eq!(report.candidate_trace_id, "trace-bbbb");
        assert_ne!(report.baseline_trace_id, report.candidate_trace_id);
    }

    #[test]
    fn trace_id_falls_back_to_first_record_id_when_meta_missing() {
        // Traces without envelope meta.trace_id (third-party imports,
        // hand-constructed fixtures, pre-1.0 SDK output) keep the
        // pre-fix behaviour: use the first record's content id. This
        // preserves backward compatibility for everything that doesn't
        // have a Session-stamped envelope.
        let b = make_trace(vec![(1, "hello")]);
        let c = make_trace(vec![(2, "world")]);
        let pricing = cost::Pricing::new();
        let report = compute_report(&b, &c, &pricing, Some(42));
        // The metadata payloads are identical so first-record content
        // ids collide here — that's the documented fallback behaviour.
        assert_eq!(report.baseline_trace_id, b[0].id);
        assert_eq!(report.candidate_trace_id, c[0].id);
    }
}
