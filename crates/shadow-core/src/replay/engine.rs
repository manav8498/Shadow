//! Replay engine.
//!
//! Given a baseline trace and an [`LlmBackend`], produce a candidate trace
//! with the same request order but LLM-derived responses freshly generated
//! by the backend. See SPEC §10 for the algorithm.

use serde_json::json;
use thiserror::Error;

use crate::agentlog::{Kind, Record};
use crate::replay::backend::{LlmBackend, LlmError};

/// Errors from [`run_replay`].
#[derive(Debug, Error)]
pub enum ReplayError {
    /// The baseline trace is empty — no root metadata record.
    #[error("baseline trace is empty\nhint: a baseline trace must start with a metadata record (SPEC §3.3)")]
    EmptyBaseline,

    /// The baseline trace does not start with a `metadata` record.
    #[error("baseline trace root is {found:?}, expected Metadata\nhint: SPEC §3.3 requires the first record to be of kind metadata")]
    BadBaselineRoot {
        /// The actually-encountered kind.
        found: Kind,
    },

    /// The backend failed on at least one request.
    #[error("backend error: {0}")]
    Backend(#[from] LlmError),
}

/// Clock abstraction so tests can pin timestamps.
pub trait Clock: Send + Sync {
    /// Return the current RFC 3339 UTC timestamp string (millisecond precision).
    fn now_iso(&self) -> String;
}

/// A clock that returns a fixed incrementing counter. Used by tests.
pub struct FixedClock {
    base: String,
    counter: std::sync::atomic::AtomicU64,
}

impl FixedClock {
    /// Build a clock that returns `"<base>#<n>"` for each call.
    pub fn new(base: impl Into<String>) -> Self {
        Self {
            base: base.into(),
            counter: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl Clock for FixedClock {
    fn now_iso(&self) -> String {
        let n = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        format!("{}#{n}", self.base)
    }
}

/// Run a replay: walk `baseline`, dispatch every `chat_request` to `backend`,
/// and produce a fresh trace with the same structure.
///
/// Algorithm (SPEC §10.1):
/// 1. Emit a new `metadata` record with `parent = None` and an envelope
///    `meta.baseline_of = baseline_root_id`.
/// 2. For each baseline record in file order:
///    - `chat_request`: re-emit with a fresh ts and parent = previous
///      output record id; then call `backend.complete(request.payload)`
///      and emit a `chat_response` whose parent = the re-emitted request.
///    - `tool_call`, `tool_result`, `error`: copy-through with fresh ts
///      and relinked parent.
///    - `chat_response`, `metadata`, `replay_summary`: skipped (the
///      backend produces responses; replay_summary is added at the end;
///      a baseline can only have one metadata record and it's the root).
/// 3. Emit a `replay_summary` at the end.
pub async fn run_replay<B: LlmBackend + ?Sized>(
    baseline: &[Record],
    backend: &B,
    clock: &dyn Clock,
) -> Result<Vec<Record>, ReplayError> {
    let baseline_root = baseline.first().ok_or(ReplayError::EmptyBaseline)?;
    if baseline_root.kind != Kind::Metadata {
        return Err(ReplayError::BadBaselineRoot {
            found: baseline_root.kind,
        });
    }

    let mut out = Vec::with_capacity(baseline.len() + 1);

    // 1. New metadata root with baseline_of pointer.
    let meta_payload = if let Some(obj) = baseline_root.payload.as_object() {
        let mut new_payload = obj.clone();
        new_payload.insert(
            "baseline_of".to_string(),
            serde_json::Value::String(baseline_root.id.clone()),
        );
        serde_json::Value::Object(new_payload)
    } else {
        json!({ "baseline_of": baseline_root.id })
    };
    let new_root = Record::new(Kind::Metadata, meta_payload, clock.now_iso(), None);
    let mut last_parent = new_root.id.clone();
    out.push(new_root);

    let mut input_count: u64 = 0;
    let mut output_count: u64 = 0;
    let mut error_count: u64 = 0;
    let start = std::time::Instant::now();

    // 2. Walk baseline.
    for (i, record) in baseline.iter().enumerate() {
        match record.kind {
            Kind::Metadata => {
                if i == 0 {
                    continue; // root already handled
                }
                // Multiple metadata records are an invariant violation, but
                // SPEC §3.3 already forbids this; we defensively copy-through.
                let copy = Record::new(
                    Kind::Metadata,
                    record.payload.clone(),
                    clock.now_iso(),
                    Some(last_parent.clone()),
                );
                last_parent = copy.id.clone();
                out.push(copy);
            }
            Kind::ChatRequest => {
                input_count += 1;
                let req = Record::new(
                    Kind::ChatRequest,
                    record.payload.clone(),
                    clock.now_iso(),
                    Some(last_parent.clone()),
                );
                let req_id = req.id.clone();
                out.push(req);
                match backend.complete(&record.payload).await {
                    Ok(response_payload) => {
                        let resp = Record::new(
                            Kind::ChatResponse,
                            response_payload,
                            clock.now_iso(),
                            Some(req_id.clone()),
                        );
                        last_parent = resp.id.clone();
                        out.push(resp);
                        output_count += 1;
                    }
                    Err(e) => {
                        error_count += 1;
                        let err = Record::new(
                            Kind::Error,
                            json!({
                                "source": "llm",
                                "code": "backend_error",
                                "message": e.to_string(),
                                "retriable": matches!(e, LlmError::Io(_)),
                            }),
                            clock.now_iso(),
                            Some(req_id.clone()),
                        );
                        last_parent = err.id.clone();
                        out.push(err);
                    }
                }
            }
            Kind::ChatResponse => {
                // Baseline responses are discarded; the backend produces
                // the candidate response. (Technically the baseline trace
                // has a chat_response for every chat_request; skipping
                // here keeps the output one-response-per-request.)
                continue;
            }
            Kind::ToolCall | Kind::ToolResult | Kind::Error => {
                let copy = Record::new(
                    record.kind,
                    record.payload.clone(),
                    clock.now_iso(),
                    Some(last_parent.clone()),
                );
                last_parent = copy.id.clone();
                out.push(copy);
            }
            Kind::ReplaySummary => continue, // never copy-through
        }
    }

    // 3. Replay summary.
    let duration_ms = start.elapsed().as_millis() as u64;
    let baseline_id = baseline_root.id.clone();
    let summary = Record::new(
        Kind::ReplaySummary,
        json!({
            "baseline_trace_id": baseline_id,
            "backend_id": backend.id(),
            "input_count": input_count,
            "output_count": output_count,
            "error_count": error_count,
            "duration_ms": duration_ms,
        }),
        clock.now_iso(),
        Some(last_parent),
    );
    out.push(summary);

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::mock::MockLlm;
    use serde_json::json;

    fn baseline_trace() -> Vec<Record> {
        let meta = Record::new(
            Kind::Metadata,
            json!({"sdk": {"name": "shadow", "version": "0.1.0"}, "tags": {"env": "demo"}}),
            "2026-04-21T10:00:00Z",
            None,
        );
        let req = Record::new(
            Kind::ChatRequest,
            json!({"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}], "params": {}}),
            "2026-04-21T10:00:00.100Z",
            Some(meta.id.clone()),
        );
        let resp = Record::new(
            Kind::ChatResponse,
            json!({"model": "claude-opus-4-7", "content": [{"text": "Hello!", "type": "text"}], "stop_reason": "end_turn", "latency_ms": 100, "usage": {"input_tokens": 5, "output_tokens": 3, "thinking_tokens": 0}}),
            "2026-04-21T10:00:00.500Z",
            Some(req.id.clone()),
        );
        vec![meta, req, resp]
    }

    #[tokio::test]
    async fn happy_path_produces_parallel_structure() {
        let baseline = baseline_trace();
        let backend = MockLlm::from_trace(&baseline);
        let clock = FixedClock::new("2026-04-22T00:00:00Z");

        let candidate = run_replay(&baseline, &backend, &clock).await.unwrap();

        // Expected structure: metadata, chat_request, chat_response, replay_summary.
        assert_eq!(candidate.len(), 4);
        assert_eq!(candidate[0].kind, Kind::Metadata);
        assert_eq!(candidate[1].kind, Kind::ChatRequest);
        assert_eq!(candidate[2].kind, Kind::ChatResponse);
        assert_eq!(candidate[3].kind, Kind::ReplaySummary);
    }

    #[tokio::test]
    async fn parent_chain_is_monotonically_linked() {
        let baseline = baseline_trace();
        let backend = MockLlm::from_trace(&baseline);
        let clock = FixedClock::new("ts");
        let candidate = run_replay(&baseline, &backend, &clock).await.unwrap();

        // First record is root (parent=None).
        assert!(candidate[0].parent.is_none());
        // Every subsequent record's parent must be an earlier record's id.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        seen.insert(candidate[0].id.clone());
        for r in &candidate[1..] {
            let parent = r.parent.as_ref().expect("non-root must have parent");
            assert!(
                seen.contains(parent),
                "unknown parent {parent} for {:?}",
                r.kind
            );
            seen.insert(r.id.clone());
        }
    }

    #[tokio::test]
    async fn baseline_of_pointer_survives_in_candidate_metadata() {
        let baseline = baseline_trace();
        let backend = MockLlm::from_trace(&baseline);
        let clock = FixedClock::new("ts");
        let candidate = run_replay(&baseline, &backend, &clock).await.unwrap();

        let meta = &candidate[0];
        let baseline_of = meta.payload.get("baseline_of").and_then(|v| v.as_str());
        assert_eq!(baseline_of, Some(baseline[0].id.as_str()));
    }

    #[tokio::test]
    async fn summary_reports_correct_counts() {
        let baseline = baseline_trace();
        let backend = MockLlm::from_trace(&baseline);
        let clock = FixedClock::new("ts");
        let candidate = run_replay(&baseline, &backend, &clock).await.unwrap();

        let summary = candidate.last().unwrap();
        assert_eq!(summary.kind, Kind::ReplaySummary);
        assert_eq!(
            summary.payload.get("input_count").unwrap().as_u64(),
            Some(1)
        );
        assert_eq!(
            summary.payload.get("output_count").unwrap().as_u64(),
            Some(1)
        );
        assert_eq!(
            summary.payload.get("error_count").unwrap().as_u64(),
            Some(0)
        );
        assert_eq!(
            summary.payload.get("backend_id").unwrap().as_str(),
            Some("mock")
        );
    }

    #[tokio::test]
    async fn empty_baseline_errors() {
        let backend = MockLlm::from_trace(&[]);
        let clock = FixedClock::new("ts");
        let err = run_replay(&[], &backend, &clock).await.unwrap_err();
        assert!(matches!(err, ReplayError::EmptyBaseline));
    }

    #[tokio::test]
    async fn non_metadata_root_errors() {
        let req = Record::new(
            Kind::ChatRequest,
            json!({"model": "x"}),
            "2026-04-21T10:00:00Z",
            None,
        );
        let backend = MockLlm::from_trace(&[]);
        let clock = FixedClock::new("ts");
        let err = run_replay(&[req], &backend, &clock).await.unwrap_err();
        match err {
            ReplayError::BadBaselineRoot { found } => assert_eq!(found, Kind::ChatRequest),
            other => panic!("expected BadBaselineRoot, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn missing_response_is_captured_as_error_record() {
        // Baseline has two requests; mock only knows about one.
        let mut baseline = baseline_trace();
        // Add a second request+response, then drop the response.
        let extra_req = Record::new(
            Kind::ChatRequest,
            json!({"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "second"}], "params": {}}),
            "2026-04-21T10:01:00Z",
            Some(baseline[2].id.clone()),
        );
        baseline.push(extra_req);

        let backend = MockLlm::from_trace(&baseline); // only first req has a response
        let clock = FixedClock::new("ts");
        let candidate = run_replay(&baseline, &backend, &clock).await.unwrap();

        // Count error records in the candidate.
        let errors: Vec<_> = candidate.iter().filter(|r| r.kind == Kind::Error).collect();
        assert_eq!(errors.len(), 1);
        assert_eq!(
            errors[0].payload.get("source").and_then(|v| v.as_str()),
            Some("llm")
        );
        // Summary counts the error.
        let summary = candidate.last().unwrap();
        assert_eq!(
            summary.payload.get("error_count").unwrap().as_u64(),
            Some(1)
        );
        assert_eq!(
            summary.payload.get("input_count").unwrap().as_u64(),
            Some(2)
        );
    }

    #[tokio::test]
    async fn tool_records_are_copied_through() {
        let meta = Record::new(
            Kind::Metadata,
            json!({"sdk": {"name": "shadow"}}),
            "2026-04-21T10:00:00Z",
            None,
        );
        let req = Record::new(
            Kind::ChatRequest,
            json!({"model": "x", "messages": [], "params": {}}),
            "2026-04-21T10:00:00.100Z",
            Some(meta.id.clone()),
        );
        let resp = Record::new(
            Kind::ChatResponse,
            json!({"model": "x", "content": [], "stop_reason": "tool_use", "latency_ms": 1, "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0}}),
            "2026-04-21T10:00:00.500Z",
            Some(req.id.clone()),
        );
        let tool_call = Record::new(
            Kind::ToolCall,
            json!({"tool_name": "search", "tool_call_id": "t1", "arguments": {}}),
            "2026-04-21T10:00:00.600Z",
            Some(resp.id.clone()),
        );
        let tool_result = Record::new(
            Kind::ToolResult,
            json!({"tool_call_id": "t1", "output": "done", "is_error": false, "latency_ms": 10}),
            "2026-04-21T10:00:00.700Z",
            Some(tool_call.id.clone()),
        );
        let baseline = vec![meta, req, resp, tool_call, tool_result];
        let backend = MockLlm::from_trace(&baseline);
        let clock = FixedClock::new("ts");
        let candidate = run_replay(&baseline, &backend, &clock).await.unwrap();
        let kinds: Vec<Kind> = candidate.iter().map(|r| r.kind).collect();
        assert_eq!(
            kinds,
            vec![
                Kind::Metadata,
                Kind::ChatRequest,
                Kind::ChatResponse,
                Kind::ToolCall,
                Kind::ToolResult,
                Kind::ReplaySummary,
            ]
        );
    }
}
