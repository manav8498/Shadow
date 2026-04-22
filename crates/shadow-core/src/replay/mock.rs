//! [`MockLlm`] — deterministic backend that replays recorded responses.
//!
//! Given a baseline trace, MockLlm indexes every `chat_response` record by
//! its parent `chat_request` id and serves it back on demand. Because the
//! request id is `sha256(canonical_json(payload))` (SPEC §6), a request
//! with the same payload as one in the baseline always hits the mock — no
//! "fuzzy matching" or fallback is attempted in strict mode.
//!
//! Use this in CI, tests, and the offline demo. For running new
//! configurations against live providers, see future backends in
//! `python/src/shadow/llm/`.

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use crate::agentlog::{Kind, Record};
use crate::replay::backend::{LlmBackend, LlmError};

/// Deterministic backend that replays recorded responses.
pub struct MockLlm {
    id: String,
    /// request_id → response payload.
    responses: HashMap<String, Value>,
}

impl MockLlm {
    /// Build from a baseline trace.
    pub fn from_trace(trace: &[Record]) -> Self {
        let mut responses = HashMap::new();
        for record in trace {
            if record.kind == Kind::ChatResponse {
                if let Some(parent_id) = &record.parent {
                    responses.insert(parent_id.clone(), record.payload.clone());
                }
            }
        }
        Self {
            id: "mock".to_string(),
            responses,
        }
    }

    /// Build from multiple traces (trace set).
    pub fn from_traces<'a, I: IntoIterator<Item = &'a [Record]>>(traces: I) -> Self {
        let mut responses = HashMap::new();
        for trace in traces {
            for record in trace {
                if record.kind == Kind::ChatResponse {
                    if let Some(parent_id) = &record.parent {
                        responses.insert(parent_id.clone(), record.payload.clone());
                    }
                }
            }
        }
        Self {
            id: "mock".to_string(),
            responses,
        }
    }

    /// Override the backend's `id()` string (useful when running multiple
    /// Mock variants in the same session).
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Number of request→response pairs the mock knows about.
    pub fn len(&self) -> usize {
        self.responses.len()
    }

    /// Whether the mock has no recorded responses at all.
    pub fn is_empty(&self) -> bool {
        self.responses.is_empty()
    }
}

#[async_trait]
impl LlmBackend for MockLlm {
    async fn complete(&self, request: &Value) -> Result<Value, LlmError> {
        let request_id = crate::agentlog::hash::content_id(request);
        self.responses
            .get(&request_id)
            .cloned()
            .ok_or(LlmError::MissingResponse(request_id))
    }

    fn id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::{hash, Kind, Record};
    use serde_json::json;

    fn tiny_trace() -> Vec<Record> {
        let meta = Record::new(
            Kind::Metadata,
            json!({"sdk": {"name": "shadow", "version": "0.1.0"}}),
            "2026-04-21T10:00:00Z",
            None,
        );
        let req_payload = json!({"model": "claude-opus-4-7", "messages": [], "params": {}});
        let req = Record::new(
            Kind::ChatRequest,
            req_payload.clone(),
            "2026-04-21T10:00:00.100Z",
            Some(meta.id.clone()),
        );
        let resp = Record::new(
            Kind::ChatResponse,
            json!({"model": "claude-opus-4-7", "content": [{"text": "Hi!", "type": "text"}], "stop_reason": "end_turn", "latency_ms": 1, "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0}}),
            "2026-04-21T10:00:00.500Z",
            Some(req.id.clone()),
        );
        vec![meta, req, resp]
    }

    #[tokio::test]
    async fn recorded_request_returns_recorded_response() {
        let trace = tiny_trace();
        let req_payload = trace[1].payload.clone();
        let expected_resp_payload = trace[2].payload.clone();

        let mock = MockLlm::from_trace(&trace);
        assert_eq!(mock.len(), 1);
        assert_eq!(mock.id(), "mock");

        let got = mock.complete(&req_payload).await.unwrap();
        assert_eq!(got, expected_resp_payload);
    }

    #[tokio::test]
    async fn unrecorded_request_returns_missing_error() {
        let trace = tiny_trace();
        let mock = MockLlm::from_trace(&trace);
        let unknown = json!({"model": "gpt-5", "messages": [], "params": {}});
        let unknown_id = hash::content_id(&unknown);
        match mock.complete(&unknown).await {
            Err(LlmError::MissingResponse(id)) => assert_eq!(id, unknown_id),
            other => panic!("expected MissingResponse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn key_by_content_id_collapses_identical_payloads() {
        // Two request-response pairs with byte-identical request payloads
        // but different timestamps — the mock should deduplicate to one
        // entry by content id.
        let trace = tiny_trace();
        let mut extended = trace.clone();
        let req2 = Record::new(
            Kind::ChatRequest,
            trace[1].payload.clone(),
            "2026-04-21T11:00:00Z",
            Some(trace[0].id.clone()),
        );
        let resp2 = Record::new(
            Kind::ChatResponse,
            trace[2].payload.clone(),
            "2026-04-21T11:00:00.500Z",
            Some(req2.id.clone()),
        );
        extended.push(req2);
        extended.push(resp2);
        let mock = MockLlm::from_trace(&extended);
        assert_eq!(mock.len(), 1); // collapsed
    }

    #[tokio::test]
    async fn from_traces_merges_multiple_sources() {
        let t1 = tiny_trace();
        // Build a second trace with a different request.
        let meta2 = Record::new(
            Kind::Metadata,
            json!({"sdk": {"name": "shadow", "version": "0.1.0"}, "tags": {"env": "other"}}),
            "2026-04-21T12:00:00Z",
            None,
        );
        let req2 = Record::new(
            Kind::ChatRequest,
            json!({"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}], "params": {}}),
            "2026-04-21T12:00:00.100Z",
            Some(meta2.id.clone()),
        );
        let resp2 = Record::new(
            Kind::ChatResponse,
            json!({"model": "claude-opus-4-7", "content": [{"text": "hello", "type": "text"}], "stop_reason": "end_turn", "latency_ms": 1, "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0}}),
            "2026-04-21T12:00:00.500Z",
            Some(req2.id.clone()),
        );
        let t2 = vec![meta2, req2, resp2];
        let mock = MockLlm::from_traces([t1.as_slice(), t2.as_slice()]);
        assert_eq!(mock.len(), 2);
    }

    #[tokio::test]
    async fn empty_trace_produces_empty_mock() {
        let mock = MockLlm::from_trace(&[]);
        assert!(mock.is_empty());
    }

    #[tokio::test]
    async fn with_id_overrides_default() {
        let mock = MockLlm::from_trace(&[]).with_id("my-mock");
        assert_eq!(mock.id(), "my-mock");
    }
}
