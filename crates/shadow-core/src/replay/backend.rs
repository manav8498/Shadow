//! The [`LlmBackend`] trait — one `complete` call maps a `chat_request`
//! payload to a `chat_response` payload.
//!
//! Keeping the trait payload-in / payload-out (rather than record-in /
//! record-out) lets the engine own the envelope (ts, parent, id) while the
//! backend only implements the LLM call itself. See CLAUDE.md §Replay
//! lifecycle.

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;

/// Errors a backend may return.
#[derive(Debug, Error)]
pub enum LlmError {
    /// The backend has no recorded response for this request (MockLlm strict).
    #[error("no recorded response for request id {0}\nhint: either re-record the baseline or run with --backend live")]
    MissingResponse(String),

    /// An I/O failure while talking to a live backend.
    #[error(
        "io error talking to LLM: {0}\nhint: check network connectivity and provider credentials"
    )]
    Io(String),

    /// The request payload failed shape validation.
    #[error("invalid request payload: {0}\nhint: check that the payload matches SPEC §4.1")]
    BadRequest(String),

    /// The backend is misconfigured.
    #[error(
        "backend misconfigured: {0}\nhint: see the backend's documentation for required fields"
    )]
    Config(String),
}

/// Pluggable LLM backend for replay.
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// Given a `chat_request` payload, return the corresponding
    /// `chat_response` payload.
    async fn complete(&self, request: &Value) -> Result<Value, LlmError>;

    /// Stable identifier for this backend (e.g. `"mock"`, `"anthropic"`).
    /// Propagated into replay summaries.
    fn id(&self) -> &str;
}
