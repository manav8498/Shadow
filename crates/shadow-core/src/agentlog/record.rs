//! Record envelope types (SPEC §3 + §4).
//!
//! The envelope is statically typed; the payload is held as a
//! [`serde_json::Value`] at this layer. Typed payload structs (`ChatRequest`,
//! `ChatResponse`, etc.) are added alongside the modules that consume them
//! (replay, diff) — they're convenience wrappers over `payload.into_typed()`
//! rather than the wire format.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::agentlog::hash;

/// The current `.agentlog` schema version (SPEC §1, §13).
pub const CURRENT_VERSION: &str = "0.1";

/// One record in an `.agentlog` file (SPEC §3.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Record {
    /// `.agentlog` schema version; must equal [`CURRENT_VERSION`] for records
    /// created by this implementation. Strict parsers reject other values.
    pub version: String,

    /// Content id: `"sha256:" + hex(sha256(canonical_json(payload)))` (SPEC §6).
    pub id: String,

    /// Record kind (SPEC §4).
    pub kind: Kind,

    /// RFC 3339 UTC timestamp with millisecond precision, e.g. `"2026-04-21T10:00:00.100Z"`.
    /// Always ends in `Z` (no numeric offset) — SPEC §3.1.
    pub ts: String,

    /// Parent record id, or `None` if this is the root (`metadata`) record.
    pub parent: Option<String>,

    /// Free-form envelope metadata. Not part of the content hash.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub meta: Option<Map<String, Value>>,

    /// The kind-specific body; SHA-256 of the canonical form of this field
    /// is the `id`.
    pub payload: Value,
}

/// Record kind discriminator (SPEC §4).
///
/// Serializes as lowercase `snake_case` to match the JSON wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Trace-level metadata; always first record, `parent: null` (SPEC §4.5).
    Metadata,
    /// Request sent to an LLM (SPEC §4.1).
    ChatRequest,
    /// Response from an LLM (SPEC §4.2).
    ChatResponse,
    /// Agent-side tool dispatch (SPEC §4.3).
    ToolCall,
    /// Tool execution result (SPEC §4.4).
    ToolResult,
    /// Error event (SPEC §4.6).
    Error,
    /// End-of-replay summary (SPEC §4.7).
    ReplaySummary,
    /// Single streaming-LLM chunk (SPEC §4.8, v0.2).
    Chunk,
    /// Framework-level harness event (SPEC §4.9, v0.2).
    HarnessEvent,
    /// Content-addressed blob reference (SPEC §4.10, v0.2).
    BlobRef,
}

impl Record {
    /// Build a new record, computing `id` from the canonical-JSON of `payload`.
    ///
    /// The `version` field is always set to [`CURRENT_VERSION`]; the caller
    /// provides `kind`, `payload`, `ts`, and `parent`. `meta` is `None` — set
    /// it via [`Record::with_meta`] if needed.
    pub fn new(kind: Kind, payload: Value, ts: impl Into<String>, parent: Option<String>) -> Self {
        let id = hash::content_id(&payload);
        Self {
            version: CURRENT_VERSION.to_string(),
            id,
            kind,
            ts: ts.into(),
            parent,
            meta: None,
            payload,
        }
    }

    /// Attach a `meta` map (builder-style).
    pub fn with_meta(mut self, meta: Map<String, Value>) -> Self {
        self.meta = Some(meta);
        self
    }

    /// Re-compute the id from the payload and compare. Returns `true` if the
    /// record has not been tampered with since it was created.
    pub fn verify_id(&self) -> bool {
        self.id == hash::content_id(&self.payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_payload() -> Value {
        json!({ "model": "claude-opus-4-7", "messages": [] })
    }

    #[test]
    fn new_computes_id_from_payload() {
        let r = Record::new(
            Kind::ChatRequest,
            sample_payload(),
            "2026-04-21T10:00:00Z",
            None,
        );
        assert_eq!(r.id, hash::content_id(&sample_payload()));
        assert_eq!(r.version, "0.1");
        assert_eq!(r.kind, Kind::ChatRequest);
        assert_eq!(r.parent, None);
        assert!(r.meta.is_none());
    }

    #[test]
    fn verify_id_is_true_for_untampered_record() {
        let r = Record::new(
            Kind::Metadata,
            json!({"sdk":{"name":"shadow","version":"0.1.0"}}),
            "2026-04-21T10:00:00Z",
            None,
        );
        assert!(r.verify_id());
    }

    #[test]
    fn verify_id_is_false_if_payload_tampered() {
        let mut r = Record::new(
            Kind::ChatRequest,
            sample_payload(),
            "2026-04-21T10:00:00Z",
            None,
        );
        // Tamper: swap the payload without recomputing id.
        r.payload = json!({ "model": "different" });
        assert!(!r.verify_id());
    }

    #[test]
    fn kind_serializes_snake_case() {
        let json = serde_json::to_string(&Kind::ChatRequest).unwrap();
        assert_eq!(json, r#""chat_request""#);
        let kind: Kind = serde_json::from_str(r#""replay_summary""#).unwrap();
        assert_eq!(kind, Kind::ReplaySummary);
    }

    #[test]
    fn all_kinds_roundtrip() {
        for kind in [
            Kind::Metadata,
            Kind::ChatRequest,
            Kind::ChatResponse,
            Kind::ToolCall,
            Kind::ToolResult,
            Kind::Error,
            Kind::ReplaySummary,
            Kind::Chunk,
            Kind::HarnessEvent,
            Kind::BlobRef,
        ] {
            let s = serde_json::to_string(&kind).unwrap();
            let back: Kind = serde_json::from_str(&s).unwrap();
            assert_eq!(kind, back);
        }
    }

    #[test]
    fn record_roundtrips_through_serde_json() {
        let original = Record::new(
            Kind::ChatRequest,
            sample_payload(),
            "2026-04-21T10:00:00.100Z",
            Some("sha256:abc".to_string()),
        );
        let wire = serde_json::to_string(&original).unwrap();
        let back: Record = serde_json::from_str(&wire).unwrap();
        assert_eq!(original, back);
    }

    #[test]
    fn meta_is_omitted_when_none() {
        // Use Kind::ChatRequest to avoid the substring "metadata" from the kind
        // confusing a naive check against the field name "meta".
        let r = Record::new(
            Kind::ChatRequest,
            json!({"model": "x"}),
            "2026-04-21T10:00:00Z",
            None,
        );
        let wire = serde_json::to_string(&r).unwrap();
        assert!(!wire.contains(r#""meta""#), "wire = {wire}");
    }

    #[test]
    fn meta_survives_roundtrip_when_set() {
        let mut meta = Map::new();
        meta.insert("session_tag".to_string(), json!("prod-agent-0"));
        let r = Record::new(Kind::Metadata, json!({}), "2026-04-21T10:00:00Z", None)
            .with_meta(meta.clone());
        let wire = serde_json::to_string(&r).unwrap();
        let back: Record = serde_json::from_str(&wire).unwrap();
        assert_eq!(back.meta, Some(meta));
    }

    #[test]
    fn new_is_independent_of_provided_ts_and_parent() {
        // Two records with the same payload but different ts/parent MUST
        // have the same id (payload-only hashing, SPEC §6.1).
        let p = sample_payload();
        let a = Record::new(Kind::ChatRequest, p.clone(), "2026-04-21T10:00:00Z", None);
        let b = Record::new(
            Kind::ChatRequest,
            p.clone(),
            "2026-12-31T23:59:59Z",
            Some("sha256:parent".to_string()),
        );
        assert_eq!(a.id, b.id);
    }
}
