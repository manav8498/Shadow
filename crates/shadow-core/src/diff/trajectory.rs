//! Axis 2: tool-call trajectory divergence.
//!
//! For each response, extract the sequence of tool-call tokens that
//! capture both the **structural shape** (tool name + sorted arg keys)
//! AND the **argument values** (8-byte digest of canonical-JSON
//! input). Compare baseline vs candidate sequences with Levenshtein
//! edit distance. Normalize by max(len(baseline_seq), len(candidate_seq))
//! so the metric is in [0, 1].
//!
//! Why include the value digest: a sequence that calls the same tools
//! in the same order with different argument values is a real
//! behavioural change (e.g. `delete_user(id="alice")` vs
//! `delete_user(id="bob")`). Without the value digest the per-axis
//! trajectory metric reports zero divergence on this case — even
//! though the alignment-based first-divergence detector picks it up
//! via its W_ARGS component. The value digest brings the per-axis
//! number in line with the alignment finding.
//!
//! The digest is the leading 8 bytes (16 hex chars) of SHA-256 over
//! the canonical-JSON serialisation of the `input` object. Birthday-
//! paradox collision probability at 16 hex chars is ~1.8e-10 for 1000
//! tool calls — negligible for any realistic agent trace.
//!
//! ## Coverage cross-references
//!
//! What this axis catches:
//! - Tool added / dropped / reordered (structural)
//! - Tool argument keys added / dropped (schema)
//! - Tool argument values changed (digest mismatch, v2.7+)
//!
//! What it does NOT catch:
//! - **Same tool sequence + same arg values + different RESPONSE
//!   text** — that's a content regression visible on the semantic
//!   axis (axis 1) and via the v2.7+ `text_chars_log` /
//!   `numeric_token_density` / `error_token_flag` dimensions of
//!   `shadow.statistical.fingerprint` (Hotelling T²).
//! - **Tool sequence policy violations** ("verify before refund",
//!   "no execute_sql without preview") — the LTLf checker
//!   (`shadow.ltl`) with `must_call_before` / `no_call` rules.
//! - **First moment of regression** — the alignment module
//!   (`shadow_core::diff::alignment`) walks both traces and points
//!   to the exact turn where divergence began, with kind
//!   classification (Structural / Decision / Style).

use sha2::{Digest, Sha256};

use crate::agentlog::Record;
use crate::diff::axes::{Axis, AxisStat};
use crate::diff::bootstrap::{median, paired_ci};

/// Length of the argument-value digest, in hex characters. 16 hex
/// chars = 64 bits = ~1.8e-10 birthday collision probability at n=1000.
const ARG_VALUE_DIGEST_HEX_LEN: usize = 16;

fn arg_value_digest(input: &serde_json::Value) -> String {
    // Canonical JSON via the `agentlog::canonical` writer would be
    // ideal; for a per-tool-call digest the simpler `serde_json::to_vec`
    // is sufficient as long as we sort keys first. We do that by walking
    // the value into a BTreeMap-backed structure before serialising.
    let canonical = canonicalise(input);
    let bytes = serde_json::to_vec(&canonical).unwrap_or_default();
    let mut h = Sha256::new();
    h.update(&bytes);
    let digest = h.finalize();
    let hex: String = digest
        .iter()
        .take(ARG_VALUE_DIGEST_HEX_LEN / 2)
        .map(|b| format!("{b:02x}"))
        .collect();
    hex
}

fn canonicalise(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut sorted: std::collections::BTreeMap<String, serde_json::Value> =
                std::collections::BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k.clone(), canonicalise(v));
            }
            serde_json::Value::Object(sorted.into_iter().collect())
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(canonicalise).collect())
        }
        other => other.clone(),
    }
}

fn tool_shape(r: &Record) -> Vec<String> {
    let content = match r.payload.get("content").and_then(|c| c.as_array()) {
        Some(arr) => arr,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    for part in content {
        if part.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
            let name = part
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("_")
                .to_string();
            let input = part.get("input").cloned().unwrap_or(serde_json::Value::Null);
            let mut keys: Vec<String> = input
                .as_object()
                .map(|o| o.keys().cloned().collect())
                .unwrap_or_default();
            keys.sort();
            let value_digest = arg_value_digest(&input);
            out.push(format!("{name}({}|{value_digest})", keys.join(",")));
        }
    }
    out
}

fn levenshtein(a: &[String], b: &[String]) -> usize {
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

fn normalized_divergence(b: &[String], c: &[String]) -> f64 {
    let denom = b.len().max(c.len());
    if denom == 0 {
        0.0
    } else {
        levenshtein(b, c) as f64 / denom as f64
    }
}

/// Compute the tool-trajectory axis.
pub fn compute(pairs: &[(&Record, &Record)], seed: Option<u64>) -> AxisStat {
    if pairs.is_empty() {
        return AxisStat::empty(Axis::Trajectory);
    }
    let baseline_zero: Vec<f64> = (0..pairs.len()).map(|_| 0.0).collect();
    let divergence: Vec<f64> = pairs
        .iter()
        .map(|(b, c)| normalized_divergence(&tool_shape(b), &tool_shape(c)))
        .collect();
    let bm = median(&baseline_zero);
    let cm = median(&divergence);
    let delta = cm - bm;
    let ci = paired_ci(
        &baseline_zero,
        &divergence,
        |bs, cs| median(cs) - median(bs),
        0,
        seed,
    );
    // Trajectory is a rate in [0, 1] measured *from* zero (identical
    // sequences → 0 divergence). The relative-delta severity used by
    // `new_value` divides by baseline_median=0.0 and always returns
    // Minor, regardless of magnitude. `new_rate` uses absolute-delta
    // thresholds, which is the honest classification for this axis.
    AxisStat::new_rate(
        Axis::Trajectory,
        bm,
        cm,
        delta,
        ci.low,
        ci.high,
        pairs.len(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::Kind;
    use serde_json::json;

    fn response_with_tools(tools: &[(&str, &[&str])]) -> Record {
        let content: Vec<serde_json::Value> = tools
            .iter()
            .map(|(name, keys)| {
                let input: serde_json::Map<String, serde_json::Value> = keys
                    .iter()
                    .map(|k| ((*k).to_string(), json!("v")))
                    .collect();
                json!({
                    "type": "tool_use",
                    "id": format!("t_{name}"),
                    "name": name,
                    "input": input,
                })
            })
            .collect();
        Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": content,
                "stop_reason": "tool_use",
                "latency_ms": 0,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        )
    }

    #[test]
    fn identical_tool_shapes_produce_zero_divergence() {
        let r = response_with_tools(&[("search_files", &["query"])]);
        let pairs = [(&r, &r)];
        let stat = compute(&pairs, Some(1));
        assert_eq!(stat.candidate_median, 0.0);
    }

    #[test]
    fn schema_change_on_same_tool_is_divergence() {
        let baseline = response_with_tools(&[("search_files", &["query"])]);
        // Candidate adds a `limit` key.
        let candidate = response_with_tools(&[("search_files", &["query", "limit"])]);
        let pairs = [(&baseline, &candidate); 10];
        let stat = compute(&pairs, Some(2));
        assert!(stat.candidate_median > 0.0);
    }

    /// Pre-fix bug: same tool, same arg KEYS, different arg VALUES
    /// produced 0 divergence on the trajectory axis. The alignment
    /// module caught it via W_ARGS, but the per-axis number lied.
    #[test]
    fn arg_value_change_on_same_tool_is_divergence() {
        let baseline = Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": [{
                    "type": "tool_use",
                    "id": "t1",
                    "name": "delete_user",
                    "input": {"id": "alice"},
                }],
                "stop_reason": "tool_use",
                "latency_ms": 0,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        );
        let candidate = Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": [{
                    "type": "tool_use",
                    "id": "t1",
                    "name": "delete_user",
                    "input": {"id": "bob"},
                }],
                "stop_reason": "tool_use",
                "latency_ms": 0,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        );
        let pairs = [(&baseline, &candidate); 10];
        let stat = compute(&pairs, Some(3));
        assert!(
            stat.candidate_median > 0.0,
            "trajectory axis must register a value change as divergence; \
             got candidate_median = {}",
            stat.candidate_median,
        );
    }

    #[test]
    fn identical_arg_values_score_zero_even_with_complex_inputs() {
        // Same tool, same nested structured input → zero divergence.
        let r = Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": [{
                    "type": "tool_use",
                    "id": "t1",
                    "name": "execute",
                    "input": {
                        "query": "SELECT * FROM users",
                        "params": {"limit": 10, "offset": 0},
                    },
                }],
                "stop_reason": "tool_use",
                "latency_ms": 0,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        );
        let pairs = [(&r, &r)];
        let stat = compute(&pairs, Some(4));
        assert_eq!(stat.candidate_median, 0.0);
    }

    /// Canonicalisation: object key order in the input must NOT cause
    /// a spurious value-change divergence.
    #[test]
    fn arg_key_order_is_canonicalised() {
        let baseline = Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": [{
                    "type": "tool_use",
                    "id": "t1",
                    "name": "log",
                    "input": {"level": "info", "msg": "hello"},
                }],
                "stop_reason": "tool_use",
                "latency_ms": 0,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        );
        let candidate = Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": [{
                    "type": "tool_use",
                    "id": "t1",
                    "name": "log",
                    // Same content, different key order.
                    "input": {"msg": "hello", "level": "info"},
                }],
                "stop_reason": "tool_use",
                "latency_ms": 0,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        );
        let pairs = [(&baseline, &candidate); 5];
        let stat = compute(&pairs, Some(5));
        assert_eq!(stat.candidate_median, 0.0);
    }

    #[test]
    fn levenshtein_basic() {
        let a = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let b = vec!["a".to_string(), "x".to_string(), "c".to_string()];
        assert_eq!(levenshtein(&a, &b), 1);
    }
}
