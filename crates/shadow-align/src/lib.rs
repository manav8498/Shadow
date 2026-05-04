//! Reusable trace-comparison primitives for AI agents.
//!
//! Standalone Rust port of `shadow.align` (Python) and
//! `@shadow-diff/align` (TypeScript). Same five-function surface,
//! same algorithms, byte-identical results on shared inputs.
//!
//! All five functions are now fully native (v0.2):
//!
//! - [`trajectory_distance`] — Levenshtein on flat tool sequences.
//! - [`tool_arg_delta`] — structural diff of two `serde_json::Value`s.
//! - [`align_traces`] — index-based pairing of `.agentlog` records,
//!   with insertions / deletions on asymmetric pair counts.
//! - [`first_divergence`] — first non-zero divergence in alignment
//!   order. `structural_drift_full` for asymmetric corpora,
//!   `structural_drift` for tool-sequence drift,
//!   `decision_drift` for response-text drift.
//! - [`top_k_divergences`] — ranked divergence list (top-K by
//!   confidence).
//!
//! These mirror the algorithms in `shadow.align` (Python) and the
//! pure-TS path in `@shadow-diff/align`. The native shadow-core
//! 9-axis differ is more thorough (it factors embedding similarity,
//! latency CDFs, etc.); for the wedge use cases (regression
//! detection, tool-trajectory drift, argument shape changes) the
//! lightweight functions here produce useful results without taking
//! a shadow-core dependency.
//!
//! # Cross-language parity
//!
//! ```
//! use shadow_align::trajectory_distance;
//! assert_eq!(trajectory_distance(&["a", "b", "c"], &["a", "b", "c"]), 0.0);
//! assert_eq!(trajectory_distance(&["a", "b"], &["x", "y"]), 1.0);
//! ```
//!
//! Same numbers as Python's `shadow.align.trajectory_distance` and
//! TypeScript's `trajectoryDistance`.

#![forbid(unsafe_code)]

use serde_json::Value;

// ---------------------------------------------------------------------------
// Public types — mirror Python dataclasses + TS interfaces.
// ---------------------------------------------------------------------------

/// One paired (baseline_index, candidate_index) entry. Either index
/// can be `None` when one side is missing — that's a structural-drift
/// turn (insertion or deletion).
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedTurn {
    pub baseline_index: Option<usize>,
    pub candidate_index: Option<usize>,
    /// 0.0 = perfect match, 1.0 = pure-gap insertion/deletion.
    pub cost: f64,
}

/// The full alignment of two traces. `total_cost` is the sum of
/// per-turn costs; lower is better.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Alignment {
    pub turns: Vec<AlignedTurn>,
    pub total_cost: f64,
}

/// One identified divergence between baseline and candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct Divergence {
    pub baseline_turn: usize,
    pub candidate_turn: usize,
    pub kind: String,
    pub primary_axis: String,
    pub explanation: String,
    pub confidence: f64,
}

/// Coarse classification of a leaf-level change between two JSON
/// values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgDeltaKind {
    Added,
    Removed,
    Changed,
    TypeChanged,
}

impl ArgDeltaKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::Removed => "removed",
            Self::Changed => "changed",
            Self::TypeChanged => "type_changed",
        }
    }
}

/// One leaf-level change between two JSON values, keyed by a
/// slash-separated JSON-pointer-like path.
#[derive(Debug, Clone, PartialEq)]
pub struct ArgDelta {
    pub path: String,
    pub kind: ArgDeltaKind,
    pub old: Option<Value>,
    pub new: Option<Value>,
}

// ---------------------------------------------------------------------------
// trajectory_distance — pure Rust, no shadow-core dependency.
// ---------------------------------------------------------------------------

/// Levenshtein edit distance between two flat sequences, normalised
/// to `[0.0, 1.0]` by the longer length.
///
/// Returns `0.0` for identical sequences, `1.0` for fully disjoint,
/// and `0.0` for `(empty, empty)`. Equality uses [`PartialEq`] so
/// any comparable element type works.
///
/// Cross-language parity: identical to
/// `shadow.align.trajectory_distance` (Python) and
/// `trajectoryDistance` (TypeScript) on the same inputs.
pub fn trajectory_distance<T: PartialEq>(a: &[T], b: &[T]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let n = a.len();
    let m = b.len();
    // Standard DP Levenshtein. O(n*m) time + memory; for very long
    // sequences we'd switch to a two-row DP, but v0.1 matches the
    // Python implementation exactly for parity.
    // Standard 2-D DP. The needless_range_loop lint suggests
    // enumerate() but the matrix DP reads more clearly with index
    // arithmetic, so we allow it locally.
    #[allow(clippy::needless_range_loop)]
    {
        let mut dp = vec![vec![0_usize; m + 1]; n + 1];
        for i in 0..=n {
            dp[i][0] = i;
        }
        for j in 0..=m {
            dp[0][j] = j;
        }
        for i in 1..=n {
            for j in 1..=m {
                let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }
        dp[n][m] as f64 / n.max(m) as f64
    }
}

// ---------------------------------------------------------------------------
// tool_arg_delta — pure Rust structural diff of two serde_json::Value.
// ---------------------------------------------------------------------------

/// Structural diff between two JSON values. Walks objects, arrays,
/// and scalars; produces typed deltas keyed by slash-separated
/// JSON-pointer paths.
///
/// Cross-language parity: byte-identical paths to
/// `shadow.align.tool_arg_delta` (Python) and `toolArgDelta`
/// (TypeScript) on the same inputs.
pub fn tool_arg_delta(a: &Value, b: &Value) -> Vec<ArgDelta> {
    let mut out = Vec::new();
    walk_arg_delta(a, b, "", &mut out);
    out
}

fn walk_arg_delta(a: &Value, b: &Value, path: &str, out: &mut Vec<ArgDelta>) {
    if a.is_null() && b.is_null() {
        return;
    }
    if a.is_null() {
        out.push(ArgDelta {
            path: pointer(path),
            kind: ArgDeltaKind::Added,
            old: None,
            new: Some(b.clone()),
        });
        return;
    }
    if b.is_null() {
        out.push(ArgDelta {
            path: pointer(path),
            kind: ArgDeltaKind::Removed,
            old: Some(a.clone()),
            new: None,
        });
        return;
    }
    // Per-scalar type discriminator that matches Python's `type()`
    // semantics: number-vs-string-vs-boolean must register as
    // `type_changed`, not `changed`.
    let ta = precise_type(a);
    let tb = precise_type(b);
    if ta != tb {
        out.push(ArgDelta {
            path: pointer(path),
            kind: ArgDeltaKind::TypeChanged,
            old: Some(a.clone()),
            new: Some(b.clone()),
        });
        return;
    }
    if let (Some(a_obj), Some(b_obj)) = (a.as_object(), b.as_object()) {
        let mut keys: Vec<&String> = a_obj.keys().chain(b_obj.keys()).collect();
        keys.sort();
        keys.dedup();
        for k in keys {
            let sub = format!("{path}/{k}");
            match (a_obj.get(k), b_obj.get(k)) {
                (Some(av), Some(bv)) => walk_arg_delta(av, bv, &sub, out),
                (None, Some(bv)) => out.push(ArgDelta {
                    path: sub,
                    kind: ArgDeltaKind::Added,
                    old: None,
                    new: Some(bv.clone()),
                }),
                (Some(av), None) => out.push(ArgDelta {
                    path: sub,
                    kind: ArgDeltaKind::Removed,
                    old: Some(av.clone()),
                    new: None,
                }),
                (None, None) => {}
            }
        }
        return;
    }
    if let (Some(a_arr), Some(b_arr)) = (a.as_array(), b.as_array()) {
        let n_a = a_arr.len();
        let n_b = b_arr.len();
        let min_len = n_a.min(n_b);
        for i in 0..min_len {
            walk_arg_delta(&a_arr[i], &b_arr[i], &format!("{path}/{i}"), out);
        }
        for (i, item) in b_arr.iter().enumerate().skip(n_a) {
            out.push(ArgDelta {
                path: format!("{path}/{i}"),
                kind: ArgDeltaKind::Added,
                old: None,
                new: Some(item.clone()),
            });
        }
        for (i, item) in a_arr.iter().enumerate().skip(n_b) {
            out.push(ArgDelta {
                path: format!("{path}/{i}"),
                kind: ArgDeltaKind::Removed,
                old: Some(item.clone()),
                new: None,
            });
        }
        return;
    }
    if a != b {
        out.push(ArgDelta {
            path: pointer(path),
            kind: ArgDeltaKind::Changed,
            old: Some(a.clone()),
            new: Some(b.clone()),
        });
    }
}

fn precise_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn pointer(path: &str) -> String {
    if path.is_empty() {
        "/".to_string()
    } else {
        path.to_string()
    }
}

// ---------------------------------------------------------------------------
// Trace-pair functions — v0.2 native (no shadow-core dependency).
//
// Records are `serde_json::Value` objects shaped like the
// `.agentlog` envelope: each item is `{ kind, payload, ... }`. We
// walk chat_request / chat_response pairs and use the existing
// `trajectory_distance` + `tool_arg_delta` primitives to detect
// divergences. Algorithm matches the pure-TS path byte-for-byte.
// ---------------------------------------------------------------------------

/// One paired chat turn extracted from a record stream.
struct ChatPair<'a> {
    response: &'a Value,
}

/// Walk a record stream and return paired (chat_request,
/// chat_response) entries in order. Pending requests without a
/// matching response are dropped (mirrors the TS path).
fn extract_chat_pairs(records: &[Value]) -> Vec<ChatPair<'_>> {
    let mut pairs: Vec<ChatPair<'_>> = Vec::new();
    let mut pending = false;
    for rec in records {
        let kind = rec.get("kind").and_then(Value::as_str).unwrap_or("");
        if kind == "chat_request" {
            pending = true;
        } else if kind == "chat_response" && pending {
            pairs.push(ChatPair { response: rec });
            pending = false;
        }
    }
    pairs
}

/// Extract tool_use names from a chat_response's content array.
/// Order-preserving — `["search", "edit"]` is different from
/// `["edit", "search"]` for trajectory_distance.
fn tool_names_from_response(response: &Value) -> Vec<String> {
    let Some(content) = response.get("payload").and_then(|p| p.get("content")) else {
        return Vec::new();
    };
    let Some(arr) = content.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|b| {
            let obj = b.as_object()?;
            if obj.get("type").and_then(Value::as_str)? != "tool_use" {
                return None;
            }
            Some(
                obj.get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            )
        })
        .collect()
}

/// Concatenate all `text` blocks in a chat_response's content. Empty
/// string when there are none.
fn text_from_response(response: &Value) -> String {
    let Some(content) = response.get("payload").and_then(|p| p.get("content")) else {
        return String::new();
    };
    let Some(arr) = content.as_array() else {
        return String::new();
    };
    let parts: Vec<&str> = arr
        .iter()
        .filter_map(|b| {
            let obj = b.as_object()?;
            if obj.get("type").and_then(Value::as_str)? != "text" {
                return None;
            }
            obj.get("text").and_then(Value::as_str)
        })
        .collect();
    parts.join("\n")
}

fn pair_cost(b: &ChatPair<'_>, c: &ChatPair<'_>) -> f64 {
    // Cost = max(toolNameDrift, textDrift). Bounded [0, 1].
    let b_tools = tool_names_from_response(b.response);
    let c_tools = tool_names_from_response(c.response);
    let tool_cost = trajectory_distance(&b_tools, &c_tools);
    let b_text = text_from_response(b.response);
    let c_text = text_from_response(c.response);
    let text_cost = if b_text == c_text {
        0.0
    } else if b_text.is_empty() || c_text.is_empty() {
        1.0
    } else {
        0.5
    };
    tool_cost.max(text_cost)
}

/// Pair every baseline chat turn to its best-match candidate turn.
///
/// v0.2: index-based pairing (turn N to turn N) — the fast path
/// Shadow's 9-axis differ uses for most real-world traces.
/// Asymmetric pair counts emit gap turns with cost 1.0.
///
/// Cross-language parity: identical alignment shape to
/// `shadow.align.align_traces` (Python) and `alignTraces`
/// (TypeScript) on the same `.agentlog` record streams.
pub fn align_traces(baseline: &[Value], candidate: &[Value]) -> Alignment {
    let base_pairs = extract_chat_pairs(baseline);
    let cand_pairs = extract_chat_pairs(candidate);
    let mut turns: Vec<AlignedTurn> = Vec::new();
    let mut total = 0.0_f64;
    let min_len = base_pairs.len().min(cand_pairs.len());
    for i in 0..min_len {
        let cost = pair_cost(&base_pairs[i], &cand_pairs[i]);
        turns.push(AlignedTurn {
            baseline_index: Some(i),
            candidate_index: Some(i),
            cost,
        });
        total += cost;
    }
    for i in min_len..base_pairs.len() {
        turns.push(AlignedTurn {
            baseline_index: Some(i),
            candidate_index: None,
            cost: 1.0,
        });
        total += 1.0;
    }
    for i in min_len..cand_pairs.len() {
        turns.push(AlignedTurn {
            baseline_index: None,
            candidate_index: Some(i),
            cost: 1.0,
        });
        total += 1.0;
    }
    Alignment {
        turns,
        total_cost: total,
    }
}

/// Find the first point at which the two traces meaningfully differ
/// in alignment order, or `None` when they agree end-to-end.
///
/// Emits one of three divergence kinds:
///
/// - `structural_drift_full` — one corpus is empty.
/// - `structural_drift` — tool sequence diverges or pair counts
///   don't match.
/// - `decision_drift` — response text differs while tool sequence
///   matches.
pub fn first_divergence(baseline: &[Value], candidate: &[Value]) -> Option<Divergence> {
    let base_pairs = extract_chat_pairs(baseline);
    let cand_pairs = extract_chat_pairs(candidate);
    if base_pairs.is_empty() && cand_pairs.is_empty() {
        return None;
    }
    if base_pairs.is_empty() || cand_pairs.is_empty() {
        return Some(Divergence {
            baseline_turn: 0,
            candidate_turn: 0,
            kind: "structural_drift_full".to_string(),
            primary_axis: "trajectory".to_string(),
            explanation: format!(
                "asymmetric corpus: baseline has {} chat pair(s), candidate has {}",
                base_pairs.len(),
                cand_pairs.len()
            ),
            confidence: 1.0,
        });
    }
    let min_len = base_pairs.len().min(cand_pairs.len());
    for i in 0..min_len {
        let b_tools = tool_names_from_response(base_pairs[i].response);
        let c_tools = tool_names_from_response(cand_pairs[i].response);
        let tool_dist = trajectory_distance(&b_tools, &c_tools);
        if tool_dist > 0.0 {
            return Some(Divergence {
                baseline_turn: i,
                candidate_turn: i,
                kind: "structural_drift".to_string(),
                primary_axis: "trajectory".to_string(),
                explanation: format!("tool sequence diverged at turn {i} (drift={tool_dist:.2})"),
                confidence: tool_dist.min(1.0),
            });
        }
        let b_text = text_from_response(base_pairs[i].response);
        let c_text = text_from_response(cand_pairs[i].response);
        if b_text != c_text {
            return Some(Divergence {
                baseline_turn: i,
                candidate_turn: i,
                kind: "decision_drift".to_string(),
                primary_axis: "semantic".to_string(),
                explanation: format!("response text diverged at turn {i}"),
                confidence: 0.7,
            });
        }
    }
    if base_pairs.len() != cand_pairs.len() {
        return Some(Divergence {
            baseline_turn: min_len,
            candidate_turn: min_len,
            kind: "structural_drift".to_string(),
            primary_axis: "trajectory".to_string(),
            explanation: format!(
                "pair-count drift: baseline={}, candidate={}",
                base_pairs.len(),
                cand_pairs.len()
            ),
            confidence: 1.0,
        });
    }
    None
}

/// Top-K ranked divergences. v0.2 walks all turns and emits one
/// `Divergence` per non-zero per-pair cost, sorted by confidence
/// descending. Returns at most `k` items.
///
/// Panics when `k < 1` (mirrors the Python and TypeScript surfaces;
/// callers should treat zero/negative k as a programming error).
pub fn top_k_divergences(baseline: &[Value], candidate: &[Value], k: usize) -> Vec<Divergence> {
    assert!(k >= 1, "k must be >= 1, got {k}");
    let base_pairs = extract_chat_pairs(baseline);
    let cand_pairs = extract_chat_pairs(candidate);
    if base_pairs.is_empty() && cand_pairs.is_empty() {
        return Vec::new();
    }
    if base_pairs.is_empty() || cand_pairs.is_empty() {
        let mut out = vec![Divergence {
            baseline_turn: 0,
            candidate_turn: 0,
            kind: "structural_drift_full".to_string(),
            primary_axis: "trajectory".to_string(),
            explanation: format!(
                "asymmetric corpus: baseline={} candidate={}",
                base_pairs.len(),
                cand_pairs.len()
            ),
            confidence: 1.0,
        }];
        out.truncate(k);
        return out;
    }
    let min_len = base_pairs.len().min(cand_pairs.len());
    let mut out: Vec<Divergence> = Vec::new();
    for i in 0..min_len {
        let b_tools = tool_names_from_response(base_pairs[i].response);
        let c_tools = tool_names_from_response(cand_pairs[i].response);
        let tool_dist = trajectory_distance(&b_tools, &c_tools);
        if tool_dist > 0.0 {
            out.push(Divergence {
                baseline_turn: i,
                candidate_turn: i,
                kind: "structural_drift".to_string(),
                primary_axis: "trajectory".to_string(),
                explanation: format!("tool sequence diverged at turn {i} (drift={tool_dist:.2})"),
                confidence: tool_dist.min(1.0),
            });
            continue;
        }
        let b_text = text_from_response(base_pairs[i].response);
        let c_text = text_from_response(cand_pairs[i].response);
        if b_text != c_text {
            out.push(Divergence {
                baseline_turn: i,
                candidate_turn: i,
                kind: "decision_drift".to_string(),
                primary_axis: "semantic".to_string(),
                explanation: format!("response text diverged at turn {i}"),
                confidence: 0.7,
            });
        }
    }
    // Sort by confidence descending (stable). NaN should never appear
    // here — confidence is bounded [0, 1] — but partial_cmp keeps
    // the sort total over the f64 lattice we actually use.
    out.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(k);
    out
}

// ---------------------------------------------------------------------------
// Unit tests — colocated to keep the public-API contract tight.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn trajectory_distance_equal_sequences() {
        assert_eq!(trajectory_distance(&["a", "b", "c"], &["a", "b", "c"]), 0.0);
    }

    #[test]
    fn trajectory_distance_disjoint_sequences() {
        assert_eq!(trajectory_distance(&["a", "b"], &["x", "y"]), 1.0);
    }

    #[test]
    fn trajectory_distance_one_substitution() {
        let d = trajectory_distance(&["a", "b", "c"], &["a", "x", "c"]);
        assert!((d - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn trajectory_distance_both_empty() {
        assert_eq!(trajectory_distance::<&str>(&[], &[]), 0.0);
    }

    #[test]
    fn trajectory_distance_one_empty() {
        assert_eq!(trajectory_distance(&["a"], &[]), 1.0);
    }

    #[test]
    fn tool_arg_delta_equal_no_deltas() {
        let a = json!({"x": 1, "y": 2});
        let b = json!({"x": 1, "y": 2});
        assert!(tool_arg_delta(&a, &b).is_empty());
    }

    #[test]
    fn tool_arg_delta_added_key() {
        let a = json!({"x": 1});
        let b = json!({"x": 1, "y": 2});
        let d = tool_arg_delta(&a, &b);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].path, "/y");
        assert_eq!(d[0].kind, ArgDeltaKind::Added);
    }

    #[test]
    fn tool_arg_delta_removed_key() {
        let a = json!({"x": 1, "y": 2});
        let b = json!({"x": 1});
        let d = tool_arg_delta(&a, &b);
        assert_eq!(d[0].kind, ArgDeltaKind::Removed);
    }

    #[test]
    fn tool_arg_delta_changed_value() {
        let a = json!({"x": 1});
        let b = json!({"x": 2});
        let d = tool_arg_delta(&a, &b);
        assert_eq!(d[0].kind, ArgDeltaKind::Changed);
    }

    #[test]
    fn tool_arg_delta_type_changed() {
        let a = json!({"x": 1});
        let b = json!({"x": "1"});
        let d = tool_arg_delta(&a, &b);
        assert_eq!(d[0].kind, ArgDeltaKind::TypeChanged);
    }

    #[test]
    fn tool_arg_delta_nested_path() {
        let a = json!({"outer": {"inner": "old"}});
        let b = json!({"outer": {"inner": "new"}});
        let d = tool_arg_delta(&a, &b);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].path, "/outer/inner");
    }

    #[test]
    fn tool_arg_delta_list_index() {
        let a = json!([1, 2, 3]);
        let b = json!([1, 9, 3]);
        let d = tool_arg_delta(&a, &b);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].path, "/1");
        assert_eq!(d[0].kind, ArgDeltaKind::Changed);
    }

    #[test]
    fn arg_delta_kind_string_repr_matches_python_and_ts() {
        assert_eq!(ArgDeltaKind::Added.as_str(), "added");
        assert_eq!(ArgDeltaKind::Removed.as_str(), "removed");
        assert_eq!(ArgDeltaKind::Changed.as_str(), "changed");
        assert_eq!(ArgDeltaKind::TypeChanged.as_str(), "type_changed");
    }

    // Helpers — synthesise minimal record streams matching the
    // `.agentlog` envelope shape (kind + payload).
    fn make_pair(tool_name: Option<&str>, text: &str) -> Vec<Value> {
        let content = if let Some(name) = tool_name {
            json!([{"type": "tool_use", "name": name, "input": {}, "id": "1"}])
        } else {
            json!([{"type": "text", "text": text}])
        };
        vec![
            json!({"kind": "chat_request", "payload": {"messages": [{"role": "user", "content": "q"}]}}),
            json!({"kind": "chat_response", "payload": {"content": content}}),
        ]
    }

    #[test]
    fn align_traces_empty_returns_empty_alignment() {
        let aln = align_traces(&[], &[]);
        assert!(aln.turns.is_empty());
        assert_eq!(aln.total_cost, 0.0);
    }

    #[test]
    fn align_traces_pairs_index_by_index_for_equal_length_inputs() {
        let mut a = make_pair(Some("search"), "");
        a.extend(make_pair(Some("summarize"), ""));
        let aln = align_traces(&a, &a);
        assert_eq!(aln.turns.len(), 2);
        assert_eq!(aln.total_cost, 0.0);
    }

    #[test]
    fn align_traces_emits_gap_turns_on_asymmetric_pair_counts() {
        let a = make_pair(Some("search"), "");
        let mut b = make_pair(Some("search"), "");
        b.extend(make_pair(Some("extra"), ""));
        let aln = align_traces(&a, &b);
        assert_eq!(aln.turns.len(), 2);
        assert_eq!(aln.turns[1].baseline_index, None);
        assert_eq!(aln.turns[1].candidate_index, Some(1));
        assert_eq!(aln.turns[1].cost, 1.0);
    }

    #[test]
    fn first_divergence_identical_traces_returns_none() {
        let recs = make_pair(Some("search"), "");
        assert!(first_divergence(&recs, &recs).is_none());
    }

    #[test]
    fn first_divergence_asymmetric_corpus_is_structural_drift_full() {
        let a = make_pair(Some("search"), "");
        let fd = first_divergence(&a, &[]).expect("divergence expected");
        assert_eq!(fd.kind, "structural_drift_full");
        assert_eq!(fd.primary_axis, "trajectory");
        assert_eq!(fd.confidence, 1.0);
    }

    #[test]
    fn first_divergence_tool_drift_is_structural_drift() {
        let a = make_pair(Some("search"), "");
        let b = make_pair(Some("summarize"), "");
        let fd = first_divergence(&a, &b).expect("divergence expected");
        assert_eq!(fd.kind, "structural_drift");
        assert_eq!(fd.primary_axis, "trajectory");
    }

    #[test]
    fn first_divergence_text_drift_is_decision_drift() {
        let a = make_pair(None, "hello");
        let b = make_pair(None, "hi there");
        let fd = first_divergence(&a, &b).expect("divergence expected");
        assert_eq!(fd.kind, "decision_drift");
        assert_eq!(fd.primary_axis, "semantic");
    }

    #[test]
    fn top_k_divergences_identical_returns_empty() {
        let recs = make_pair(Some("search"), "");
        assert!(top_k_divergences(&recs, &recs, 5).is_empty());
    }

    #[test]
    fn top_k_divergences_caps_at_k() {
        let mut a = make_pair(Some("a"), "");
        a.extend(make_pair(Some("b"), ""));
        a.extend(make_pair(Some("c"), ""));
        let mut b = make_pair(Some("x"), "");
        b.extend(make_pair(Some("y"), ""));
        b.extend(make_pair(Some("z"), ""));
        let out = top_k_divergences(&a, &b, 2);
        assert!(out.len() <= 2);
    }

    #[test]
    #[should_panic(expected = "k must be >= 1")]
    fn top_k_divergences_panics_on_zero_k() {
        let _ = top_k_divergences(&[], &[], 0);
    }
}
