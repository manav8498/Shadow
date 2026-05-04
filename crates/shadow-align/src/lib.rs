//! Reusable trace-comparison primitives for AI agents.
//!
//! Standalone Rust port of `shadow.align` (Python) and
//! `@shadow-diff/align` (TypeScript). Same five-function surface,
//! same algorithms, byte-identical results on shared inputs.
//!
//! Two functions ship as fully-native Rust:
//!
//! - [`trajectory_distance`] — Levenshtein on flat tool sequences.
//! - [`tool_arg_delta`] — structural diff of two `serde_json::Value`s.
//!
//! Three functions (`align_traces`, `first_divergence`,
//! `top_k_divergences`) require Shadow's 9-axis differ. v0.1 of this
//! crate exposes them as the [`Divergence`] / [`Alignment`] types
//! plus stub functions that return `None` / empty results — for the
//! full implementation, callers depend on `shadow-core` directly. A
//! future v0.2 wires the crates so `shadow-align` re-exports the
//! Rust differ's alignment functions natively.
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
// Trace-pair functions — v0.1 stubs.
//
// `align_traces`, `first_divergence`, `top_k_divergences` require
// the 9-axis differ from `shadow-core`. Callers that have it should
// depend on `shadow-core` directly for now; future v0.2 of this
// crate adds a feature flag (`with-shadow-core`) that re-exports
// the differ's alignment surface.
// ---------------------------------------------------------------------------

/// v0.1 stub. Returns an empty `Alignment`. For the real per-turn
/// pairing, depend on `shadow-core::diff::compute_diff_report`.
pub fn align_traces<T>(_baseline: &[T], _candidate: &[T]) -> Alignment {
    Alignment::default()
}

/// v0.1 stub. Returns `None`. For the real first-divergence
/// detection, depend on `shadow-core::diff::alignment::detect`.
pub fn first_divergence<T>(_baseline: &[T], _candidate: &[T]) -> Option<Divergence> {
    None
}

/// v0.1 stub. Returns an empty Vec. For the real top-K ranking,
/// depend on `shadow-core::diff::alignment::detect_top_k`.
pub fn top_k_divergences<T>(_baseline: &[T], _candidate: &[T], _k: usize) -> Vec<Divergence> {
    Vec::new()
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

    #[test]
    fn align_traces_returns_empty_v01_stub() {
        let aln = align_traces::<&str>(&[], &[]);
        assert!(aln.turns.is_empty());
        assert_eq!(aln.total_cost, 0.0);
    }

    #[test]
    fn first_divergence_returns_none_v01_stub() {
        assert!(first_divergence::<&str>(&[], &[]).is_none());
    }
}
