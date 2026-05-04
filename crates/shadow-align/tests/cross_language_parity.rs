//! Cross-language parity test for `shadow-align` (Rust) vs.
//! `shadow.align` (Python) and `@shadow-diff/align` (TypeScript).
//!
//! Each test mirrors a Python/TS test one-for-one. Outputs must
//! match byte-for-byte (modulo language-native naming) on shared
//! inputs. If any number drifts here, fix it in all three ports
//! — drift between languages is the bug.

// Integration tests live in their own crate, so the package-level
// `lints.clippy` block in Cargo.toml applies here too. Tests are
// allowed to .expect() / .unwrap() / panic!; pattern-matching every
// Option in test code obscures intent.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use serde_json::{json, Value};
use shadow_align::{
    align_traces, first_divergence, tool_arg_delta, top_k_divergences, trajectory_distance,
    ArgDeltaKind,
};

/// Helper that mirrors the `makePair` helpers in the Python and TS
/// test suites: a chat_request followed by a chat_response with
/// either a tool_use or text content block.
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
fn trajectory_distance_parity_with_python_and_ts() {
    // Direct mirror of test_align_library.py / align.test.ts.
    let cases: &[(&[&str], &[&str], f64)] = &[
        (&["a", "b", "c"], &["a", "b", "c"], 0.0),
        (&["a", "b"], &["x", "y"], 1.0),
        (&["a", "b", "c"], &["a", "x", "c"], 1.0 / 3.0),
        (&[], &[], 0.0),
        (&["a"], &[], 1.0),
    ];
    for (a, b, expected) in cases {
        let got = trajectory_distance(a, b);
        assert!(
            (got - expected).abs() < 1e-5,
            "trajectory_distance({a:?}, {b:?}) = {got}, expected {expected}",
        );
    }
}

#[test]
fn tool_arg_delta_added_key_matches_python() {
    let a = json!({"a": 1});
    let b = json!({"a": 1, "b": 2});
    let d = tool_arg_delta(&a, &b);
    assert_eq!(d.len(), 1);
    assert_eq!(d[0].path, "/b");
    assert_eq!(d[0].kind.as_str(), "added");
}

#[test]
fn tool_arg_delta_type_changed_matches_python() {
    // Python returns 'type_changed' for int->str. TS port had a
    // bug in the first cut; this test pins parity.
    let a = json!({"x": 1});
    let b = json!({"x": "1"});
    let d = tool_arg_delta(&a, &b);
    assert_eq!(d.len(), 1);
    assert_eq!(d[0].kind.as_str(), "type_changed");
}

#[test]
fn tool_arg_delta_nested_path_matches_python() {
    let a = json!({"outer": {"inner": "old"}});
    let b = json!({"outer": {"inner": "new"}});
    let d = tool_arg_delta(&a, &b);
    assert_eq!(d.len(), 1);
    assert_eq!(d[0].path, "/outer/inner");
    assert_eq!(d[0].kind.as_str(), "changed");
}

#[test]
fn tool_arg_delta_list_with_appended_matches_python() {
    let a = json!({"items": [1, 2]});
    let b = json!({"items": [1, 2, 3]});
    let d = tool_arg_delta(&a, &b);
    assert_eq!(d.len(), 1);
    assert_eq!(d[0].path, "/items/2");
    assert_eq!(d[0].kind.as_str(), "added");
}

#[test]
fn tool_arg_delta_deeply_nested_stack_safe() {
    // Build 100-deep nested object. Python handles this; Rust must
    // too without overflowing the recursion stack.
    let mut a = json!("old");
    let mut b = json!("new");
    for _ in 0..100 {
        a = json!({ "inner": a });
        b = json!({ "inner": b });
    }
    let d = tool_arg_delta(&a, &b);
    assert_eq!(d.len(), 1);
    let segs: Vec<&str> = d[0].path.split('/').filter(|s| !s.is_empty()).collect();
    assert_eq!(segs.len(), 100);
}

#[test]
fn tool_arg_delta_kinds_string_repr_matches_other_ports() {
    assert_eq!(ArgDeltaKind::Added.as_str(), "added");
    assert_eq!(ArgDeltaKind::Removed.as_str(), "removed");
    assert_eq!(ArgDeltaKind::Changed.as_str(), "changed");
    assert_eq!(ArgDeltaKind::TypeChanged.as_str(), "type_changed");
}

// ---------------------------------------------------------------------------
// align_traces / first_divergence / top_k_divergences cross-language parity.
// Mirrors the Python `test_align_library.py` and TypeScript
// `align.test.ts` cases line-for-line. If any kind/axis string drifts
// here, fix it in all three ports.
// ---------------------------------------------------------------------------

#[test]
fn align_traces_empty_inputs_return_empty_alignment() {
    let aln = align_traces(&[], &[]);
    assert!(aln.turns.is_empty());
    assert_eq!(aln.total_cost, 0.0);
}

#[test]
fn align_traces_pairs_index_by_index_matching_other_ports() {
    let mut a = make_pair(Some("search"), "");
    a.extend(make_pair(Some("summarize"), ""));
    let aln = align_traces(&a, &a);
    assert_eq!(aln.turns.len(), 2);
    assert_eq!(aln.total_cost, 0.0);
}

#[test]
fn align_traces_emits_gap_turn_with_cost_one_on_asymmetric_pairs() {
    let a = make_pair(Some("search"), "");
    let mut b = make_pair(Some("search"), "");
    b.extend(make_pair(Some("extra"), ""));
    let aln = align_traces(&a, &b);
    assert_eq!(aln.turns.len(), 2);
    assert!(aln.turns[1].baseline_index.is_none());
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
    // Same kind string Python/TS emit. Drift at this assertion is the bug.
    let a = make_pair(Some("search"), "");
    let fd = first_divergence(&a, &[]).expect("divergence expected");
    assert_eq!(fd.kind, "structural_drift_full");
    assert_eq!(fd.primary_axis, "trajectory");
}

#[test]
fn first_divergence_tool_drift_is_structural_drift_on_trajectory() {
    let a = make_pair(Some("search"), "");
    let b = make_pair(Some("summarize"), "");
    let fd = first_divergence(&a, &b).expect("divergence expected");
    assert_eq!(fd.kind, "structural_drift");
    assert_eq!(fd.primary_axis, "trajectory");
}

#[test]
fn first_divergence_text_drift_is_decision_drift_on_semantic() {
    let a = make_pair(None, "hello");
    let b = make_pair(None, "hi there");
    let fd = first_divergence(&a, &b).expect("divergence expected");
    assert_eq!(fd.kind, "decision_drift");
    assert_eq!(fd.primary_axis, "semantic");
}

#[test]
fn top_k_divergences_caps_at_k_matching_other_ports() {
    let mut a = make_pair(Some("a"), "");
    a.extend(make_pair(Some("b"), ""));
    a.extend(make_pair(Some("c"), ""));
    let mut b = make_pair(Some("x"), "");
    b.extend(make_pair(Some("y"), ""));
    b.extend(make_pair(Some("z"), ""));
    let out = top_k_divergences(&a, &b, 2);
    assert!(out.len() <= 2);
}
