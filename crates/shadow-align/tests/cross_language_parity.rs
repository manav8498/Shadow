//! Cross-language parity test for `shadow-align` (Rust) vs.
//! `shadow.align` (Python) and `@shadow-diff/align` (TypeScript).
//!
//! Each test mirrors a Python/TS test one-for-one. Outputs must
//! match byte-for-byte (modulo language-native naming) on shared
//! inputs. If any number drifts here, fix it in all three ports
//! — drift between languages is the bug.

use serde_json::json;
use shadow_align::{tool_arg_delta, trajectory_distance, ArgDeltaKind};

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
