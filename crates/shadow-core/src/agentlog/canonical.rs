//! Canonical JSON serialization per SPEC §5.
//!
//! Given a [`serde_json::Value`], produce the canonical byte sequence that
//! is the input to SHA-256 content addressing (§6). Rules:
//!
//! 1. Object keys sorted by NFC-normalized UTF-8 bytes (§5.1).
//! 2. No whitespace between tokens (§5.1).
//! 3. Strings NFC-normalized before emission (§5.2).
//! 4. Numbers in shortest round-trip decimal form (§5.3).
//! 5. `-0` normalizes to `0` (§5.3).

use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

/// Serialize a [`serde_json::Value`] to canonical bytes (SPEC §5).
///
/// Infallible: `serde_json::Value` cannot hold `NaN` / `±Infinity`, so the
/// only way a number is un-representable has already been ruled out before
/// this function is called.
pub fn to_bytes(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    write_value(&mut out, value);
    out
}

fn write_value(out: &mut Vec<u8>, value: &Value) {
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => write_number(out, n),
        Value::String(s) => write_string(out, s),
        Value::Array(arr) => {
            out.push(b'[');
            for (i, v) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_value(out, v);
            }
            out.push(b']');
        }
        Value::Object(map) => {
            out.push(b'{');
            // NFC-normalize keys first so equivalent forms collapse, then
            // sort by the normalized UTF-8 bytes.
            let mut entries: Vec<(String, &Value)> = map
                .iter()
                .map(|(k, v)| (k.nfc().collect::<String>(), v))
                .collect();
            entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            for (i, (k, v)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_string(out, k);
                out.push(b':');
                write_value(out, v);
            }
            out.push(b'}');
        }
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.push(b'"');
    let normalized: String = s.nfc().collect();
    for c in normalized.chars() {
        match c {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            '\u{08}' => out.extend_from_slice(b"\\b"),
            '\u{0c}' => out.extend_from_slice(b"\\f"),
            c if (c as u32) < 0x20 => {
                // Non-shorthand control char → lowercase \u00XX escape.
                let code = c as u32;
                let buf = [
                    b'\\',
                    b'u',
                    b'0',
                    b'0',
                    hex_nibble((code >> 4) as u8),
                    hex_nibble((code & 0xF) as u8),
                ];
                out.extend_from_slice(&buf);
            }
            c => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out.push(b'"');
}

fn hex_nibble(n: u8) -> u8 {
    debug_assert!(n < 16);
    match n {
        0..=9 => b'0' + n,
        _ => b'a' + (n - 10),
    }
}

fn write_number(out: &mut Vec<u8>, n: &serde_json::Number) {
    // Prefer integer forms when available — serde_json tracks whether the
    // number was parsed as an integer.
    if let Some(i) = n.as_i64() {
        out.extend_from_slice(i.to_string().as_bytes());
        return;
    }
    if let Some(u) = n.as_u64() {
        out.extend_from_slice(u.to_string().as_bytes());
        return;
    }
    if let Some(f) = n.as_f64() {
        // SPEC §5.3: -0 → 0.
        if f == 0.0 {
            out.push(b'0');
            return;
        }
        // Rust's default `{}` for f64 is the shortest round-trip form:
        // 1.0 -> "1", 1.5 -> "1.5", 1e20 -> "100000000000000000000".
        // JSON uses lowercase `e` for scientific notation; Rust does too,
        // so no conversion needed. `is_finite()` is a guard even though
        // serde_json::Number can't hold NaN/Infinity by construction.
        if f.is_finite() {
            let s = format!("{f}");
            out.extend_from_slice(s.as_bytes());
            return;
        }
    }
    // If we got here, the number is in an unexpected state. Emit "null" so
    // canonical output remains syntactically valid JSON. Callers won't hit
    // this because serde_json::Number is closed over i64/u64/f64-finite.
    out.extend_from_slice(b"null");
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sorts_object_keys() {
        assert_eq!(
            to_bytes(&json!({"b": 2, "a": 1})),
            br#"{"a":1,"b":2}"#.to_vec()
        );
    }

    #[test]
    fn recursive_sorting() {
        assert_eq!(
            to_bytes(&json!({"b": {"z": 1, "y": 2}, "a": 1})),
            br#"{"a":1,"b":{"y":2,"z":1}}"#.to_vec()
        );
    }

    #[test]
    fn no_whitespace_in_arrays() {
        assert_eq!(
            to_bytes(&json!({"a": [1, 2, 3]})),
            br#"{"a":[1,2,3]}"#.to_vec()
        );
    }

    #[test]
    fn booleans_and_null() {
        assert_eq!(to_bytes(&json!(true)), b"true".to_vec());
        assert_eq!(to_bytes(&json!(false)), b"false".to_vec());
        assert_eq!(to_bytes(&json!(null)), b"null".to_vec());
    }

    #[test]
    fn integer_numbers() {
        assert_eq!(to_bytes(&json!(42)), b"42".to_vec());
        assert_eq!(to_bytes(&json!(-17)), b"-17".to_vec());
        assert_eq!(to_bytes(&json!(0)), b"0".to_vec());
    }

    #[test]
    fn float_that_is_an_integer_emits_as_integer() {
        // `1.00` parses as f64 1.0 — must emit as "1", not "1.0".
        let v: Value = serde_json::from_str("1.00").unwrap();
        assert_eq!(to_bytes(&v), b"1".to_vec());
    }

    #[test]
    fn fractional_float() {
        assert_eq!(to_bytes(&json!(1.5)), b"1.5".to_vec());
        assert_eq!(to_bytes(&json!(0.1)), b"0.1".to_vec());
    }

    #[test]
    fn negative_zero_normalizes_to_zero() {
        let v: Value = serde_json::from_str("-0.0").unwrap();
        assert_eq!(to_bytes(&v), b"0".to_vec());
    }

    #[test]
    fn string_mandatory_escapes() {
        assert_eq!(
            to_bytes(&json!({"x": "a\"b\\c"})),
            br#"{"x":"a\"b\\c"}"#.to_vec()
        );
    }

    #[test]
    fn string_control_chars_use_shorthand_when_available() {
        assert_eq!(to_bytes(&json!("\n")), br#""\n""#.to_vec());
        assert_eq!(to_bytes(&json!("\t")), br#""\t""#.to_vec());
        assert_eq!(to_bytes(&json!("\r")), br#""\r""#.to_vec());
    }

    #[test]
    fn string_other_control_chars_use_u00xx() {
        // U+0001 has no shorthand; must emit the lowercase \u0001 escape.
        let mut expected_01 = Vec::new();
        expected_01.extend_from_slice(b"\"\\u0001\"");
        assert_eq!(to_bytes(&json!("\u{01}")), expected_01);
        // U+001F is the highest control-char that must be escaped.
        let mut expected_1f = Vec::new();
        expected_1f.extend_from_slice(b"\"\\u001f\"");
        assert_eq!(to_bytes(&json!("\u{1f}")), expected_1f);
    }

    #[test]
    fn non_ascii_emitted_literally() {
        // SPEC §5.2: non-ASCII emitted as literal UTF-8, not \uXXXX.
        // U+00E9 = é = bytes c3 a9.
        let out = to_bytes(&json!("é"));
        assert_eq!(out, &[b'"', 0xc3, 0xa9, b'"']);
    }

    #[test]
    fn utf8_nfc_collapses_equivalent_forms() {
        // "é" precomposed (U+00E9) vs decomposed (U+0065 U+0301) MUST
        // serialize to the same bytes after NFC.
        let decomposed = "e\u{0301}";
        let precomposed = "\u{00e9}";
        assert_eq!(to_bytes(&json!(decomposed)), to_bytes(&json!(precomposed)));
    }

    #[test]
    fn utf8_nfc_applied_to_object_keys() {
        // An object with both the decomposed and precomposed form as keys
        // has TWO distinct keys in serde_json::Value, but after NFC they
        // collide. Policy: emit both, sorted — they'll be byte-identical
        // after normalization, so the sort is stable and both survive.
        // (The §5.1 "unique keys" rule is a producer requirement, not
        // something canonical serialization enforces after-the-fact.)
        // The relevant invariant here: NFC is applied before the sort, so
        // equivalent forms sort together.
        let v = json!({ "é": 1, "e\u{0301}": 2 });
        let out = to_bytes(&v);
        let s = std::str::from_utf8(&out).unwrap();
        // Both keys become "é" after NFC, so the output has two identical
        // keys. That's intentionally left to producers to police. Assert
        // the keys are adjacent in the output (prefix + same-key + prefix).
        assert!(s.starts_with(r#"{"é":"#));
    }

    #[test]
    fn idempotent_roundtrip() {
        let v = json!({"b": 2, "a": {"d": 3, "c": 4}, "arr": [{"y": 1, "x": 2}]});
        let once = to_bytes(&v);
        let reparsed: Value = serde_json::from_slice(&once).unwrap();
        let twice = to_bytes(&reparsed);
        assert_eq!(once, twice);
    }

    #[test]
    fn spec_5_6_known_vector_canonical_bytes() {
        // SPEC §5.6 Conformance test case: {"hello":"world"} canonical bytes.
        let payload = json!({"hello": "world"});
        assert_eq!(to_bytes(&payload), br#"{"hello":"world"}"#.to_vec());
    }
}
