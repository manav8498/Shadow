//! SHA-256 content addressing per SPEC §6.
//!
//! The content id for a record is `sha256(canonical_json(payload))`, lower-case
//! hex, prefixed with the string `"sha256:"`. Because the hash is over the
//! canonical form (§5), two payloads that are semantically equivalent
//! (e.g. the same keys in a different order, or a string with an
//! alternative Unicode normalization form) map to the same id.

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::agentlog::canonical;

/// Prefix applied to every content id. Parsers use it to disambiguate
/// future hash algorithms (e.g. `"blake3:"`) if we ever add them.
pub const ID_PREFIX: &str = "sha256:";

/// Length of the hex digest portion of an id (64 hex chars for SHA-256).
pub const HEX_LEN: usize = 64;

/// Compute the content id for a payload.
///
/// The input is the `payload` field of a `.agentlog` envelope — NOT the
/// whole envelope. See SPEC §6.1 for why the envelope is excluded.
pub fn content_id(payload: &Value) -> String {
    let bytes = canonical::to_bytes(payload);
    let digest = Sha256::digest(&bytes);
    let mut out = String::with_capacity(ID_PREFIX.len() + HEX_LEN);
    out.push_str(ID_PREFIX);
    for byte in digest {
        out.push(nibble(byte >> 4));
        out.push(nibble(byte & 0xF));
    }
    out
}

/// True if `s` is a syntactically valid content id (`sha256:` + 64 lowercase hex).
pub fn is_valid(s: &str) -> bool {
    if !s.starts_with(ID_PREFIX) {
        return false;
    }
    let hex = &s[ID_PREFIX.len()..];
    hex.len() == HEX_LEN && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn nibble(n: u8) -> char {
    debug_assert!(n < 16);
    match n {
        0..=9 => (b'0' + n) as char,
        _ => (b'a' + (n - 10)) as char,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn spec_5_6_known_vector() {
        // The single normative cross-implementation vector — SPEC §5.6.
        let payload = json!({"hello": "world"});
        assert_eq!(
            content_id(&payload),
            "sha256:93a23971a914e5eacbf0a8d25154cda309c3c1c72fbb9914d47c60f3cb681588"
        );
    }

    #[test]
    fn id_is_prefixed_and_64_hex_chars() {
        let id = content_id(&json!(null));
        assert!(id.starts_with("sha256:"));
        let hex = &id[ID_PREFIX.len()..];
        assert_eq!(hex.len(), HEX_LEN);
        assert!(hex
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn determinism_across_calls() {
        let p = json!({"model": "claude-opus-4-7", "temperature": 0.2});
        assert_eq!(content_id(&p), content_id(&p));
    }

    #[test]
    fn equivalent_payloads_hash_equal() {
        // Same content, different key order. Canonicalization should sort
        // them, so the hashes match.
        let a = json!({"a": 1, "b": 2});
        let b = json!({"b": 2, "a": 1});
        assert_eq!(content_id(&a), content_id(&b));
    }

    #[test]
    fn nfc_equivalence_produces_equal_id() {
        let decomposed = json!({"key": "e\u{0301}clair"});
        let precomposed = json!({"key": "\u{00e9}clair"});
        assert_eq!(content_id(&decomposed), content_id(&precomposed));
    }

    #[test]
    fn distinct_payloads_hash_different() {
        let a = json!({"a": 1});
        let b = json!({"a": 2});
        assert_ne!(content_id(&a), content_id(&b));
    }

    #[test]
    fn is_valid_accepts_well_formed_id() {
        assert!(is_valid(
            "sha256:93a23971a914e5eacbf0a8d25154cda309c3c1c72fbb9914d47c60f3cb681588"
        ));
    }

    #[test]
    fn is_valid_rejects_wrong_prefix() {
        assert!(!is_valid(
            "md5:93a23971a914e5eacbf0a8d25154cda309c3c1c72fbb9914d47c60f3cb681588"
        ));
        assert!(!is_valid(
            "93a23971a914e5eacbf0a8d25154cda309c3c1c72fbb9914d47c60f3cb681588"
        ));
    }

    #[test]
    fn is_valid_rejects_wrong_length() {
        assert!(!is_valid("sha256:abcd"));
        assert!(!is_valid(&format!("sha256:{}", "a".repeat(63))));
        assert!(!is_valid(&format!("sha256:{}", "a".repeat(65))));
    }

    #[test]
    fn is_valid_rejects_uppercase_hex() {
        assert!(!is_valid(
            "sha256:93A23971A914E5EACBF0A8D25154CDA309C3C1C72FBB9914D47C60F3CB681588"
        ));
    }
}
