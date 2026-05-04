//! napi-rs bindings for `shadow-align`. Exposes the pure-Rust
//! `trajectory_distance` and `tool_arg_delta` functions to
//! Node/TypeScript callers as native bindings — ~25× faster than
//! the pure-TS implementations on large inputs (per Phase 8 perf
//! measurements).
//!
//! Build:
//!     cd typescript && npm run build:native
//!
//! Output: `typescript/native/index.<platform>.node` plus the
//! cross-platform loader stub.

use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Levenshtein edit distance between two flat string sequences,
/// normalised to `[0.0, 1.0]` by the longer length. Identical to
/// `shadow_align::trajectory_distance` in pure Rust.
#[napi]
pub fn trajectory_distance(a: Vec<String>, b: Vec<String>) -> f64 {
    let a_refs: Vec<&str> = a.iter().map(|s| s.as_str()).collect();
    let b_refs: Vec<&str> = b.iter().map(|s| s.as_str()).collect();
    shadow_align::trajectory_distance(&a_refs, &b_refs)
}

/// Structural diff of two JSON values (received as JSON strings
/// since napi-rs's JsUnknown is awkward to walk). Returns a JSON
/// string of the delta list.
///
/// TS-side wrapper parses both directions so callers see a
/// typed `ArgDelta[]` matching the pure-TS surface exactly.
#[napi]
pub fn tool_arg_delta_json(a_json: String, b_json: String) -> Result<String> {
    let a: serde_json::Value = serde_json::from_str(&a_json)
        .map_err(|e| Error::new(Status::InvalidArg, format!("a_json parse: {}", e)))?;
    let b: serde_json::Value = serde_json::from_str(&b_json)
        .map_err(|e| Error::new(Status::InvalidArg, format!("b_json parse: {}", e)))?;
    let deltas = shadow_align::tool_arg_delta(&a, &b);
    let out: Vec<serde_json::Value> = deltas
        .into_iter()
        .map(|d| {
            serde_json::json!({
                "path": d.path,
                "kind": d.kind.as_str(),
                "old": d.old,
                "new": d.new,
            })
        })
        .collect();
    serde_json::to_string(&out)
        .map_err(|e| Error::new(Status::GenericFailure, format!("serialize: {}", e)))
}

/// Constant identifier surfaced to the TS loader so it can verify
/// it's loading a compatible binding.
#[napi]
pub fn binding_version() -> String {
    "shadow-align-napi/0.1".to_string()
}
