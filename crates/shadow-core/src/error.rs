//! Error types for `shadow-core`.
//!
//! Every module defines its own typed error and flattens it into this
//! top-level enum via `#[from]`. User-facing messages end with a `hint:` line
//! so callers know what to do next.

use thiserror::Error;

/// The top-level error type returned by `shadow-core`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Wrapped `std::io::Error` from filesystem or network operations.
    #[error("io error: {0}\nhint: check file permissions and disk space")]
    Io(#[from] std::io::Error),

    /// JSON parse or serialization error (see SPEC §3 for the envelope schema).
    #[error(
        "json error: {0}\nhint: verify the record matches the .agentlog envelope from SPEC §3"
    )]
    Json(#[from] serde_json::Error),

    /// SQLite error from the index store.
    #[error("sqlite error: {0}\nhint: the .shadow/index.sqlite file may be corrupt; delete and re-index")]
    Sqlite(#[from] rusqlite::Error),

    /// Catch-all for domain errors that predate dedicated variants.
    /// Phase-2 implementations will replace most `Other` uses with typed variants.
    #[error("{0}")]
    Other(String),
}
