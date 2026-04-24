//! `shadow-core` — the Rust core of the Shadow tool.
//!
//! See `SPEC.md` for the `.agentlog` format and `CONTRIBUTING.md` §Architecture for
//! how the modules in this crate compose into the end-to-end pipeline.
//!
//! Submodules:
//! - [`agentlog`] — canonical JSON, SHA-256 content addressing, streaming
//!   JSONL parser + writer for the `.agentlog` record format.
//! - [`diff`] — the nine behavioural axes, bootstrap CIs, severity
//!   scoring, first-divergence detection, recommendations, report renderer.
//! - [`replay`] — the `LlmBackend` trait + replay engine.
//! - [`store`] — content-addressed blob store + SQLite index.
//! - [`error`] — typed error hierarchy shared across the crate.

#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod agentlog;
pub mod diff;
pub mod error;
pub mod replay;
pub mod store;

#[cfg(feature = "python")]
pub mod python;

pub use error::Error;

/// Library version, matches the workspace `[package].version`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
