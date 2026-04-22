//! `shadow-core` — the Rust core of the Shadow tool.
//!
//! See `SPEC.md` for the `.agentlog` format and `CLAUDE.md` §Architecture for
//! how the modules in this crate compose into the end-to-end pipeline.
//!
//! Phase-0 stub: only the module tree is declared. Each module's public
//! surface lands during Phase 2 per the plan.

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
