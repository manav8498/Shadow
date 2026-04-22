//! `.agentlog` record types, canonical JSON, content hashing, parser, writer.
//!
//! Phase-2 fills this in per SPEC §3–§6. This is a Phase-0 stub.

pub mod canonical;
pub mod hash;
pub mod parser;
pub mod record;
pub mod writer;

pub use record::{Kind, Record, CURRENT_VERSION};
