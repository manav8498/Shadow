//! `.agentlog` record types, canonical JSON, content hashing, parser, writer.
//!
//! Implements SPEC §3-§6: the record envelope, payload variants,
//! canonical JSON serialisation, SHA-256 content addressing, and the
//! streaming JSONL parser/writer.

pub mod canonical;
pub mod hash;
pub mod parser;
pub mod record;
pub mod writer;

pub use record::{Kind, Record, CURRENT_VERSION};
