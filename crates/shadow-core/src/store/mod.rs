//! Content-addressed filesystem blob store and SQLite index.
//!
//! See SPEC.md §8 and SPEC §8 (sharding) for the on-disk format.

pub mod fs;
pub mod sqlite;

pub use fs::{Store, StoreError};
pub use sqlite::{Index, IndexError, ReplayRecord, TraceRecord};
