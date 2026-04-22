//! Content-addressed filesystem blob store and SQLite index.
//!
//! See CLAUDE.md §Storage layout and SPEC §8 (sharding) for the on-disk format.

pub mod fs;

pub use fs::{Store, StoreError};

// sqlite index lands in the next commit.
