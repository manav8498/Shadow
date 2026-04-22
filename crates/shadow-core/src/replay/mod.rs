//! Replay engine and the `LlmBackend` trait.
//!
//! See CLAUDE.md §Replay for the lifecycle and SPEC §10 for the algorithm.

pub mod backend;
pub mod engine;
pub mod mock;

pub use backend::{LlmBackend, LlmError};
pub use engine::{run_replay, Clock, FixedClock, ReplayError};
pub use mock::MockLlm;
