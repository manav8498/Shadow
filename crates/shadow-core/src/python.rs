//! PyO3 bindings (enabled via the `python` feature).
//!
//! Phase-3 populates this module with `#[pyclass]` wrappers and the
//! `parse_agentlog`, `write_agentlog`, and `compute_diff` functions. The
//! resulting extension module is exposed to Python as `shadow._core`.

#![allow(unused_imports)]

use pyo3::prelude::*;

/// Placeholder so the extension module can be imported before Phase 3 lands.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", crate::VERSION)?;
    Ok(())
}
