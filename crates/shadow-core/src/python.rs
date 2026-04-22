//! PyO3 bindings — the `shadow._core` Python extension module.
//!
//! Exposed surface:
//! - `parse_agentlog(bytes) -> list[dict]`
//! - `write_agentlog(list[dict]) -> bytes`
//! - `canonical_bytes(payload_dict) -> bytes`  (SPEC §5)
//! - `content_id(payload_dict) -> str`         (SPEC §6)
//! - `compute_diff_report(baseline, candidate, pricing, seed) -> dict`
//!
//! Everything is dict-oriented on the Python side; `pythonize` handles
//! the serde_json::Value ↔ PyObject conversion. Type hints for Python
//! callers live in `python/src/shadow/_core.pyi`.
//
// clippy::useless_conversion fires on the `?` operator in PyResult chains
// where `?` does an identity PyErr→PyErr conversion via `From`. That's a
// standard PyO3 pattern (every PyO3 API returns PyResult); suppressing here
// keeps PyO3-idiomatic code readable without sprinkling allows everywhere.
#![allow(clippy::useless_conversion)]

use std::io::Cursor;

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use pythonize::{depythonize, pythonize};

use crate::agentlog::{hash, parser, writer, Record};
use crate::diff::{compute_report, cost::Pricing};

/// Parse a `.agentlog` byte blob into a list of record dicts.
#[pyfunction]
fn parse_agentlog<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyBytes>,
) -> PyResult<Bound<'py, PyList>> {
    let bytes = data.as_bytes();
    let records =
        parser::parse_all(Cursor::new(bytes)).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let out = PyList::empty_bound(py);
    for r in records {
        let v = serde_json::to_value(&r).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let obj = pythonize(py, &v).map_err(|e| PyValueError::new_err(e.to_string()))?;
        out.append(obj)?;
    }
    Ok(out)
}

/// Serialize a list of record dicts into `.agentlog` bytes.
#[pyfunction]
fn write_agentlog<'py>(
    py: Python<'py>,
    records: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut parsed: Vec<Record> = Vec::with_capacity(records.len());
    for item in records.iter() {
        let v: serde_json::Value =
            depythonize(&item).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let r: Record =
            serde_json::from_value(v).map_err(|e| PyValueError::new_err(e.to_string()))?;
        parsed.push(r);
    }
    let mut buf = Vec::new();
    writer::write_all(&mut buf, &parsed).map_err(|e| PyIOError::new_err(e.to_string()))?;
    Ok(PyBytes::new_bound(py, &buf))
}

/// Canonical-JSON byte sequence for a payload (SPEC §5).
#[pyfunction]
fn canonical_bytes<'py>(
    py: Python<'py>,
    payload: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    let v: serde_json::Value =
        depythonize(payload).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let bytes = crate::agentlog::canonical::to_bytes(&v);
    Ok(PyBytes::new_bound(py, &bytes))
}

/// Content id for a payload dict (SPEC §6).
#[pyfunction]
fn content_id(payload: &Bound<'_, PyAny>) -> PyResult<String> {
    let v: serde_json::Value =
        depythonize(payload).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(hash::content_id(&v))
}

/// Compute a nine-axis diff between two traces.
///
/// `baseline` and `candidate` are lists of record dicts (as produced by
/// [`parse_agentlog`]). `pricing` is a `dict[str, tuple[float, float]]`
/// mapping model name → (price_per_input_token, price_per_output_token).
/// `seed` is an optional RNG seed for reproducible bootstrap CIs.
#[pyfunction]
#[pyo3(signature = (baseline, candidate, pricing=None, seed=None))]
fn compute_diff_report<'py>(
    py: Python<'py>,
    baseline: &Bound<'py, PyList>,
    candidate: &Bound<'py, PyList>,
    pricing: Option<&Bound<'py, PyDict>>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyAny>> {
    let baseline_records = pylist_to_records(baseline)?;
    let candidate_records = pylist_to_records(candidate)?;

    let mut price_map = Pricing::new();
    if let Some(dict) = pricing {
        for (k, v) in dict.iter() {
            let key: String = k
                .extract()
                .map_err(|e| PyValueError::new_err(format!("pricing key: {e}")))?;
            let pair: (f64, f64) = v
                .extract()
                .map_err(|e| PyValueError::new_err(format!("pricing value: {e}")))?;
            price_map.insert(key, pair);
        }
    }

    let report = compute_report(&baseline_records, &candidate_records, &price_map, seed);
    let v = serde_json::to_value(&report).map_err(|e| PyValueError::new_err(e.to_string()))?;
    pythonize(py, &v).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn pylist_to_records(list: &Bound<'_, PyList>) -> PyResult<Vec<Record>> {
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let v: serde_json::Value =
            depythonize(&item).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let r: Record =
            serde_json::from_value(v).map_err(|e| PyValueError::new_err(e.to_string()))?;
        out.push(r);
    }
    Ok(out)
}

/// The entry point the Python interpreter calls when `shadow._core` is
/// imported. Registers every function above.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", crate::VERSION)?;
    m.add("SPEC_VERSION", crate::agentlog::CURRENT_VERSION)?;
    m.add_function(wrap_pyfunction!(parse_agentlog, m)?)?;
    m.add_function(wrap_pyfunction!(write_agentlog, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(content_id, m)?)?;
    m.add_function(wrap_pyfunction!(compute_diff_report, m)?)?;
    Ok(())
}
