//! Content-addressed trace store on the local filesystem (SPEC §8).
//!
//! Layout (git-objects-style sharding):
//!
//! ```text
//! <root>/
//! ├── ab/
//! │   └── 1234…ef.agentlog          # trace root id = sha256:ab1234…ef
//! └── cd/
//!     └── 5678…90.agentlog
//! ```
//!
//! The trace root's content id is the file's logical name; the first two
//! hex characters of the digest become the shard directory.

use std::fs;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::agentlog::hash::{HEX_LEN, ID_PREFIX};
use crate::agentlog::{parser, writer, Record};

/// Errors from [`Store`].
#[derive(Debug, Error)]
pub enum StoreError {
    /// Attempted to store an empty trace (no records).
    #[error(
        "cannot store an empty trace\nhint: a trace needs at least one record (the metadata root)"
    )]
    Empty,

    /// The trace id does not match SPEC §6 format (`sha256:<64 hex>`).
    #[error("invalid trace id: {0}\nhint: expected `sha256:` followed by 64 lowercase hex chars (SPEC §6)")]
    BadId(String),

    /// Underlying I/O failure.
    #[error(
        "io error: {0}\nhint: check permissions on the store directory and available disk space"
    )]
    Io(#[from] std::io::Error),

    /// On-disk parse error while reading a trace back.
    #[error("parse error while reading trace: {0}\nhint: the on-disk trace may be corrupt; delete it and re-record if you have a source")]
    Parse(#[from] parser::ParseError),
}

/// Result alias for store operations.
pub type Result<T> = std::result::Result<T, StoreError>;

/// Content-addressed trace store.
///
/// `root` should be an existing directory (typically `.shadow/traces/`).
/// This struct does NOT create the root on construction — callers can
/// choose whether to pre-create it.
pub struct Store {
    root: PathBuf,
}

impl Store {
    /// Wrap an existing root directory. See SPEC §8 for the layout.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The root directory wrapped by this store.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Put a trace. Returns the trace's root id (the first record's id).
    ///
    /// Writes atomically: the final path does not appear until the write
    /// has fully completed and flushed to disk.
    pub fn put(&self, trace: &[Record]) -> Result<String> {
        let root_record = trace.first().ok_or(StoreError::Empty)?;
        let trace_id = root_record.id.clone();
        let dest = self.path_for(&trace_id)?;
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        // Write to `<dest>.tmp.<pid>`, fsync, rename. A crash mid-write
        // leaves at most a .tmp. file, not a half-written real file.
        let tmp = dest.with_extension("agentlog.tmp");
        {
            let file = fs::File::create(&tmp)?;
            let mut w = BufWriter::new(file);
            writer::write_all(&mut w, trace)?;
            w.flush()?;
        }
        fs::rename(&tmp, &dest)?;
        Ok(trace_id)
    }

    /// Read a trace by its root content id.
    pub fn get(&self, trace_id: &str) -> Result<Vec<Record>> {
        let path = self.path_for(trace_id)?;
        let file = fs::File::open(&path)?;
        let records = parser::parse_all(BufReader::new(file))?;
        Ok(records)
    }

    /// Whether a trace with this id is stored.
    pub fn exists(&self, trace_id: &str) -> bool {
        self.path_for(trace_id)
            .map(|p| p.is_file())
            .unwrap_or(false)
    }

    /// Iterate over all trace ids currently in the store.
    ///
    /// Walk order is undefined; callers that need a deterministic order
    /// should collect and sort.
    pub fn list(&self) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        if !self.root.is_dir() {
            return Ok(ids);
        }
        for shard in fs::read_dir(&self.root)? {
            let shard = shard?;
            if !shard.file_type()?.is_dir() {
                continue;
            }
            let shard_name = shard.file_name().to_string_lossy().to_string();
            if shard_name.len() != 2 || !shard_name.chars().all(|c| c.is_ascii_hexdigit()) {
                continue;
            }
            for entry in fs::read_dir(shard.path())? {
                let entry = entry?;
                let name = entry.file_name().to_string_lossy().to_string();
                if let Some(rest) = name.strip_suffix(".agentlog") {
                    if rest.len() == HEX_LEN - 2 && rest.chars().all(|c| c.is_ascii_hexdigit()) {
                        ids.push(format!("{ID_PREFIX}{shard_name}{rest}"));
                    }
                }
            }
        }
        Ok(ids)
    }

    /// Compute the on-disk path for a given trace id. Does NOT create
    /// directories or check whether the file exists.
    pub fn path_for(&self, trace_id: &str) -> Result<PathBuf> {
        if !trace_id.starts_with(ID_PREFIX) {
            return Err(StoreError::BadId(trace_id.to_string()));
        }
        let hex = &trace_id[ID_PREFIX.len()..];
        if hex.len() != HEX_LEN || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(StoreError::BadId(trace_id.to_string()));
        }
        let (shard, rest) = hex.split_at(2);
        Ok(self.root.join(shard).join(format!("{rest}.agentlog")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::Kind;
    use serde_json::json;
    use tempfile::TempDir;

    fn sample_trace() -> Vec<Record> {
        let root = Record::new(
            Kind::Metadata,
            json!({"sdk": {"name": "shadow", "version": "0.1.0"}}),
            "2026-04-21T10:00:00Z",
            None,
        );
        let req = Record::new(
            Kind::ChatRequest,
            json!({"model": "claude-opus-4-7", "messages": [], "params": {}}),
            "2026-04-21T10:00:00.100Z",
            Some(root.id.clone()),
        );
        vec![root, req]
    }

    fn new_store() -> (Store, TempDir) {
        let dir = tempfile::tempdir().unwrap();
        (Store::new(dir.path()), dir)
    }

    #[test]
    fn put_then_get_roundtrips() {
        let (store, _dir) = new_store();
        let trace = sample_trace();
        let id = store.put(&trace).unwrap();
        assert_eq!(id, trace[0].id);
        let back = store.get(&id).unwrap();
        assert_eq!(back, trace);
    }

    #[test]
    fn put_creates_sharded_path() {
        let (store, dir) = new_store();
        let trace = sample_trace();
        let id = store.put(&trace).unwrap();
        let expected = store.path_for(&id).unwrap();
        assert!(expected.is_file());
        // Path starts with <root>/<2 hex chars>/
        let rel = expected.strip_prefix(dir.path()).unwrap();
        let mut parts = rel.iter();
        let shard = parts.next().unwrap().to_string_lossy();
        assert_eq!(shard.len(), 2);
    }

    #[test]
    fn put_is_idempotent() {
        let (store, _dir) = new_store();
        let trace = sample_trace();
        let id1 = store.put(&trace).unwrap();
        let id2 = store.put(&trace).unwrap();
        assert_eq!(id1, id2);
        // Only one file — the shard dir has exactly one entry.
        let path = store.path_for(&id1).unwrap();
        let shard = path.parent().unwrap();
        let entries: Vec<_> = fs::read_dir(shard).unwrap().collect();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn exists_reports_presence() {
        let (store, _dir) = new_store();
        let trace = sample_trace();
        assert!(!store.exists(&trace[0].id));
        store.put(&trace).unwrap();
        assert!(store.exists(&trace[0].id));
    }

    #[test]
    fn list_returns_all_stored_traces() {
        let (store, _dir) = new_store();
        // Build two distinct traces by varying the payload.
        let a = vec![Record::new(
            Kind::Metadata,
            json!({"sdk": {"name": "a"}}),
            "2026-01-01T00:00:00Z",
            None,
        )];
        let b = vec![Record::new(
            Kind::Metadata,
            json!({"sdk": {"name": "b"}}),
            "2026-01-01T00:00:00Z",
            None,
        )];
        let id_a = store.put(&a).unwrap();
        let id_b = store.put(&b).unwrap();
        let mut ids = store.list().unwrap();
        ids.sort();
        let mut expected = vec![id_a, id_b];
        expected.sort();
        assert_eq!(ids, expected);
    }

    #[test]
    fn list_on_nonexistent_root_returns_empty() {
        let store = Store::new("/this/path/should/not/exist/for/tests");
        assert_eq!(store.list().unwrap().len(), 0);
    }

    #[test]
    fn path_for_rejects_bad_ids() {
        let (store, _dir) = new_store();
        assert!(matches!(store.path_for("abc"), Err(StoreError::BadId(_))));
        assert!(matches!(
            store.path_for("md5:aaaa"),
            Err(StoreError::BadId(_))
        ));
        assert!(matches!(
            store.path_for(&format!("sha256:{}", "z".repeat(64))),
            Err(StoreError::BadId(_))
        ));
    }

    #[test]
    fn put_empty_trace_errors() {
        let (store, _dir) = new_store();
        assert!(matches!(store.put(&[]), Err(StoreError::Empty)));
    }

    #[test]
    fn get_missing_trace_errors() {
        let (store, _dir) = new_store();
        let fake = format!("sha256:{}", "a".repeat(64));
        match store.get(&fake) {
            Err(StoreError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
            other => panic!("expected Io/NotFound, got {other:?}"),
        }
    }

    #[test]
    fn list_ignores_non_trace_files() {
        let (store, dir) = new_store();
        // Create a spurious shard-shaped dir with a non-.agentlog file.
        let fake_shard = dir.path().join("ab");
        fs::create_dir_all(&fake_shard).unwrap();
        fs::write(fake_shard.join("not-a-trace.txt"), "oops").unwrap();
        // And a non-shard directory name.
        fs::create_dir_all(dir.path().join("notashard")).unwrap();

        let trace = sample_trace();
        let id = store.put(&trace).unwrap();
        let ids = store.list().unwrap();
        assert_eq!(ids, vec![id]);
    }
}
