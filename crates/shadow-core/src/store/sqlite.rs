//! SQLite index over stored traces (SPEC.md §8).
//!
//! The filesystem store ([`super::fs::Store`]) is the authoritative source of
//! truth for trace content; this index just lets `shadow` answer questions
//! like "which traces have tag env=prod?" without scanning every file on
//! disk. The `bundled` feature of rusqlite is on, so there's no system
//! `sqlite3` dependency.

use std::collections::HashMap;
use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension};
use thiserror::Error;

const SCHEMA_SQL: &str = include_str!("schema.sql");

/// Errors from [`Index`].
#[derive(Debug, Error)]
pub enum IndexError {
    /// Underlying rusqlite failure.
    #[error("sqlite error: {0}\nhint: the .shadow/index.sqlite file may be corrupt; delete it and re-register your traces")]
    Sqlite(#[from] rusqlite::Error),
}

/// Result alias for index operations.
pub type Result<T> = std::result::Result<T, IndexError>;

/// SQLite-backed trace index. Cheap to construct; use one per process or
/// thread.
pub struct Index {
    conn: Connection,
}

impl Index {
    /// Open (or create) a SQLite database at `path`. Applies the schema
    /// idempotently — safe to call on an existing DB.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(SCHEMA_SQL)?;
        Ok(Self { conn })
    }

    /// Open an in-memory index (useful for tests).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA_SQL)?;
        Ok(Self { conn })
    }

    /// Register a trace. Existing rows with the same `id` are replaced
    /// (idempotent insert). Tags are cleared and re-inserted.
    pub fn register_trace(&mut self, trace: &TraceRecord) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT OR REPLACE INTO traces (id, created_at, session_tag, root_record_id) VALUES (?1, ?2, ?3, ?4)",
            params![trace.id, trace.created_at, trace.session_tag, trace.root_record_id],
        )?;
        tx.execute("DELETE FROM tags WHERE trace_id = ?1", params![trace.id])?;
        {
            let mut stmt =
                tx.prepare("INSERT INTO tags (trace_id, key, value) VALUES (?1, ?2, ?3)")?;
            for (k, v) in &trace.tags {
                stmt.execute(params![trace.id, k, v])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Look up a trace by id.
    pub fn get_trace(&self, id: &str) -> Result<Option<TraceRecord>> {
        let row: Option<(String, i64, Option<String>, String)> = self
            .conn
            .query_row(
                "SELECT id, created_at, session_tag, root_record_id FROM traces WHERE id = ?1",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()?;
        let Some((id, created_at, session_tag, root_record_id)) = row else {
            return Ok(None);
        };
        let tags = self.tags_for(&id)?;
        Ok(Some(TraceRecord {
            id,
            created_at,
            session_tag,
            root_record_id,
            tags,
        }))
    }

    /// Trace ids with `tags.key = :key AND tags.value = :value`.
    pub fn find_by_tag(&self, key: &str, value: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT trace_id FROM tags WHERE key = ?1 AND value = ?2 ORDER BY trace_id")?;
        let ids: std::result::Result<Vec<String>, _> = stmt
            .query_map(params![key, value], |r| r.get::<_, String>(0))?
            .collect();
        Ok(ids?)
    }

    /// Trace ids with `traces.session_tag = :tag`.
    pub fn find_by_session_tag(&self, tag: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM traces WHERE session_tag = ?1 ORDER BY created_at DESC")?;
        let ids: std::result::Result<Vec<String>, _> = stmt
            .query_map(params![tag], |r| r.get::<_, String>(0))?
            .collect();
        Ok(ids?)
    }

    /// The `limit` most recently-created trace ids.
    pub fn recent(&self, limit: u32) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM traces ORDER BY created_at DESC LIMIT ?1")?;
        let ids: std::result::Result<Vec<String>, _> = stmt
            .query_map(params![limit], |r| r.get::<_, String>(0))?
            .collect();
        Ok(ids?)
    }

    /// Register a replay.
    pub fn register_replay(&mut self, replay: &ReplayRecord) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO replays (id, baseline_trace_id, config_hash, outcome_record_id, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                replay.id,
                replay.baseline_trace_id,
                replay.config_hash,
                replay.outcome_record_id,
                replay.created_at
            ],
        )?;
        Ok(())
    }

    /// Replay records whose baseline is `baseline_trace_id`.
    pub fn replays_of(&self, baseline_trace_id: &str) -> Result<Vec<ReplayRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, baseline_trace_id, config_hash, outcome_record_id, created_at FROM replays WHERE baseline_trace_id = ?1 ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map(params![baseline_trace_id], |r| {
            Ok(ReplayRecord {
                id: r.get(0)?,
                baseline_trace_id: r.get(1)?,
                config_hash: r.get(2)?,
                outcome_record_id: r.get(3)?,
                created_at: r.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    fn tags_for(&self, trace_id: &str) -> Result<HashMap<String, String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT key, value FROM tags WHERE trace_id = ?1")?;
        let mut map = HashMap::new();
        for row in stmt.query_map(params![trace_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })? {
            let (k, v) = row?;
            map.insert(k, v);
        }
        Ok(map)
    }
}

/// One row in the `traces` table (plus its tags).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceRecord {
    /// Trace id (content id of the root record).
    pub id: String,
    /// Unix epoch millis.
    pub created_at: i64,
    /// Optional session tag (matches `metadata.payload.tags.session_tag`).
    pub session_tag: Option<String>,
    /// Root record id (== `id` for canonical traces, but we store it
    /// separately so the invariant is queryable).
    pub root_record_id: String,
    /// Tags as a key→value map.
    pub tags: HashMap<String, String>,
}

/// One row in the `replays` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayRecord {
    /// Replay id (a UUID or content hash — producer's choice).
    pub id: String,
    /// Baseline trace this replay was run against.
    pub baseline_trace_id: String,
    /// Content hash of the candidate config that drove the replay.
    pub config_hash: String,
    /// Content id of the `replay_summary` record, if one has been written.
    pub outcome_record_id: Option<String>,
    /// Unix epoch millis.
    pub created_at: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trace(id: &str, session_tag: Option<&str>) -> TraceRecord {
        TraceRecord {
            id: id.to_string(),
            created_at: 1_700_000_000_000,
            session_tag: session_tag.map(ToString::to_string),
            root_record_id: id.to_string(),
            tags: HashMap::new(),
        }
    }

    #[test]
    fn open_in_memory_applies_schema() {
        let idx = Index::open_in_memory().unwrap();
        // Schema exists if we can query the table without error.
        let count: i64 = idx
            .conn
            .query_row("SELECT COUNT(*) FROM traces", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn register_and_get_trace() {
        let mut idx = Index::open_in_memory().unwrap();
        let mut trace = make_trace("sha256:aaaa", Some("prod-agent-0"));
        trace.tags.insert("env".to_string(), "prod".to_string());
        trace
            .tags
            .insert("region".to_string(), "us-east".to_string());
        idx.register_trace(&trace).unwrap();

        let back = idx.get_trace("sha256:aaaa").unwrap().unwrap();
        assert_eq!(back.id, trace.id);
        assert_eq!(back.session_tag.as_deref(), Some("prod-agent-0"));
        assert_eq!(back.tags, trace.tags);
    }

    #[test]
    fn get_missing_trace_returns_none() {
        let idx = Index::open_in_memory().unwrap();
        assert!(idx.get_trace("sha256:does-not-exist").unwrap().is_none());
    }

    #[test]
    fn register_is_idempotent_and_refreshes_tags() {
        let mut idx = Index::open_in_memory().unwrap();
        let mut trace = make_trace("sha256:aaaa", None);
        trace.tags.insert("env".to_string(), "prod".to_string());
        idx.register_trace(&trace).unwrap();

        // Re-register with different tags — old ones should go.
        trace.tags.clear();
        trace.tags.insert("env".to_string(), "dev".to_string());
        idx.register_trace(&trace).unwrap();

        let back = idx.get_trace("sha256:aaaa").unwrap().unwrap();
        assert_eq!(back.tags.get("env").map(String::as_str), Some("dev"));
        assert_eq!(back.tags.len(), 1);
    }

    #[test]
    fn find_by_tag() {
        let mut idx = Index::open_in_memory().unwrap();
        let mut a = make_trace("sha256:a", None);
        a.tags.insert("env".into(), "prod".into());
        let mut b = make_trace("sha256:b", None);
        b.tags.insert("env".into(), "prod".into());
        let mut c = make_trace("sha256:c", None);
        c.tags.insert("env".into(), "dev".into());
        for t in [&a, &b, &c] {
            idx.register_trace(t).unwrap();
        }
        let mut prod = idx.find_by_tag("env", "prod").unwrap();
        prod.sort();
        assert_eq!(prod, vec!["sha256:a", "sha256:b"]);
        assert_eq!(idx.find_by_tag("env", "dev").unwrap(), vec!["sha256:c"]);
        assert_eq!(idx.find_by_tag("env", "staging").unwrap().len(), 0);
    }

    #[test]
    fn find_by_session_tag_and_recent_respect_ordering() {
        let mut idx = Index::open_in_memory().unwrap();
        // Insert in order old → new; `recent` must return new first.
        for (i, id) in ["sha256:a", "sha256:b", "sha256:c"].iter().enumerate() {
            let mut t = make_trace(id, Some("agent-0"));
            t.created_at = 1_700_000_000_000 + i as i64 * 1000;
            idx.register_trace(&t).unwrap();
        }
        assert_eq!(
            idx.find_by_session_tag("agent-0").unwrap(),
            vec!["sha256:c", "sha256:b", "sha256:a"]
        );
        assert_eq!(idx.recent(2).unwrap(), vec!["sha256:c", "sha256:b"]);
    }

    #[test]
    fn register_and_query_replays() {
        let mut idx = Index::open_in_memory().unwrap();
        let trace = make_trace("sha256:baseline", None);
        idx.register_trace(&trace).unwrap();
        for i in 0..3 {
            idx.register_replay(&ReplayRecord {
                id: format!("replay-{i}"),
                baseline_trace_id: trace.id.clone(),
                config_hash: format!("sha256:cfg-{i}"),
                outcome_record_id: None,
                created_at: 1_700_000_000_000 + i * 1000,
            })
            .unwrap();
        }
        let replays = idx.replays_of(&trace.id).unwrap();
        assert_eq!(replays.len(), 3);
        // Descending by created_at.
        assert!(replays[0].created_at > replays[2].created_at);
    }

    #[test]
    fn cascade_delete_of_trace_removes_tags_and_replays() {
        let mut idx = Index::open_in_memory().unwrap();
        let mut trace = make_trace("sha256:x", None);
        trace.tags.insert("k".into(), "v".into());
        idx.register_trace(&trace).unwrap();
        idx.register_replay(&ReplayRecord {
            id: "r0".into(),
            baseline_trace_id: trace.id.clone(),
            config_hash: "sha256:c".into(),
            outcome_record_id: None,
            created_at: 1,
        })
        .unwrap();
        idx.conn
            .execute("DELETE FROM traces WHERE id = ?1", params![trace.id])
            .unwrap();
        assert!(idx.find_by_tag("k", "v").unwrap().is_empty());
        assert!(idx.replays_of(&trace.id).unwrap().is_empty());
    }
}
