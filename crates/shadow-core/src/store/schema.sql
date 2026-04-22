-- Shadow SQLite index schema, v0.1.
-- Loaded via `include_str!` from store::sqlite. See CLAUDE.md §5.
-- No migrations in v0.1; schema bumps require a bump of SHADOW_SCHEMA_VERSION.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS traces (
    id              TEXT PRIMARY KEY,
    created_at      INTEGER NOT NULL,    -- Unix epoch millis
    session_tag     TEXT,
    root_record_id  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS traces_session_tag_idx ON traces(session_tag);
CREATE INDEX IF NOT EXISTS traces_created_at_idx ON traces(created_at);

CREATE TABLE IF NOT EXISTS tags (
    trace_id  TEXT NOT NULL,
    key       TEXT NOT NULL,
    value     TEXT NOT NULL,
    PRIMARY KEY (trace_id, key),
    FOREIGN KEY (trace_id) REFERENCES traces(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS tags_kv_idx ON tags(key, value);

CREATE TABLE IF NOT EXISTS replays (
    id                  TEXT PRIMARY KEY,
    baseline_trace_id   TEXT NOT NULL,
    config_hash         TEXT NOT NULL,
    outcome_record_id   TEXT,
    created_at          INTEGER NOT NULL,
    FOREIGN KEY (baseline_trace_id) REFERENCES traces(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS replays_baseline_idx ON replays(baseline_trace_id);
CREATE INDEX IF NOT EXISTS replays_config_idx ON replays(config_hash);
