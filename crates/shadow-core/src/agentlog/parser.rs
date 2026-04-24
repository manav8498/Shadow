//! Streaming JSONL parser for `.agentlog` files.
//!
//! Each line of an `.agentlog` file is one JSON object conforming to the
//! envelope schema from SPEC §3. This module provides a zero-copy-ish
//! iterator that yields one [`Record`] per line (or a typed error with the
//! 1-based line number where the failure occurred). It does NOT enforce
//! trace-level invariants like "first record is metadata" or "parents
//! point backward" — those are enforced by callers that actually need
//! them (the replay engine, the differ) rather than at parse time.

use std::io::{BufRead, Read};

use thiserror::Error;

use crate::agentlog::Record;

/// Maximum bytes per JSONL line. Default covers real agent traces
/// with long tool outputs + conversational context (observed p99 is
/// ~50 KB per record); the ceiling catches runaway inputs early
/// rather than OOMing deep inside `read_line`.
///
/// Tunable via [`Parser::with_max_line_bytes`] — callers that ingest
/// legitimately bigger records (multimodal payloads, massive tool
/// results) can raise it explicitly.
pub const DEFAULT_MAX_LINE_BYTES: usize = 16 * 1024 * 1024;

/// Maximum total bytes per trace. Hard cap to prevent a malicious
/// or truncated file from exhausting memory during a `parse_all`
/// collect. At 1 GB, covers months of production traffic for most
/// agents while still refusing obvious denial-of-service payloads.
pub const DEFAULT_MAX_TOTAL_BYTES: usize = 1024 * 1024 * 1024;

/// Errors produced by the streaming parser.
#[derive(Debug, Error)]
pub enum ParseError {
    /// The underlying reader failed.
    #[error("io error on line {line}: {source}\nhint: check that the file is readable and not truncated mid-line")]
    Io {
        /// 1-based line number where the error surfaced.
        line: usize,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// A line was not valid JSON or did not match the [`Record`] schema.
    #[error("parse error on line {line}: {source}\nhint: verify the record matches the envelope schema (SPEC §3)")]
    Json {
        /// 1-based line number where the error surfaced.
        line: usize,
        /// The underlying serde_json error.
        #[source]
        source: serde_json::Error,
    },

    /// A single JSONL line exceeded [`Parser`]'s per-line byte budget.
    #[error("line {line} exceeds byte limit ({bytes} > {limit})\nhint: raise via Parser::with_max_line_bytes or check for corrupt input")]
    LineTooLarge {
        /// 1-based line number where the limit was hit.
        line: usize,
        /// Actual byte count observed.
        bytes: usize,
        /// Configured limit.
        limit: usize,
    },

    /// The trace's total byte count exceeded the configured ceiling.
    #[error("trace exceeds total byte limit ({bytes} > {limit})\nhint: raise via Parser::with_max_total_bytes or split the trace")]
    TraceTooLarge {
        /// Accumulated byte count at the point of failure.
        bytes: usize,
        /// Configured limit.
        limit: usize,
    },
}

/// Streaming parser. One instance processes one `.agentlog` stream.
///
/// Usage:
/// ```no_run
/// # use std::io::BufReader;
/// # use std::fs::File;
/// # use shadow_core::agentlog::parser::Parser;
/// let file = File::open("trace.agentlog").unwrap();
/// for result in Parser::new(BufReader::new(file)) {
///     let record = result.unwrap();
///     println!("{}", record.id);
/// }
/// ```
pub struct Parser<R> {
    reader: R,
    line: usize,
    buffer: String,
    done: bool,
    max_line_bytes: usize,
    max_total_bytes: usize,
    total_bytes: usize,
}

impl<R: BufRead> Parser<R> {
    /// Wrap a buffered reader with default resource bounds.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line: 0,
            buffer: String::new(),
            done: false,
            max_line_bytes: DEFAULT_MAX_LINE_BYTES,
            max_total_bytes: DEFAULT_MAX_TOTAL_BYTES,
            total_bytes: 0,
        }
    }

    /// Override the per-line byte cap. Default: [`DEFAULT_MAX_LINE_BYTES`].
    pub fn with_max_line_bytes(mut self, n: usize) -> Self {
        self.max_line_bytes = n;
        self
    }

    /// Override the whole-trace byte cap. Default: [`DEFAULT_MAX_TOTAL_BYTES`].
    pub fn with_max_total_bytes(mut self, n: usize) -> Self {
        self.max_total_bytes = n;
        self
    }
}

impl<R: BufRead> Iterator for Parser<R> {
    type Item = Result<Record, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        loop {
            self.buffer.clear();
            self.line += 1;
            // Bound the bytes read per line to protect against a
            // malformed stream with no newline that would otherwise
            // grow `buffer` unbounded. We pass a `&mut R` explicitly
            // to `Read::take` (rather than using the inherent
            // auto-deref method) so the reborrow — not the underlying
            // reader — is what moves into the Take adapter. The
            // resulting `Take<&mut R>` implements BufRead through the
            // blanket impl, so `read_line` still works.
            let reader_ref: &mut R = &mut self.reader;
            let mut bounded: std::io::Take<&mut R> =
                Read::take(reader_ref, self.max_line_bytes as u64 + 1);
            match bounded.read_line(&mut self.buffer) {
                Ok(0) => {
                    self.done = true;
                    return None;
                }
                Ok(bytes) => {
                    self.total_bytes = self.total_bytes.saturating_add(bytes);
                    if bytes > self.max_line_bytes {
                        self.done = true;
                        return Some(Err(ParseError::LineTooLarge {
                            line: self.line,
                            bytes,
                            limit: self.max_line_bytes,
                        }));
                    }
                    if self.total_bytes > self.max_total_bytes {
                        self.done = true;
                        return Some(Err(ParseError::TraceTooLarge {
                            bytes: self.total_bytes,
                            limit: self.max_total_bytes,
                        }));
                    }
                    // Skip blank lines (defensive — valid `.agentlog` files
                    // don't have them, but we'd rather tolerate a stray
                    // trailing newline than error-spam on it).
                    let trimmed = self.buffer.trim_end_matches(['\r', '\n']);
                    if trimmed.is_empty() {
                        continue;
                    }
                    let parsed = serde_json::from_str::<Record>(trimmed);
                    return Some(parsed.map_err(|e| ParseError::Json {
                        line: self.line,
                        source: e,
                    }));
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(ParseError::Io {
                        line: self.line,
                        source: e,
                    }));
                }
            }
        }
    }
}

/// Parse an entire `.agentlog` stream into a `Vec<Record>`.
///
/// Collects into memory — use [`Parser`] directly for large files.
pub fn parse_all<R: BufRead>(reader: R) -> Result<Vec<Record>, ParseError> {
    Parser::new(reader).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::{Kind, Record};
    use serde_json::json;
    use std::io::Cursor;

    fn make_record(kind: Kind, payload: serde_json::Value) -> Record {
        Record::new(kind, payload, "2026-04-21T10:00:00Z", None)
    }

    fn to_jsonl(records: &[Record]) -> String {
        let mut out = String::new();
        for r in records {
            out.push_str(&serde_json::to_string(r).unwrap());
            out.push('\n');
        }
        out
    }

    #[test]
    fn parses_a_single_record() {
        let r = make_record(Kind::ChatRequest, json!({"model": "a"}));
        let wire = to_jsonl(std::slice::from_ref(&r));
        let parsed: Vec<Record> = parse_all(Cursor::new(wire)).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].id, r.id);
    }

    #[test]
    fn parses_multiple_records_in_order() {
        let records = vec![
            make_record(Kind::Metadata, json!({"sdk": {"name": "shadow"}})),
            make_record(Kind::ChatRequest, json!({"model": "a"})),
            make_record(
                Kind::ChatResponse,
                json!({"model": "a", "stop_reason": "end_turn"}),
            ),
        ];
        let wire = to_jsonl(&records);
        let parsed = parse_all(Cursor::new(wire)).unwrap();
        assert_eq!(parsed.len(), 3);
        for (a, b) in records.iter().zip(parsed.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.kind, b.kind);
        }
    }

    #[test]
    fn blank_lines_are_skipped() {
        let r = make_record(Kind::ChatRequest, json!({"model": "a"}));
        let wire = format!("\n{}\n\n", serde_json::to_string(&r).unwrap());
        let parsed = parse_all(Cursor::new(wire)).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn rejects_a_line_longer_than_the_configured_limit() {
        // Build a valid record with one field of ~2kB.
        let big = "x".repeat(2048);
        let r = make_record(Kind::ChatRequest, json!({"model": "a", "pad": big}));
        let wire = to_jsonl(std::slice::from_ref(&r));

        // With a 1024-byte limit, the parser should error out with
        // `LineTooLarge` rather than silently pass through.
        let mut it = Parser::new(Cursor::new(wire)).with_max_line_bytes(1024);
        match it.next().unwrap() {
            Err(ParseError::LineTooLarge { limit, bytes, .. }) => {
                assert_eq!(limit, 1024);
                assert!(bytes > 1024);
            }
            other => panic!("expected LineTooLarge, got {:?}", other),
        }
        // Parser is `done` after the error — no further records.
        assert!(it.next().is_none());
    }

    #[test]
    fn rejects_total_trace_exceeding_byte_cap() {
        // Build three 800-byte records. With a 1500-byte total cap,
        // the second one should trip TraceTooLarge.
        let pad = "y".repeat(700);
        let r = make_record(Kind::ChatRequest, json!({"model": "a", "pad": pad}));
        let wire = to_jsonl(&[r.clone(), r.clone(), r.clone()]);

        let mut it = Parser::new(Cursor::new(wire)).with_max_total_bytes(1500);
        // First record: fits under 1500 cumulative.
        assert!(it.next().unwrap().is_ok());
        // Second record: pushes total over 1500 → error.
        match it.next().unwrap() {
            Err(ParseError::TraceTooLarge { limit, bytes }) => {
                assert_eq!(limit, 1500);
                assert!(bytes > 1500);
            }
            other => panic!("expected TraceTooLarge, got {:?}", other),
        }
        assert!(it.next().is_none());
    }

    #[test]
    fn reports_line_number_on_malformed_line() {
        // Two records, with a malformed middle line.
        let r = make_record(Kind::ChatRequest, json!({"model": "a"}));
        let wire = format!(
            "{}\nnot-valid-json\n{}\n",
            serde_json::to_string(&r).unwrap(),
            serde_json::to_string(&r).unwrap()
        );
        let mut it = Parser::new(Cursor::new(wire));
        assert!(it.next().unwrap().is_ok()); // line 1 ok
        match it.next().unwrap() {
            Err(ParseError::Json { line, .. }) => assert_eq!(line, 2),
            other => panic!("expected Json error on line 2, got {other:?}"),
        }
    }

    #[test]
    fn reports_line_number_on_schema_mismatch() {
        // Valid JSON but missing required envelope fields.
        let wire = r#"{"not_a_record": true}"#.to_string() + "\n";
        let mut it = Parser::new(Cursor::new(wire));
        match it.next().unwrap() {
            Err(ParseError::Json { line, .. }) => assert_eq!(line, 1),
            other => panic!("expected Json error on line 1, got {other:?}"),
        }
    }

    #[test]
    fn empty_input_yields_empty_vec() {
        let parsed = parse_all(Cursor::new(String::new())).unwrap();
        assert_eq!(parsed.len(), 0);
    }

    #[test]
    fn handles_trailing_newline() {
        let r = make_record(Kind::ChatRequest, json!({"model": "a"}));
        let wire = serde_json::to_string(&r).unwrap() + "\n";
        let parsed = parse_all(Cursor::new(wire)).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn handles_crlf_line_endings() {
        let r = make_record(Kind::ChatRequest, json!({"model": "a"}));
        let wire = format!("{}\r\n", serde_json::to_string(&r).unwrap());
        let parsed = parse_all(Cursor::new(wire)).unwrap();
        assert_eq!(parsed.len(), 1);
    }
}
