//! Streaming JSONL parser for `.agentlog` files.
//!
//! Each line of an `.agentlog` file is one JSON object conforming to the
//! envelope schema from SPEC §3. This module provides a zero-copy-ish
//! iterator that yields one [`Record`] per line (or a typed error with the
//! 1-based line number where the failure occurred). It does NOT enforce
//! trace-level invariants like "first record is metadata" or "parents
//! point backward" — those belong in a separate validator (Phase 3).

use std::io::BufRead;

use thiserror::Error;

use crate::agentlog::Record;

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
}

impl<R: BufRead> Parser<R> {
    /// Wrap a buffered reader.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line: 0,
            buffer: String::new(),
            done: false,
        }
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
            match self.reader.read_line(&mut self.buffer) {
                Ok(0) => {
                    self.done = true;
                    return None;
                }
                Ok(_) => {
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
