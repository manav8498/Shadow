//! Streaming JSONL writer for `.agentlog` files.
//!
//! Records are serialized with `serde_json` (struct-field order, not
//! canonical — SPEC §5 canonical form applies to the payload hash, not
//! to wire representation). One `\n` is emitted after each record,
//! including the last, so the file always ends with a newline.

use std::io::Write;

use crate::agentlog::Record;

/// Write a single record followed by a newline.
pub fn write_record<W: Write>(writer: &mut W, record: &Record) -> std::io::Result<()> {
    serde_json::to_writer(&mut *writer, record)?;
    writer.write_all(b"\n")?;
    Ok(())
}

/// Write an iterator of records, one per line.
///
/// Any record whose `verify_id()` returns `false` is written anyway — this
/// function is a pure serializer, not a validator. Use
/// [`Record::verify_id`](crate::agentlog::Record::verify_id) before calling
/// if you want to refuse tampered records.
pub fn write_all<'a, W: Write, I: IntoIterator<Item = &'a Record>>(
    writer: &mut W,
    records: I,
) -> std::io::Result<()> {
    for r in records {
        write_record(writer, r)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::parser::parse_all;
    use crate::agentlog::{Kind, Record};
    use serde_json::json;
    use std::io::Cursor;

    fn sample(kind: Kind, payload: serde_json::Value) -> Record {
        Record::new(kind, payload, "2026-04-21T10:00:00Z", None)
    }

    #[test]
    fn writes_one_record_with_trailing_newline() {
        let r = sample(Kind::ChatRequest, json!({"model": "a"}));
        let mut buf = Vec::new();
        write_record(&mut buf, &r).unwrap();
        assert_eq!(buf.last(), Some(&b'\n'));
        // Exactly one newline in the output (JSONL line separator).
        assert_eq!(buf.iter().filter(|b| **b == b'\n').count(), 1);
    }

    #[test]
    fn roundtrips_through_parser() {
        let records = vec![
            sample(Kind::Metadata, json!({"sdk": {"name": "shadow"}})),
            sample(Kind::ChatRequest, json!({"model": "a"})),
            sample(
                Kind::ChatResponse,
                json!({"model": "a", "stop_reason": "end_turn"}),
            ),
            sample(
                Kind::ToolCall,
                json!({"tool_name": "search", "tool_call_id": "t1", "arguments": {}}),
            ),
        ];
        let mut buf = Vec::new();
        write_all(&mut buf, &records).unwrap();
        let back = parse_all(Cursor::new(buf)).unwrap();
        assert_eq!(back, records);
    }

    #[test]
    fn roundtrip_preserves_content_id() {
        // If the writer ever started doing something weird to payloads
        // (e.g. reordering keys), the id would drift. Pin this down.
        let r = sample(Kind::ChatRequest, json!({"b": 2, "a": 1}));
        let before = r.id.clone();
        let mut buf = Vec::new();
        write_record(&mut buf, &r).unwrap();
        let back = parse_all(Cursor::new(buf)).unwrap();
        assert_eq!(back[0].id, before);
        assert!(back[0].verify_id());
    }

    #[test]
    fn empty_input_writes_nothing() {
        let mut buf = Vec::new();
        let empty: Vec<&Record> = Vec::new();
        write_all(&mut buf, empty).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn records_are_separated_by_exactly_one_newline() {
        let records = vec![
            sample(Kind::ChatRequest, json!({"n": 1})),
            sample(Kind::ChatRequest, json!({"n": 2})),
        ];
        let mut buf = Vec::new();
        write_all(&mut buf, &records).unwrap();
        let s = std::str::from_utf8(&buf).unwrap();
        // Two records → two newlines.
        assert_eq!(s.matches('\n').count(), 2);
        // No blank lines.
        assert!(!s.contains("\n\n"));
    }
}
