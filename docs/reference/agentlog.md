# The .agentlog format

Shadow's on-disk trace format. JSON-Lines, content-addressed, streaming-safe.

Full spec: [SPEC.md](https://github.com/manav8498/Shadow/blob/main/SPEC.md)
(25 pages, 5 worked examples, normative).

## Envelope

Every record:

```json
{
  "version": "0.1",
  "id": "sha256:<hex>",
  "kind": "metadata|chat_request|chat_response|tool_call|tool_result|error|...",
  "ts": "2026-04-24T10:00:00.000Z",
  "parent": "sha256:<parent_id>" or null,
  "payload": { ... }
}
```

- `id` is `sha256:` + hex(SHA-256(canonical_json(payload))). Two
  records with identical payloads have identical ids, content
  addressing.
- `parent` links records into a DAG. Metadata records have `null`;
  all others point back.
- Canonical JSON: sorted keys, no whitespace, UTF-8, RFC 8259 numbers.

## Payload kinds

See `SPEC.md §4` for the full schema of each. Common ones:

- `metadata`: session root. Carries SDK info, runtime, tags, tool
  schemas.
- `chat_request`: what was sent to the model.
- `chat_response`: what came back. Carries `content` (list of blocks),
  `stop_reason`, `latency_ms`, `usage`.
- `error`: failure record with retriable flag.

## File layout

Traces live under `.shadow/traces/<id[:2]>/<id[2:]>.agentlog`
(git-objects-style sharding). SQLite index at `.shadow/index.sqlite`.

## Size limits

Parser defaults (v1.1+):

- `DEFAULT_MAX_LINE_BYTES` = 16 MiB (per-record cap)
- `DEFAULT_MAX_TOTAL_BYTES` = 1 GiB (whole-trace cap)

Both tunable per-`Parser` via `with_max_line_bytes` / `with_max_total_bytes`
for legitimately larger records (multimodal payloads, batch ingest).
