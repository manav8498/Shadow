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
- `parent` links records into a DAG. The session-root `metadata`
  record has `null`; all others point back. Subsequent `metadata`
  records (used as session-boundary markers) have a non-null parent.
- Canonical JSON: sorted keys, no whitespace, UTF-8, RFC 8259 numbers.

## Payload kinds

The `version` field stays `"0.1"` (envelope is unchanged). Three
record kinds were added in v2.3 inside that envelope.

**v0.1 kinds** (see `SPEC.md §4.1` – `§4.7`):

- `metadata`: session root or session-boundary marker. Carries SDK
  info, runtime, tags, tool schemas.
- `chat_request`: what was sent to the model.
- `chat_response`: what came back. Carries `content` (list of blocks),
  `stop_reason`, `latency_ms`, `usage`.
- `tool_call`: a synthesised or recorded tool invocation.
- `tool_result`: the value the tool returned.
- `error`: failure record with retriable flag.
- `replay_summary`: written by `shadow replay` to mark a candidate
  trace's outcome.

**v0.2 record-kind extensions** (see `SPEC.md §4.8` – `§4.10`):

- `chunk`: one chunk of a streaming model response. Payload carries
  `chunk_index`, `time_unix_nano` (absolute, so long-stream replay
  doesn't drift), `delta` (provider-shape passthrough), and an
  optional `is_final` flag.
- `harness_event`: a single record kind for harness-side events with
  a `category` discriminator over a closed taxonomy: `retry`,
  `rate_limit`, `model_switch`, `context_trim`, `cache`, `guardrail`,
  `budget`, `stream_interrupt`, `tool_lifecycle`. Carries
  `severity` (`info`/`warning`/`error`/`fatal`) and a free-form
  `reason`.
- `blob_ref`: content-addressed binary reference. Carries `blob_id`
  (sha256 of bytes), `mime`, `size_bytes`, optional
  `agentlog-blob://` URI, optional 64-bit dHash `phash` for the cheap
  similarity tier, optional `embedding` slot for the semantic tier.

A v0.1-only parser is allowed to skip records whose `kind` it doesn't
recognise.

## File layout

Traces live under `.shadow/traces/<id[:2]>/<id[2:]>.agentlog`
(git-objects-style sharding). SQLite index at `.shadow/index.sqlite`.

## Size limits

Parser defaults (v1.1+):

- `DEFAULT_MAX_LINE_BYTES` = 16 MiB (per-record cap)
- `DEFAULT_MAX_TOTAL_BYTES` = 1 GiB (whole-trace cap)

Both tunable per-`Parser` via `with_max_line_bytes` / `with_max_total_bytes`
for legitimately larger records (multimodal payloads, batch ingest).
