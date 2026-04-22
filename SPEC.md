# `.agentlog` Specification — Version 0.1

> **Status:** Draft. Published with Shadow v0.1.0. Stable on `0.x.y` means
> no breaking changes within a minor version.
>
> **License:** Apache-2.0 (see `LICENSE-SPEC`).
>
> **Editor:** Shadow contributors.
>
> **Author-time date:** 2026-04-21.

## §1 Scope and non-goals

### §1.1 What `.agentlog` is

`.agentlog` is a file format and a semantic model for **traces produced by
LLM-powered agents**. One `.agentlog` file represents one logical agent
session — the full sequence of chat requests, chat responses, tool calls,
tool results, and errors that make up an agent's interaction with an LLM
(and optionally, external tools).

The format is:

- **JSON Lines (JSONL)** — one record per line, streaming-safe.
- **Content-addressed** — every record's `id` is the SHA-256 of a
  canonical serialization of its payload (§6). Identical payloads produce
  identical ids.
- **OpenTelemetry GenAI–compatible** — every field maps cleanly onto
  [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
  (§7), so traces can be exported to any OTel collector.
- **Privacy-first** — the format has first-class redaction markers (§9)
  and implementations MUST redact by default.
- **Replay-friendly** — records carry enough information (§10) that a
  conforming replayer can re-run the session against a new model or
  config and produce a behaviorally-comparable trace.

### §1.2 Non-goals

- **Cross-language SDK.** `.agentlog` is a file format. Shadow v0.1 ships
  a Python SDK and a Rust core. Implementations in other languages are
  welcome but out of scope for v0.1.
- **Streaming mid-response PII redaction.** Redaction happens at record
  boundaries (§9), not mid-token. Tools that need token-level redaction
  should preprocess before handing strings to Shadow.
- **Encryption at rest.** Use filesystem permissions, full-disk
  encryption, or a wrapper tool.
- **Binary compaction.** JSONL is verbose but diff-friendly and
  debuggable. Compact binary formats (msgpack, CBOR) may be explored in
  v0.2 as an optional encoding.
- **Auto-classified PII beyond regex patterns.** The default redactor is
  deterministic regex (§9). ML-based detectors are out of scope for v0.1
  — bring your own via the redactor plugin interface.

### §1.3 Relationship to Shadow

The Shadow tool uses `.agentlog` as its on-disk storage format, but
`.agentlog` itself is **tool-agnostic**. Another implementation could
produce `.agentlog` files that Shadow can replay and diff, as long as
they satisfy §3–§6.

## §2 Terminology

| Term | Meaning |
|------|---------|
| **Record** | One JSON object on one line of a `.agentlog` file. |
| **Payload** | The `payload` field of a record — the kind-specific body that is the subject of content-addressing (§6). |
| **Envelope** | Everything in the record other than the payload: `version`, `id`, `kind`, `ts`, `parent`, optional `meta`. |
| **Trace** | One complete `.agentlog` file. One logical agent session. |
| **Trace set** | A collection of traces referred to together (e.g. "last week's prod traffic"). |
| **Content hash** | SHA-256 over the canonical serialization (§5) of the payload. See §6. |
| **Content id** | `"sha256:" + hex_lowercase(content_hash)`. |
| **Canonical form** | The bytes produced by canonical serialization (§5). |
| **Backend** | An implementation of the LlmBackend trait (Rust) / protocol (Python). |
| **Replay** | The act of re-running a baseline trace against a new config, producing a candidate trace. See §10. |
| **Redaction** | Pattern-based replacement of sensitive substrings (§9). |

Key words "MUST", "MUST NOT", "SHOULD", "SHOULD NOT", and "MAY" in this
document are to be interpreted as described in
[RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

## §3 Record envelope

Every record has the same envelope shape. Conforming encoders MUST emit
all required fields; conforming parsers MUST reject records that omit
them or use wrong types.

```json
{
  "version": "0.1",
  "id": "sha256:9b74c9897bac770ffc029102a200c5de2c7ff7e5f8f8b8b5f6e1e4a9f8e0d1c2",
  "kind": "chat_request",
  "ts": "2026-04-21T22:00:00.123Z",
  "parent": null,
  "meta": { "session_tag": "prod-agent-0" },
  "payload": { /* kind-specific, see §4 */ }
}
```

### §3.1 Field definitions

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `version` | string | yes | `.agentlog` schema version. This document defines `"0.1"`. Parsers MUST reject unknown major versions in strict mode; MAY accept in permissive mode. |
| `id` | string | yes | Content id (§6). Must match `sha256:[0-9a-f]{64}`. |
| `kind` | string | yes | One of the kinds listed in §4. |
| `ts` | string | yes | RFC 3339 UTC timestamp with millisecond precision. MUST end in `Z` (not a numeric offset). |
| `parent` | string \| null | yes | Content id of the parent record in this trace, or `null` if the record is a root (typically only the first `metadata` record is a root). |
| `meta` | object | no | Free-form envelope-level metadata. Encoders MAY omit. Not part of the content hash. |
| `payload` | object | yes | Kind-specific body; see §4. Exactly this field is hashed for `id` (§6). |

### §3.2 What is NOT in the envelope (and why)

- **No trace id.** A trace is identified by the content id of its first
  record (the `metadata` record). You don't need a separate `trace_id`.
- **No sequence number.** File order IS the sequence. JSONL parsers see
  records in the order they were written.
- **No author / workspace / tenant.** Put those in `meta` or in the
  `metadata` payload (§4.5) if you need them.

### §3.3 Record ordering rules

Within one `.agentlog` file:

- The first record MUST be of kind `metadata` (§4.5) and MUST have
  `parent: null`. This is the trace's root.
- Every subsequent record MUST have a `parent` pointing to a record that
  appears earlier in the same file. Forward references are forbidden.
- `ts` values MUST be monotonically non-decreasing. Two records MAY have
  the same `ts` if they occurred within the same millisecond.
- A trace MUST NOT contain more than one `metadata` record.

## §4 Payload types

Each record kind specifies a fixed payload schema. Unknown fields inside
a payload MUST be ignored by consumers but preserved by round-tripping
implementations (i.e., don't drop them on re-serialization).

### §4.1 `chat_request`

A request sent to an LLM. Parent is typically a `metadata` (for the first
turn) or a `tool_result` (for subsequent turns after tool use).

```json
{
  "model": "claude-opus-4-7",
  "messages": [
    { "role": "system", "content": "You are a careful code reviewer." },
    { "role": "user",   "content": "Review the following diff: ..." }
  ],
  "tools": [
    {
      "name": "search_files",
      "description": "Search for files matching a glob.",
      "input_schema": {
        "type": "object",
        "properties": { "query": { "type": "string" } },
        "required": ["query"]
      }
    }
  ],
  "params": {
    "temperature": 0.2,
    "top_p": 1.0,
    "max_tokens": 1024,
    "stop": null
  },
  "system_fingerprint": null
}
```

| Field | Type | Required | Notes |
|-------|------|:--------:|-------|
| `model` | string | yes | Provider-native model id (e.g. `"claude-opus-4-7"`, `"gpt-5.3"`). |
| `messages` | array | yes | In provider message format. `role` ∈ `system / user / assistant / tool`. `content` is either a string or an array of content parts (text / tool_use / tool_result / thinking, as per the provider). |
| `tools` | array | no | Tool definitions. Omit when empty; implementations MUST treat omission and `[]` as equivalent. |
| `params` | object | yes | Sampling parameters. See §4.1.1. |
| `system_fingerprint` | string \| null | no | Optional provider-specific build id. |

#### §4.1.1 `params`

Conforming agents SHOULD populate:

- `temperature` (number, 0–2) — MAY be omitted if using provider default.
- `top_p` (number, 0–1)
- `max_tokens` (integer, >0)
- `stop` (array of string \| null)

Providers have extensions (`top_k`, `presence_penalty`, etc.) — these are
recorded verbatim as additional keys. Implementations MUST preserve them
on round-trip.

### §4.2 `chat_response`

A response from the LLM to a `chat_request`. Parent MUST be the request.

```json
{
  "model": "claude-opus-4-7",
  "content": [
    { "type": "text", "text": "I'll start by searching for..." },
    {
      "type": "tool_use",
      "id": "toolu_01A9F...",
      "name": "search_files",
      "input": { "query": "*.rs" }
    }
  ],
  "stop_reason": "tool_use",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 412,
    "output_tokens": 87,
    "thinking_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0
  },
  "latency_ms": 3421
}
```

| Field | Type | Required | Notes |
|-------|------|:--------:|-------|
| `model` | string | yes | Echo of the responding model. |
| `content` | array | yes | List of content parts. Shapes: `{type: "text", text}`, `{type: "tool_use", id, name, input}`, `{type: "thinking", text}` (opt-in), `{type: "image", ...}` (future). |
| `stop_reason` | string | yes | One of: `end_turn`, `tool_use`, `max_tokens`, `stop_sequence`, `content_filter`, `error`. |
| `stop_sequence` | string \| null | no | The matched stop string, if any. |
| `usage` | object | yes | Token counts. `thinking_tokens` defaults to 0. |
| `latency_ms` | integer | yes | Wall-clock end-to-end, receiving-side. |

### §4.3 `tool_call`

Emitted when an agent dispatches a tool call. This is distinct from a
`chat_response.content` `tool_use` part: the `tool_call` record is the
agent-side event, whereas `tool_use` inside a `chat_response` is the
model's request.

```json
{
  "tool_name": "search_files",
  "tool_call_id": "toolu_01A9F...",
  "arguments": { "query": "*.rs" }
}
```

| Field | Type | Required | Notes |
|-------|------|:--------:|-------|
| `tool_name` | string | yes | |
| `tool_call_id` | string | yes | MUST match the model's `tool_use.id`. |
| `arguments` | object | yes | JSON-serializable arguments as dispatched. |

### §4.4 `tool_result`

The result of a tool call. Parent is the `tool_call` record.

```json
{
  "tool_call_id": "toolu_01A9F...",
  "output": "README.md\nCargo.toml\n...",
  "is_error": false,
  "latency_ms": 12
}
```

| Field | Type | Required | Notes |
|-------|------|:--------:|-------|
| `tool_call_id` | string | yes | |
| `output` | string \| object | yes | Stringly output or structured JSON. |
| `is_error` | boolean | yes | |
| `latency_ms` | integer | yes | |

### §4.5 `metadata`

Trace-level metadata. MUST be the first record in a file.

```json
{
  "tags": { "env": "prod", "agent_id": "demo-0", "git_sha": "98f2c48" },
  "sdk": { "name": "shadow", "version": "0.1.0" },
  "runtime": { "python": "3.11.9", "platform": "darwin-arm64" },
  "baseline_of": null
}
```

| Field | Type | Required | Notes |
|-------|------|:--------:|-------|
| `tags` | object | no | String→string free-form. Indexed by the SQLite store. |
| `sdk` | object | yes | `{name, version}` of the producer. |
| `runtime` | object | no | Runtime version fingerprint. |
| `baseline_of` | string \| null | no | Content id of a baseline trace this trace is a replay of. `null` for original runs. |

### §4.6 `error`

An error produced by the SDK, the LLM, or a tool.

```json
{
  "source": "llm",
  "code": "rate_limit_exceeded",
  "message": "Too many requests",
  "retriable": true,
  "upstream_status": 429
}
```

| Field | Type | Required | Notes |
|-------|------|:--------:|-------|
| `source` | string | yes | One of `sdk`, `llm`, `tool`, `user`. |
| `code` | string | yes | Stable identifier. |
| `message` | string | yes | Human-readable. |
| `retriable` | boolean | yes | |
| `upstream_status` | integer \| null | no | HTTP status if applicable. |

### §4.7 `replay_summary`

Emitted at the end of a replay run.

```json
{
  "baseline_trace_id": "sha256:abc...",
  "candidate_config_hash": "sha256:def...",
  "input_count": 42,
  "output_count": 42,
  "error_count": 0,
  "duration_ms": 1534
}
```

## §5 Canonical JSON serialization

The content hash in §6 is computed over the **canonical form** of the
payload. Canonicalization is lossless for the value space defined here:
the same logical payload always serializes to the same bytes.

This section defines the canonicalization. Implementations MAY use
[RFC 8785](https://www.rfc-editor.org/rfc/rfc8785) (JCS) directly; the
rules below are equivalent to JCS with one clarification (§5.4).

### §5.1 Structure

- **Objects:** keys sorted lexicographically by their UTF-8 byte sequence
  (equivalent to Unicode code-point order for ASCII keys). `{` `}` with
  no surrounding whitespace. `:` between key and value with no
  whitespace. `,` between members with no whitespace. Duplicate keys are
  forbidden.
- **Arrays:** `[` `]` with no surrounding whitespace. `,` between
  elements with no whitespace. Order is preserved.
- **`true`, `false`, `null`:** literal lowercase.

### §5.2 Strings

- UTF-8 encoded. String value is quoted with `"`.
- Mandatory escapes: `"` → `\"`, `\` → `\\`, and control characters
  (U+0000 through U+001F) as `\u00XX` (lowercase hex).
- All other characters MUST be emitted as literal UTF-8. In particular,
  do **not** emit `\uXXXX` for non-ASCII characters.
- Strings MUST be valid Unicode. Lone surrogates are illegal.

### §5.3 Numbers

The number canonicalization follows RFC 8785 §3.2.2.3:

- `NaN` and `±Infinity` are forbidden (JSON does not represent them).
- `-0` normalizes to `0`.
- Integers in the range [-2^53, 2^53] are emitted without a decimal
  point or exponent (e.g. `42`, `-17`).
- Other numbers are emitted in the shortest form that round-trips to the
  same IEEE-754 double-precision value: no trailing zeros, no leading
  zeros, lowercase `e` for exponents, and at most one decimal point.

### §5.4 Application clarification (not in RFC 8785)

- Payloads MUST NOT contain JSON values other than string, number, true,
  false, null, object, array. If a producer has, e.g., a Python
  `Decimal`, it MUST convert to a JSON number before canonicalization.
- Keys MUST be strings (no non-string JSON object keys); this is already
  required by the JSON grammar.

### §5.5 Examples

Input:

```json
{"b": 2, "a": {"z": 1, "y": 2}}
```

Canonical output:

```
{"a":{"y":2,"z":1},"b":2}
```

Input:

```json
{"price": 1.00, "ratio": 0.1}
```

Canonical output:

```
{"price":1,"ratio":0.1}
```

## §6 Content addressing

The record's `id` field is computed as:

```
id = "sha256:" + lowercase_hex( sha256( canonical_json( payload ) ) )
```

That is:

1. Take only the `payload` field (not the envelope).
2. Canonicalize per §5 into a UTF-8 byte sequence.
3. Compute the SHA-256 digest of those bytes.
4. Lowercase-hex encode (64 hex chars).
5. Prefix with `"sha256:"`.

### §6.1 Why payload-only?

The envelope contains `ts` (unique per occurrence) and `parent` (varies
by trace context). Including them in the hash would defeat the primary
purpose of content addressing: **two semantically-identical requests
should dedupe to the same blob**. For replay and for storage efficiency
this matters — e.g., a 500-request prompt-cache evaluation becomes 1
blob + 500 envelope references.

### §6.2 Known-vector test

Given `payload = {"hello":"world"}` (already canonical):

```
canonical_bytes = b'{"hello":"world"}'
sha256(canonical_bytes) = 93a23971a914e5eacbf0a8d25154cda309c3c1c72fbb9914d47c60f3cb681588
id = "sha256:93a23971a914e5eacbf0a8d25154cda309c3c1c72fbb9914d47c60f3cb681588"
```

Conforming implementations MUST produce exactly this id for this
payload. `shadow-core`'s test suite pins this vector in
`crates/shadow-core/tests/canonical_vectors.rs`.

### §6.3 Collision handling

SHA-256 is collision-resistant at cryptographic levels (2^128 work).
Conforming implementations MAY assume that distinct ids imply distinct
payloads. If a producer encounters an on-disk record whose file name id
conflicts with the hash of a different payload, it MUST refuse to write
and surface the conflict (this would indicate corruption or an attack).

## §7 OpenTelemetry GenAI mapping

`.agentlog` records map cleanly to
[OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
so a trace can be exported to an OTel collector. The Rust core ships
with an OTel exporter (v0.2).

### §7.1 Request attributes

| `.agentlog` (chat_request) | OTel GenAI attribute |
|---|---|
| `payload.model` | `gen_ai.request.model` |
| `payload.model` (first segment) | `gen_ai.system` (e.g. `anthropic`, `openai`, `google`) |
| `payload.params.temperature` | `gen_ai.request.temperature` |
| `payload.params.top_p` | `gen_ai.request.top_p` |
| `payload.params.max_tokens` | `gen_ai.request.max_tokens` |
| `payload.params.stop` | `gen_ai.request.stop_sequences` |

### §7.2 Response attributes

| `.agentlog` (chat_response) | OTel GenAI attribute |
|---|---|
| `payload.model` | `gen_ai.response.model` |
| `payload.stop_reason` | `gen_ai.response.finish_reasons` (singleton array) |
| `payload.usage.input_tokens` | `gen_ai.usage.input_tokens` |
| `payload.usage.output_tokens` | `gen_ai.usage.output_tokens` |

### §7.3 Span structure

A trace's first record (metadata) becomes a root span. Each
`chat_request` → `chat_response` pair becomes a child span with the
request as span start and the response as span end; if a `chat_response`
contains `tool_use`, subsequent `tool_call` / `tool_result` pairs become
child spans of that chat span.

## §8 File layout and sharding

Shadow stores traces in a git-objects-style sharded layout. Given a
trace whose first record (metadata) has content id
`sha256:ab1234...`:

```
.shadow/traces/ab/1234....agentlog
```

The first two hex characters of the trace root's content id are the
shard prefix; the rest is the filename (plus `.agentlog`).

### §8.1 One-trace-per-file

Every `.agentlog` file is exactly one trace. Concatenating two
`.agentlog` files does NOT produce a valid `.agentlog` file: the
resulting file would have two `metadata` records, which violates §3.3.

### §8.2 Trace sets

A trace set (§2) is a directory (or glob pattern) listing multiple
`.agentlog` files. Shadow's SQLite index (`.shadow/index.sqlite`)
catalogs trace sets with user-defined tags; see `CLAUDE.md` §Storage
layout.

## §9 Redaction

### §9.1 Required behaviors

Conforming SDKs MUST:

1. Apply redaction **before canonicalization** (so the hash reflects the
   redacted content, not the raw one).
2. Enable redaction by default.
3. Expose a per-field allowlist that suppresses redaction on specific
   JSON paths (e.g., `payload.messages[*].content`).

### §9.2 Default pattern set

The default redactor matches and replaces:

| Pattern name | Regex (Python/Rust flavor) | Replacement |
|---|---|---|
| `openai_api_key` | `sk-[A-Za-z0-9]{20,}` | `[REDACTED:openai_api_key]` |
| `anthropic_api_key` | `sk-ant-[A-Za-z0-9\-_]{20,}` | `[REDACTED:anthropic_api_key]` |
| `email` | `[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}` | `[REDACTED:email]` |
| `phone_e164` | `\+[1-9]\d{1,14}(?!\d)` | `[REDACTED:phone]` |
| `credit_card` | 13–19 digit sequences that pass the Luhn check | `[REDACTED:credit_card]` |

Implementations MAY add patterns. They MUST NOT remove default patterns
silently (removing a pattern is a breaking change for users relying on
it, and SHOULD be surfaced via a log line).

### §9.3 Redaction stamp

When a redactor modifies a record, it MUST set
`envelope.meta.redacted = true`. Consumers can use this to distinguish
clean-from-source records from redacted ones without re-scanning.

### §9.4 Non-regex redactors

The SDK MAY accept user-supplied redactors implementing the same
interface. The redaction order is: user redactors first, then the
default pattern set, then canonicalization.

## §10 Replay semantics

### §10.1 Algorithm

Given a **baseline** trace set `B` and a **candidate config** `C`:

```
for trace b in B:
    emit metadata record m' with parent=null, meta.baseline_of = b.metadata.id
    for record r in b (in file order):
        if r.kind == "chat_request":
            prompt_hash = sha256(canonical_json(r.payload)) # = r.id
            response    = backend.complete(r.payload, C)
            emit chat_request r' (same payload, new envelope ts, parent = m'.id or previous)
            emit chat_response with parent = r'.id
        elif r.kind in ("tool_call", "tool_result", "error", "metadata"):
            copy-through with new envelope ts and parent relink
    emit replay_summary
```

### §10.2 Backend modes

- **Mock strict (default for `shadow diff` / CI):** The backend holds a
  `{prompt_hash → chat_response}` map loaded from `B`. If a replay
  request misses, it errors. Guarantees deterministic behavior.
- **Mock loose:** On miss, invoke a fallback (identity: return the
  original response). Useful when the candidate config does not change
  the request payload and you're measuring non-response axes (e.g.,
  cost under a different pricing table).
- **Live:** Hit the real provider. Used when the candidate's model_id
  differs from baseline so prompt_hash misses are expected.

Implementations MUST document which mode they default to. Shadow's
`shadow replay` defaults to **mock strict** when baseline and candidate
share the same `model`, and **live** (with an explicit `--backend live`
flag) otherwise.

### §10.3 Prompt-hash determinism

Because `id = sha256(canonical_json(payload))` (§6) and
canonicalization is deterministic (§5), identical requests across runs
produce identical ids. This is the foundation of mock-mode replay: the
backend just needs to index recorded responses by `request.id`.

### §10.4 Ordering and concurrency

Replay preserves record order within a trace. If a backend returns
responses out of order (async batching), the engine reorders them back
to match the baseline order before writing the candidate trace.

## §11 Compatibility with existing tools

`.agentlog` is designed to be exportable to — and importable from —
popular LLM-observability products. v0.1 does not ship these bridges;
they are described here to inform v0.2 design.

### §11.1 Langfuse

Langfuse's trace/span model has `traceId`, `spanId`, `parentObservationId`,
`input`, `output`, `model`. Mapping:

- `traceId` ← trace-root metadata record id (the first record's `id`)
- `spanId` ← individual record's `id`
- `parentObservationId` ← record's `parent`
- `input` ← `chat_request.messages`
- `output` ← `chat_response.content`
- `model` ← `chat_request.model`

### §11.2 Braintrust

Braintrust uses `experiments` and `logs`. A Shadow replay result maps to
a Braintrust experiment; each baseline-candidate pair becomes a log
entry. Nine-axis DiffReport fields map to Braintrust custom scorers.

### §11.3 LangSmith

LangSmith's `Run` objects map to `chat_request`/`chat_response` pairs.
LangSmith has first-class tool-call support matching §4.3/§4.4.

### §11.4 OpenTelemetry

Direct — see §7. Any OTel-compatible backend (Tempo, Jaeger, Honeycomb)
can consume exported `.agentlog` traces once Shadow's OTel exporter
lands in v0.2.

## §12 Worked examples

Five end-to-end records / traces. All examples use canonical JSON (§5).
The `ts` values and ids are real; re-compute them to verify.

### §12.1 Minimal chat

A single request and a single response. No tools.

```jsonl
{"version":"0.1","id":"sha256:5e43c6a3a0e5c8c82d4b19d7d2e8f8a1a7a2c3d4e5f60718293a4b5c6d7e8f90","kind":"metadata","ts":"2026-04-21T10:00:00.000Z","parent":null,"payload":{"sdk":{"name":"shadow","version":"0.1.0"},"tags":{"env":"dev"}}}
{"version":"0.1","id":"sha256:a8c6e9b4f3d217e58c192d3b4e5f607180293a4b5c6d7e8f9011223344556677","kind":"chat_request","ts":"2026-04-21T10:00:00.100Z","parent":"sha256:5e43c6a3a0e5c8c82d4b19d7d2e8f8a1a7a2c3d4e5f60718293a4b5c6d7e8f90","payload":{"messages":[{"content":"Say hi.","role":"user"}],"model":"claude-opus-4-7","params":{"max_tokens":64,"temperature":0.2,"top_p":1.0}}}
{"version":"0.1","id":"sha256:f1e2d3c4b5a69788796a5b4c3d2e1f0099887766554433221100ffeeddccbbaa","kind":"chat_response","ts":"2026-04-21T10:00:00.432Z","parent":"sha256:a8c6e9b4f3d217e58c192d3b4e5f607180293a4b5c6d7e8f9011223344556677","payload":{"content":[{"text":"Hi!","type":"text"}],"latency_ms":332,"model":"claude-opus-4-7","stop_reason":"end_turn","usage":{"input_tokens":9,"output_tokens":2,"thinking_tokens":0}}}
```

Note: the example `id` values are illustrative placeholders for
formatting; §6.2 is the only normative test vector. Consumers validate
ids by re-hashing the payload.

### §12.2 Streaming response

Streaming responses are recorded as **one `chat_response` record with
the aggregated content**, not a record per token. Stream timing lives in
`payload.stream_timings` (optional array of `{token_index, ts_delta_ms}`).

```jsonl
{"version":"0.1","id":"sha256:222...","kind":"chat_response","ts":"2026-04-21T10:05:00.600Z","parent":"sha256:111...","payload":{"content":[{"text":"The capital of France is Paris.","type":"text"}],"latency_ms":642,"model":"claude-opus-4-7","stop_reason":"end_turn","stream_timings":[{"token_index":0,"ts_delta_ms":25},{"token_index":1,"ts_delta_ms":57},{"token_index":2,"ts_delta_ms":82}],"usage":{"input_tokens":20,"output_tokens":8,"thinking_tokens":0}}}
```

### §12.3 Tool call round trip

Agent asks LLM, LLM requests a tool call, agent dispatches, tool
returns, agent sends result back to LLM, LLM replies.

```jsonl
{"version":"0.1","id":"sha256:aaa...","kind":"metadata","ts":"2026-04-21T11:00:00.000Z","parent":null,"payload":{"sdk":{"name":"shadow","version":"0.1.0"},"tags":{"env":"demo"}}}
{"version":"0.1","id":"sha256:bbb...","kind":"chat_request","ts":"2026-04-21T11:00:00.050Z","parent":"sha256:aaa...","payload":{"messages":[{"content":"Find all Rust files in this repo.","role":"user"}],"model":"claude-opus-4-7","params":{"max_tokens":256,"temperature":0.2,"top_p":1.0},"tools":[{"description":"Search for files matching a glob.","input_schema":{"properties":{"query":{"type":"string"}},"required":["query"],"type":"object"},"name":"search_files"}]}}
{"version":"0.1","id":"sha256:ccc...","kind":"chat_response","ts":"2026-04-21T11:00:00.510Z","parent":"sha256:bbb...","payload":{"content":[{"text":"I'll search for Rust files.","type":"text"},{"id":"toolu_01","input":{"query":"*.rs"},"name":"search_files","type":"tool_use"}],"latency_ms":460,"model":"claude-opus-4-7","stop_reason":"tool_use","usage":{"input_tokens":86,"output_tokens":23,"thinking_tokens":0}}}
{"version":"0.1","id":"sha256:ddd...","kind":"tool_call","ts":"2026-04-21T11:00:00.520Z","parent":"sha256:ccc...","payload":{"arguments":{"query":"*.rs"},"tool_call_id":"toolu_01","tool_name":"search_files"}}
{"version":"0.1","id":"sha256:eee...","kind":"tool_result","ts":"2026-04-21T11:00:00.538Z","parent":"sha256:ddd...","payload":{"is_error":false,"latency_ms":18,"output":"crates/shadow-core/src/lib.rs\ncrates/shadow-core/src/error.rs","tool_call_id":"toolu_01"}}
{"version":"0.1","id":"sha256:fff...","kind":"chat_request","ts":"2026-04-21T11:00:00.539Z","parent":"sha256:eee...","payload":{"messages":[{"content":"Find all Rust files in this repo.","role":"user"},{"content":[{"text":"I'll search for Rust files.","type":"text"},{"id":"toolu_01","input":{"query":"*.rs"},"name":"search_files","type":"tool_use"}],"role":"assistant"},{"content":[{"content":"crates/shadow-core/src/lib.rs\ncrates/shadow-core/src/error.rs","tool_use_id":"toolu_01","type":"tool_result"}],"role":"user"}],"model":"claude-opus-4-7","params":{"max_tokens":256,"temperature":0.2,"top_p":1.0},"tools":[{"description":"Search for files matching a glob.","input_schema":{"properties":{"query":{"type":"string"}},"required":["query"],"type":"object"},"name":"search_files"}]}}
{"version":"0.1","id":"sha256:ggg...","kind":"chat_response","ts":"2026-04-21T11:00:01.024Z","parent":"sha256:fff...","payload":{"content":[{"text":"Found 2 Rust files: `crates/shadow-core/src/lib.rs` and `crates/shadow-core/src/error.rs`.","type":"text"}],"latency_ms":485,"model":"claude-opus-4-7","stop_reason":"end_turn","usage":{"input_tokens":140,"output_tokens":34,"thinking_tokens":0}}}
```

### §12.4 Multi-turn agent (condensed)

Three-turn loop, with the second turn using a `thinking` content part.
Abbreviated for brevity (ids shown as `sha256:hN`).

```jsonl
{"version":"0.1","id":"sha256:h0","kind":"metadata","ts":"2026-04-21T12:00:00Z","parent":null,"payload":{"sdk":{"name":"shadow","version":"0.1.0"},"tags":{"agent_id":"demo-0","env":"prod"}}}
{"version":"0.1","id":"sha256:h1","kind":"chat_request","ts":"2026-04-21T12:00:00.100Z","parent":"sha256:h0","payload":{"messages":[{"content":"Plan a 3-step refactor.","role":"user"}],"model":"claude-opus-4-7","params":{"max_tokens":512,"temperature":0.2,"top_p":1.0}}}
{"version":"0.1","id":"sha256:h2","kind":"chat_response","ts":"2026-04-21T12:00:00.800Z","parent":"sha256:h1","payload":{"content":[{"text":"Breaking this into three manageable steps...","type":"thinking"},{"text":"Step 1: extract the parser. Step 2: add the differ. Step 3: wire the CLI.","type":"text"}],"latency_ms":701,"model":"claude-opus-4-7","stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":48,"thinking_tokens":22}}}
{"version":"0.1","id":"sha256:h3","kind":"chat_request","ts":"2026-04-21T12:00:01.100Z","parent":"sha256:h2","payload":{"messages":[{"content":"Plan a 3-step refactor.","role":"user"},{"content":[{"text":"Step 1: extract the parser. Step 2: add the differ. Step 3: wire the CLI.","type":"text"}],"role":"assistant"},{"content":"Expand step 2.","role":"user"}],"model":"claude-opus-4-7","params":{"max_tokens":512,"temperature":0.2,"top_p":1.0}}}
{"version":"0.1","id":"sha256:h4","kind":"chat_response","ts":"2026-04-21T12:00:01.900Z","parent":"sha256:h3","payload":{"content":[{"text":"Step 2 — add the differ: bootstrap CI, implement the nine axes, add the markdown renderer.","type":"text"}],"latency_ms":802,"model":"claude-opus-4-7","stop_reason":"end_turn","usage":{"input_tokens":72,"output_tokens":24,"thinking_tokens":0}}}
{"version":"0.1","id":"sha256:h5","kind":"chat_request","ts":"2026-04-21T12:00:02.100Z","parent":"sha256:h4","payload":{"messages":[{"content":"Plan a 3-step refactor.","role":"user"},{"content":[{"text":"Step 1: ... Step 3: wire the CLI.","type":"text"}],"role":"assistant"},{"content":"Expand step 2.","role":"user"},{"content":[{"text":"Step 2 — add the differ...","type":"text"}],"role":"assistant"},{"content":"Done. Thanks.","role":"user"}],"model":"claude-opus-4-7","params":{"max_tokens":64,"temperature":0.2,"top_p":1.0}}}
{"version":"0.1","id":"sha256:h6","kind":"chat_response","ts":"2026-04-21T12:00:02.500Z","parent":"sha256:h5","payload":{"content":[{"text":"You're welcome.","type":"text"}],"latency_ms":401,"model":"claude-opus-4-7","stop_reason":"end_turn","usage":{"input_tokens":120,"output_tokens":5,"thinking_tokens":0}}}
```

### §12.5 Failure trace

A `chat_request` that was rate-limited, with a recorded `error` record.

```jsonl
{"version":"0.1","id":"sha256:e0","kind":"metadata","ts":"2026-04-21T13:00:00Z","parent":null,"payload":{"sdk":{"name":"shadow","version":"0.1.0"},"tags":{"env":"prod"}}}
{"version":"0.1","id":"sha256:e1","kind":"chat_request","ts":"2026-04-21T13:00:00.050Z","parent":"sha256:e0","payload":{"messages":[{"content":"Summarize this document: ...","role":"user"}],"model":"claude-opus-4-7","params":{"max_tokens":2048,"temperature":0.2,"top_p":1.0}}}
{"version":"0.1","id":"sha256:e2","kind":"error","ts":"2026-04-21T13:00:00.120Z","parent":"sha256:e1","payload":{"code":"rate_limit_exceeded","message":"Please retry after 60s.","retriable":true,"source":"llm","upstream_status":429}}
```

The replay engine records errors as first-class events; the nine-axis
differ (§3 of `CLAUDE.md`) treats `error.code` as part of the
refusal/safety axis when `source == "llm"` and the code indicates a
safety filter.

## §13 Versioning and compatibility

- This document defines version `"0.1"`.
- Schema changes within `0.x.y` MUST be additive (new fields, new kinds)
  and MUST preserve backward compatibility: a `0.1` parser reading a
  `0.1.1` file ignores unknown fields.
- Breaking changes (removed fields, renamed kinds, altered canonical
  rules) require a `1.0` bump.
- Parsers MUST implement a `strict` mode (reject unknown versions /
  unknown kinds) and MAY implement `permissive` mode (warn and
  continue). Shadow's Rust core defaults to strict.
- The canonical-JSON rules in §5 are frozen within a major version. A
  change to §5 is always a breaking change.

## §14 Change log

| Version | Date | Changes |
|---|---|---|
| 0.1 | 2026-04-21 | Initial draft. Released with Shadow v0.1.0. |

---

## Appendix A — Conformance test matrix

Implementations claiming conformance MUST pass:

1. **Known-vector hash test** (§6.2): `{"hello":"world"}` hashes to
   `sha256:93a23971a914e5eacbf0a8d25154cda309c3c1c72fbb9914d47c60f3cb681588`.
2. **Canonical round-trip**: parse any record, re-serialize its
   payload canonically, recompute the id — the result equals the
   original id.
3. **Envelope validation**: reject a record missing any required field
   (§3.1).
4. **Order rules** (§3.3): reject a file whose first record is not
   `metadata` or whose records reference non-existent / forward
   parents.
5. **Redaction by default** (§9.1): the SDK exports a Session API that,
   when created without configuration, produces records with
   `envelope.meta.redacted = true` whenever a default pattern matches.

The `shadow-core` test suite (`cargo test --test conformance`) runs
these tests on every commit.

## Appendix B — Glossary cross-reference

- **JCS** — RFC 8785, JSON Canonicalization Scheme. See §5.
- **OTel GenAI** — OpenTelemetry GenAI semantic conventions. See §7.
- **Content id** — `"sha256:" + lowercase_hex(sha256(canonical_json(payload)))`.
- **Shadow** — the implementation that ships this spec.
