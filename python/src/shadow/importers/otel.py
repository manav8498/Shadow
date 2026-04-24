"""OpenTelemetry OTLP/JSON -> Shadow `.agentlog` importer.

This is the network-effect converter: ANY OpenTelemetry-instrumented
agent (Python, Go, Rust, Node, Java — thousands of existing
deployments) can export OTLP/JSON and pipe it into Shadow without
rewriting instrumentation.

Semconv compatibility
---------------------

Supports the full stack of GenAI semantic conventions the OTel
ecosystem has shipped since v1.28:

- **v1.40.0 (current, Development/experimental)** — structured JSON
  messages under `gen_ai.input.messages` / `gen_ai.output.messages`
  (carried either as span attributes or inside the
  `gen_ai.client.inference.operation.details` event), the provider
  rename (`gen_ai.provider.name` superseding `gen_ai.system`), the
  cache-token attributes (`gen_ai.usage.cache_read.input_tokens` and
  `cache_creation.input_tokens`), and the retrieval / agent span
  additions.
- **v1.37-v1.39** — structured messages land; per-message events
  (`gen_ai.user.message`, `gen_ai.assistant.message`, etc.) become
  deprecated. The importer still parses those events when present so
  older traces round-trip cleanly.
- **v1.28-v1.36** — the flat indexed `gen_ai.prompt.N.role/content`
  and `gen_ai.completion.N.content` shape, plus `gen_ai.system`. Kept
  as a fallback for traces from implementers that haven't tracked the
  v1.37 restructure (OpenLLMetry still defaults to this shape as of
  April 2026).

All three shapes are normalised to the single
v1.40-style ``{role, parts, tool_calls?, tool_call_id?}`` model before
being emitted as Shadow records. Consumers never see the layering.

Attribute mapping (complete)
----------------------------

Request:
  gen_ai.request.model                -> request.model
  gen_ai.request.temperature          -> request.params.temperature
  gen_ai.request.top_p                -> request.params.top_p
  gen_ai.request.top_k                -> request.params.top_k
  gen_ai.request.max_tokens           -> request.params.max_tokens
  gen_ai.request.frequency_penalty    -> request.params.frequency_penalty
  gen_ai.request.presence_penalty     -> request.params.presence_penalty
  gen_ai.request.stop_sequences       -> request.params.stop
  gen_ai.request.seed                 -> request.params.seed
  gen_ai.request.choice.count         -> request.params.n
  gen_ai.request.encoding_formats     -> request.params.encoding_formats
  gen_ai.tool.definitions             -> request.tools
  gen_ai.system_instructions          -> request.messages (prepended as system)
  gen_ai.input.messages               -> request.messages
  gen_ai.prompt.N.*                   -> request.messages (legacy)

Response:
  gen_ai.response.id                  -> response.id
  gen_ai.response.model               -> response.model
  gen_ai.response.finish_reasons      -> response.stop_reason
  gen_ai.output.type                  -> response.output_type
  gen_ai.output.messages              -> response.content
  gen_ai.completion.N.*               -> response.content (legacy)
  gen_ai.conversation.id              -> response.conversation_id
  gen_ai.usage.input_tokens           -> response.usage.input_tokens
  gen_ai.usage.output_tokens          -> response.usage.output_tokens
  gen_ai.usage.cache_read.input_tokens     -> response.usage.cached_input_tokens
  gen_ai.usage.cache_creation.input_tokens -> response.usage.cached_input_tokens (added)
  gen_ai.usage.cached_input_tokens    -> response.usage.cached_input_tokens (legacy)
  gen_ai.usage.reasoning_tokens       -> response.usage.thinking_tokens
  gen_ai.usage.thinking_tokens        -> response.usage.thinking_tokens (legacy)
  error.type / gen_ai.error.type      -> response.error.type

Metadata:
  gen_ai.provider.name / gen_ai.system -> metadata.provider
  gen_ai.agent.id / .name / .description / .version -> metadata.agent
  gen_ai.evaluation.* events           -> metadata.evaluations
  server.address / server.port         -> metadata.server

Shadow's IDs are SHA-256 of the canonical payload. OTel trace_id and
span_id are preserved in `meta.otel_trace_id` / `meta.otel_span_id`
for traceability back to the source.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

from shadow import __version__, _core
from shadow.errors import ShadowConfigError

OTEL_FORMAT = "otel"

# v1.40 spec values for `gen_ai.operation.name`. Kept as constants so
# the taxonomy lives in one place when the spec evolves (v1.41+).
_OP_CHAT = "chat"
_OP_TEXT_COMPLETION = "text_completion"
_OP_GENERATE = "generate_content"
_OP_EMBEDDINGS = "embeddings"
_OP_EXECUTE_TOOL = "execute_tool"
_OP_INVOKE_AGENT = "invoke_agent"
_OP_CREATE_AGENT = "create_agent"

# All operation names that produce a chat_request + chat_response pair.
# Embeddings are intentionally excluded — they have no response content
# that maps onto Shadow's content-block model.
_CHAT_OPERATIONS = frozenset({_OP_CHAT, _OP_TEXT_COMPLETION, _OP_GENERATE})


def otel_to_agentlog(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert an OTLP/JSON payload to Shadow records."""
    resource_spans = data.get("resourceSpans")
    if not isinstance(resource_spans, list):
        raise ShadowConfigError(
            "OTLP input missing top-level `resourceSpans` list.\n"
            "hint: pipe output of `otel-cli export` or any collector to a file."
        )

    spans = _flatten_spans(resource_spans)
    # Deterministic order: sort by startTime so repeated imports of the
    # same OTLP payload produce identical content-addressed IDs.
    spans.sort(key=lambda s: int(s.get("startTimeUnixNano") or 0))

    resource_attrs = _collect_resource_attrs(resource_spans)

    meta_payload: dict[str, Any] = {
        "sdk": {"name": "shadow", "version": __version__},
        "imported_from": OTEL_FORMAT,
        "source": {"format": OTEL_FORMAT, "semconv": "opentelemetry-gen-ai-v1.40"},
    }
    _augment_metadata_from_spans(meta_payload, spans, resource_attrs)

    meta_id = _core.content_id(meta_payload)
    records: list[dict[str, Any]] = [
        {
            "version": "0.1",
            "id": meta_id,
            "kind": "metadata",
            "ts": _now_iso(),
            "parent": None,
            "payload": meta_payload,
        }
    ]
    last_parent = meta_id

    chat_spans = [s for s in spans if _is_chat_span(s)]
    for span in chat_spans:
        attrs = _attrs_as_dict(span.get("attributes", []))
        events = span.get("events", []) or []
        # v1.37+ inference-details event can carry the messages attached to
        # the span as a side-channel. Merge its attributes on top of the
        # span's attribute bag so downstream extractors don't care which
        # side carried them.
        for ev in events:
            if ev.get("name") == "gen_ai.client.inference.operation.details":
                for k, v in _attrs_as_dict(ev.get("attributes", []) or []).items():
                    attrs.setdefault(k, v)

        start_ts = _nano_to_iso(span.get("startTimeUnixNano"))
        end_ts = _nano_to_iso(span.get("endTimeUnixNano"))
        req_payload = _span_to_request_payload(span, attrs, events)
        req_id = _core.content_id(req_payload)
        records.append(
            _envelope(
                "chat_request",
                req_id,
                start_ts,
                last_parent,
                req_payload,
                otel_span_id=span.get("spanId", ""),
                otel_trace_id=span.get("traceId", ""),
            )
        )
        resp_payload = _span_to_response_payload(span, attrs, events, start_ts, end_ts)
        resp_id = _core.content_id(resp_payload)
        records.append(
            _envelope(
                "chat_response",
                resp_id,
                end_ts,
                req_id,
                resp_payload,
                otel_span_id=span.get("spanId", ""),
                otel_trace_id=span.get("traceId", ""),
            )
        )
        last_parent = resp_id

    return records


# ---- span collection ------------------------------------------------------


def _flatten_spans(resource_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rs in resource_spans:
        for ss in rs.get("scopeSpans", []):
            out.extend(ss.get("spans", []))
    return out


def _collect_resource_attrs(resource_spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge `resource.attributes` from every resourceSpans block.

    OTLP allows a single payload to contain multiple resources (typical
    when a collector batches spans from several services). We merge them
    into one dict, last-wins, because for a single agent session the
    resource is almost always identical across spans.
    """
    merged: dict[str, Any] = {}
    for rs in resource_spans:
        merged.update(_attrs_as_dict((rs.get("resource") or {}).get("attributes", []) or []))
    return merged


def _is_chat_span(span: dict[str, Any]) -> bool:
    """True if this span describes an LLM inference call we should import.

    Preference order:
    1. `gen_ai.operation.name` in {chat, text_completion, generate_content}
       (v1.37+ canonical signal)
    2. `gen_ai.request.model` present AND no `gen_ai.operation.name`
       pointing elsewhere (v1.28-v1.36 fallback)
    3. span name prefix `gen_ai.chat` / `gen_ai.text_completion`
       (some instrumenters set only the name)
    """
    attrs = _attrs_as_dict(span.get("attributes", []))
    op = attrs.get("gen_ai.operation.name")
    if isinstance(op, str) and op in _CHAT_OPERATIONS:
        return True
    if isinstance(op, str) and op and op not in _CHAT_OPERATIONS:
        # Explicitly non-chat operations (execute_tool, invoke_agent,
        # create_agent, embeddings, ...) are handled elsewhere or skipped.
        return False
    if "gen_ai.request.model" in attrs:
        return True
    name = str(span.get("name", ""))
    return name.startswith("gen_ai.chat") or name.startswith("gen_ai.text_completion")


# ---- attribute decoding ---------------------------------------------------


def _attrs_as_dict(attrs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for a in attrs:
        key = a.get("key")
        if not isinstance(key, str):
            continue
        val = a.get("value") or {}
        out[key] = _decode_any_value(val)
    return out


def _decode_any_value(v: dict[str, Any]) -> Any:
    """Decode the OTLP AnyValue oneof into a Python scalar / list / dict."""
    if "stringValue" in v:
        return v["stringValue"]
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "intValue" in v:
        try:
            return int(v["intValue"])
        except (TypeError, ValueError):
            return v["intValue"]
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "arrayValue" in v:
        return [_decode_any_value(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {
            kv.get("key", ""): _decode_any_value(kv.get("value") or {})
            for kv in v["kvlistValue"].get("values", [])
        }
    if "bytesValue" in v:
        # Rare but allowed; return as hex string for JSON-compat.
        raw = v["bytesValue"]
        return raw if isinstance(raw, str) else str(raw)
    return None


def _maybe_parse_json(val: Any) -> Any:
    """Some instrumenters stringify structured attributes (JSON-in-string).

    v1.37 explicitly allows `gen_ai.input.messages` to be serialised as a
    JSON string OR a structured kvlist. We attempt to parse strings that
    look like JSON and return the parsed value; non-JSON strings pass
    through unchanged.
    """
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s:
        return val
    if s[0] not in "[{":
        return val
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return val


# ---- request payload ------------------------------------------------------


def _span_to_request_payload(
    span: dict[str, Any], attrs: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any]:
    model = str(attrs.get("gen_ai.request.model", ""))
    messages = _extract_messages(attrs, events, direction="input")
    params = _extract_request_params(attrs)
    tools = _extract_tool_definitions(attrs)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "params": params,
    }
    if tools:
        payload["tools"] = tools
    conv_id = attrs.get("gen_ai.conversation.id")
    if isinstance(conv_id, str) and conv_id:
        payload["conversation_id"] = conv_id
    return payload


def _extract_request_params(attrs: dict[str, Any]) -> dict[str, Any]:
    """All documented v1.40 request params that Shadow tracks."""
    params: dict[str, Any] = {}
    numeric_map = (
        ("gen_ai.request.temperature", "temperature", float),
        ("gen_ai.request.top_p", "top_p", float),
        ("gen_ai.request.top_k", "top_k", float),
        ("gen_ai.request.max_tokens", "max_tokens", int),
        ("gen_ai.request.frequency_penalty", "frequency_penalty", float),
        ("gen_ai.request.presence_penalty", "presence_penalty", float),
        ("gen_ai.request.seed", "seed", int),
        ("gen_ai.request.choice.count", "n", int),
    )
    for src, dst, caster in numeric_map:
        raw = attrs.get(src)
        if raw is None:
            continue
        try:
            params[dst] = caster(raw)
        except (TypeError, ValueError):
            # Instrumenters occasionally stringify numerics; leave raw.
            params[dst] = raw
    stop = attrs.get("gen_ai.request.stop_sequences")
    if isinstance(stop, list) and stop:
        params["stop"] = [str(x) for x in stop]
    enc = attrs.get("gen_ai.request.encoding_formats")
    if isinstance(enc, list) and enc:
        params["encoding_formats"] = [str(x) for x in enc]
    return params


def _extract_tool_definitions(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """`gen_ai.tool.definitions` is a list of {type, name, description, parameters}."""
    raw = _maybe_parse_json(attrs.get("gen_ai.tool.definitions"))
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for t in raw:
        if isinstance(t, dict):
            out.append(t)
    return out


# ---- response payload -----------------------------------------------------


def _span_to_response_payload(
    span: dict[str, Any],
    attrs: dict[str, Any],
    events: list[dict[str, Any]],
    start_ts: str,
    end_ts: str,
) -> dict[str, Any]:
    model = str(attrs.get("gen_ai.response.model", attrs.get("gen_ai.request.model", "")))
    content = _extract_messages(attrs, events, direction="output")
    # Response messages live under `gen_ai.output.messages`; each
    # message's parts + tool_calls become Shadow content blocks.
    blocks = _messages_to_response_content_blocks(content)
    stop_reason = _extract_stop_reason(attrs, span)
    latency_ms = _duration_ms(start_ts, end_ts)
    usage = _extract_usage(attrs)

    payload: dict[str, Any] = {
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
        "latency_ms": latency_ms,
        "usage": usage,
    }
    resp_id = attrs.get("gen_ai.response.id")
    if isinstance(resp_id, str) and resp_id:
        payload["response_id"] = resp_id
    out_type = attrs.get("gen_ai.output.type")
    if isinstance(out_type, str) and out_type:
        payload["output_type"] = out_type
    err_type = attrs.get("error.type") or attrs.get("gen_ai.error.type")
    if isinstance(err_type, str) and err_type:
        payload["error"] = {"type": err_type}
    return payload


def _extract_usage(attrs: dict[str, Any]) -> dict[str, int]:
    """Normalise the usage-token attribute zoo into Shadow's three keys."""

    def _int(key: str, *aliases: str) -> int:
        for k in (key, *aliases):
            if k in attrs and attrs[k] is not None:
                try:
                    return int(attrs[k])
                except (TypeError, ValueError):
                    continue
        return 0

    # cache_read + cache_creation both end up in cached_input_tokens —
    # distinguishing them matters for pricing, but Shadow's cost axis
    # already has that level via the pricing table.
    cache_read = _int("gen_ai.usage.cache_read.input_tokens", "gen_ai.usage.cached_input_tokens")
    cache_write = _int("gen_ai.usage.cache_creation.input_tokens")
    reasoning = _int("gen_ai.usage.reasoning_tokens", "gen_ai.usage.thinking_tokens")

    usage: dict[str, int] = {
        "input_tokens": _int("gen_ai.usage.input_tokens"),
        "output_tokens": _int("gen_ai.usage.output_tokens"),
        "thinking_tokens": reasoning,
    }
    if cache_read or cache_write:
        usage["cached_input_tokens"] = cache_read + cache_write
    return usage


# ---- message extraction (v1.40 + v1.37 + legacy) --------------------------


def _extract_messages(
    attrs: dict[str, Any], events: list[dict[str, Any]], *, direction: str
) -> list[dict[str, Any]]:
    """Return a v1.40-normalised list of messages.

    direction="input"  → `gen_ai.input.messages` / `gen_ai.prompt.N.*`
    direction="output" → `gen_ai.output.messages` / `gen_ai.completion.N.*`
    """
    # Preferred path: v1.37+ structured messages attribute.
    structured_key = "gen_ai.input.messages" if direction == "input" else "gen_ai.output.messages"
    structured = _maybe_parse_json(attrs.get(structured_key))
    if isinstance(structured, list) and structured:
        normalised = [_normalise_message(m) for m in structured if isinstance(m, dict)]
        if direction == "input":
            sys_prefix = _extract_system_instructions(attrs)
            if sys_prefix:
                return [*sys_prefix, *normalised]
        return normalised

    # Intermediate path: deprecated per-message events
    # (gen_ai.user.message, gen_ai.assistant.message, gen_ai.tool.message,
    # gen_ai.system.message, gen_ai.choice).
    event_messages = _messages_from_deprecated_events(events, direction=direction)
    if event_messages:
        if direction == "input":
            sys_prefix = _extract_system_instructions(attrs)
            if sys_prefix:
                return [*sys_prefix, *event_messages]
        return event_messages

    # Legacy path: v1.28-v1.36 flat indexed attributes.
    legacy = _extract_legacy_messages(
        attrs,
        prefix="gen_ai.prompt." if direction == "input" else "gen_ai.completion.",
    )
    if direction == "input":
        sys_prefix = _extract_system_instructions(attrs)
        if sys_prefix:
            return [*sys_prefix, *legacy]
    return legacy


def _normalise_message(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce a v1.37/v1.40 message into Shadow's shape.

    Input shape: {role, parts, tool_calls?, tool_call_id?}
    Shadow shape: {role, content, tool_calls?, tool_call_id?}
        where content is either a string (single text part) or a list of
        typed content blocks.
    """
    role = str(raw.get("role", "user"))
    parts = raw.get("parts")
    content: Any
    if isinstance(parts, list):
        blocks: list[dict[str, Any]] = []
        for p in parts:
            if not isinstance(p, dict):
                continue
            ptype = str(p.get("type", "text"))
            if ptype == "text":
                blocks.append({"type": "text", "text": str(p.get("content", ""))})
            elif ptype == "image":
                blocks.append({"type": "image", "source": p.get("content") or p.get("source") or p})
            elif ptype == "tool_use":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(p.get("id", "")),
                        "name": str(p.get("name", "")),
                        "input": p.get("arguments") or p.get("input") or {},
                    }
                )
            elif ptype == "tool_result":
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": str(p.get("id", "") or p.get("tool_call_id", "")),
                        "content": p.get("content") or p.get("result") or "",
                    }
                )
            else:
                # Unknown part type passes through verbatim so the data isn't lost.
                blocks.append(p)
        # Collapse a single text block back into a plain string — that's
        # the shape Shadow's instrumentation emits for simple chats, and
        # it keeps the agentlog readable.
        if len(blocks) == 1 and blocks[0].get("type") == "text":
            content = blocks[0]["text"]
        else:
            content = blocks
    else:
        content = raw.get("content", "")

    out: dict[str, Any] = {"role": role, "content": content}
    # Assistant tool calls (v1.40 top-level on the message).
    tool_calls = raw.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        out["tool_calls"] = tool_calls
    # Tool-role messages carry the id of the tool_use they answer.
    tcid = raw.get("tool_call_id")
    if isinstance(tcid, str) and tcid:
        out["tool_call_id"] = tcid
    return out


def _extract_system_instructions(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """`gen_ai.system_instructions` can be a string or a parts array."""
    raw = _maybe_parse_json(attrs.get("gen_ai.system_instructions"))
    if raw is None:
        return []
    if isinstance(raw, str):
        if not raw.strip():
            return []
        return [{"role": "system", "content": raw}]
    if isinstance(raw, list):
        texts: list[str] = []
        for part in raw:
            if isinstance(part, dict) and isinstance(part.get("content"), str):
                texts.append(part["content"])
            elif isinstance(part, str):
                texts.append(part)
        if not texts:
            return []
        return [{"role": "system", "content": "\n".join(texts)}]
    return []


def _messages_from_deprecated_events(
    events: list[dict[str, Any]], *, direction: str
) -> list[dict[str, Any]]:
    """Reconstruct messages from the v1.28-v1.36 per-message event model."""
    role_by_event = {
        "gen_ai.user.message": "user",
        "gen_ai.system.message": "system",
        "gen_ai.assistant.message": "assistant",
        "gen_ai.tool.message": "tool",
    }
    keep_roles = {"user", "system", "tool"} if direction == "input" else {"assistant"}
    out: list[dict[str, Any]] = []
    for ev in events:
        name = str(ev.get("name", ""))
        if name == "gen_ai.choice" and direction == "output":
            ev_attrs = _attrs_as_dict(ev.get("attributes", []) or [])
            choice_msg = _maybe_parse_json(ev_attrs.get("message"))
            if isinstance(choice_msg, dict):
                out.append(_normalise_message(choice_msg))
            continue
        role = role_by_event.get(name)
        if role is None or role not in keep_roles:
            continue
        ev_attrs = _attrs_as_dict(ev.get("attributes", []) or [])
        content = ev_attrs.get("content")
        tc = _maybe_parse_json(ev_attrs.get("tool_calls"))
        built: dict[str, Any] = {"role": role, "content": content if content is not None else ""}
        if isinstance(tc, list) and tc:
            built["tool_calls"] = tc
        tcid = ev_attrs.get("id") or ev_attrs.get("tool_call_id")
        if isinstance(tcid, str) and tcid:
            built["tool_call_id"] = tcid
        out.append(built)
    return out


def _extract_legacy_messages(attrs: dict[str, Any], *, prefix: str) -> list[dict[str, Any]]:
    """Collect v1.28-v1.36 `gen_ai.prompt.N.*` / `gen_ai.completion.N.*` attrs."""
    by_idx: dict[int, dict[str, Any]] = {}
    for key, val in attrs.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        if "." not in rest:
            continue
        idx_str, field = rest.split(".", 1)
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        by_idx.setdefault(idx, {})[field] = val
    messages: list[dict[str, Any]] = []
    for i in sorted(by_idx.keys()):
        m = by_idx[i]
        role_default = "user" if prefix.endswith("prompt.") else "assistant"
        messages.append({"role": str(m.get("role", role_default)), "content": m.get("content", "")})
    return messages


def _messages_to_response_content_blocks(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten output-message list into Shadow's chat_response.content.

    A response message typically has role=assistant with a text part and
    optional tool_calls. We surface text parts as ``{"type":"text"}``
    blocks and each tool_call as a ``{"type":"tool_use"}`` block.
    """
    blocks: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict):
                    blocks.append(b)
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(tc.get("id", "")),
                    "name": str(tc.get("name") or tc.get("function", {}).get("name", "")),
                    "input": tc.get("arguments") or tc.get("input") or {},
                }
            )
    if not blocks:
        blocks = [{"type": "text", "text": ""}]
    return blocks


# ---- misc extractors ------------------------------------------------------


def _extract_stop_reason(attrs: dict[str, Any], span: dict[str, Any]) -> str:
    """Map `gen_ai.response.finish_reasons` (list) to Shadow's single stop string.

    Takes the first entry. When no finish_reason is present, infer from
    span.status: OTel status code 2 = error. Defaults to ``end_turn``.
    """
    fr = attrs.get("gen_ai.response.finish_reasons")
    if isinstance(fr, list) and fr:
        return str(fr[0])
    if isinstance(fr, str) and fr:
        return fr
    # error.type is authoritative when set.
    err = attrs.get("error.type") or attrs.get("gen_ai.error.type")
    if isinstance(err, str) and err:
        return "error"
    status_code = (span.get("status") or {}).get("code", 1)
    if status_code == 2:
        return "error"
    return "end_turn"


def _augment_metadata_from_spans(
    meta_payload: dict[str, Any],
    spans: list[dict[str, Any]],
    resource_attrs: dict[str, Any],
) -> None:
    """Attach provider, server, agent, and evaluation info to metadata.

    The metadata record is Shadow's single top-of-trace summary; OTel
    splits the same info across resource attributes, span attributes,
    and events. We flatten so `shadow diff` / `shadow check-policy`
    don't need to know the OTel topology.
    """
    providers: set[str] = set()
    servers: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []

    for span in spans:
        attrs = _attrs_as_dict(span.get("attributes", []))
        provider = attrs.get("gen_ai.provider.name") or attrs.get("gen_ai.system")
        if isinstance(provider, str) and provider:
            providers.add(provider)
        srv_addr = attrs.get("server.address")
        if isinstance(srv_addr, str) and srv_addr:
            servers.append(
                {
                    "address": srv_addr,
                    "port": attrs.get("server.port"),
                }
            )
        op = attrs.get("gen_ai.operation.name")
        if op in (_OP_CREATE_AGENT, _OP_INVOKE_AGENT):
            agent = {
                "id": attrs.get("gen_ai.agent.id"),
                "name": attrs.get("gen_ai.agent.name"),
                "description": attrs.get("gen_ai.agent.description"),
                "version": attrs.get("gen_ai.agent.version"),
            }
            agent = {k: v for k, v in agent.items() if v is not None}
            if agent:
                agents.append(agent)
        for ev in span.get("events", []) or []:
            if ev.get("name") != "gen_ai.evaluation.result":
                continue
            ev_attrs = _attrs_as_dict(ev.get("attributes", []) or [])
            evaluations.append(
                {
                    "name": ev_attrs.get("gen_ai.evaluation.name"),
                    "score": ev_attrs.get("gen_ai.evaluation.score.value"),
                    "label": ev_attrs.get("gen_ai.evaluation.score.label"),
                    "explanation": ev_attrs.get("gen_ai.evaluation.explanation"),
                }
            )

    if providers:
        meta_payload["provider"] = (
            sorted(providers)[0] if len(providers) == 1 else sorted(providers)
        )
    if servers:
        # De-dup identical (address, port) pairs.
        seen: set[tuple[str, Any]] = set()
        unique: list[dict[str, Any]] = []
        for s in servers:
            key = (s["address"], s.get("port"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(s)
        meta_payload["server"] = unique[0] if len(unique) == 1 else unique
    if agents:
        meta_payload["agents"] = agents
    if evaluations:
        meta_payload["evaluations"] = evaluations
    # Resource attributes (service.name etc) preserved under source.
    svc = resource_attrs.get("service.name")
    if isinstance(svc, str) and svc:
        meta_payload.setdefault("source", {})["service_name"] = svc


# ---- time / envelope helpers ---------------------------------------------


def _nano_to_iso(ns: Any) -> str:
    if ns is None:
        return _now_iso()
    try:
        n = int(ns)
        secs = n // 1_000_000_000
        ms = (n // 1_000_000) % 1000
        dt = datetime.datetime.fromtimestamp(secs, tz=datetime.UTC)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"
    except (TypeError, ValueError):
        return _now_iso()


def _duration_ms(start_ts: str, end_ts: str) -> int:
    try:
        start_dt = datetime.datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
        end_dt = datetime.datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
        return max(0, int((end_dt - start_dt).total_seconds() * 1000))
    except ValueError:
        return 0


def _envelope(
    kind: str,
    record_id: str,
    ts: str,
    parent: str | None,
    payload: dict[str, Any],
    *,
    otel_span_id: str,
    otel_trace_id: str,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if otel_trace_id:
        meta["otel_trace_id"] = otel_trace_id
    if otel_span_id:
        meta["otel_span_id"] = otel_span_id
    env: dict[str, Any] = {
        "version": "0.1",
        "id": record_id,
        "kind": kind,
        "ts": ts,
        "parent": parent,
        "payload": payload,
    }
    if meta:
        env["meta"] = meta
    return env


def _now_iso() -> str:
    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


__all__ = ["OTEL_FORMAT", "otel_to_agentlog"]
