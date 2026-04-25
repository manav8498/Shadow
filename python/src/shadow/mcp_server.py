"""MCP server mode for Shadow.

Exposes Shadow's diff, bisect, policy, and schema-watch capabilities as
Model Context Protocol tools. Agentic CLIs that speak MCP (Claude Code,
Cursor, Zed, Claude Desktop, and any other MCP client) can invoke
Shadow as a tool without knowing anything about the CLI, by connecting
to `shadow mcp-serve` over stdio.

Tools exposed:

- `shadow_diff`  compute a nine-axis behavioral diff between two
  .agentlog files and return the full DiffReport JSON.
- `shadow_check_policy`  check a trace against a YAML or JSON policy
  and return regressions + fixes.
- `shadow_schema_watch`  classify tool-schema changes between two
  configs (breaking / risky / additive / neutral).
- `shadow_token_diff`  per-turn token distribution summary.
- `shadow_summarise`  plain-English summary of a saved DiffReport.
- `shadow_certify`  generate an Agent Behavior Certificate (ABOM) for
  a release trace and write it to disk.
- `shadow_verify_cert`  verify a certificate's content-addressed
  cert_id matches the body; tampering surfaces as `ok: false`.

A typical usage inside Claude Code or Cursor:

    shadow_diff(baseline="./baseline.agentlog",
                candidate="./candidate.agentlog",
                policy_path="./policy.yaml")

The host agent gets back a JSON object with rows, recommendations,
first-divergence, and (if a policy was supplied) regressions. It can
summarise that for the user, show a table, or use the signal to decide
whether to proceed with a merge.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from shadow import __version__, _core
from shadow.errors import ShadowError

SERVER_NAME = "shadow"


def _load_agentlog(path_str: str) -> list[dict[str, Any]]:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no such .agentlog file: {path}")
    return _core.parse_agentlog(path.read_bytes())


def _load_policy_file(path_str: str) -> list[Any]:
    from shadow.hierarchical import load_policy

    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no such policy file: {path}")
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return load_policy(raw)


async def _handle_diff(arguments: dict[str, Any]) -> dict[str, Any]:
    baseline = _load_agentlog(arguments["baseline"])
    candidate = _load_agentlog(arguments["candidate"])
    pricing = arguments.get("pricing")
    seed = int(arguments.get("seed", 42))

    price_map: dict[str, Any] | None = None
    if pricing:
        pricing_path = Path(pricing).expanduser().resolve()
        if pricing_path.is_file():
            price_map = json.loads(pricing_path.read_text())

    report = _core.compute_diff_report(baseline, candidate, price_map, seed)

    # Optional: check policy
    policy_path = arguments.get("policy_path")
    if policy_path:
        from shadow.hierarchical import policy_diff

        rules = _load_policy_file(policy_path)
        pdiff = policy_diff(baseline, candidate, rules)
        report["policy_diff"] = pdiff.to_dict()

    return report


async def _handle_check_policy(arguments: dict[str, Any]) -> dict[str, Any]:
    from shadow.hierarchical import policy_diff

    baseline = _load_agentlog(arguments["baseline"])
    candidate = _load_agentlog(arguments["candidate"])
    rules = _load_policy_file(arguments["policy_path"])
    pdiff = policy_diff(baseline, candidate, rules)
    return pdiff.to_dict()


async def _handle_token_diff(arguments: dict[str, Any]) -> dict[str, Any]:
    from shadow.hierarchical import token_diff

    baseline = _load_agentlog(arguments["baseline"])
    candidate = _load_agentlog(arguments["candidate"])
    top_k = int(arguments.get("top_k_pairs", 10))
    tdiff = token_diff(baseline, candidate, top_k_pairs=top_k)
    return tdiff.to_dict()


async def _handle_schema_watch(arguments: dict[str, Any]) -> dict[str, Any]:
    from shadow.schema_watch import classify_schema_changes

    cfg_a = Path(arguments["config_a"]).expanduser().resolve()
    cfg_b = Path(arguments["config_b"]).expanduser().resolve()
    if not cfg_a.is_file():
        raise FileNotFoundError(f"no such config: {cfg_a}")
    if not cfg_b.is_file():
        raise FileNotFoundError(f"no such config: {cfg_b}")
    import yaml

    a = yaml.safe_load(cfg_a.read_text())
    b = yaml.safe_load(cfg_b.read_text())
    result = classify_schema_changes(a, b)
    return result.to_dict() if hasattr(result, "to_dict") else result


async def _handle_summarise(arguments: dict[str, Any]) -> dict[str, Any]:
    from shadow.report.summary import summarise_report

    report_path = Path(arguments["report_json"]).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"no such report.json: {report_path}")
    report = json.loads(report_path.read_text())
    summary = summarise_report(report)
    return {"summary": summary}


async def _handle_certify(arguments: dict[str, Any]) -> dict[str, Any]:
    from shadow.certify import build_certificate

    trace = _load_agentlog(arguments["trace"])
    baseline_trace: list[dict[str, Any]] | None = None
    if arguments.get("baseline"):
        baseline_trace = _load_agentlog(arguments["baseline"])
    policy_path: Path | None = None
    if arguments.get("policy_path"):
        policy_path = Path(arguments["policy_path"]).expanduser().resolve()
        if not policy_path.is_file():
            raise FileNotFoundError(f"no such policy file: {policy_path}")
    cert = build_certificate(
        trace=trace,
        agent_id=str(arguments["agent_id"]),
        policy_path=policy_path,
        baseline_trace=baseline_trace,
    )
    output_path = Path(arguments["output"]).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cert.to_dict(), indent=2) + "\n")
    return {"cert_id": cert.cert_id, "output": str(output_path), "cert": cert.to_dict()}


async def _handle_verify_cert(arguments: dict[str, Any]) -> dict[str, Any]:
    from shadow.certify import verify_certificate

    cert_path = Path(arguments["cert"]).expanduser().resolve()
    if not cert_path.is_file():
        raise FileNotFoundError(f"no such certificate file: {cert_path}")
    payload = json.loads(cert_path.read_text())
    ok, detail = verify_certificate(payload)
    return {"ok": ok, "detail": detail, "cert_id": payload.get("cert_id")}


TOOL_HANDLERS = {
    "shadow_diff": _handle_diff,
    "shadow_check_policy": _handle_check_policy,
    "shadow_token_diff": _handle_token_diff,
    "shadow_schema_watch": _handle_schema_watch,
    "shadow_summarise": _handle_summarise,
    "shadow_certify": _handle_certify,
    "shadow_verify_cert": _handle_verify_cert,
}


def _build_tools() -> list[Any]:
    """Return the list of tool descriptors Shadow exposes."""
    from mcp.types import Tool

    return [
        Tool(
            name="shadow_diff",
            title="Nine-axis behavioral diff",
            description=(
                "Compute a nine-axis behavioral diff between two .agentlog "
                "trace files. Returns rows for semantic, trajectory, safety, "
                "verbosity, latency, cost, reasoning, judge, conformance; "
                "plus first-divergence, top regressive pairs, and "
                "recommendations. Pass `policy_path` to also enforce a YAML "
                "policy and include regressions in the response."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "baseline": {
                        "type": "string",
                        "description": "path to the baseline .agentlog file",
                    },
                    "candidate": {
                        "type": "string",
                        "description": "path to the candidate .agentlog file",
                    },
                    "pricing": {
                        "type": "string",
                        "description": "optional path to a pricing.json file for the cost axis",
                    },
                    "policy_path": {
                        "type": "string",
                        "description": "optional path to a policy YAML/JSON file",
                    },
                    "seed": {"type": "integer", "description": "bootstrap RNG seed (default 42)"},
                },
                "required": ["baseline", "candidate"],
            },
        ),
        Tool(
            name="shadow_check_policy",
            title="Policy-rule check",
            description=(
                "Check two traces against a YAML or JSON policy file. "
                "Supported rule kinds: must_call_before, must_call_once, "
                "no_call, max_turns, required_stop_reason, "
                "max_total_tokens, must_include_text, forbidden_text, "
                "must_match_json_schema, must_remain_consistent, "
                "must_followup, must_be_grounded. "
                "Each rule accepts `scope: trace` (default) or "
                "`scope: session` — session-scoped rules are evaluated "
                "independently per user-initiated session, which is "
                "almost always what you want on multi-ticket production "
                "traces. Returns baseline_violations, "
                "candidate_violations, regressions, and fixes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "baseline": {"type": "string"},
                    "candidate": {"type": "string"},
                    "policy_path": {"type": "string"},
                },
                "required": ["baseline", "candidate", "policy_path"],
            },
        ),
        Tool(
            name="shadow_token_diff",
            title="Token-level distribution diff",
            description=(
                "Per-dimension token distribution summary (input / output / "
                "thinking) between two traces, with top-k worst per-pair "
                "deltas. Useful for catching verbose-candidate or "
                "thinking-token blowups."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "baseline": {"type": "string"},
                    "candidate": {"type": "string"},
                    "top_k_pairs": {"type": "integer", "description": "default 10"},
                },
                "required": ["baseline", "candidate"],
            },
        ),
        Tool(
            name="shadow_schema_watch",
            title="Tool-schema change classifier",
            description=(
                "Classify tool-schema changes between two configs without "
                "replaying. Returns changes tiered into breaking / risky / "
                "additive / neutral."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "config_a": {"type": "string", "description": "path to config A YAML"},
                    "config_b": {"type": "string", "description": "path to config B YAML"},
                },
                "required": ["config_a", "config_b"],
            },
        ),
        Tool(
            name="shadow_summarise",
            title="Plain-English diff summary",
            description=(
                "Given a DiffReport JSON produced by shadow_diff (or "
                "`shadow diff --output-json`), produce a short paragraph "
                "summarising the regression for a reviewer. Deterministic, "
                "no LLM calls."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "report_json": {"type": "string", "description": "path to a saved report.json"},
                },
                "required": ["report_json"],
            },
        ),
        Tool(
            name="shadow_certify",
            title="Generate an Agent Behavior Certificate (ABOM)",
            description=(
                "Generate a content-addressed Agent Behavior Certificate "
                "for a release trace. The certificate captures the trace's "
                "content-id, all distinct models, content-ids of system "
                "prompts, content-ids of tool schemas, optional policy "
                "hash, and an optional baseline-vs-candidate nine-axis "
                "regression-suite rollup. Writes to `output` and returns "
                "the cert_id plus the full certificate body. The result "
                "is self-verifying via shadow_verify_cert."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "trace": {
                        "type": "string",
                        "description": "path to the release trace .agentlog file",
                    },
                    "agent_id": {
                        "type": "string",
                        "description": (
                            "stable identifier for the release, e.g. 'refund-agent@2.3.0'"
                        ),
                    },
                    "output": {
                        "type": "string",
                        "description": "path to write the certificate JSON",
                    },
                    "policy_path": {
                        "type": "string",
                        "description": "optional policy YAML/JSON path; recorded as policy_hash",
                    },
                    "baseline": {
                        "type": "string",
                        "description": "optional baseline .agentlog; folded into regression_suite",
                    },
                },
                "required": ["trace", "agent_id", "output"],
            },
        ),
        Tool(
            name="shadow_verify_cert",
            title="Verify an Agent Behavior Certificate",
            description=(
                "Verify a certificate's content-addressed cert_id matches "
                "the body. Returns `ok: true` when consistent, `ok: false` "
                "with a `detail` reason when tampered, malformed, or of an "
                "unsupported cert_version."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "cert": {
                        "type": "string",
                        "description": "path to a certificate JSON produced by shadow_certify",
                    },
                },
                "required": ["cert"],
            },
        ),
    ]


async def _call_tool_impl(name: str, arguments: dict[str, Any]) -> list[Any]:
    """Dispatch the tool call; always return a list of TextContent."""
    from mcp.types import TextContent

    if name not in TOOL_HANDLERS:
        payload = {"error": f"unknown tool: {name}", "available": sorted(TOOL_HANDLERS)}
        return [TextContent(type="text", text=json.dumps(payload))]

    try:
        result = await TOOL_HANDLERS[name](arguments)
    except (ShadowError, FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        payload = {"error": str(exc), "kind": type(exc).__name__}
        return [TextContent(type="text", text=json.dumps(payload))]
    except Exception as exc:
        payload = {"error": f"internal error: {exc}", "kind": type(exc).__name__}
        return [TextContent(type="text", text=json.dumps(payload))]

    return [TextContent(type="text", text=json.dumps(result, default=str))]


async def serve() -> None:
    """Run the Shadow MCP server over stdio until the client disconnects."""
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    server = Server(SERVER_NAME, version=__version__)

    @server.list_tools()
    async def _list_tools() -> list[Any]:
        return _build_tools()

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[Any]:
        return await _call_tool_impl(name, arguments or {})

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Entry point for `shadow mcp-serve`."""
    asyncio.run(serve())


__all__ = ["TOOL_HANDLERS", "main", "serve"]
