"""Push a Shadow `.agentlog` to Datadog via OTLP/HTTP.

Datadog's ingest endpoint accepts OTLP/HTTP POSTs at
`https://trace.agent.{site}/api/v0.2/traces`. We convert the agentlog
to OTLP via `shadow export --format otel`, then POST it.

Usage:
    DD_API_KEY=... DD_SITE=datadoghq.com \
        python push_to_datadog.py <trace.agentlog>

For local/CI tests, point `DD_ENDPOINT` at a mock (see test_push.py).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python" / "src"))

import httpx

from shadow import _core
from shadow.otel import agentlog_to_otel


def push(agentlog_path: Path, endpoint: str, api_key: str) -> httpx.Response:
    records = _core.parse_agentlog(agentlog_path.read_bytes())
    otlp = agentlog_to_otel(records)
    return httpx.post(
        endpoint,
        json=otlp,
        headers={
            "Content-Type": "application/json",
            "DD-API-KEY": api_key,
        },
        timeout=30.0,
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: push_to_datadog.py <trace.agentlog>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    site = os.environ.get("DD_SITE", "datadoghq.com")
    endpoint = os.environ.get(
        "DD_ENDPOINT",
        f"https://trace.agent.{site}/api/v0.2/traces",
    )
    api_key = os.environ.get("DD_API_KEY")
    if not api_key:
        print("DD_API_KEY not set", file=sys.stderr)
        return 2
    resp = push(path, endpoint, api_key)
    print(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return 0 if resp.is_success else 1


def _unused_any(_: Any) -> None: ...


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
