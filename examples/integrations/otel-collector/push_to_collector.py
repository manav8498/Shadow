"""Push a Shadow `.agentlog` to a local OpenTelemetry Collector.

Requires `otelcol-contrib --config collector-config.yaml` running and
listening on localhost:4318 (OTLP/HTTP).

    python push_to_collector.py <trace.agentlog>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python" / "src"))

import httpx

from shadow import _core
from shadow.otel import agentlog_to_otel


def push(agentlog_path: Path, url: str) -> httpx.Response:
    records = _core.parse_agentlog(agentlog_path.read_bytes())
    otlp = agentlog_to_otel(records)
    return httpx.post(
        url,
        json=otlp,
        headers={"Content-Type": "application/json"},
        timeout=30.0,
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: push_to_collector.py <trace.agentlog>", file=sys.stderr)
        return 2
    url = os.environ.get("OTEL_COLLECTOR_URL", "http://localhost:4318/v1/traces")
    resp = push(Path(argv[1]), url)
    print(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return 0 if resp.is_success else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
