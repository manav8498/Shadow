"""Push Shadow `.agentlog` records to Splunk via the HTTP Event Collector (HEC).

Splunk HEC accepts one JSON event per POST, or a batch of events in a single
POST when newline-delimited. We emit one event per `.agentlog` record.

Usage:
    SPLUNK_HEC_URL=https://<host>:8088/services/collector \
        SPLUNK_HEC_TOKEN=<token> \
        python push_to_splunk_hec.py <trace.agentlog>
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python" / "src"))

import httpx

from shadow import _core


def push(agentlog_path: Path, url: str, token: str) -> httpx.Response:
    records = _core.parse_agentlog(agentlog_path.read_bytes())
    # One Splunk event per Shadow record. `time` is epoch seconds.
    lines = []
    for r in records:
        lines.append(
            json.dumps(
                {
                    "time": time.time(),
                    "source": "shadow",
                    "sourcetype": "shadow:agentlog",
                    "event": r,
                }
            )
        )
    return httpx.post(
        url,
        content="\n".join(lines),
        headers={
            "Authorization": f"Splunk {token}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
        verify=os.environ.get("SPLUNK_VERIFY_TLS", "true").lower() != "false",
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: push_to_splunk_hec.py <trace.agentlog>", file=sys.stderr)
        return 2
    url = os.environ.get("SPLUNK_HEC_URL")
    token = os.environ.get("SPLUNK_HEC_TOKEN")
    if not url or not token:
        print("SPLUNK_HEC_URL + SPLUNK_HEC_TOKEN required", file=sys.stderr)
        return 2
    resp = push(Path(argv[1]), url, token)
    print(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return 0 if resp.is_success else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
