"""Integration tests: Shadow → Datadog / Splunk / OTel Collector.

Each test starts a local mock receiver on a free port, points the
example script at it via env var, and verifies the payload shape.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import typing
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from shadow.sdk import Session

INTEGRATION_DIR = Path(__file__).resolve().parents[2] / "examples" / "integrations"
sys.path.insert(0, str(INTEGRATION_DIR))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_tiny_trace(path: Path) -> None:
    with Session(output_path=path, auto_instrument=False, session_tag="integration") as s:
        s.record_chat(
            request={"model": "x", "messages": [{"role": "user", "content": "hi"}], "params": {}},
            response={
                "model": "x",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "latency_ms": 100,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )


class _CollectingHandler(BaseHTTPRequestHandler):
    """Collect every POSTed body + headers for assertion."""

    received: typing.ClassVar[list[tuple[dict[str, str], bytes]]] = []

    def do_POST(self) -> None:  # noqa: N802  # stdlib BaseHTTPRequestHandler hook name
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        _CollectingHandler.received.append((dict(self.headers), body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, *args: Any, **kwargs: Any) -> None:
        return  # silence stdout


def _start_mock(port: int) -> HTTPServer:
    _CollectingHandler.received = []
    server = HTTPServer(("127.0.0.1", port), _CollectingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ---------------------------------------------------------------------------


def test_datadog_push_sends_otlp_with_api_key_header(tmp_path: Path) -> None:
    sys.path.insert(0, str(INTEGRATION_DIR / "datadog"))
    from push_to_datadog import push  # type: ignore[import-not-found]

    port = _free_port()
    server = _start_mock(port)
    try:
        trace = tmp_path / "t.agentlog"
        _write_tiny_trace(trace)
        resp = push(trace, f"http://127.0.0.1:{port}/traces", "fake-dd-key")
        assert resp.status_code == 200
    finally:
        server.shutdown()

    assert len(_CollectingHandler.received) == 1
    headers, body = _CollectingHandler.received[0]
    assert headers.get("DD-API-KEY") == "fake-dd-key" or headers.get("Dd-Api-Key") == "fake-dd-key"
    payload = json.loads(body)
    assert "resourceSpans" in payload
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 1  # one chat pair → one span


def test_splunk_hec_push_sends_one_event_per_record(tmp_path: Path) -> None:
    sys.path.insert(0, str(INTEGRATION_DIR / "splunk"))
    from push_to_splunk_hec import push  # type: ignore[import-not-found]

    port = _free_port()
    server = _start_mock(port)
    try:
        trace = tmp_path / "t.agentlog"
        _write_tiny_trace(trace)
        resp = push(trace, f"http://127.0.0.1:{port}/services/collector", "fake-token")
        assert resp.status_code == 200
    finally:
        server.shutdown()

    headers, body = _CollectingHandler.received[0]
    assert headers.get("Authorization", "").startswith("Splunk fake-token")
    # Body is newline-delimited JSON — 1 line per record (metadata + 2 chat records).
    lines = body.decode().strip().splitlines()
    assert len(lines) == 3
    for raw in lines:
        evt = json.loads(raw)
        assert evt["source"] == "shadow"
        assert evt["sourcetype"] == "shadow:agentlog"
        assert "event" in evt


def test_otel_collector_push_sends_otlp(tmp_path: Path) -> None:
    sys.path.insert(0, str(INTEGRATION_DIR / "otel-collector"))
    from push_to_collector import push  # type: ignore[import-not-found]

    port = _free_port()
    server = _start_mock(port)
    try:
        trace = tmp_path / "t.agentlog"
        _write_tiny_trace(trace)
        resp = push(trace, f"http://127.0.0.1:{port}/v1/traces")
        assert resp.status_code == 200
    finally:
        server.shutdown()

    _, body = _CollectingHandler.received[0]
    payload = json.loads(body)
    assert "resourceSpans" in payload


def test_datadog_push_without_key_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(INTEGRATION_DIR / "datadog"))
    from push_to_datadog import main  # type: ignore[import-not-found]

    monkeypatch.delenv("DD_API_KEY", raising=False)
    trace = tmp_path / "t.agentlog"
    _write_tiny_trace(trace)
    assert main(["prog", str(trace)]) == 2
