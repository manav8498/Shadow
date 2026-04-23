"""Concurrent-clients stress for `shadow serve`.

Spins up a real uvicorn server in a background thread, opens 10
concurrent WebSocket clients, and asserts every client receives
notifications when new traces land.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path

import pytest

fastapi_installed: bool
try:
    import fastapi  # type: ignore[import-not-found]  # noqa: F401
    import uvicorn  # type: ignore[import-not-found]
    import websockets  # type: ignore[import-not-found]

    fastapi_installed = True
except ImportError:
    fastapi_installed = False

pytestmark = pytest.mark.skipif(
    not fastapi_installed, reason="requires shadow[serve] + websockets client"
)

from shadow.sdk import Session  # noqa: E402


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_trace(path: Path, text: str) -> None:
    with Session(output_path=path, auto_instrument=False, session_tag="stress") as s:
        s.record_chat(
            request={"model": "x", "messages": [], "params": {}},
            response={
                "model": "x",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "latency_ms": 100,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )


@pytest.mark.asyncio
async def test_ten_concurrent_ws_clients_each_get_notifications(tmp_path: Path) -> None:
    from shadow.serve import build_app

    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()

    port = _free_port()
    app = build_app(tmp_path)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    # Wait for uvicorn to bind.
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("uvicorn did not bind within 2.5s")

    try:
        # Preload with one trace so all clients see it on connection.
        _write_trace(traces_dir / "warmup.agentlog", "warmup")

        num_clients = 10
        received: list[list[str]] = [[] for _ in range(num_clients)]

        async def client(idx: int) -> None:
            uri = f"ws://127.0.0.1:{port}/ws"
            async with websockets.connect(uri) as ws:
                try:
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                        data = json.loads(raw)
                        if data.get("type") == "trace":
                            received[idx].append(data["trace"]["path"])
                            if len(received[idx]) >= 2:
                                return
                except TimeoutError:
                    return

        # Start all clients concurrently; after 1s, drop a second trace.
        tasks = [asyncio.create_task(client(i)) for i in range(num_clients)]
        await asyncio.sleep(1.2)
        _write_trace(traces_dir / "post_connect.agentlog", "second")
        await asyncio.gather(*tasks)

        # Every client should have received BOTH traces.
        for i, paths in enumerate(received):
            assert any(p.endswith("warmup.agentlog") for p in paths), f"client {i} missed warmup"
            assert any(
                p.endswith("post_connect.agentlog") for p in paths
            ), f"client {i} missed second"
    finally:
        server.should_exit = True
        server_thread.join(timeout=5.0)


@pytest.mark.asyncio
async def test_concurrent_rest_calls_dont_race(tmp_path: Path) -> None:
    """Many parallel /api/diff calls should all return valid reports."""
    from fastapi.testclient import TestClient

    from shadow.serve import build_app

    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_trace(traces_dir / "a.agentlog", "one")
    _write_trace(traces_dir / "b.agentlog", "two")

    app = build_app(tmp_path)
    client = TestClient(app)

    async def fire() -> dict:
        r = client.get(
            f"/api/diff?baseline={traces_dir / 'a.agentlog'}"
            f"&candidate={traces_dir / 'b.agentlog'}"
        )
        return r.json()

    results = await asyncio.gather(*[fire() for _ in range(20)])
    for r in results:
        assert len(r["rows"]) == 9
