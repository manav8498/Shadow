"""Tests for the `shadow serve` dashboard (FastAPI app)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

fastapi_installed: bool
try:
    from fastapi.testclient import TestClient  # type: ignore[import-not-found]

    fastapi_installed = True
except ImportError:
    fastapi_installed = False


pytestmark = pytest.mark.skipif(
    not fastapi_installed, reason="fastapi not installed (shadow[serve] extra)"
)

from shadow.sdk import Session  # noqa: E402


def _write_trace(path: Path, text: str = "hi") -> None:
    with Session(output_path=path, auto_instrument=False, session_tag="demo") as s:
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


def test_serve_index_renders(tmp_path: Path) -> None:
    from shadow.serve import build_app

    app = build_app(tmp_path)
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "Shadow" in r.text
    assert "/api/traces" in r.text  # JS references it


def test_serve_lists_traces_from_the_root(tmp_path: Path) -> None:
    from shadow.serve import build_app

    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_trace(traces_dir / "a.agentlog", "hello")
    _write_trace(traces_dir / "b.agentlog", "world")

    app = build_app(tmp_path)
    client = TestClient(app)
    r = client.get("/api/traces")
    assert r.status_code == 200
    data = r.json()
    paths = {t["path"] for t in data["traces"]}
    assert str(traces_dir / "a.agentlog") in paths
    assert str(traces_dir / "b.agentlog") in paths
    for t in data["traces"]:
        assert t["session_tag"] == "demo"
        assert t["records"] >= 1


def test_serve_diff_endpoint_returns_nine_axes(tmp_path: Path) -> None:
    from shadow.serve import build_app

    base = tmp_path / "base.agentlog"
    cand = tmp_path / "cand.agentlog"
    _write_trace(base, "hello")
    _write_trace(cand, "goodbye")

    app = build_app(tmp_path)
    client = TestClient(app)
    r = client.get(f"/api/diff?baseline={base}&candidate={cand}")
    assert r.status_code == 200
    report = r.json()
    assert len(report["rows"]) == 9


def test_serve_api_diff_rejects_path_traversal(tmp_path: Path) -> None:
    """SECURITY: /api/diff must not read arbitrary files outside root."""
    from shadow.serve import build_app

    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_trace(traces_dir / "ok.agentlog", "hi")

    app = build_app(tmp_path)
    client = TestClient(app)
    # Attempt path traversal.
    r = client.get("/api/diff?baseline=/etc/passwd&candidate=/etc/passwd")
    assert r.status_code == 400, r.text
    assert "outside" in r.text.lower() or "path" in r.text.lower()

    # Relative traversal from a valid-looking path.
    bad = str(tmp_path / "traces" / ".." / ".." / ".." / "etc" / "passwd")
    r = client.get(f"/api/diff?baseline={bad}&candidate={bad}")
    assert r.status_code == 400

    # Legitimate request still works.
    path = str(traces_dir / "ok.agentlog")
    r = client.get(f"/api/diff?baseline={path}&candidate={path}")
    assert r.status_code == 200


def test_serve_api_diff_404s_on_nonexistent_trace(tmp_path: Path) -> None:
    from shadow.serve import build_app

    app = build_app(tmp_path)
    client = TestClient(app)
    # Under root but does not exist.
    missing = str(tmp_path / "nope.agentlog")
    r = client.get(f"/api/diff?baseline={missing}&candidate={missing}")
    assert r.status_code == 404


def test_serve_tolerates_missing_traces_dir(tmp_path: Path) -> None:
    from shadow.serve import build_app

    app = build_app(tmp_path / "nonexistent")
    client = TestClient(app)
    r = client.get("/api/traces")
    assert r.status_code == 200
    assert r.json() == {"traces": []}


def test_serve_cli_registered_in_app() -> None:
    """The `serve` subcommand is callable (function exists + wired)."""
    from shadow.cli.app import app as cli_app
    from shadow.cli.app import serve as serve_fn

    # The function is defined and decorated with @app.command().
    assert callable(serve_fn)
    # The command appears in typer's registry (name may be derived from fn name).
    fn_names = {getattr(cmd.callback, "__name__", None) for cmd in cli_app.registered_commands}
    assert "serve" in fn_names


def test_serve_websocket_delivers_trace_notifications(tmp_path: Path) -> None:
    from shadow.serve import build_app

    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    # Start with one trace, then add one after the WS opens.
    _write_trace(traces_dir / "a.agentlog")

    app = build_app(tmp_path)
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        # Initial scan should push the existing trace.
        msg_text = ws.receive_text()
        msg = json.loads(msg_text)
        assert msg["type"] == "trace"
        assert msg["trace"]["path"].endswith("a.agentlog")
