"""FastAPI app for `shadow serve` — live diff dashboard."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from shadow import _core
from shadow.errors import ShadowBackendError


def build_app(root: Path) -> Any:
    """Build a FastAPI app tailing the given `.shadow/` directory.

    Separated from `serve` so tests can mount the app into a test client
    without running uvicorn.
    """
    try:
        import fastapi
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as e:
        raise ShadowBackendError("fastapi not installed\nhint: pip install 'shadow[serve]'") from e

    WebSocketDisconnect = fastapi.WebSocketDisconnect  # noqa: N806 — class alias

    app = fastapi.FastAPI(title="Shadow", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML)

    @app.get("/api/traces")
    async def list_traces() -> JSONResponse:
        traces = _list_traces(root)
        return JSONResponse({"traces": traces})

    @app.get("/api/diff")
    async def diff(baseline: str, candidate: str) -> JSONResponse:
        # Restrict the readable-path set to files under `root`. Without this
        # guard, any authenticated caller can read arbitrary files on the
        # host via `?baseline=../../etc/passwd` (SECURITY).
        try:
            b_path = _safe_under_root(baseline, root)
            c_path = _safe_under_root(candidate, root)
        except ValueError as e:
            return JSONResponse({"error": "path outside root", "detail": str(e)}, status_code=400)
        if not b_path.is_file() or not c_path.is_file():
            return JSONResponse({"error": "trace not found"}, status_code=404)
        b = _core.parse_agentlog(b_path.read_bytes())
        c = _core.parse_agentlog(c_path.read_bytes())
        report = _core.compute_diff_report(b, c, None, 42)
        return JSONResponse(report)

    async def tail_ws(websocket: Any) -> None:  # Starlette WebSocket
        await websocket.accept()
        seen: set[str] = set()
        try:
            while True:
                current = _list_traces(root)
                paths = {t["path"] for t in current}
                new = [t for t in current if t["path"] not in seen]
                for t in new:
                    await websocket.send_json({"type": "trace", "trace": t})
                seen = paths
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            return
        except Exception:
            return

    # Register the WebSocket on the underlying Starlette router directly to
    # bypass FastAPI's dependency-injection machinery, which misinterprets
    # the `websocket` parameter as a query param on our typer/click combo.
    app.router.add_websocket_route("/ws", tail_ws)

    return app


def _safe_under_root(user_path: str, root: Path) -> Path:
    """Resolve `user_path`; raise ValueError if it escapes `root`.

    Defends against path-traversal (`../../etc/passwd`) and symlink escape.
    Accepts absolute paths only when they already sit under `root.resolve()`.
    """
    candidate = Path(user_path).resolve()
    root_abs = root.resolve()
    try:
        candidate.relative_to(root_abs)
    except ValueError as e:
        raise ValueError(f"{user_path} is outside {root_abs}") from e
    return candidate


def _list_traces(root: Path) -> list[dict[str, Any]]:
    """Return metadata for every `.agentlog` under `root/traces/`."""
    traces_dir = root / "traces"
    if not traces_dir.is_dir():
        # Fall back to any `.agentlog` directly under root.
        traces_dir = root
    out: list[dict[str, Any]] = []
    for p in sorted(traces_dir.rglob("*.agentlog")):
        try:
            records = _core.parse_agentlog(p.read_bytes())
        except Exception:
            continue
        root_rec = records[0] if records else None
        n_req = sum(1 for r in records if r.get("kind") == "chat_request")
        n_resp = sum(1 for r in records if r.get("kind") == "chat_response")
        out.append(
            {
                "path": str(p),
                "size": p.stat().st_size,
                "records": len(records),
                "chat_requests": n_req,
                "chat_responses": n_resp,
                "session_tag": ((root_rec or {}).get("meta") or {}).get("session_tag"),
                "trace_id": ((root_rec or {}).get("meta") or {}).get("trace_id"),
            }
        )
    return out


def serve(root: Path = Path(".shadow"), host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the dashboard with uvicorn (blocking)."""
    try:
        import uvicorn
    except ImportError as e:
        raise ShadowBackendError("uvicorn not installed\nhint: pip install 'shadow[serve]'") from e
    app = build_app(root)
    uvicorn.run(app, host=host, port=port, log_level="info")


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Shadow — live</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
           margin: 2rem; max-width: 1100px; }
    h1 { margin: 0 0 0.5rem 0; }
    .muted { color: #888; font-size: 0.9rem; }
    table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
    th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #333; }
    th { background: rgba(127,127,127,0.08); }
    .sev-none { color: #6c6; }
    .sev-minor { color: #cc6; }
    .sev-moderate { color: #e90; }
    .sev-severe { color: #e55; font-weight: bold; }
    .flag { font-size: 0.75rem; color: #999; }
    form { margin: 1rem 0; display: flex; gap: 0.5rem; }
    input, button, select { font-size: 1rem; padding: 0.3rem 0.6rem; }
    pre { background: rgba(127,127,127,0.08); padding: 0.5rem; border-radius: 4px;
          overflow-x: auto; }
  </style>
</head>
<body>
  <h1>Shadow</h1>
  <p class="muted">Live behavioral diff — traces tailed from <code id="root"></code></p>

  <h2>Traces</h2>
  <table id="traces"><thead><tr>
    <th>path</th><th>records</th><th>req</th><th>resp</th><th>tag</th><th>trace_id</th>
  </tr></thead><tbody></tbody></table>

  <h2>Run a diff</h2>
  <form id="diff-form">
    <select id="baseline"><option value="">baseline…</option></select>
    <select id="candidate"><option value="">candidate…</option></select>
    <button type="submit">Diff</button>
  </form>
  <div id="report"></div>

  <script>
    const SEV_EMOJI = {none: '🟢', minor: '🟡', moderate: '🟠', severe: '🔴'};
    let traces = [];

    async function refresh() {
      const r = await fetch('/api/traces');
      const data = await r.json();
      traces = data.traces;
      renderTraces();
    }

    function renderTraces() {
      const tbody = document.querySelector('#traces tbody');
      tbody.innerHTML = '';
      for (const t of traces) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td><code>${t.path}</code></td>`
          + `<td>${t.records}</td><td>${t.chat_requests}</td>`
          + `<td>${t.chat_responses}</td>`
          + `<td>${t.session_tag || '—'}</td>`
          + `<td class="muted">${(t.trace_id || '').slice(0, 12)}…</td>`;
        tbody.appendChild(tr);
      }
      for (const sel of ['baseline', 'candidate']) {
        const el = document.getElementById(sel);
        el.innerHTML = `<option value="">${sel}…</option>` +
          traces.map(t => `<option value="${t.path}">${t.path}</option>`).join('');
      }
    }

    document.getElementById('diff-form').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const b = document.getElementById('baseline').value;
      const c = document.getElementById('candidate').value;
      if (!b || !c) return;
      const r = await fetch(`/api/diff?baseline=${encodeURIComponent(b)}`
        + `&candidate=${encodeURIComponent(c)}`);
      const report = await r.json();
      renderReport(report);
    });

    function renderReport(report) {
      const rows = report.rows.map(r => {
        const sev = r.severity || 'none';
        const flags = (r.flags || []).join(',') || '—';
        return `<tr>
          <td>${r.axis}</td>
          <td>${r.baseline_median.toFixed(3)}</td>
          <td>${r.candidate_median.toFixed(3)}</td>
          <td>${r.delta >= 0 ? '+' : ''}${r.delta.toFixed(3)}</td>
          <td>[${r.ci95_low.toFixed(2)}, ${r.ci95_high.toFixed(2)}]</td>
          <td class="sev-${sev}">${SEV_EMOJI[sev] || ''} ${sev}</td>
          <td class="flag">${flags}</td>
          <td>${r.n}</td>
        </tr>`;
      }).join('');
      document.getElementById('report').innerHTML = `
        <table>
          <thead><tr>
            <th>axis</th><th>baseline</th><th>candidate</th><th>delta</th>
            <th>95% CI</th><th>severity</th><th>flags</th><th>n</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>`;
    }

    // Live updates.
    try {
      const ws = new WebSocket(`ws://${location.host}/ws`);
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'trace') refresh();
      };
    } catch (_) { /* ok if WS not available */ }

    refresh();
  </script>
</body>
</html>"""
