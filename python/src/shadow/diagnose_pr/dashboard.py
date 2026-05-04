"""Hosted dashboard surface for `shadow diagnose-pr` reports.

A small FastAPI app that loads a `report.json` (or directory of
them) and renders a browsable web page with the verdict, dominant
cause, blast radius, top causes, per-trace diagnoses, and the
suggested fix.

Spec §1.3 explicitly listed dashboards as a non-goal for the
strategic-pivot wedge, but a thin local-server surface is useful
for two real workflows:

  1. Reviewing a CI-produced report.json without copy-pasting
     markdown into a chat tool.
  2. Sharing a result with someone who can't run the CLI.

Run via the `shadow dashboard` CLI command:

    shadow dashboard --report .shadow/diagnose-pr/report.json --port 8080

The server is single-process, single-report, no auth — meant for
local use or behind your own reverse proxy. Don't expose to the
public internet without a proxy doing auth + TLS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Shadow diagnose-pr — {verdict_upper}</title>
  <style>
    :root {{
      --fg: #1a1a1a;
      --bg: #ffffff;
      --muted: #666;
      --border: #e2e2e2;
      --code-bg: #f7f7f7;
      --ship: #16a34a;
      --probe: #ca8a04;
      --hold: #ea580c;
      --stop: #dc2626;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --fg: #e8e8e8;
        --bg: #0d0d0d;
        --muted: #9a9a9a;
        --border: #2a2a2a;
        --code-bg: #1a1a1a;
      }}
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      max-width: 880px;
      margin: 2rem auto;
      padding: 0 1.5rem;
      color: var(--fg);
      background: var(--bg);
      line-height: 1.55;
    }}
    h1 {{
      font-size: 1.6rem;
      margin-bottom: 0.3rem;
    }}
    .verdict {{
      display: inline-block;
      padding: 0.25rem 0.75rem;
      border-radius: 6px;
      color: white;
      font-weight: 600;
      font-size: 0.95rem;
      letter-spacing: 0.04em;
    }}
    .verdict-ship  {{ background: var(--ship); }}
    .verdict-probe {{ background: var(--probe); }}
    .verdict-hold  {{ background: var(--hold); }}
    .verdict-stop  {{ background: var(--stop); }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 0.75rem;
      margin: 1.5rem 0;
    }}
    .stat-card {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.9rem 1rem;
    }}
    .stat-label {{ color: var(--muted); font-size: 0.85rem; }}
    .stat-value {{ font-size: 1.5rem; font-weight: 600; margin-top: 0.2rem; }}
    h2 {{
      margin-top: 2rem;
      font-size: 1.15rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.4rem;
    }}
    code {{
      background: var(--code-bg);
      padding: 0.1rem 0.4rem;
      border-radius: 3px;
      font-size: 0.9em;
    }}
    pre {{
      background: var(--code-bg);
      padding: 1rem;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 0.85rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
    }}
    th, td {{
      text-align: left;
      padding: 0.5rem 0.75rem;
      border-bottom: 1px solid var(--border);
      font-size: 0.9rem;
    }}
    th {{ font-weight: 600; color: var(--muted); }}
    .delta-id {{ font-family: monospace; }}
    .flag {{
      display: inline-block;
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 0.1rem 0.5rem;
      font-size: 0.8rem;
      margin-right: 0.5rem;
    }}
    footer {{
      margin-top: 3rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.85rem;
    }}
  </style>
</head>
<body>
  <h1>Shadow diagnose-pr</h1>
  <p>
    <span class="verdict verdict-{verdict}">{verdict_upper}</span>
    {flags_html}
  </p>
  <p style="color: var(--muted); margin-top: 0.5rem;">{verdict_blurb}</p>

  <div class="stat-grid">
    <div class="stat-card">
      <div class="stat-label">Affected traces</div>
      <div class="stat-value">{affected} / {total}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Blast radius</div>
      <div class="stat-value">{blast_pct}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Policy violations</div>
      <div class="stat-value">{new_violations}</div>
    </div>
  </div>

  {dominant_cause_html}

  {policy_html}

  {top_causes_html}

  {suggested_fix_html}

  <h2>Per-trace diagnoses</h2>
  <table>
    <thead>
      <tr>
        <th>Trace ID</th>
        <th>Affected</th>
        <th>Worst axis</th>
        <th>Policy violations</th>
      </tr>
    </thead>
    <tbody>
      {trace_rows}
    </tbody>
  </table>

  <footer>
    Schema: <code>{schema_version}</code>.
    Source report: <code>{source_path}</code>.
  </footer>
</body>
</html>
"""


_VERDICT_BLURBS = {
    "ship": "No behavior regression detected against the production-like trace sample.",
    "probe": "Behavior changed but the effect is uncertain (CI crosses zero).",
    "hold": "This PR changes agent behavior with measurable effect.",
    "stop": "This PR violates a critical policy and must not merge as-is.",
}


def _esc(text: str) -> str:
    """HTML-escape a string for safe insertion into the template."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_report_html(report: dict[str, Any], source_path: str = "") -> str:
    """Render a diagnose-pr `report.json` (already-parsed dict) to
    a self-contained HTML page string."""
    verdict = str(report.get("verdict", "ship"))
    total = int(report.get("total_traces", 0))
    affected = int(report.get("affected_traces", 0))
    blast = float(report.get("blast_radius", 0.0))
    flags = report.get("flags") or []
    flags_html = "".join(f'<span class="flag">{_esc(f)}</span>' for f in flags)

    dom = report.get("dominant_cause")
    if dom:
        ate = dom.get("ate")
        ci_low = dom.get("ci_low")
        ci_high = dom.get("ci_high")
        e_value = dom.get("e_value")
        ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]" if ci_low is not None and ci_high is not None else "n/a"
        e_str = f"{e_value:.1f}" if e_value is not None else "n/a"
        ate_str = f"{ate:+.2f}" if isinstance(ate, (int, float)) else "n/a"
        dominant_cause_html = (
            f"<h2>Dominant cause</h2>"
            f'<p><code class="delta-id">{_esc(dom.get("delta_id", ""))}</code> on '
            f'axis <code>{_esc(dom.get("axis", ""))}</code></p>'
            f"<table>"
            f"<tr><th>ATE</th><td>{ate_str}</td></tr>"
            f"<tr><th>95% CI</th><td>{ci_str}</td></tr>"
            f"<tr><th>E-value</th><td>{e_str}</td></tr>"
            f'<tr><th>Confidence</th><td>{dom.get("confidence", 0.5):.1f}</td></tr>'
            f"</table>"
        )
    else:
        dominant_cause_html = ""

    worst_rule = report.get("worst_policy_rule")
    n_viols = int(report.get("new_policy_violations", 0))
    if worst_rule and n_viols > 0:
        policy_html = (
            f"<h2>Why it matters</h2>"
            f"<p>{n_viols} traces violate the <code>{_esc(worst_rule)}</code> policy rule.</p>"
        )
    else:
        policy_html = ""

    top_causes = report.get("top_causes") or []
    if top_causes:
        # Sort by |ate|*confidence; show top 5.
        ranked = sorted(
            top_causes,
            key=lambda c: abs(float(c.get("ate", 0.0))) * float(c.get("confidence", 0.5)),
            reverse=True,
        )[:5]
        rows = "\n".join(
            f'<tr><td><code class="delta-id">{_esc(c.get("delta_id", ""))}</code></td>'
            f'<td>{_esc(c.get("axis", ""))}</td>'
            f'<td>{float(c.get("ate", 0.0)):+.2f}</td>'
            f'<td>{float(c.get("confidence", 0.5)):.1f}</td></tr>'
            for c in ranked
        )
        top_causes_html = (
            f"<h2>Top causes</h2>"
            f"<table><thead><tr><th>Delta</th><th>Axis</th><th>ATE</th><th>Confidence</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    else:
        top_causes_html = ""

    suggested = report.get("suggested_fix")
    if suggested:
        suggested_fix_html = f"<h2>Suggested fix</h2><p>{_esc(suggested)}</p>"
    else:
        suggested_fix_html = ""

    diagnoses = report.get("trace_diagnoses") or []
    if diagnoses:
        trace_rows = "\n".join(
            f'<tr><td><code>{_esc(d.get("trace_id", "")[:24])}…</code></td>'
            f'<td>{"yes" if d.get("affected") else "no"}</td>'
            f'<td>{_esc(d.get("worst_axis") or "—")}</td>'
            f'<td>{len(d.get("policy_violations") or [])}</td></tr>'
            for d in diagnoses[:50]
        )
    else:
        trace_rows = '<tr><td colspan="4" style="color: var(--muted);">no per-trace diagnoses</td></tr>'

    return _HTML_TEMPLATE.format(
        verdict=verdict,
        verdict_upper=verdict.upper(),
        verdict_blurb=_esc(_VERDICT_BLURBS.get(verdict, "")),
        flags_html=flags_html,
        total=total,
        affected=affected,
        blast_pct=f"{blast * 100:.1f}%",
        new_violations=n_viols,
        dominant_cause_html=dominant_cause_html,
        policy_html=policy_html,
        top_causes_html=top_causes_html,
        suggested_fix_html=suggested_fix_html,
        trace_rows=trace_rows,
        schema_version=_esc(report.get("schema_version", "diagnose-pr/v0.1")),
        source_path=_esc(source_path),
    )


def build_app(report_path: Path) -> Any:
    """Build the FastAPI app that serves the dashboard.

    Imported lazily — `shadow.diagnose_pr.dashboard` doesn't pull
    fastapi at module load time. Only the `shadow dashboard`
    command + this build_app pull it in.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as exc:
        raise RuntimeError(
            "shadow dashboard requires fastapi. Install with: "
            "pip install 'shadow-diff[dashboard]' "
            "(or: pip install fastapi uvicorn)"
        ) from exc

    app = FastAPI(
        title="Shadow diagnose-pr dashboard",
        version="0.1.0",
        docs_url=None,  # don't expose /docs by default — local tool
        redoc_url=None,
    )

    def _load_report() -> dict[str, Any]:
        if not report_path.is_file():
            raise HTTPException(404, f"report not found: {report_path}")
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"could not parse report: {e}") from e

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return render_report_html(_load_report(), source_path=str(report_path))

    @app.get("/report.json", response_class=JSONResponse)
    def raw_report() -> dict[str, Any]:
        return _load_report()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


__all__ = ["build_app", "render_report_html"]
