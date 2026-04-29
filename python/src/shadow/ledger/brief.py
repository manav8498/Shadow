"""Formatters and webhook posting for ``shadow brief``.

Three formats over the same :class:`LedgerView`:

    * ``terminal`` — plain text suitable for echoing into a Slack
      thread, an email, or a CI artifact. No box drawing, copies
      cleanly across most terminals.
    * ``markdown`` — GitHub-flavoured Markdown. Trace ids appear as
      code spans so a reader can copy them straight back into
      ``shadow trail``.
    * ``slack`` — Block Kit JSON. Posts via stdlib ``urllib``; no
      new runtime dependency.

The formatters are pure (string in / string out). The webhook poster
is the only function in this module that does network I/O; it fails
soft on errors so a missing webhook never breaks the surrounding loop.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

from shadow.ledger.view import LedgerView, relative_time

# ---------------------------------------------------------------------------
# Public formatter API
# ---------------------------------------------------------------------------


def format_brief_terminal(view: LedgerView) -> str:
    """Plain-text brief suitable for piping into other tools."""
    return _render_text(view, link_traces=False)


def format_brief_markdown(view: LedgerView) -> str:
    """GitHub-flavoured Markdown brief."""
    return _render_markdown(view)


def format_brief_slack(view: LedgerView) -> dict[str, Any]:
    """Block Kit payload ready for ``urllib`` to POST.

    Wraps the markdown brief in a single mrkdwn section block plus a
    short context footer. Slack's mrkdwn dialect renders most of
    what GitHub Markdown does, with a few exceptions (no headings, no
    nested lists). The markdown formatter intentionally avoids both.
    """
    body = _render_markdown(view, slack_compat=True)
    return {
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": body},
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": ("_Verifiable: every trace id is a SHA-256 prefix._"),
                    }
                ],
            },
        ]
    }


# ---------------------------------------------------------------------------
# Webhook poster
# ---------------------------------------------------------------------------


def post_to_slack(webhook_url: str, payload: dict[str, Any]) -> tuple[bool, str]:
    """POST a Block Kit payload to a Slack incoming webhook.

    Returns ``(ok, message)``. Never raises — network or HTTP errors
    surface as ``ok=False`` with a short reason string so callers can
    log a warning and continue. Uses stdlib only so this module adds
    zero new runtime dependencies.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace").strip()
            ok = 200 <= resp.status < 300
            return ok, body or f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, f"network error: {e.reason}"
    except (TimeoutError, OSError) as e:
        return False, f"network error: {e}"


# ---------------------------------------------------------------------------
# Plain-text rendering
# ---------------------------------------------------------------------------


def _render_text(view: LedgerView, *, link_traces: bool) -> str:
    """Plain ASCII brief. ``link_traces`` is ignored — kept as a
    parameter so future integrations can swap rendering without
    changing the public function signature.
    """
    del link_traces  # placeholder for symmetry with the markdown side
    lines: list[str] = []
    lines.append(f"Shadow brief — {_date_part(view.now)}")
    lines.append("=" * 40)

    # Pass-rate summary
    pr = view.pass_rate
    if pr.total > 0:
        lines.append(
            f"Anchor pass rate   {pr.successes} of {pr.total} calls   "
            f"{pr.display_rate}   95% CI {pr.display_ci}"
        )
    else:
        lines.append("Anchor pass rate   no calls logged yet")

    # Most-concerning entry
    if view.most_concerning is not None:
        mc = view.most_concerning
        when = relative_time(mc.timestamp, now=view.now) if view.now else "—"
        summary = mc.driver_summary or (mc.primary_axis or "regression")
        lines.append(
            f"Most concerning    {mc.anchor_id}   {_text_tier(mc.tier)}   " f"{summary}   ({when})"
        )

    if not view.entries:
        lines.append("")
        lines.append("(no entries — opt in with `shadow call --log` or `shadow log <report>`)")
        return "\n".join(lines)

    # Recent entries — up to 5 lines, terse
    lines.append("")
    lines.append("Recent")
    for e in view.entries[:5]:
        when = relative_time(e.timestamp, now=view.now) if view.now else "—"
        tier = _text_tier(e.tier)
        summary = e.driver_summary or (e.primary_axis or "—")
        lines.append(f"  {e.anchor_id:<10} {when:<12} {e.kind:<6} {tier:<6} {summary}")

    # Action suggestion
    lines.append("")
    if view.most_concerning is not None and view.most_concerning.anchor_id:
        lines.append(f"Next: shadow trail {view.most_concerning.anchor_id}")
    else:
        lines.append("Next: shadow ledger")
    return "\n".join(lines)


def _text_tier(tier: str | None) -> str:
    """ASCII renderings of the tier label for plain-text mode."""
    if tier is None:
        return "—"
    return tier


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_markdown(view: LedgerView, *, slack_compat: bool = False) -> str:
    """GitHub-flavoured Markdown. ``slack_compat=True`` swaps a few
    constructs for ones Slack's mrkdwn dialect understands.
    """
    lines: list[str] = []
    header_prefix = "*" if slack_compat else "**"

    if slack_compat:
        # Slack mrkdwn doesn't render `#` headings; use bold instead.
        lines.append(f"{header_prefix}Shadow brief — {_date_part(view.now)}{header_prefix}")
    else:
        lines.append(f"### Shadow brief — {_date_part(view.now)}")

    lines.append("")

    pr = view.pass_rate
    if pr.total > 0:
        lines.append(
            f"{header_prefix}Anchor pass rate{header_prefix} "
            f"{pr.successes} of {pr.total} calls — "
            f"{pr.display_rate} (95% CI {pr.display_ci})"
        )
    else:
        lines.append(f"{header_prefix}Anchor pass rate{header_prefix} no calls logged yet")

    if view.most_concerning is not None:
        mc = view.most_concerning
        when = relative_time(mc.timestamp, now=view.now) if view.now else "—"
        summary = mc.driver_summary or (mc.primary_axis or "regression")
        lines.append("")
        lines.append(
            f"{header_prefix}Most concerning{header_prefix} "
            f"`{mc.anchor_id}` — {summary} "
            f"({_text_tier(mc.tier)}, {when})"
        )

    if view.entries:
        lines.append("")
        lines.append(f"{header_prefix}Recent{header_prefix}")
        for e in view.entries[:5]:
            when = relative_time(e.timestamp, now=view.now) if view.now else "—"
            summary = e.driver_summary or (e.primary_axis or "—")
            tier = _text_tier(e.tier)
            lines.append(f"- `{e.anchor_id}` ({when}) — {e.kind} / {tier} — {summary}")

        lines.append("")
        if view.most_concerning is not None and view.most_concerning.anchor_id:
            cmd = f"shadow trail {view.most_concerning.anchor_id}"
        else:
            cmd = "shadow ledger"
        lines.append(f"{header_prefix}Next{header_prefix} `{cmd}`")
    else:
        lines.append("")
        lines.append("_No entries — opt in with `shadow call --log` or `shadow log <report>`._")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _date_part(now: datetime | None) -> str:
    """ISO date portion of a timestamp, or ``"now"`` when missing."""
    if now is None:
        return "now"
    return now.date().isoformat()
