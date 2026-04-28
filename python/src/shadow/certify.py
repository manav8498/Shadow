"""Agent Behavior Certificate (ABOM): a release artefact that captures
what behavior shipped.

The same idea as a software SBOM, applied to AI agents. Given a trace
(produced from a real recording or a CI replay), this module produces
a JSON certificate that proves:

- which model + provider was in use
- which system prompts were sent (content-addressed)
- which tool schemas were exposed (content-addressed)
- which policy rules were enforced (content-addressed via the policy file)
- the per-axis severity rollup of the candidate against an optional
  baseline (the "regression suite" result)
- a single content-addressed ``cert_id`` over the whole certificate

Verification is deterministic: re-hash the payload, compare to the
``cert_id`` field. No PKI, no signing — that's a separate concern
covered in v1.8 (sigstore / cosign keyless integration).

The certificate format is intentionally small and human-readable so it
can live in a release branch alongside the artefact it describes.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from shadow import _core
from shadow.errors import ShadowConfigError

CERT_VERSION = "0.2"
_CERT_VERSION_0_1 = "0.1"
"""Bumped on backwards-incompatible certificate-shape changes."""


@dataclass
class AgentCertificate:
    """The ABOM payload. Serialise with :meth:`to_dict`."""

    cert_version: str
    agent_id: str
    released_at: str
    """ISO 8601 timestamp in UTC, second precision."""
    trace_id: str
    """Content-id of the trace's root metadata record."""
    models: list[str] = field(default_factory=list)
    """All distinct model ids observed in chat_request records."""
    prompt_hashes: list[str] = field(default_factory=list)
    """Content-ids of all distinct system prompts observed."""
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    """Per-tool ``{name, hash}`` pairs (hash over the canonical JSON
    of the tool schema). Empty when no tools were exposed."""
    policy_hash: str | None = None
    """Content-id of the policy YAML/JSON contents, or None if no
    policy was supplied."""
    regression_suite: dict[str, Any] | None = None
    """The nine-axis severity rollup from a baseline-vs-candidate diff,
    or None if no baseline was supplied. Shape:
    ``{"baseline_trace_id": str, "axes": [{axis, severity, delta}, ...],
    "conformal": {ConformalCoverageReport dict} | None}``."""
    cert_id: str = ""
    """Content-id of the rest of the certificate. Computed last; empty
    until :meth:`finalise` is called."""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # cert_id at the end for readability when humans read the JSON.
        cert_id = d.pop("cert_id")
        d["cert_id"] = cert_id
        return d


def build_certificate(
    *,
    trace: list[dict[str, Any]],
    agent_id: str,
    policy_path: Path | None = None,
    baseline_trace: list[dict[str, Any]] | None = None,
    pricing: dict[str, tuple[float, float]] | None = None,
    seed: int = 42,
    released_at: datetime.datetime | None = None,
    conformal_coverage: float | None = None,
    conformal_confidence: float = 0.95,
) -> AgentCertificate:
    """Construct an :class:`AgentCertificate` from a trace.

    ``trace`` is the list of records as parsed by ``_core.parse_agentlog``.
    ``baseline_trace`` is optional; when supplied, the nine-axis diff
    between baseline and the trace is folded into ``regression_suite``.
    ``policy_path`` is optional; when supplied, its content-id (sha256
    of its bytes) is recorded.

    ``conformal_coverage`` is optional; when supplied (e.g. 0.90),
    a conformal prediction coverage bound is computed and embedded in
    ``regression_suite["conformal"]``. Requires ``baseline_trace``.
    ``conformal_confidence`` is the PAC confidence level (default 0.95).

    The function does not perform replay — the caller is expected to
    have produced ``trace`` by recording or by calling Shadow's replay
    engine. Certifying a trace that was never validated is a
    user-error, not something this function can detect.
    """
    if not trace:
        raise ShadowConfigError("cannot certify an empty trace")
    if trace[0].get("kind") != "metadata":
        raise ShadowConfigError("trace must start with a metadata record")

    root = trace[0]
    trace_id = str(root["id"])

    # Extract distinct models, system prompts, and tools from the trace.
    models: list[str] = []
    seen_models: set[str] = set()
    prompt_hashes: list[str] = []
    seen_prompts: set[str] = set()
    tool_schemas: list[dict[str, Any]] = []
    seen_tools: set[str] = set()

    for rec in trace:
        if rec.get("kind") != "chat_request":
            continue
        payload = rec.get("payload") or {}
        m = payload.get("model")
        if isinstance(m, str) and m and m not in seen_models:
            models.append(m)
            seen_models.add(m)
        # System prompts live in the messages array as the first
        # role:system entry. There can be more than one across the
        # trace if the agent rebuilds context between sessions.
        for msg in payload.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content:
                pid = _core.content_id({"system_prompt": content})
                if pid not in seen_prompts:
                    prompt_hashes.append(pid)
                    seen_prompts.add(pid)
        # Tool schemas, if any.
        for tool in payload.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name") or "<unnamed>")
            tid = _core.content_id(tool)
            if tid not in seen_tools:
                tool_schemas.append({"name": name, "hash": tid})
                seen_tools.add(tid)

    policy_hash: str | None = None
    if policy_path is not None:
        try:
            data = policy_path.read_bytes()
        except OSError as e:
            raise ShadowConfigError(f"could not read policy file: {e}") from e
        policy_hash = "sha256:" + hashlib.sha256(data).hexdigest()

    regression_suite: dict[str, Any] | None = None
    if baseline_trace is not None:
        report = _core.compute_diff_report(baseline_trace, trace, pricing, seed)
        baseline_id = "<unknown>"
        if baseline_trace and baseline_trace[0].get("kind") == "metadata":
            baseline_id = str(baseline_trace[0]["id"])
        axis_rows = [
            {
                "axis": str(row.get("axis")),
                "severity": str(row.get("severity")),
                "delta": row.get("delta"),
                "n": row.get("n"),
                "ci95_low": row.get("ci95_low"),
                "ci95_high": row.get("ci95_high"),
            }
            for row in (report.get("rows") or [])
        ]
        conformal: dict[str, Any] | None = None
        if conformal_coverage is not None:
            from shadow.conformal import build_conformal_coverage

            conformal_report = build_conformal_coverage(
                axis_rows=axis_rows,
                target_coverage=conformal_coverage,
                confidence=conformal_confidence,
            )
            conformal = conformal_report.to_dict()
        regression_suite = {
            "baseline_trace_id": baseline_id,
            "axes": [
                {"axis": r["axis"], "severity": r["severity"], "delta": r["delta"]}
                for r in axis_rows
            ],
            "conformal": conformal,
        }

    if released_at is None:
        released_at = datetime.datetime.now(datetime.UTC)

    cert = AgentCertificate(
        cert_version=CERT_VERSION,
        agent_id=agent_id,
        released_at=released_at.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        trace_id=trace_id,
        models=models,
        prompt_hashes=prompt_hashes,
        tool_schemas=tool_schemas,
        policy_hash=policy_hash,
        regression_suite=regression_suite,
    )
    cert.cert_id = _hash_payload(cert)
    return cert


def verify_certificate(payload: dict[str, Any]) -> tuple[bool, str]:
    """Verify a certificate's internal consistency.

    Returns ``(ok, detail)`` where ``ok`` is ``True`` if the supplied
    ``cert_id`` matches the recomputed hash of the rest of the payload.
    On mismatch, ``detail`` carries a short reason. Does not validate
    that the trace, baseline, or policy still exist on disk.
    """
    if not isinstance(payload, dict):
        return False, "certificate must be a JSON object"
    supported_versions = {CERT_VERSION, _CERT_VERSION_0_1}
    if payload.get("cert_version") not in supported_versions:
        return (
            False,
            f"unsupported cert_version {payload.get('cert_version')!r}; "
            f"expected one of {sorted(supported_versions)}",
        )
    claimed = payload.get("cert_id")
    if not isinstance(claimed, str) or not claimed.startswith("sha256:"):
        return False, "missing or malformed cert_id"
    # Reconstruct as an AgentCertificate so the canonicalisation is the
    # same as the one used at build time. AgentCertificate fields drive
    # the order; unknown extras in payload would be ignored.
    try:
        rebuilt = AgentCertificate(
            cert_version=str(payload["cert_version"]),
            agent_id=str(payload.get("agent_id") or ""),
            released_at=str(payload.get("released_at") or ""),
            trace_id=str(payload.get("trace_id") or ""),
            models=list(payload.get("models") or []),
            prompt_hashes=list(payload.get("prompt_hashes") or []),
            tool_schemas=list(payload.get("tool_schemas") or []),
            policy_hash=payload.get("policy_hash"),
            regression_suite=payload.get("regression_suite"),
        )
    except (KeyError, TypeError, ValueError) as e:
        return False, f"certificate payload could not be parsed: {e}"
    expected = _hash_payload(rebuilt)
    if expected != claimed:
        return False, f"cert_id mismatch: claimed {claimed}, recomputed {expected}"
    return True, "certificate is internally consistent"


def render_terminal(cert: AgentCertificate) -> str:
    """Human-readable terminal rendering of a certificate."""
    lines: list[str] = []
    lines.append(f"Agent Behavior Certificate ({cert.cert_version})")
    lines.append(f"  agent_id    : {cert.agent_id}")
    lines.append(f"  released_at : {cert.released_at}")
    lines.append(f"  cert_id     : {cert.cert_id}")
    lines.append(f"  trace_id    : {cert.trace_id}")
    lines.append(f"  models      : {', '.join(cert.models) if cert.models else '<none>'}")
    lines.append(f"  prompts     : {len(cert.prompt_hashes)} distinct system prompt(s)")
    if cert.tool_schemas:
        lines.append(f"  tools       : {len(cert.tool_schemas)} schema(s)")
        for ts in cert.tool_schemas:
            lines.append(f"    - {ts['name']}  ({ts['hash'][:19]}...)")
    else:
        lines.append("  tools       : <none>")
    lines.append(f"  policy_hash : {cert.policy_hash or '<none>'}")
    if cert.regression_suite is not None:
        axes = cert.regression_suite.get("axes") or []
        worst = _worst_severity(axes)
        lines.append(f"  baseline    : {cert.regression_suite.get('baseline_trace_id')}")
        lines.append(f"  worst axis  : {worst}")
        conformal = cert.regression_suite.get("conformal")
        if conformal:
            cov = conformal.get("target_coverage", 0)
            conf = conformal.get("confidence", 0)
            worst_ax = conformal.get("worst_axis", "")
            n_cal = conformal.get("n_calibration", 0)
            suf = conformal.get("sufficient_n", False)
            n_min = conformal.get("n_min", 0)
            lines.append(
                f"  conformal   : {cov:.0%} coverage @ {conf:.0%} PAC confidence "
                f"(n={n_cal}, binding={worst_ax})" + ("" if suf else f"  ⚠ n < n_min={n_min}")
            )
            for ax_row in (conformal.get("axes") or [])[:3]:
                lines.append(
                    f"    · {ax_row['axis']:<22}  q̂={ax_row['q_hat']:.4f}  "
                    f"coverage={ax_row['achieved_coverage']:.2%}"
                )
    else:
        lines.append("  baseline    : <not provided — no regression suite>")
    return "\n".join(lines)


def _worst_severity(axes: list[dict[str, Any]]) -> str:
    rank = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
    worst = max((rank.get(str(a.get("severity")), 0) for a in axes), default=0)
    return next((k for k, v in rank.items() if v == worst), "none")


def _hash_payload(cert: AgentCertificate) -> str:
    """Content-id of the certificate body (everything except cert_id)."""
    body = asdict(cert)
    body.pop("cert_id", None)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "CERT_VERSION",
    "AgentCertificate",
    "build_certificate",
    "render_terminal",
    "verify_certificate",
]
