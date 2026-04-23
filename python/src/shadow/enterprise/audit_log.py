"""Tamper-evident append-only audit log.

Each event is appended to an `.auditlog` file as one canonical-JSON
line (reusing Shadow's SPEC §5 canonical-JSON rules). Every event's
`prev_hash` is the SHA-256 of the *previous* line's canonical bytes;
the chain's tail hash is a Merkle-tree-style proof that no event was
inserted, deleted, or reordered.

Intended consumers:

- SOC 2 CC7.1 auditor evidence: "what operations were performed, by
  whom, in what order?"
- Incident-response: if a secret leak is suspected, the audit log
  shows exactly which Session wrote which records when.
- Forensic replay: the chain's integrity can be verified offline
  without trusting the host.

Not intended for PII storage: audit events record *operations*
(who/what/when), not user data. The trace store (.agentlog) is where
redacted payloads live.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from shadow import _core


@dataclass(frozen=True)
class AuditEvent:
    """One audit entry. Fields are load-bearing; don't reorder."""

    ts: str  # RFC 3339 UTC
    actor: str  # e.g. "user:alice" | "service:shadow-cli" | "unknown"
    action: str  # e.g. "session.open" | "diff.compute" | "trace.read"
    resource: str  # e.g. "/path/to/trace.agentlog"
    outcome: str  # "ok" | "denied" | "error"
    reason: str = ""  # optional free text
    prev_hash: str = ""  # sha256 of prior canonical event; "" for first entry
    extra: dict[str, Any] = field(default_factory=dict)


class AuditLog:
    """Append-only chain of [`AuditEvent`]s.

    Parameters
    ----------
    path:
        Path to the `.auditlog` file. Created if missing. Appended to,
        never rewritten.
    actor:
        Default principal stamped on events if `record(...)` doesn't
        override. Use `"system"` for in-process automation.

    Thread safety: `record` is *not* thread-safe by itself; wrap in a
    `threading.Lock` or equivalent if called from multiple threads in
    the same process.
    """

    def __init__(self, path: Path | str, actor: str = "system") -> None:
        self._path = Path(path)
        self._default_actor = actor
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        action: str,
        resource: str,
        outcome: str = "ok",
        *,
        actor: str | None = None,
        reason: str = "",
        extra: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Append a new event; return the stored event with `prev_hash` set."""
        prev_hash = self._tail_hash()
        event = AuditEvent(
            ts=_now_iso(),
            actor=actor or self._default_actor,
            action=action,
            resource=resource,
            outcome=outcome,
            reason=reason,
            prev_hash=prev_hash,
            extra=dict(extra or {}),
        )
        line = _canonical(asdict(event))
        with self._path.open("ab") as f:
            f.write(line)
            f.write(b"\n")
        return event

    def events(self) -> list[AuditEvent]:
        """Parse the entire log (for verification / reporting)."""
        out: list[AuditEvent] = []
        for raw in self._path.read_bytes().splitlines():
            if not raw.strip():
                continue
            d = json.loads(raw)
            out.append(AuditEvent(**d))
        return out

    def verify(self) -> tuple[bool, str]:
        """Walk the chain; return (is_valid, reason).

        Returns `(True, "")` iff every event's `prev_hash` matches the
        sha256 of the preceding canonical line. First event must have
        `prev_hash == ""`.
        """
        lines = [raw for raw in self._path.read_bytes().splitlines() if raw.strip()]
        expected = ""
        for i, raw in enumerate(lines):
            try:
                d = json.loads(raw)
            except json.JSONDecodeError as e:
                return False, f"line {i}: invalid JSON ({e})"
            actual = d.get("prev_hash", "")
            if actual != expected:
                return (
                    False,
                    f"line {i}: prev_hash mismatch "
                    f"(got {actual[:16]}…, expected {expected[:16]}…)",
                )
            expected = sha256(_canonical(d)).hexdigest()
        return True, ""

    def _tail_hash(self) -> str:
        """SHA-256 of the last recorded canonical line, or "" if empty."""
        lines = self._path.read_bytes().splitlines()
        if not lines:
            return ""
        for raw in reversed(lines):
            if raw.strip():
                return sha256(raw).hexdigest()
        return ""


# ---------------------------------------------------------------------------


def _canonical(obj: Any) -> bytes:
    """Reuse Shadow's canonical-JSON serializer from the Rust core."""
    return bytes(_core.canonical_bytes(obj))


def _now_iso() -> str:
    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


__all__ = ["AuditEvent", "AuditLog"]
