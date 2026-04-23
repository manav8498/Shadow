"""Tests for the SOC 2 audit-log + access-log middleware."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shadow.enterprise import AuditLog


def test_audit_log_chain_is_verifiable(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "a.auditlog")
    log.record("session.open", "/traces/one.agentlog")
    log.record("diff.compute", "/api/diff", extra={"n": 3})
    log.record("session.close", "/traces/one.agentlog")

    events = log.events()
    assert len(events) == 3
    assert events[0].prev_hash == ""  # first entry has no predecessor
    assert events[1].prev_hash != ""  # chained
    assert events[2].prev_hash != events[1].prev_hash

    ok, reason = log.verify()
    assert ok, reason


def test_audit_log_tampering_is_detected(tmp_path: Path) -> None:
    """Modify a committed line; verify() must reject."""
    log = AuditLog(tmp_path / "a.auditlog")
    log.record("a", "r1")
    log.record("b", "r2")
    log.record("c", "r3")

    # Tamper with the middle entry's `reason` field.
    lines = (tmp_path / "a.auditlog").read_bytes().splitlines()
    middle = json.loads(lines[1])
    middle["reason"] = "FORGED"
    lines[1] = json.dumps(middle).encode()
    (tmp_path / "a.auditlog").write_bytes(b"\n".join(lines) + b"\n")

    ok, reason = log.verify()
    assert not ok
    assert "line 2" in reason  # the third line's prev_hash no longer matches


def test_audit_log_deletion_is_detected(tmp_path: Path) -> None:
    """Deleting a middle entry breaks the chain."""
    log = AuditLog(tmp_path / "a.auditlog")
    log.record("a", "r1")
    log.record("b", "r2")
    log.record("c", "r3")

    lines = (tmp_path / "a.auditlog").read_bytes().splitlines()
    # Remove the middle line.
    tampered = [lines[0], lines[2]]
    (tmp_path / "a.auditlog").write_bytes(b"\n".join(tampered) + b"\n")

    ok, _ = log.verify()
    assert not ok


def test_audit_log_records_actor_and_outcome(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "a.auditlog", actor="service:ci")
    log.record("trace.read", "/x.agentlog")
    log.record(
        "diff.compute",
        "/api/diff",
        outcome="denied",
        actor="user:alice",
        reason="rate-limited",
        extra={"status": 429},
    )
    events = log.events()
    assert events[0].actor == "service:ci"
    assert events[0].outcome == "ok"
    assert events[1].actor == "user:alice"
    assert events[1].outcome == "denied"
    assert events[1].extra == {"status": 429}


def test_access_log_middleware_records_each_request(tmp_path: Path) -> None:
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from shadow.enterprise import AccessLogMiddleware

    audit = AuditLog(tmp_path / "a.auditlog", actor="svc:test")
    app = fastapi.FastAPI()
    app.add_middleware(AccessLogMiddleware, audit_log=audit)

    @app.get("/hello")
    async def hello() -> dict:
        return {"ok": True}

    client = TestClient(app)
    r = client.get("/hello", headers={"X-Shadow-Principal": "user:alice"})
    assert r.status_code == 200
    client.get("/missing", headers={"X-Shadow-Principal": "user:bob"})

    events = audit.events()
    # Two entries, one OK, one error.
    assert len(events) == 2
    assert events[0].action == "http.GET"
    assert events[0].resource == "/hello"
    assert events[0].outcome == "ok"
    assert events[0].actor == "user:alice"
    assert events[0].extra["status_code"] == 200

    assert events[1].resource == "/missing"
    assert events[1].outcome == "error"
    assert events[1].actor == "user:bob"

    ok, reason = audit.verify()
    assert ok, reason


def test_access_log_falls_back_to_ip_when_no_principal_header(tmp_path: Path) -> None:
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from shadow.enterprise import AccessLogMiddleware

    audit = AuditLog(tmp_path / "a.auditlog")
    app = fastapi.FastAPI()
    app.add_middleware(AccessLogMiddleware, audit_log=audit)

    @app.get("/")
    async def root() -> dict:
        return {}

    TestClient(app).get("/")
    events = audit.events()
    assert len(events) == 1
    # TestClient reports a peer host of `testclient` or an IP string.
    assert events[0].actor
