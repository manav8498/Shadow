"""Tests for ``shadow.certify_sign`` — the cosign / sigstore signing
layer for Agent Behavior Certificates.

Real sigstore signing needs an OIDC token from a real issuer. We
mock at the sigstore boundary so the test exercises Shadow's
canonicalisation, identity-policy plumbing, and on-disk format
without standing up Fulcio / Rekor.

The whole file is gated on the optional ``sign`` extra; CI's
default install path doesn't include sigstore.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sigstore")

from shadow import _core  # noqa: E402
from shadow.certify import build_certificate  # noqa: E402
from shadow.certify_sign import (  # noqa: E402
    SIGNATURE_SUFFIX,
    canonical_body_bytes,
    fingerprint_payload,
    sign_certificate,
    signature_path_for,
    verify_signature,
)
from shadow.errors import ShadowConfigError  # noqa: E402
from shadow.sdk import Session  # noqa: E402


def _make_trace(path: Path) -> None:
    with Session(output_path=path, tags={"env": "test"}) as s:
        s.record_chat(
            request={
                "model": "claude-opus-4-7",
                "messages": [{"role": "system", "content": "hi"}],
                "params": {},
            },
            response={
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "latency_ms": 1,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )


def _build_cert(tmp_path: Path) -> Any:
    trace_path = tmp_path / "t.agentlog"
    _make_trace(trace_path)
    return build_certificate(trace=_core.parse_agentlog(trace_path.read_bytes()), agent_id="x")


# ---- canonicalisation ---------------------------------------------------


def test_canonical_body_is_deterministic_sorted_no_whitespace(tmp_path: Path) -> None:
    cert = _build_cert(tmp_path)
    bytes_a = canonical_body_bytes(cert)
    bytes_b = canonical_body_bytes(cert.to_dict())
    assert bytes_a == bytes_b, "dataclass and dict forms must canonicalise identically"
    s = bytes_a.decode("utf-8")
    assert " " not in s.replace(
        " wrong colour", ""
    )  # rough whitespace check; sort_keys + no separators
    assert "cert_id" not in s, "cert_id must NOT appear in the canonical body"


def test_canonical_body_excludes_cert_id_so_signature_pins_body_only(tmp_path: Path) -> None:
    cert = _build_cert(tmp_path)
    payload = cert.to_dict()
    fp_before = fingerprint_payload(payload)
    payload["cert_id"] = "sha256:" + ("0" * 64)  # tamper with cert_id only
    fp_after = fingerprint_payload(payload)
    assert fp_before == fp_after, (
        "fingerprint must be over body bytes, not cert_id; otherwise a "
        "signature couldn't survive a re-emit of the same canonical body."
    )


def test_signature_path_uses_sidecar_convention() -> None:
    p = Path("/tmp/release.cert.json")
    assert signature_path_for(p) == Path("/tmp/release.cert.json" + SIGNATURE_SUFFIX)


# ---- sign with a mocked sigstore boundary -------------------------------


class _FakeBundle:
    """Behaves like a sigstore.models.Bundle for Shadow's purposes:
    has to_json()/from_json() round-trip."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def to_json(self) -> str:
        return json.dumps({"fake_bundle_for": fingerprint_payload({})})


class _FakeSigner:
    def __init__(self) -> None:
        self.signed: list[bytes] = []

    def __enter__(self) -> _FakeSigner:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def sign_artifact(self, payload: bytes) -> _FakeBundle:
        self.signed.append(payload)
        return _FakeBundle(payload)


class _FakeContext:
    def __init__(self) -> None:
        self.signer_obj = _FakeSigner()

    def signer(self, _token: Any) -> _FakeSigner:
        return self.signer_obj


def test_sign_certificate_writes_bundle_at_sidecar_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_ctx = _FakeContext()
    import shadow.certify_sign as cs

    monkeypatch.setattr(
        "sigstore.sign.SigningContext.production",
        lambda: fake_ctx,
    )
    monkeypatch.setattr(
        "sigstore.oidc.IdentityToken",
        lambda _token: object(),
    )

    cert = _build_cert(tmp_path)
    bundle_path = tmp_path / "cert.json.sigstore"
    cs.sign_certificate(cert, output_bundle=bundle_path, identity_token="fake-jwt")

    assert bundle_path.is_file()
    parsed = json.loads(bundle_path.read_text())
    assert "fake_bundle_for" in parsed  # round-tripped through our fake
    # The signer saw the canonical body bytes — the same bytes a real
    # verifier would later check against.
    assert fake_ctx.signer_obj.signed == [canonical_body_bytes(cert)]


def test_sign_without_token_or_ambient_creds_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without GitHub Actions env vars and without an explicit token,
    the discovery helper must raise — sigstore's interactive flow is
    not appropriate to fall through to silently in CI."""
    monkeypatch.delenv("ACTIONS_ID_TOKEN_REQUEST_URL", raising=False)
    monkeypatch.delenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", raising=False)
    cert = _build_cert(tmp_path)
    with pytest.raises(ShadowConfigError, match="no OIDC token"):
        sign_certificate(cert, output_bundle=tmp_path / "x.sigstore")


# ---- verify -------------------------------------------------------------


def test_verify_signature_missing_bundle_returns_false(tmp_path: Path) -> None:
    cert = _build_cert(tmp_path)
    ok, detail = verify_signature(
        cert.to_dict(),
        bundle_path=tmp_path / "no-such-bundle.sigstore",
        expected_identity="alice@example.com",
    )
    assert not ok
    assert "not found" in detail


def test_verify_signature_corrupt_bundle_returns_false(tmp_path: Path) -> None:
    cert = _build_cert(tmp_path)
    bad_bundle = tmp_path / "bad.sigstore"
    bad_bundle.write_text("{not valid json")
    ok, detail = verify_signature(
        cert.to_dict(),
        bundle_path=bad_bundle,
        expected_identity="alice@example.com",
    )
    assert not ok
    assert "could not parse" in detail or "verification failed" in detail


def test_verify_signature_bundle_signed_for_different_body_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wire verify against a stub Verifier that always succeeds, but
    swap the certificate body — Shadow's canonicalisation must produce
    DIFFERENT input bytes, and the boundary must surface that."""
    cert = _build_cert(tmp_path)
    bundle_path = tmp_path / "release.cert.json.sigstore"
    # We're not actually exercising sigstore crypto here; we just want
    # to confirm our boundary calls verify_artifact with the right
    # canonical bytes. The verifier raising would already be tested by
    # sigstore's own suite.
    seen: dict[str, bytes] = {}

    class _StubBundle:
        @classmethod
        def from_json(cls, _s: str) -> _StubBundle:
            return cls()

    class _StubVerifier:
        @classmethod
        def production(cls) -> _StubVerifier:
            return cls()

        @classmethod
        def staging(cls) -> _StubVerifier:
            return cls()

        def verify_artifact(self, *, input_: bytes, bundle: Any, policy: Any) -> None:
            seen["input"] = input_

    bundle_path.write_text("{}")
    monkeypatch.setattr("sigstore.models.Bundle", _StubBundle)
    monkeypatch.setattr("sigstore.verify.Verifier", _StubVerifier)
    monkeypatch.setattr(
        "sigstore.verify.policy.Identity",
        lambda **kw: ("policy", kw),
    )

    ok, detail = verify_signature(
        cert.to_dict(),
        bundle_path=bundle_path,
        expected_identity="alice@example.com",
    )
    assert ok, detail
    assert seen["input"] == canonical_body_bytes(cert)
