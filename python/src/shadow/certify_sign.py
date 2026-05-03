"""Cosign / sigstore signing for Agent Behavior Certificates.

Layered ON TOP of the content-addressed certificate format from
:mod:`shadow.certify`. The certificate body is what gets signed —
``cert_id`` already pins the body, the signature pins the body to a
real signer identity issued by Fulcio.

Optional. The ``shadow-diff[sign]`` extra adds ``sigstore>=3.0,<4`` as
a runtime dep. Without it, importing this module raises a clear error
that points the user at the extra.

Two flows:

- **Keyless** (recommended): no long-lived signing keys. Sigstore
  fetches a short-lived signing certificate from Fulcio, scoped to
  an OIDC identity (Github Actions OIDC token in CI; interactive
  browser flow for local users). The Bundle written alongside the
  certificate contains the signature, the signing certificate, and
  a Rekor transparency-log entry — anyone with the certificate
  bundle can verify it later without trusting Shadow.

- **Verification** uses the same sigstore Bundle. The signed payload
  is the canonical bytes of the certificate body (everything except
  ``cert_id`` — the same canonicalisation :mod:`shadow.certify` uses
  to compute ``cert_id``). Verification additionally requires an
  expected ``identity`` (email or workflow URL) so a leaked Bundle
  signed by an attacker doesn't pass.

This module's surface is small on purpose. The sigstore client
handles the cryptography; Shadow handles canonicalisation,
identity-policy plumbing, and the on-disk format.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shadow.errors import ShadowConfigError

if TYPE_CHECKING:
    from shadow.certify import AgentCertificate


SIGNATURE_SUFFIX = ".sigstore"
"""Sidecar suffix for the sigstore Bundle. ``release.cert.json`` →
``release.cert.json.sigstore``. Matches sigstore-python's CLI default."""


_SIGSTORE_INSTALL_HINT = (
    "sigstore is not installed. Install the signing extra: " "`pip install 'shadow-diff[sign]'`"
)


def _require_sigstore() -> None:
    """Raise a clear error when sigstore isn't installed.

    Defers the import so the rest of Shadow doesn't need sigstore
    on the install path.
    """
    try:
        import sigstore  # noqa: F401
    except ImportError as exc:
        raise ShadowConfigError(_SIGSTORE_INSTALL_HINT) from exc


def canonical_body_bytes(cert: AgentCertificate | dict[str, Any]) -> bytes:
    """Return the canonical bytes of a certificate body — everything
    except ``cert_id``. This is what gets signed AND what
    :func:`shadow.certify._hash_payload` hashes for ``cert_id``, so
    signature and content-id agree on the same bytes.

    Accepts either an :class:`AgentCertificate` dataclass instance
    OR the dict produced by ``to_dict()``. Useful for verifying a
    signature against a certificate read from disk.
    """
    body = dict(cert) if isinstance(cert, dict) else asdict(cert)
    body.pop("cert_id", None)
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sign_certificate(
    cert: AgentCertificate,
    *,
    output_bundle: Path,
    identity_token: str | None = None,
    staging: bool = False,
) -> None:
    """Sign a certificate's canonical body and write the sigstore
    Bundle to ``output_bundle``.

    ``identity_token`` overrides the default OIDC discovery. When
    ``None``, sigstore-python uses ambient credentials: GitHub
    Actions' ``ACTIONS_ID_TOKEN_REQUEST_*`` env vars in CI, an
    interactive browser flow for local use.

    ``staging=True`` targets sigstore's staging instance instead of
    production. Useful for tests and integration smoke checks; do not
    use for real releases.
    """
    _require_sigstore()
    from sigstore.oidc import IdentityToken
    from sigstore.sign import SigningContext

    payload = canonical_body_bytes(cert)
    # sigstore 4.0 removed the SigningContext.{production,staging}
    # convenience class methods in favour of going through a
    # ClientTrustConfig (which gained .production / .staging factories
    # in 4.x). Detect the 4.x API by the presence of
    # SigningContext.from_trust_config — ClientTrustConfig itself is
    # importable on both majors but only the 4.x version has the
    # factory methods we need. Fall back to the legacy path for
    # sigstore 3.x. The [sign] extra ranges over both majors via
    # `sigstore>=3.0,<5`.
    if hasattr(SigningContext, "from_trust_config"):
        from sigstore.sign import ClientTrustConfig  # 4.x

        trust_config = ClientTrustConfig.staging() if staging else ClientTrustConfig.production()
        ctx = SigningContext.from_trust_config(trust_config)
    else:
        # sigstore 3.x: old-style class methods on SigningContext
        # are the only path.
        ctx = SigningContext.staging() if staging else SigningContext.production()  # type: ignore[attr-defined]
    if identity_token is not None:
        token = IdentityToken(identity_token)
    else:
        token = _discover_identity_token()
    with ctx.signer(token) as signer:
        bundle = signer.sign_artifact(payload)
    output_bundle.parent.mkdir(parents=True, exist_ok=True)
    output_bundle.write_text(bundle.to_json())


def verify_signature(
    cert_payload: dict[str, Any],
    *,
    bundle_path: Path,
    expected_identity: str,
    expected_issuer: str | None = None,
    staging: bool = False,
) -> tuple[bool, str]:
    """Verify the sigstore signature on a certificate matches its
    canonical body AND comes from the expected signer identity.

    Returns ``(ok, detail)``. ``ok=True`` requires:
      1. The Bundle's signature verifies the canonical body bytes.
      2. The Bundle's certificate identity matches
         ``expected_identity`` (typically an email or a Github
         workflow URL like ``https://github.com/org/repo/.github/
         workflows/release.yml@refs/tags/v1.8.0``).
      3. Optionally, the OIDC issuer matches ``expected_issuer``.

    A leaked Bundle signed by another identity will fail step 2 even
    if its cryptography is otherwise valid — that's the whole point
    of the identity-bound flow.
    """
    _require_sigstore()
    # sigstore 4.x raises a typed `InvalidBundle` (subclass of
    # `sigstore.errors.Error`) on corrupt bundle JSON, where 3.x raised
    # plain `ValueError`. Catching the umbrella `Error` keeps both
    # majors working without coupling this code to a specific class
    # path that may move again.
    from sigstore.errors import Error as SigstoreError
    from sigstore.models import Bundle
    from sigstore.verify import Verifier
    from sigstore.verify.policy import Identity

    if not bundle_path.is_file():
        return False, f"signature bundle not found: {bundle_path}"
    try:
        bundle = Bundle.from_json(bundle_path.read_text())
    except (OSError, ValueError, SigstoreError) as exc:
        return False, f"could not parse sigstore bundle: {exc}"

    verifier = Verifier.staging() if staging else Verifier.production()
    if expected_issuer is None:
        # Default to the GitHub Actions OIDC issuer because that's the
        # common production path. Local interactive flows use a
        # different issuer; callers must override.
        expected_issuer = "https://token.actions.githubusercontent.com"
    policy = Identity(identity=expected_identity, issuer=expected_issuer)

    payload = canonical_body_bytes(cert_payload)
    try:
        verifier.verify_artifact(input_=payload, bundle=bundle, policy=policy)
    except Exception as exc:
        return False, f"signature verification failed: {type(exc).__name__}: {exc}"
    return True, "signature verified against canonical body"


def _discover_identity_token() -> Any:
    """Best-effort OIDC token discovery.

    GitHub Actions populates ``ACTIONS_ID_TOKEN_REQUEST_TOKEN`` and
    ``ACTIONS_ID_TOKEN_REQUEST_URL`` for any job that requests
    ``id-token: write`` permission. When both are present we fetch
    a sigstore-scoped token and return it. Otherwise we fall through
    to sigstore's interactive flow (browser) by raising so the caller
    can let sigstore prompt.
    """
    token_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    token_secret = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if token_url and token_secret:
        from sigstore.oidc import IdentityToken, detect_credential

        # detect_credential() handles the GH Actions exchange, including
        # adding the sigstore audience.
        token_str = detect_credential()
        if token_str is None:
            raise ShadowConfigError(
                "ACTIONS_ID_TOKEN_REQUEST_* present but detect_credential() "
                "returned no token. Ensure the workflow grants id-token: write."
            )
        return IdentityToken(token_str)
    raise ShadowConfigError(
        "no OIDC token discovered. Either run inside GitHub Actions with "
        "`permissions: id-token: write`, or pass --identity-token explicitly."
    )


def signature_path_for(cert_path: Path) -> Path:
    """Conventional sidecar path: ``foo.cert.json`` → ``foo.cert.json.sigstore``."""
    return cert_path.parent / (cert_path.name + SIGNATURE_SUFFIX)


def fingerprint_payload(cert: AgentCertificate | dict[str, Any]) -> str:
    """Hex sha256 of the canonical body bytes.

    Useful for log output. Equal to the body part of ``cert_id`` (which
    is just ``"sha256:" + this``).
    """
    return hashlib.sha256(canonical_body_bytes(cert)).hexdigest()


__all__ = [
    "SIGNATURE_SUFFIX",
    "canonical_body_bytes",
    "fingerprint_payload",
    "sign_certificate",
    "signature_path_for",
    "verify_signature",
]
