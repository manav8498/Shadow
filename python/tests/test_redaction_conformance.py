"""Redaction conformance matrix — documents what the default Redactor catches.

This file is the source of truth for what Shadow's default redaction
covers. Auditors and data-protection reviewers should be able to read
the test names and assertions to see exactly which PII classes are
redacted and which are explicitly NOT covered (documented gaps).

Coverage categories (SOC 2 CC6.1 / CC6.7 / HIPAA §164.514 / GDPR Art.4):

  COVERED (redacted by default):
    - OpenAI API keys (sk-…)
    - Anthropic API keys (sk-ant-…)
    - Email addresses (RFC 5321-ish)
    - Phone numbers in E.164 format
    - Credit-card PANs (Luhn-validated)

  DOCUMENTED GAPS (not yet redacted — callers must provide custom rules):
    - US SSN (9-digit)
    - IBAN (bank accounts)
    - IPv4 / IPv6 addresses
    - Dates of birth in free text
    - Arbitrary national IDs (driver's license, passport numbers)
    - Internal / proprietary identifiers (vary per enterprise)

Every `test_redacts_*` asserts a class IS redacted. Every `test_documents_gap_*`
asserts a class is NOT redacted today — changing that is a deliberate
coverage expansion and should be accompanied by a CHANGELOG entry.
"""

from __future__ import annotations

from shadow.redact import Redactor

# ---------------------------------------------------------------------------
# COVERED classes
# ---------------------------------------------------------------------------


def test_redacts_openai_sk_key() -> None:
    r = Redactor()
    out = r.redact_value("key=sk-proj-ABC123def456ghi789JKL012mno345")
    assert "sk-proj-ABC123def456ghi789JKL012mno345" not in out


def test_redacts_anthropic_sk_ant_key() -> None:
    r = Redactor()
    out = r.redact_value("Authorization: Bearer sk-ant-api03-deadbeef" + "a" * 50)
    assert "sk-ant-api03-deadbeef" not in out


def test_redacts_email() -> None:
    r = Redactor()
    out = r.redact_value("Contact alice.smith@example.com for details")
    assert "alice.smith@example.com" not in out


def test_redacts_e164_phone() -> None:
    r = Redactor()
    out = r.redact_value("Call +14155551234 to confirm")
    assert "+14155551234" not in out


def test_redacts_credit_card_luhn_valid() -> None:
    """4111-1111-1111-1111 is the canonical Luhn-valid Visa test PAN."""
    r = Redactor()
    out = r.redact_value("card 4111-1111-1111-1111 on file")
    assert "4111-1111-1111-1111" not in out


def test_does_not_redact_lookalikes_that_fail_luhn() -> None:
    """16-digit strings that fail Luhn should pass through."""
    r = Redactor()
    out = r.redact_value("tracking 1234567890123456 shipment")
    # Luhn-fails → not a card → not redacted.
    assert "1234567890123456" in out


def test_does_not_redact_20_digit_luhn_valid_strings() -> None:
    """Regression: the grouped-alternative regex previously matched up to
    24 digits with no upper-bound Luhn check. A 20-digit Luhn-passing
    string like "0000 4111 1111 1111 1111" (leading zeros preserve Luhn
    on the Visa test PAN) got incorrectly redacted. Max legitimate PAN
    is 19 digits per ISO/IEC 7812."""
    r = Redactor()
    out = r.redact_value("identifier 0000 4111 1111 1111 1111 not a card")
    # Not redacted — 20 digits is outside the legitimate PAN range.
    assert "0000 4111 1111 1111 1111" in out
    # Sanity: the underlying 16-digit test PAN alone IS still redacted.
    out2 = r.redact_value("card 4111 1111 1111 1111 on file")
    assert "4111 1111 1111 1111" not in out2


# ---------------------------------------------------------------------------
# Additional classes added after a security audit round.
# ---------------------------------------------------------------------------


def test_redacts_aws_access_key_id() -> None:
    r = Redactor()
    out = r.redact_value("creds: AKIAIOSFODNN7EXAMPLE for bucket x")
    assert "AKIAIOSFODNN7EXAMPLE" not in out


def test_redacts_github_personal_access_token() -> None:
    r = Redactor()
    fake = "ghp_" + "a" * 36
    out = r.redact_value(f"token={fake}")
    assert fake not in out


def test_redacts_pem_private_key() -> None:
    r = Redactor()
    body = "X" * 60
    pem = "-----BEGIN RSA PRIVATE KEY-----\n" f"{body}\n" "-----END RSA PRIVATE KEY-----"
    out = r.redact_value(f"here is a key:\n{pem}\nafter")
    assert "BEGIN RSA PRIVATE KEY" not in out
    assert body not in out


def test_redacts_jwt_token() -> None:
    r = Redactor()
    # Canonical hand-rolled JWT shape — three base64url parts, dots between.
    jwt = (
        "eyJhbGciOiJIUzI1NiJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    out = r.redact_value(f"Authorization: Bearer {jwt}")
    assert jwt not in out


# ---------------------------------------------------------------------------
# DOCUMENTED GAPS — if these start passing, update the coverage matrix.
# ---------------------------------------------------------------------------


def test_documents_gap_ssn_not_redacted() -> None:
    r = Redactor()
    out = r.redact_value("SSN 123-45-6789 on file")
    # NOT redacted today. If this breaks, extend the coverage matrix.
    assert "123-45-6789" in out


def test_documents_gap_iban_not_redacted() -> None:
    r = Redactor()
    out = r.redact_value("IBAN DE89370400440532013000 for wire")
    assert "DE89370400440532013000" in out


def test_documents_gap_ipv4_not_redacted() -> None:
    r = Redactor()
    out = r.redact_value("from 192.168.1.42 accessed /admin")
    assert "192.168.1.42" in out


def test_documents_gap_date_of_birth_not_redacted() -> None:
    r = Redactor()
    out = r.redact_value("DOB: 1985-07-14, primary care: Kaiser")
    assert "1985-07-14" in out


# ---------------------------------------------------------------------------
# Nested structures are recursively redacted.
# ---------------------------------------------------------------------------


def test_redacts_nested_dict() -> None:
    r = Redactor()
    payload = {
        "user": {"email": "bob@example.com", "phone": "+447911123456"},
        "cards": [{"pan": "4111-1111-1111-1111"}],
    }
    red = r.redact_value(payload)
    flat = str(red)
    assert "bob@example.com" not in flat
    assert "+447911123456" not in flat
    assert "4111-1111-1111-1111" not in flat


def test_redaction_is_idempotent() -> None:
    """Running redact on already-redacted output produces the same result."""
    r = Redactor()
    once = r.redact_value({"email": "a@b.com"})
    twice = r.redact_value(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Allowlist escape hatch: per-key bypass.
# ---------------------------------------------------------------------------


def test_allowlist_bypasses_redaction() -> None:
    payload = {"internal_email": "ops@company.com", "user_email": "alice@x.com"}
    # Default Redactor: both redacted.
    both_red = Redactor().redact_value(payload)
    assert "ops@company.com" not in str(both_red)
    assert "alice@x.com" not in str(both_red)
    # Redactor with allowlist on "internal_email": only user_email is redacted.
    one_red = Redactor(allowlist_keys=frozenset({"internal_email"})).redact_value(payload)
    assert "ops@company.com" in str(one_red)
    assert "alice@x.com" not in str(one_red)
