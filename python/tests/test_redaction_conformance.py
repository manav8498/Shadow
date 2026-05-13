"""Redaction conformance matrix — documents what the default Redactor catches.

This file is the source of truth for what Shadow's default redaction
covers. Auditors and data-protection reviewers should be able to read
the test names and assertions to see exactly which PII classes are
redacted and which are explicitly NOT covered (documented gaps).

Coverage categories (SOC 2 CC6.1 / CC6.7 / HIPAA §164.514 / GDPR Art.4):

  COVERED (redacted by default):
    - OpenAI API keys (sk-…)
    - Anthropic API keys (sk-ant-…)
    - GitHub tokens (ghp_ / gho_ / ghu_ / ghs_ / ghr_)
    - AWS access key ids (AKIA / ASIA / AIDA / AROA)
    - JWT tokens
    - PEM private keys
    - Email addresses (RFC 5321-ish)
    - Phone numbers in E.164 format
    - Credit-card PANs (Luhn-validated)
    - US SSN in dashed `XXX-XX-XXXX` form (v3.2.5+)
    - IBAN (v3.2.5+; grouped + compact forms)
    - IPv4 addresses with valid octet ranges (v3.2.5+)
    - IPv6 addresses, full and `::`-compressed forms (v3.2.5+)

  DOCUMENTED GAPS (not yet redacted — callers must provide custom rules):
    - US SSN in bare 9-digit form (collides with order ids, hashes)
    - Dates of birth in free text (collides with any date)
    - Arbitrary national IDs (UK NI number, Canadian SIN, French INSEE,
      Indian Aadhaar — each is country-specific; adopters provide
      patterns matching the regulations they operate under)
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
# DOCUMENTED GAPS — coverage tests for the remaining gaps live alongside
# the v3.2.5 positive tests at the bottom of this file. The classes
# that ARE still gaps (DOB, bare-digit SSN, country-specific national
# IDs, internal IDs) keep `test_documents_gap_*` tests so a future
# coverage expansion has a single place to update.
# ---------------------------------------------------------------------------


def test_documents_gap_date_of_birth_not_redacted() -> None:
    """DOB collides with every other date in free text. A regex-based
    redactor cannot tell `DOB: 1985-07-14` apart from
    `policy effective 1985-07-14`. Adopters that handle PHI under
    HIPAA / health-data regulation provide a domain-specific pattern
    that uses surrounding context (e.g. the word `DOB` itself)."""
    out = Redactor().redact_value("DOB: 1985-07-14, primary care: Kaiser")
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


# ---------------------------------------------------------------------------
# v3.2.5 patterns added in response to external review: SSN, IBAN, IPv4, IPv6.
# ---------------------------------------------------------------------------


def test_redacts_us_ssn_dashed_form() -> None:
    """`XXX-XX-XXXX` is the canonical US SSN format."""
    out = Redactor().redact_value("user SSN: 123-45-6789 for the form")
    assert "123-45-6789" not in out
    assert "us_ssn" in out


def test_documents_gap_bare_9_digit_ssn_is_not_redacted() -> None:
    """Documented gap: bare 9-digit strings are ambiguous (order ids,
    hash prefixes, zip+4 concatenations). Only the dashed form is
    recognised. Adopters whose data carries bare-digit SSNs add a
    domain-specific pattern. Changing this requires a CHANGELOG entry
    + opt-in for backward compat."""
    out = Redactor().redact_value("order 123456789 shipped")
    assert "123456789" in out


def test_redacts_iban_grouped_form() -> None:
    """IBAN with the printed-on-statement spacing every 4 chars."""
    out = Redactor().redact_value("wire to GB82 WEST 1234 5698 7654 32 today")
    assert "GB82 WEST 1234 5698 7654 32" not in out
    assert "iban" in out


def test_redacts_iban_compact_form() -> None:
    """Same IBAN with no spaces (API JSON form)."""
    out = Redactor().redact_value("DE89370400440532013000 routed")
    assert "DE89370400440532013000" not in out
    assert "iban" in out


def test_redacts_ipv4_address() -> None:
    """Standard dotted-quad IPv4."""
    out = Redactor().redact_value("connect from 203.0.113.42 at 09:00")
    assert "203.0.113.42" not in out
    assert "ipv4" in out


def test_does_not_redact_invalid_ipv4_octets() -> None:
    """Octets above 255 are not valid IPv4 addresses and must not match.
    Regex bound, not a semantic check — guards against over-redaction
    on free-form numbers."""
    out = Redactor().redact_value("version 999.999.999.999 of fake.example")
    assert "999.999.999.999" in out


def test_redacts_ipv6_full_form() -> None:
    """Full eight-group IPv6."""
    out = Redactor().redact_value("client 2001:0db8:85a3:0000:0000:8a2e:0370:7334 connected")
    assert "2001:0db8:85a3:0000:0000:8a2e:0370:7334" not in out
    assert "ipv6" in out


def test_redacts_ipv6_compressed_form() -> None:
    """`::`-compressed IPv6."""
    out = Redactor().redact_value("gateway 2001:db8::1 visible")
    assert "2001:db8::1" not in out
    assert "ipv6" in out
