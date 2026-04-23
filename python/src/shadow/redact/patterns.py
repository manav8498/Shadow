"""Default regex pattern set for shadow.redact (SPEC §9.2)."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Pattern:
    """One named redaction pattern."""

    name: str
    regex: re.Pattern[str]
    """Compiled regex. Match groups are ignored — the whole match is replaced."""
    replacement: str
    """Replacement string (usually `[REDACTED:<name>]`)."""


# Each pattern's replacement follows the `[REDACTED:<name>]` convention from
# SPEC §9.2. Named groups are not used; the whole match is replaced.

OPENAI_API_KEY = Pattern(
    name="openai_api_key",
    # OpenAI keys come in several formats:
    #   - Legacy:      sk-<48+ alphanumerics>
    #   - Project:     sk-proj-<alphanumeric + hyphen + underscore, ≥20 chars>
    #   - Admin/svc:   sk-svcacct-…, sk-admin-…
    # We explicitly exclude `sk-ant-` (handled by ANTHROPIC_API_KEY).
    regex=re.compile(r"sk-(?!ant-)(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}"),
    replacement="[REDACTED:openai_api_key]",
)

ANTHROPIC_API_KEY = Pattern(
    name="anthropic_api_key",
    regex=re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    replacement="[REDACTED:anthropic_api_key]",
)

EMAIL = Pattern(
    name="email",
    regex=re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    replacement="[REDACTED:email]",
)

PHONE_E164 = Pattern(
    name="phone",
    regex=re.compile(r"\+[1-9]\d{9,14}(?!\d)"),
    replacement="[REDACTED:phone]",
)

# Credit card redaction is regex + Luhn check — the regex catches candidates,
# the Luhn check in `redactor.py` rejects non-cards (e.g. "1234567890123456").
# Supports both contiguous digits AND hyphen/space-separated groups (the
# common user-visible rendering: "4111-1111-1111-1111", "4111 1111 1111 1111").
#
# Both alternatives are capped at 19 digits (ISO/IEC 7812 PAN max). Without
# the cap on the grouped alternative, a Luhn-passing 20-digit string like
# "0000 4111 1111 1111 1111" (leading zeros preserve Luhn) would be
# incorrectly flagged — see tests/test_redaction_conformance.py.
#
# Grouped-max breakdown: 4 (first group) + 3x4 (up to 3 more full groups)
# + 3 (trailing partial group) = 19 digits.
CREDIT_CARD_CANDIDATE = Pattern(
    name="credit_card",
    # Three alternatives, all capped at 19 digits (ISO/IEC 7812):
    #   1. Contiguous 13-19 digits (no separators)
    #   2. Visa/MC-style 4-4-4(-4) groups (8-19 digits with separators)
    #   3. Amex 15-digit 4-6-5 layout (`3412-345678-90123`)
    regex=re.compile(
        r"(?<![0-9])(?:"
        r"\d{13,19}"
        r"|\d{4}(?:[\s\-]\d{4}){2,3}\d{0,3}"
        r"|\d{4}[\s\-]\d{6}[\s\-]\d{5}"
        r")(?![0-9])"
    ),
    replacement="[REDACTED:credit_card]",
)

AWS_ACCESS_KEY_ID = Pattern(
    name="aws_access_key_id",
    # AWS access key IDs: AKIA / ASIA / AIDA / AROA prefix + 16 uppercase alnum.
    regex=re.compile(r"\b(?:AKIA|ASIA|AIDA|AROA)[0-9A-Z]{16}\b"),
    replacement="[REDACTED:aws_access_key_id]",
)

GITHUB_TOKEN = Pattern(
    name="github_token",
    # GitHub personal access / OAuth / installation / server-to-server / refresh tokens.
    regex=re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,251}\b"),
    replacement="[REDACTED:github_token]",
)

PEM_PRIVATE_KEY = Pattern(
    name="private_key",
    # Covers RSA / EC / ED25519 / OpenSSH / generic PEM-armoured private keys.
    regex=re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY-----"
        r"[\s\S]{20,}?"
        r"-----END (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY-----"
    ),
    replacement="[REDACTED:private_key]",
)

JWT_TOKEN = Pattern(
    name="jwt",
    # Three base64url segments separated by dots. Minimum lengths reflect a
    # realistic JWT header (~20 chars), payload (~20), and signature (~20).
    regex=re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{20,}\b"),
    replacement="[REDACTED:jwt]",
)

DEFAULT_PATTERNS: tuple[Pattern, ...] = (
    # Order matters: longer / more specific prefixes first so partial
    # matches from shorter patterns don't win (e.g. sk-ant- vs sk-).
    PEM_PRIVATE_KEY,
    JWT_TOKEN,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    AWS_ACCESS_KEY_ID,
    GITHUB_TOKEN,
    EMAIL,
    PHONE_E164,
    CREDIT_CARD_CANDIDATE,
)


def luhn_valid(digits: str) -> bool:
    """Standard Luhn check. Accepts 13-19 digit numbers (ISO/IEC 7812 PAN range)."""
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    reverse_digits = list(reversed(digits))
    for i, ch in enumerate(reverse_digits):
        if not ch.isdigit():
            return False
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0
