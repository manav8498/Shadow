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
    regex=re.compile(r"sk-(?!ant-)[A-Za-z0-9]{20,}"),
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
CREDIT_CARD_CANDIDATE = Pattern(
    name="credit_card",
    regex=re.compile(r"(?<![0-9])\d{13,19}(?![0-9])"),
    replacement="[REDACTED:credit_card]",
)

DEFAULT_PATTERNS: tuple[Pattern, ...] = (
    ANTHROPIC_API_KEY,  # must come before OPENAI_API_KEY (sk-ant- subset of sk-)
    OPENAI_API_KEY,
    EMAIL,
    PHONE_E164,
    CREDIT_CARD_CANDIDATE,
)


def luhn_valid(digits: str) -> bool:
    """Standard Luhn check. Accepts 13-19 digit numbers."""
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
    return total % 10 == 0 and len(digits) >= 13
