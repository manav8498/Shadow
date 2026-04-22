"""Regex-based redaction for LLM trace records (SPEC §9)."""

from shadow.redact.patterns import (
    ANTHROPIC_API_KEY,
    CREDIT_CARD_CANDIDATE,
    DEFAULT_PATTERNS,
    EMAIL,
    OPENAI_API_KEY,
    PHONE_E164,
    Pattern,
    luhn_valid,
)
from shadow.redact.redactor import Redactor

__all__ = [
    "ANTHROPIC_API_KEY",
    "CREDIT_CARD_CANDIDATE",
    "DEFAULT_PATTERNS",
    "EMAIL",
    "OPENAI_API_KEY",
    "PHONE_E164",
    "Pattern",
    "Redactor",
    "luhn_valid",
]
