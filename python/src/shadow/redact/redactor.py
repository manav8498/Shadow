"""Regex-based redactor for shadow — applied to every record's payload
before canonicalization (SPEC §9.1)."""

from __future__ import annotations

from typing import Any

from shadow.redact.patterns import (
    CREDIT_CARD_CANDIDATE,
    DEFAULT_PATTERNS,
    Pattern,
    luhn_valid,
)

JsonValue = Any  # recursive; mypy doesn't express it cleanly


class Redactor:
    """Apply regex-based redaction to strings, dicts, and lists.

    Parameters
    ----------
    patterns:
        Patterns to apply. Defaults to [`DEFAULT_PATTERNS`] (SPEC §9.2).
    allowlist_keys:
        Top-level or nested keys whose *string* values are NOT redacted.
        This is a name-based allowlist (no path expressions) to keep v0.1
        simple.
    """

    def __init__(
        self,
        patterns: tuple[Pattern, ...] = DEFAULT_PATTERNS,
        allowlist_keys: frozenset[str] = frozenset(),
    ) -> None:
        self._patterns = patterns
        self._allowlist = allowlist_keys
        # Track whether the last redact* call modified anything. Used by the
        # SDK to stamp `envelope.meta.redacted = True` per SPEC §9.3.
        self._last_modified: bool = False

    @property
    def last_modified(self) -> bool:
        """True iff the last redact* call modified its input."""
        return self._last_modified

    def redact_text(self, text: str) -> str:
        """Redact all patterns in a single string."""
        out = text
        modified = False
        for pat in self._patterns:
            if pat.name == CREDIT_CARD_CANDIDATE.name:
                out, cc_changed = self._redact_credit_card(out)
                modified = modified or cc_changed
            else:
                new_out, n = pat.regex.subn(pat.replacement, out)
                if n > 0:
                    modified = True
                    out = new_out
        self._last_modified = self._last_modified or modified
        return out

    def redact_value(self, value: JsonValue) -> JsonValue:
        """Recursively redact a JSON-compatible value.

        Strings are passed through `redact_text`. Lists and dicts are
        recursed into; keys in `allowlist_keys` skip redaction for their
        direct string values (but nested dicts/lists under them are still
        recursed).
        """
        self._last_modified = False
        return self._redact_inner(value)

    def _redact_inner(self, value: JsonValue, key_context: str | None = None) -> JsonValue:
        if isinstance(value, str):
            if key_context is not None and key_context in self._allowlist:
                return value
            return self.redact_text(value)
        if isinstance(value, dict):
            return {k: self._redact_inner(v, key_context=k) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact_inner(v) for v in value]
        # int, float, bool, None — no text inside, pass through
        return value

    def _redact_credit_card(self, text: str) -> tuple[str, bool]:
        """Redact Luhn-valid credit-card-like number sequences only."""
        pat = CREDIT_CARD_CANDIDATE
        modified = False
        out: list[str] = []
        last = 0
        for match in pat.regex.finditer(text):
            if luhn_valid(match.group()):
                out.append(text[last : match.start()])
                out.append(pat.replacement)
                last = match.end()
                modified = True
        out.append(text[last:])
        return ("".join(out), modified) if modified else (text, False)
