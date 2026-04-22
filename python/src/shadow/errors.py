"""Exception hierarchy for shadow. All user-facing errors subclass ShadowError."""

from __future__ import annotations


class ShadowError(Exception):
    """Base class for all shadow exceptions.

    CLI callers catch this and print the message with a coloured prefix,
    then exit 1. Library callers can catch specific subclasses.
    """


class ShadowConfigError(ShadowError):
    """Invalid or missing shadow configuration."""


class ShadowParseError(ShadowError):
    """A `.agentlog` file could not be parsed."""


class ShadowBackendError(ShadowError):
    """An LLM backend (mock or live) failed."""


class ShadowRedactionError(ShadowError):
    """Redaction layer failed (e.g. user provided a bad allowlist path)."""
