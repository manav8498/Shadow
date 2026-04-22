"""Shadow — Git-native behavioral diff and shadow deployment for LLM agents."""

from __future__ import annotations

from shadow import _core
from shadow.errors import (
    ShadowBackendError,
    ShadowConfigError,
    ShadowError,
    ShadowParseError,
    ShadowRedactionError,
)

__version__ = "0.1.0"
SPEC_VERSION: str = _core.SPEC_VERSION

__all__ = [
    "SPEC_VERSION",
    "ShadowBackendError",
    "ShadowConfigError",
    "ShadowError",
    "ShadowParseError",
    "ShadowRedactionError",
    "__version__",
    "_core",
]
