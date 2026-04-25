"""Shadow — Git-native behavioral diff and shadow deployment for LLM agents."""

from __future__ import annotations

import sys as _sys

from shadow.errors import (
    ShadowBackendError,
    ShadowConfigError,
    ShadowError,
    ShadowParseError,
    ShadowRedactionError,
)

# Bail early with a clear message if the extension was built for a newer
# Python than the one importing it. The cryptic
# `symbol not found '_PyType_GetName'` from abi3 mismatch has burned more
# than one real user — surface the actual cause.
try:
    from shadow import _core
except ImportError as _e:  # pragma: no cover - tested via a subprocess
    _msg = str(_e)
    if "_PyType_GetName" in _msg or "abi3" in _msg or "symbol not found" in _msg:
        _py = f"{_sys.version_info.major}.{_sys.version_info.minor}"
        raise ImportError(
            f"shadow._core requires Python 3.11+ (you are on {_py}).\n"
            "hint: create a 3.11+ venv, e.g. `python3.12 -m venv .venv && "
            "source .venv/bin/activate && pip install -e python/`"
        ) from _e
    raise

__version__ = "2.0.1"
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
