"""FastAPI middleware that logs every `shadow serve` request.

Emits one [`AuditEvent`] per request with:
  - action:  `http.{method}` (e.g. `http.GET`)
  - resource: the URL path
  - outcome: `ok` if status < 400, `error` otherwise
  - actor:  `X-Shadow-Principal` header if set, else peer IP
  - extra: status_code, latency_ms

Install on the dashboard app:

    from shadow.enterprise import AccessLogMiddleware, AuditLog
    audit = AuditLog(".shadow/audit.auditlog")
    app.add_middleware(AccessLogMiddleware, audit_log=audit)

Auditors read the resulting `.auditlog` to satisfy SOC 2 CC7.2 ("access
to system resources is authenticated and authorized") and CC7.3
("processing activities are logged").
"""

from __future__ import annotations

import time
from typing import Any

from shadow.enterprise.audit_log import AuditLog

try:
    from starlette.middleware.base import (
        BaseHTTPMiddleware,  # type: ignore[import-not-found, unused-ignore]
    )
    from starlette.requests import Request  # type: ignore[import-not-found, unused-ignore]
    from starlette.responses import Response  # type: ignore[import-not-found, unused-ignore]

    _STARLETTE_INSTALLED = True
except ImportError:  # pragma: no cover
    _STARLETTE_INSTALLED = False
    BaseHTTPMiddleware = object  # type: ignore[assignment,misc]


class AccessLogMiddleware(BaseHTTPMiddleware):  # type: ignore[misc, unused-ignore]
    """Record one audit event per HTTP request."""

    def __init__(self, app: Any, audit_log: AuditLog) -> None:
        if not _STARLETTE_INSTALLED:
            raise RuntimeError(
                "starlette not installed — required for AccessLogMiddleware.\n"
                "hint: pip install 'shadow[serve]'"
            )
        super().__init__(app)
        self._audit = audit_log

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        start = time.perf_counter()
        outcome = "ok"
        status = 0
        error_reason = ""
        try:
            response = await call_next(request)
            status = response.status_code
            if status >= 400:
                outcome = "error"
            return response  # type: ignore[no-any-return]
        except Exception as e:
            outcome = "error"
            # A malicious/broken exception whose __str__ raises would
            # crash the middleware and bypass the audit log. Guard that.
            try:
                error_reason = f"{type(e).__name__}: {e}"[:240]
            except Exception:
                error_reason = f"{type(e).__name__}: <unable to format>"
            raise
        finally:
            latency_ms = int((time.perf_counter() - start) * 1000)
            principal = request.headers.get("x-shadow-principal") or (
                request.client.host if request.client else "unknown"
            )
            self._audit.record(
                action=f"http.{request.method}",
                resource=request.url.path,
                outcome=outcome,
                actor=principal,
                reason=error_reason,
                extra={"status_code": status, "latency_ms": latency_ms},
            )


__all__ = ["AccessLogMiddleware"]
