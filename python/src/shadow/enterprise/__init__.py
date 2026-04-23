"""Enterprise-readiness primitives for Shadow.

These modules exist to make Shadow credible during a SOC 2 Type II
audit of a customer who has adopted it. They do NOT make Shadow
"SOC 2 certified" — that's a property of the deploying organization,
not of OSS code. What they DO provide:

- `audit_log`: a tamper-evident append-only event journal with each
  event chained to the prior via SHA-256 (same content-addressing
  scheme Shadow already uses for `.agentlog`).
- `access_log`: FastAPI middleware that records every dashboard
  request with timestamp, caller IP, principal, and outcome.
- `redaction_conformance`: a test suite + documented coverage matrix
  enumerating which PII classes Shadow's default redactor catches
  and which it doesn't.

See `docs/SOC2-READINESS.md` for how auditors read these artifacts.
"""

from shadow.enterprise.access_log import AccessLogMiddleware
from shadow.enterprise.audit_log import AuditEvent, AuditLog

__all__ = ["AccessLogMiddleware", "AuditEvent", "AuditLog"]
