# Shadow, SOC 2 readiness notes

This doc is for customers deploying Shadow and for their auditors. Shadow is
open-source software; **it is not a SOC 2 certified service**. The adopting
organization retains responsibility for their own audit. This doc describes the
primitives Shadow provides that help you satisfy the relevant Trust Services
Criteria.

## What Shadow ships that maps to SOC 2

| Criterion | Shadow provides | Location |
|---|---|---|
| **CC6.1** (logical access controls) | Redaction at record boundaries; per-key allowlist | `shadow.redact.Redactor` |
| **CC6.7** (data transmission confidentiality) | Keys and PII redacted before canonical-JSON serialization | `shadow.sdk.Session._redact` |
| **CC7.1** (system monitoring) | Tamper-evident append-only audit log with SHA-256 chain | `shadow.enterprise.AuditLog` |
| **CC7.2** (authenticated + authorized access) | HTTP principal via `X-Shadow-Principal` header + IP fallback | `shadow.enterprise.AccessLogMiddleware` |
| **CC7.3** (processing logged) | One audit event per dashboard request | same |
| **CC7.4** (anomaly response) | Audit log `verify()` detects tampering, deletion, reordering | `AuditLog.verify()` |
| **CC8.1** (change management) | Content-addressed traces (SHA-256 ids), any modification changes the id | spec §6 |

## Audit-log data model

Every event is an append-only line of canonical JSON in a `.auditlog` file.
Fields:

- `ts`, RFC 3339 UTC timestamp, millisecond precision
- `actor`, `user:<name>` | `service:<name>` | IP | `"unknown"`
- `action`, canonical action verb (e.g. `session.open`, `http.GET`, `trace.read`)
- `resource`, path / URL / trace id being acted on
- `outcome`, `ok` | `denied` | `error`
- `reason`, optional free text (max 240 chars for the access-log middleware)
- `prev_hash`, SHA-256 of the canonical bytes of the *previous* event, or `""` for the first

The chain is verifiable offline: `AuditLog(path).verify() → (bool, reason)`.
Any tampering, modification, deletion, reordering, breaks a subsequent
`prev_hash` and is detected.

## Access-log middleware

Install on the `shadow serve` dashboard (or any FastAPI app):

```python
from shadow.enterprise import AuditLog, AccessLogMiddleware

audit = AuditLog(".shadow/audit.auditlog", actor="service:shadow-serve")
app.add_middleware(AccessLogMiddleware, audit_log=audit)
```

This records one audit event per request:
- `action = "http.GET"` / `"http.POST"` etc.
- `resource = request.url.path`
- `actor = X-Shadow-Principal header || client IP`
- `outcome = "ok"` if status<400 else `"error"`
- `extra = {"status_code": int, "latency_ms": int}`

## Redaction coverage matrix

The default `Redactor()` catches these classes **before** content is hashed
into `.agentlog`:

| Class | Pattern | Behavior |
|---|---|---|
| OpenAI API key | `sk-(proj-\|svcacct-\|admin-)?…` (≥20 chars) | replaced with `[REDACTED:openai_api_key]` |
| Anthropic API key | `sk-ant-…` (≥20 chars) | replaced with `[REDACTED:anthropic_api_key]` |
| Email | RFC-5321-ish | `[REDACTED:email]` |
| Phone (E.164) | `+<country><digits>` | `[REDACTED:phone]` |
| Credit card | Luhn-valid, contiguous OR hyphen/space-separated | `[REDACTED:credit_card]` |

**Gaps**, classes NOT redacted by default. Customers needing these must
install custom `Redactor` patterns OR rely on field-level application-layer
redaction before handing payloads to `Session.record_chat`:

- US SSN (9-digit)
- IBAN (bank account numbers)
- IPv4 / IPv6 addresses
- Free-text dates of birth
- Driver's license / passport / national IDs
- Internal / proprietary employee or customer IDs

The conformance test suite at `python/tests/test_redaction_conformance.py`
asserts both what IS covered and what is NOT covered. Changing
either is a deliberate coverage expansion and must be accompanied by a
CHANGELOG entry.

## Per-key allowlist

Some keys (e.g. `internal_email` sent from service A to service B) are safe
to leave un-redacted. Configure at Redactor construction:

```python
from shadow.redact import Redactor
r = Redactor(allowlist_keys=frozenset({"internal_email", "trace_id"}))
```

## What Shadow does NOT do (customer responsibility)

- **Encryption at rest.** `.agentlog` and `.auditlog` files are cleartext on
  disk. Use full-disk encryption, encrypted volumes, or deploy to a KMS-backed
  filesystem per your requirements.
- **Authentication.** The dashboard middleware logs the caller via the
  `X-Shadow-Principal` header, but it does not verify the identity. Terminate
  authN at your ingress (OAuth2, mTLS, header-injection from a SSO proxy).
- **Authorization.** Per-resource RBAC is out of scope for v0.1. Callers
  with access to the `.shadow/` dir can read anything under it.
- **Retention / data-lifecycle.** Shadow writes files; your organisation's
  retention policy is out of scope.
- **Key rotation.** API keys Shadow may record are redacted before write;
  rotation of YOUR application's keys is your operational concern.

## Evidence bundle for auditors

When handing evidence to a SOC 2 auditor, assemble:

1. The `docs/SOC2-READINESS.md` file (this doc) as scope.
2. The `python/tests/test_redaction_conformance.py` output (last run).
3. The `python/tests/test_enterprise_audit.py` output (tamper detection).
4. A sample `.auditlog` from a production period + the result of
   `AuditLog(path).verify()` showing `(True, "")`.
5. The deployment topology showing where encryption-at-rest, authN, and
   retention policies are enforced (customer-specific).
