# Security audit preparation

This is the pre-audit dossier. It exists so that a third-party
security firm engaging with Shadow can pick up the work cold, without
spending the first week reverse-engineering the architecture.

We have **not** had a third-party audit. The closest we have is
internal review across multiple external reproduction passes. This
page documents what we'd hand the firm on day one.

## Scope of the auditable surface

Shadow ships three artifacts:

| Artifact | Surface | Public attack surface |
|---|---|---|
| `shadow-diff` Python wheel + sdist | `python/src/shadow/` | CLI parses untrusted `.agentlog` JSONL + YAML policy files |
| `shadow-diff` npm package | `typescript/src/` | Auto-instrumentation patches consumer SDK prototypes; no network surface |
| `shadow-diff` Rust crate (`shadow-core` and `shadow-align`) | `crates/` | PyO3 binding consumes `bytes` from Python; no network surface |

The CLI is **local-only by design**. It does not bind to a network
port, send telemetry without opt-in, or store credentials. The two
ways Shadow touches the network:

1. **`--backend live`** in `shadow diagnose-pr` calls OpenAI through
   the user's own `OPENAI_API_KEY` env var.
2. **`shadow certify --sign`** calls Sigstore's keyless OIDC flow via
   the user's GitHub Actions OIDC token in CI.

Both are user-opt-in and use the user's credentials. Shadow itself
holds no secrets.

## Threat model

We classify threats using STRIDE, scoped to the realistic deployment
shapes (local CLI; GitHub Action runner; library import in a user's
process).

### Spoofing

| Threat | Mitigation | Confidence |
|---|---|---|
| Attacker forges a `.agentlog` file | Content-addressed via `sha256(canonical_bytes(payload))`. `shadow baseline verify` (fixed in v3.1.3) recomputes content_id and raises on mismatch. | High |
| Attacker forges a release artifact | Sigstore keyless signing on every release; signature + cert published alongside artifact. | High |
| Attacker forges a release commit | Git history is the canonical record; release-please reads tags from a SHA-pinned action. | Medium (depends on GitHub identity controls) |

### Tampering

| Threat | Mitigation | Confidence |
|---|---|---|
| Trace tampering (payload edit, stale id) | `compute_baseline_hash` re-derives content_id per record; fails on mismatch. Pinned by `test_compute_baseline_hash_detects_payload_tamper_with_stale_id`. | High |
| Workflow tampering via mutable Action tags | All third-party actions SHA-pinned (v3.2.1). Dependabot ecosystem `github-actions` tracks future bumps. | High |
| Dependency tampering at install time | `uv.lock` + `requirements-locked.txt` published per release; opt-in install via `uv sync --frozen`. | Medium (default install path is permissive by design) |

### Repudiation

| Threat | Mitigation | Confidence |
|---|---|---|
| User claims they didn't approve a baseline | Baseline hash is committed to `shadow.yaml` in git; baseline-approval is an explicit `shadow baseline update --force`. The commit is the receipt. | High |
| Adopter claims they never ran Shadow | `.agentlog` files carry monotonic timestamps + content-addressed ids; the trace is its own non-repudiation record. | Medium |

### Information disclosure

| Threat | Mitigation | Confidence |
|---|---|---|
| Secret leakage through recorded traces | Built-in regex redactor covers OpenAI / Anthropic API keys, AWS access keys, OAuth tokens, PEM private keys, AWS access keys, emails, phone numbers, SSNs, IBANs. Property-tested. | Medium (custom secret patterns are adopter responsibility) |
| Trace upload to a SaaS without consent | Shadow does not upload anywhere. `--backend live` is opt-in and uses the user's API key. | High |
| Telemetry leakage | Telemetry off by default. First-run prompt + `SHADOW_TELEMETRY=off` kill switch. Fields are documented; no traces, prompts, or response text ever sent. | High |

### Denial of service

| Threat | Mitigation | Confidence |
|---|---|---|
| Malicious `.agentlog` file (zip-bomb-style JSONL) | Streaming parser; no whole-file buffering. Per-record size cap on JSONL parse. | Medium |
| Malicious policy YAML | YAML loaded with `yaml.safe_load`; rules validated against a known kinds + params schema; unknown keys ignored. | Medium |
| `--backend live` runaway spend | `--max-cost USD` cap; trips before exceeding the cap; OPENAI_API_KEY never accepted as a CLI arg. | High |

### Elevation of privilege

| Threat | Mitigation | Confidence |
|---|---|---|
| Untrusted prompt content causing code execution | No `eval` / `exec` in Shadow's code paths on user-supplied input. Auto-instrumentation patches use closure capture, never `setattr` on user-controlled names. | High |
| Untrusted policy YAML causing code execution | No `python/object` tags or YAML constructors registered. `yaml.safe_load` only. | High |

## Control inventory

### Secrets

| Control | Status |
|---|---|
| No hardcoded secrets in source | ✅ verified by `git-secrets` config + `shadow scan` pattern set; CI runs both |
| No secrets in test fixtures | ✅ fixtures use synthetic IDs (`sha256:000…`, `sk-fake`, `AKIAIOSFODNN7EXAMPLE`) |
| Secret-redaction unit tests | ✅ `python/tests/test_redact*.py` (property-based via hypothesis) |
| Custom secret patterns documented for adopters | ⚠️ documented in `docs/features/redaction.md` but adopters need to add domain-specific patterns themselves |

### Dependency management

| Control | Status |
|---|---|
| Lockfile published per release | ✅ `uv.lock` + `requirements-locked.txt` at repo root |
| `cargo audit` / `pip-audit` / `npm audit` in CI | ✅ runs on PR + main |
| Documented accept-with-justification for advisories | ✅ `.cargo/audit.toml` + `docs/SUPPLY-CHAIN.md` |
| SBOM published per release | ✅ CycloneDX 1.5 for Python + Rust + TS |
| Renovate / Dependabot | ✅ Dependabot for all four ecosystems |

### Build + release integrity

| Control | Status |
|---|---|
| Reproducible builds | ⚠️ Python wheels are reproducible-ish via maturin; Rust is reproducible with `--locked`; TS bundles vary by Node version |
| Provenance attestation (SLSA) | ✅ `actions/attest-build-provenance` on every release artifact |
| Sigstore signatures | ✅ keyless cosign on every release; `.sig` + `.crt` alongside artifacts |
| SHA-pinned CI actions | ✅ as of v3.2.1; Dependabot tracks bumps |
| Branch protection on `main` | ✅ requires green CI + signed-off-by; `allow_force_pushes: false` |

### Operational

| Control | Status |
|---|---|
| Documented disclosure policy | ✅ `SECURITY.md` |
| Time-bound disclosure SLO | ⚠️ documented as "best-effort"; not a calendar commitment |
| Released CVE history | ✅ none reported to date |
| Documented incident response | ❌ none — adopt-as-needed when first incident occurs |

## What an auditor would still need to verify

The audit-prep work in this document is necessary but not sufficient.
An audit firm would still:

1. Reproduce the threat model claims by adversarial testing (fuzz the
   JSONL parser, attempt YAML deserialization escapes, attempt regex
   bypass on the redactor).
2. Review the Rust core for memory-safety violations (unsafe blocks,
   FFI boundaries with PyO3).
3. Review the PyO3 binding for reference-counting bugs, GIL
   discipline, and panic-safety across the FFI boundary.
4. Static-analyse the dependency tree for known-vulnerable transitive
   crates (RUSTSEC catches most; the firm would extend that with
   private databases).
5. Review the GitHub Actions workflows for token-scope escalation
   paths (we assert least-privilege but they verify it).

We expect the audit to find issues — that's the point. This document
exists so the firm spends their time on (1)–(5) rather than
re-deriving the architecture.

## Engagement

If you're an enterprise reviewer who needs Shadow audited:

1. Use this dossier to scope the engagement with your preferred firm.
2. The maintainer is reachable at the email in `SECURITY.md`.
3. We will cooperate with audit firms engaged by adopters as long as
   the engagement is in good faith (we receive the findings under
   coordinated-disclosure terms; we publish a summary in
   `docs/security/audit-<year>.md`).

Funding for an audit is welcome but not required — a firm that does
the work pro-bono or on the adopter's tab is equally welcome.
