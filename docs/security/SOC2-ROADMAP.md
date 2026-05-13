# SOC 2 roadmap

This is **not** a SOC 2 attestation. Shadow is not a SOC 2 service.

Read [`docs/SOC2-READINESS.md`](../SOC2-READINESS.md) for the formal
position. This page exists to map out what would change if Shadow
ever stood up a managed service or a `Production/Stable` story that
required SOC 2 — and what the existing CLI tool already satisfies
without an attestation.

## Why Shadow probably does NOT need SOC 2

The CLI tool ships as software, not as a service. The Trust Services
Criteria (TSC) that SOC 2 covers (Security, Availability, Processing
Integrity, Confidentiality, Privacy) are scoped to a *service
organisation*. Shadow as installed and run inside the adopter's
infrastructure is a *vendor* but not a *service organisation*.

The relevant standard for Shadow-as-shipped-software is closer to:

- **Supply-chain integrity** (SLSA level 3 attestation, sigstore
  signatures, SBOM) — covered.
- **Vulnerability disclosure** (SECURITY.md, time-bound triage,
  CVE issuance if needed) — partially covered.
- **Source code review** (third-party audit, see
  `docs/security/AUDIT-PREP.md`) — not yet covered.

Enterprise procurement teams sometimes ask for SOC 2 even from
software vendors. That request is usually a process-control proxy
for "we want to know you take security seriously." The answer is to
point them at **SUPPLY-CHAIN.md + AUDIT-PREP.md + SECURITY.md**
together — those address the actual concerns SOC 2 maps to, in a
form that's relevant for a CLI tool.

## When Shadow WOULD need SOC 2

If Shadow ships any of these, SOC 2 Type 1 becomes a near-term
requirement:

- A hosted dashboard service that accepts `.agentlog` uploads.
- A managed `shadow record` collector with a HTTP ingestion endpoint.
- A SaaS for `diagnose-pr` that runs the causal-attribution backend
  on the user's behalf.
- A marketplace for policy packs or behavior certificates.

None of those are on the v0.x roadmap. If any of them lands, the
SOC 2 path follows the gap analysis below.

## Gap analysis (TSC Common Criteria)

The criteria below are the SOC 2 Common Criteria (CC) series. ✅
means we have the control in a form that an auditor would accept;
⚠️ means we have something partial; ❌ means absent.

### CC1 — Control environment

| Criterion | Status | Notes |
|---|---|---|
| CC1.1 Integrity and ethical values | ⚠️ | No employee handbook; project has one maintainer. |
| CC1.2 Board oversight | ❌ | No board. Single-maintainer project. |
| CC1.3 Org structure + reporting lines | ❌ | N/A at current size. |
| CC1.4 Commitment to competence | ⚠️ | Maintainer's prior experience documented in profile; no formal training program. |
| CC1.5 Accountability | ⚠️ | Disclosure policy exists; no formal accountability framework. |

### CC2 — Communication and information

| Criterion | Status | Notes |
|---|---|---|
| CC2.1 Internal information | ⚠️ | Single-maintainer; everything is in git or the GitHub issues tracker. |
| CC2.2 Internal communication | ⚠️ | Discord / email; no formal channel. |
| CC2.3 External communication | ✅ | SECURITY.md, public CHANGELOG, GitHub Issues + Discussions. |

### CC3 — Risk assessment

| Criterion | Status | Notes |
|---|---|---|
| CC3.1 Risk identification | ✅ | `docs/security/AUDIT-PREP.md` carries the threat model in STRIDE form. |
| CC3.2 Risk assessment | ⚠️ | Threat model is qualitative; no quantified risk-register. |
| CC3.3 Fraud risk | ❌ | N/A for an OSS project. |
| CC3.4 Change in environment | ⚠️ | CHANGELOG documents change; no formal change-impact review. |

### CC4 — Monitoring activities

| Criterion | Status | Notes |
|---|---|---|
| CC4.1 Ongoing evaluations | ⚠️ | External-review cycles drive most of this; not on a calendar. |
| CC4.2 Control deficiencies | ✅ | Issues + CHANGELOG document deficiencies and fixes. |

### CC5 — Control activities

| Criterion | Status | Notes |
|---|---|---|
| CC5.1 Control selection | ✅ | Documented in SUPPLY-CHAIN.md and AUDIT-PREP.md. |
| CC5.2 General IT controls | ⚠️ | GitHub-hosted; controls are GitHub's, not ours. |
| CC5.3 Deployment of policies | ⚠️ | Policies in `docs/`; no compliance-tracking dashboard. |

### CC6 — Logical access controls

| Criterion | Status | Notes |
|---|---|---|
| CC6.1 Logical access | N/A | CLI runs in adopter's environment; we control no access. |
| CC6.2 Provisioning + de-provisioning | N/A | Same. |
| CC6.3 Authentication | N/A | Same. |
| CC6.6 Vulnerability identification | ✅ | cargo audit / pip-audit / npm audit in CI. |
| CC6.7 Data transmission | ⚠️ | Telemetry off by default; uses HTTPS when on. |
| CC6.8 Software prevention | ⚠️ | Auto-instrumentation patches consumer SDKs; documented in code. |

### CC7 — System operations

| Criterion | Status | Notes |
|---|---|---|
| CC7.1 Detection of events | N/A | No service to monitor. |
| CC7.2 Anomaly monitoring | N/A | Same. |
| CC7.3 Evaluating events | ⚠️ | Issues are the channel; no formal evaluation matrix. |
| CC7.4 Recovery | ⚠️ | Releases are immutable; recovery is "pull the bad version, ship a patch." Documented? No. |

### CC8 — Change management

| Criterion | Status | Notes |
|---|---|---|
| CC8.1 Change authorisation | ⚠️ | Single maintainer; auto-merge after CI green. PR review optional. |

### CC9 — Risk mitigation

| Criterion | Status | Notes |
|---|---|---|
| CC9.1 Vendor management | ⚠️ | Dependency pinning + audit, but no formal vendor list. |
| CC9.2 Business resumption | N/A | Single-maintainer OSS project; no SLA. |

## If you need Shadow SOC 2 certified

The honest path:

1. **Wait for v1.0** when Shadow is `Production/Stable` (see
   `ROADMAP-TO-PRODUCTION.md`). The maintainer count and process
   maturity in CC1–CC5 are the gating items for a useful attestation.
2. **Or pay for the audit firm yourself** under your own
   procurement process, with Shadow's maintainer cooperating on
   evidence gathering. We will produce: dep tree, threat model,
   change history, CHANGELOG narrative, sigstore-verifiable release
   chain, SBOM history.
3. **Or use a SOC 2 wrapper.** Adopters running Shadow in their own
   SOC 2-covered environment (AWS account, GitHub Enterprise org)
   often satisfy auditor questions by attesting that the tool runs
   inside their own controlled environment — the tool itself doesn't
   need SOC 2.

Path 3 is the right answer for almost everyone right now.

## What we will NOT do

- **Self-attest SOC 2.** Self-attestation isn't a SOC 2 report;
  claiming it would be misleading.
- **Mock SOC 2 by adding checkboxes without controls.** The gap
  analysis above is honest; we'd rather show the gap than pretend it
  isn't there.
- **Hide behind "we're open-source so SOC 2 doesn't apply."** It does
  apply when there's a service involved; it doesn't for a CLI; the
  distinction is what this doc is for.
