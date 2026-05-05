# Security policy

The canonical security policy for Shadow lives at the repo root, in [`SECURITY.md`](https://github.com/manav8498/Shadow/blob/main/SECURITY.md). It covers:

- Supported versions (currently `3.x` and the latest `2.9.x`)
- How to report a vulnerability via private GitHub Security Advisory
- Acknowledgement and fix timelines (72-hour ack, 30-day high-severity fix target)
- What's in scope / out of scope
- The v1.1 hardening pass (parser limits, env-var hygiene, path resolution, supply-chain integrity)

Please open advisories against the upstream repo, not via this docs site.

## Companion pages

- [Production-readiness brief](security/production-readiness.md) — 30-minute CISO-readable summary: data flow, on-disk artifacts, redaction patterns, signing chain, offline operation, threat model.
- [Supply-chain integrity](SUPPLY-CHAIN.md) — Sigstore signing, SLSA build provenance, CycloneDX SBOMs per language surface, and how to verify a release artifact.
- [SOC 2 readiness notes](SOC2-READINESS.md) — primitives Shadow ships that map to Trust Services Criteria when deploying inside a SOC-2-scoped environment.
