# Roadmap to Production / Stable

Shadow declares `Development Status :: 4 - Beta` in
[`python/pyproject.toml`](../python/pyproject.toml). This page exists so
prospective adopters and reviewers can see, in writing, what would
justify a bump to `5 - Production/Stable`.

We keep the Beta classifier intentionally. The repo is three weeks
old, has shipped multiple patch releases per day during four review
cycles, and has single-digit external adoption. A `Production/Stable`
classifier on PyPI is read by enterprise procurement as "the dust has
settled." It hasn't.

## What we will require, all of these

### Adoption signals

- [ ] **≥ 90 days** since the most recent breaking API change or behavior change in default-backend output.
- [ ] **≥ 25 distinct GitHub stars** from accounts unaffiliated with the maintainer.
- [ ] **≥ 5 forks** with at least one PR or issue interaction.
- [ ] **≥ 3 named production adopters** (case-studies in this repo's `docs/adopters/`, written by the adopter, linkable).
- [ ] **PyPI download volume** above 1,000 / week sustained for 4 weeks (an order of magnitude above current).

### Stability signals

- [ ] **≥ 30 consecutive days** with no patch release that ships a regression-class bug fix (i.e. no `fix:` commits whose changelog entry mentions "external review", "regression", or "audit").
- [ ] **Empirical Type-I / Type-II rate validation** on the 9-axis differ across 1,000+ paired-response samples, with results published in `docs/theory/statistical-power.md`.
- [ ] **Cross-language byte-parity tests** (Python ↔ TypeScript ↔ Rust) green on every release for 90 consecutive days.
- [ ] **Test count growth without coverage loss** — coverage gate at ≥85% line-level for both Python and Rust, sustained through three minor releases.

### Security signals

- [ ] **Independent third-party security audit** completed by a recognised firm (Trail of Bits, NCC Group, Cure53, Doyensec, or equivalent). Report published or summarised in `docs/security/audit-<year>.md`.
- [ ] **SBOM published per release** in CycloneDX format (already shipping; this stays as a guarantee, not a goal).
- [ ] **Sigstore signature verification** on every released artifact (already shipping).
- [ ] **No `RUSTSEC`, `pip-audit`, or `npm audit` advisory** open for more than 30 days without an explicit, documented accept-with-justification entry.

### Compliance signals

- [ ] **SOC 2 Type 1 attestation** for the project's own release infrastructure. Required only if Shadow ships a hosted service; the CLI tool itself does not require SOC 2 for adopter compliance (see `docs/security/SOC2-ROADMAP.md`).
- [ ] **DPA template** in `docs/legal/` for adopters who need a data-processing addendum.

### Process signals

- [ ] **Maintainer count ≥ 2** with commit access. A single-maintainer project does not meet bus-factor expectations for `Production/Stable`.
- [ ] **Documented security disclosure policy** in `SECURITY.md` (already shipping).
- [ ] **Public CHANGELOG** in `keepachangelog` format (already shipping).
- [ ] **Semantic versioning enforced** (already shipping via release-please).

## What we do NOT require

These would be nice-to-haves but are not blockers:

- A specific number of GitHub issues closed.
- A specific number of CI minutes consumed.
- ISO 27001, FedRAMP, HIPAA, or other certifications above SOC 2 (those are adopter-specific).
- A paid support tier.
- A managed cloud offering.

## When the bump happens

The PyPI classifier in `python/pyproject.toml` line 34 changes from
`Development Status :: 4 - Beta` to `Development Status :: 5 - Production/Stable`
in a single commit, alongside a `v1.0.0` release tag. The
corresponding CHANGELOG entry will reference the exact box on the
list above that the bump satisfies.

Until then, Shadow stays Beta. That is the honest classification.

## Open questions

If you're an enterprise reviewer who needs Shadow at Production-Stable
sooner than the criteria above suggest, the conversation Shadow's
maintainers want to have is:

1. Which specific criteria are blockers for your procurement?
2. Which would you accept as "documented gap with mitigation"?
3. Are you willing to fund the gap closure (e.g. sponsoring the third-party audit)?

That conversation belongs in a GitHub Discussion or a direct email,
not in this doc. The list above is the menu; the conversation picks
which items to escalate.
