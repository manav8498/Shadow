# Governance

Shadow is a small project at v0.1. This document describes who decides
what, how decisions are made, and how that will evolve.

## Current model, BDFL with lazy consensus

At v0.1, one person (the repo owner) has final say on:

- Merging to `main`
- Releases and versioning
- Acceptance or rejection of proposals
- Code of Conduct enforcement

In practice most decisions are made by **lazy consensus**: proposals on
issues or pull requests land if no maintainer objects within a
reasonable window (typically 5 business days for non-trivial changes).

## Who is a maintainer

Maintainers are listed in [MAINTAINERS.md](./MAINTAINERS.md). A maintainer
has commit access, review authority for their area (per CODEOWNERS), and
a vote in governance decisions.

New maintainers are invited by existing maintainers based on sustained
high-quality contribution, typically at least 3 months of regular
engagement and meaningful code / review / triage work.

## How to propose a change

1. **Small changes** (bug fixes, docs, test improvements): open a PR.
   No RFC required.
2. **Non-trivial changes** (new features, public-API changes, behaviour
   changes that could break downstream users): open a GitHub **Discussion**
   with `[RFC]` in the title. Wait for maintainer signoff before spending
   significant implementation time.
3. **Spec changes** (`SPEC.md`, the `.agentlog` format, or anything that
   could break `.agentlog` compatibility): RFC required. Discussion must
   stay open for a minimum of 14 days.

## Scaling up

When we reach >3 active maintainers, this document will be revised to
adopt a lightweight **Technical Steering Committee** model, majority
vote on disagreements, with a documented appeal path. The CNCF TAG-level
project governance template is the likely reference.

## Code of Conduct enforcement

Reports of Code of Conduct violations go to the address in
[CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md). Enforcement decisions are
made by the repo owner until a TSC exists.

## Open questions

- Foundation neutrality (CNCF / OpenJS / Linux Foundation): not planned for now
  until we have sustained external contribution.
- Trademark ownership: see [TRADEMARK.md](./TRADEMARK.md). Will transfer
  to a neutral foundation when one exists.
