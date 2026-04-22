# Security Policy

## Supported versions

Shadow is pre-1.0. Only the latest `0.x` release receives security patches.
Upgrades within `0.x.y` remain backward-compatible at the `.agentlog`
format level (see `SPEC.md §13`).

| Version | Supported |
|---------|:---------:|
| `0.1.x` | ✅ |
| older   | ❌ |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.** A public
issue tips off attackers and doesn't give us time to ship a fix.

Instead, please open a [private security advisory](https://github.com/manav8498/Shadow/security/advisories/new)
on the GitHub repo. Include:

- A brief description of the vulnerability.
- Steps to reproduce (ideally a minimal repro repo or script).
- The Shadow version / commit SHA and your platform (OS, Rust + Python
  versions).
- Your expected impact (confidentiality / integrity / availability /
  supply chain) and severity estimate.

We will acknowledge within **72 hours** and aim to ship a fix within
**30 days** for high-severity issues, **90 days** for lower severity. We'll
coordinate a disclosure date with you.

## What's in scope

- **Any code path that handles untrusted `.agentlog` files.** The parser,
  canonicalization, and content-hashing routines are supply-chain critical.
- **The Python SDK's redaction layer** — if a default regex mis-fires and
  leaks PII/secrets through a reasonable input, that's in scope.
- **The `shadow` CLI** invoked from CI or from a developer's shell.
- **The `.github/actions/shadow-action`** composite action — it writes to
  PR comments using the caller's `github-token`.

## What's out of scope

- Bugs that require the attacker to be the *author* of the `.agentlog`
  file AND the consumer system admin AND the Judge. (I.e., threat models
  where the attacker already controls all layers.)
- Issues in third-party dependencies — please report those upstream. We
  pin every direct dep to an exact version; if an upstream CVE lands, file
  an issue here so we bump the pin.
- Denial of service from a single maliciously-crafted trace file: we
  accept `>2^20`-iteration bootstrap exhaustion as a known limit. If you
  find sub-polynomial blow-ups (`O(n^3)` on `n`-byte input, etc.),
  please do report.
- The `Judge` axis behaviour when the user supplies a malicious Judge.
  The Judge is user-supplied code; treat it as you'd treat any
  user-supplied callback.

## Disclosure philosophy

We default to coordinated disclosure with credit to the reporter (unless
you'd prefer anonymity). Once a fix ships, we publish an advisory in
`CHANGELOG.md` under the release header, with:

- A description of the vulnerability and impact.
- The fixing commit SHA.
- Credits to the reporter.
- Any user-facing remediation steps.

## Encryption

GitHub's [private security advisory](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
channel is end-to-end between you and the maintainers and is the
preferred transport. If you need a separate encrypted channel, open
the advisory with a note asking for one and we'll coordinate.
