# Security Policy

Shadow has not had a formal third-party security audit. The v1.1 hardening pass covers resource exhaustion, path traversal, and subprocess injection, but that is not a substitute for a penetration test. If you run Shadow on untrusted inputs in a multi-tenant context, commission an audit.

## Supported versions

| Version | Supported |
|---------|:---------:|
| `1.x`   | ✅ latest minor receives security patches |
| `0.x`   | ❌ |

Upgrades within `1.x.y` stay backward compatible at the `.agentlog` format level. See `SPEC.md §13`.

## Reporting a vulnerability

Please do not open a public GitHub issue for security problems. A public issue tips off attackers and does not give us time to ship a fix.

Open a [private security advisory](https://github.com/manav8498/Shadow/security/advisories/new) on the GitHub repo instead. Include:

- A brief description of the vulnerability.
- Steps to reproduce (a minimal repro repo or script if you can).
- The Shadow version or commit SHA and your platform (OS, Rust and Python versions).
- Your expected impact (confidentiality, integrity, availability, supply chain) and severity estimate.

We acknowledge within 72 hours and aim to ship a fix within 30 days for high-severity issues and 90 days for lower severity. We coordinate a disclosure date with you.

## What's in scope

- Any code path that handles untrusted `.agentlog` files. The parser, canonicalization, and content-hashing routines are supply-chain critical.
- The Python SDK's redaction layer. If a default regex misfires and leaks PII or secrets through a reasonable input, that is in scope.
- The `shadow` CLI invoked from CI or a developer's shell.
- The `.github/actions/shadow-action` composite action. It writes to PR comments using the caller's `github-token`.

## What's out of scope

- Bugs that need the attacker to be the author of the `.agentlog` file and the consumer's system admin and the Judge. Threat models where the attacker already controls all layers are not interesting.
- Issues in third-party dependencies. Report those upstream. If an upstream CVE lands, file an issue here so we bump the pin.
- Denial of service from a single maliciously crafted trace. A `>2^20`-iteration bootstrap exhaustion is a known limit. If you find sub-polynomial blow-ups (`O(n^3)` on `n`-byte input and similar), please do report.
- The `Judge` axis behaviour when the user supplies a malicious Judge. The Judge is user code. Treat it like any user-supplied callback.

## Disclosure

We default to coordinated disclosure with credit to the reporter (anonymous if you prefer). Once a fix ships, we publish an advisory in `CHANGELOG.md` under the release header with:

- A description of the vulnerability and impact.
- The fixing commit SHA.
- Credits to the reporter.
- Any user-facing remediation steps.

## Encryption

GitHub's [private security advisory](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) channel is end-to-end between you and the maintainers. That is the preferred transport. If you need a separate encrypted channel, open the advisory with a note asking for one and we will coordinate.

## v1.1 hardening pass

A concrete hardening pass shipped alongside v1.1. Each item names the attack surface, the mitigation, and what is not addressed so you know where your own risk posture has to carry the load.

### 1. `.agentlog` parser, resource exhaustion

**Risk.** A malicious or truncated `.agentlog` could exhaust memory via a newline-free stream (unbounded `read_line`), a deeply nested JSON payload, or a multi-gigabyte file.

**Hardened.**
- `DEFAULT_MAX_LINE_BYTES = 16 MiB`. Per-line cap enforced via `Read::take`, so a newline-free stream errors out at the cap rather than growing the buffer.
- `DEFAULT_MAX_TOTAL_BYTES = 1 GiB`. Whole-trace cap that keeps `parse_all` from accumulating an unbounded `Vec<Record>`.
- Both limits are tunable per-`Parser` via `with_max_line_bytes` and `with_max_total_bytes`. Callers ingesting legitimate larger records (multimodal payloads, batch ingest) can raise it.
- New typed errors `LineTooLarge { line, bytes, limit }` and `TraceTooLarge { bytes, limit }`. Actionable, not silent.

**Not hardened.**
- JSON depth limits beyond serde_json's default recursion guard. Out of scope for a local-first tool. For server-side ingestion, wrap the parser in your own JSON sanity layer.

### 2. `shadow record`, env-var driven writes

**Risk.** `SHADOW_SESSION_OUTPUT` lets the shim write wherever the invoking user can. A poisoned shell profile could redirect output somewhere unwanted.

**Hardened.**
- `shadow record` writability-preflights the output path in the parent before spawning the child. Test-locked via `test_shadow_record_fails_fast_on_unwritable_output_path`.
- Env handling: no user-input splat. Only `SHADOW_SESSION_OUTPUT`, `SHADOW_SESSION_TAGS`, and an augmented `PYTHONPATH` are set.

**Not hardened.**
- The shim does not verify the output path is inside the caller's project dir. Same posture as any env-var-driven test runner.

### 3. `shadow init` and `shadow quickstart`, path arguments

**Risk.** Path traversal via `../../../etc/` style arguments. Always scoped to what the invoking user can write, so low impact, but bad form and worth closing.

**Hardened.**
- Both commands use `Path.resolve()` before any write. Symlink tricks resolve before the write check. Writes to paths outside the current working directory fail cleanly.

**Not hardened.**
- We do not disallow absolute paths outright. Users legitimately want to scaffold into `~/projects/foo/`.

### 4. Subprocess argv, shell injection

Not a real risk. `subprocess.run(args, shell=False, env=env)` is the only spawn path. No shell-string concatenation. The user's argv is passed verbatim as an argument vector. Documented for reviewers who want to confirm by inspection.

### 5. Supply chain, wheel integrity

**Hardened.**
- Every wheel on PyPI is signed via Sigstore cosign (keyless OIDC). Attestations published alongside the GitHub Release.
- CycloneDX 1.5 SBOMs attached to each release (Python, Rust, and TypeScript components).
- PyPI uploads use Trusted Publisher (OIDC). No long-lived token stored anywhere.

**Not hardened.**
- We do not ship reproducible builds. Two consecutive `maturin build` invocations on the same commit produce byte-identical wheels on the same machine, but not necessarily across machines.

---

If you find something this pass missed, please open a private security advisory as described above. Anything reported in good faith will be credited (or kept anonymous at your preference) in the release notes when the fix ships.
