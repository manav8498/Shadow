# Security Policy

> **Honest scope note (as of v1.2.x):** Shadow has **not** had a formal
> third-party security audit. The v1.1 hardening pass addresses resource-exhaustion, path-traversal, and subprocess-
> injection classes of bug, but is not a substitute for a penetration
> test by a qualified firm. If you run Shadow on untrusted inputs in
> a multi-tenant context, commission an audit.

## Supported versions

| Version | Supported |
|---------|:---------:|
| `1.x`   | ✅ latest minor receives security patches |
| `0.x`   | ❌ |

Upgrades within `1.x.y` remain backward-compatible at the `.agentlog`
format level (see `SPEC.md §13`).

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

## v1.1 hardening pass — what changed

A concrete hardening pass was shipped alongside v1.1. Each item below
names the attack surface, the mitigation, and (importantly) what is
**not** addressed so you know where your own risk posture has to
carry the load.

### 1. `.agentlog` parser — resource exhaustion

**Risk**: a malicious or truncated `.agentlog` could exhaust memory
via a newline-free stream (unbounded `read_line`), a deeply-nested
JSON payload, or a multi-gigabyte file.

**Hardened**:
- `DEFAULT_MAX_LINE_BYTES = 16 MiB` — per-line cap enforced via
  `Read::take`, so a newline-free stream errors out at the cap
  rather than growing the buffer.
- `DEFAULT_MAX_TOTAL_BYTES = 1 GiB` — whole-trace cap that keeps
  `parse_all` from accumulating an unbounded `Vec<Record>`.
- Both limits are tunable per-`Parser` via `with_max_line_bytes` /
  `with_max_total_bytes`; callers ingesting legitimate larger
  records (multimodal payloads, batch ingest) can raise explicitly.
- New typed errors `LineTooLarge { line, bytes, limit }` and
  `TraceTooLarge { bytes, limit }` — actionable, not silent.

**NOT hardened**:
- JSON depth limits beyond serde_json's default recursion guard.
  Out of scope for a local-first tool; for server-side ingestion,
  wrap the parser in your own JSON sanity layer.

### 2. `shadow record` — env-var driven writes

**Risk**: `SHADOW_SESSION_OUTPUT` lets the shim write wherever the
invoking user can. A poisoned shell profile could redirect output
somewhere unwanted.

**Hardened** (v0.3.1 + v1.1):
- `shadow record` writable-preflights the output path in the parent
  before spawning the child. Test-locked via
  `test_shadow_record_fails_fast_on_unwritable_output_path`.
- Env handling: no user-input splat; only `SHADOW_SESSION_OUTPUT`,
  `SHADOW_SESSION_TAGS`, and an augmented `PYTHONPATH` are set
  explicitly.

**NOT hardened**:
- The shim doesn't verify the output path is inside the caller's
  project dir. Same posture as any env-var-driven test runner.

### 3. `shadow init` / `shadow quickstart` — path arguments

**Risk**: path-traversal via `../../../etc/` style arguments. Always
scoped to what the invoking user can write, so low-impact — but bad
form and worth closing.

**Hardened**:
- Both commands use `Path.resolve()` before any write, so symlink
  tricks resolve before the write check. Writes to paths outside
  the current working directory fail cleanly.

**NOT hardened**:
- We don't disallow absolute paths outright — users legitimately
  want to scaffold into `~/projects/foo/`.

### 4. Subprocess argv — shell injection

**Not a real risk**: `subprocess.run(args, shell=False, env=env)` is
the only spawn path. No shell-string concatenation. The user's argv
is passed verbatim as an argument vector. Documented for reviewers
who want to confirm by inspection.

### 5. Supply-chain — wheel integrity

**Hardened** (v0.2.2):
- Every wheel on PyPI is signed via Sigstore cosign (keyless OIDC).
  Attestations published alongside the GitHub Release.
- CycloneDX 1.5 SBOMs attached to each release (Python + Rust +
  TypeScript components).
- PyPI uploads use Trusted Publisher (OIDC) — no long-lived token
  stored anywhere.

**NOT hardened**:
- We don't ship reproducible builds. Two consecutive `maturin
  build` invocations on the same commit produce byte-identical
  wheels on the same machine, but not necessarily across machines.

---

**If you find something this pass missed**, please open a private
security advisory as described above. Anything reported in good
faith will be credited (or kept anonymous at your preference) in the
release notes when the fix ships.
