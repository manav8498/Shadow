# Supply-chain integrity

Every Shadow release artifact is:

1. **Built by GitHub Actions** on `release.yml` (pinned runners, pinned actions).
2. **Signed with sigstore cosign** via keyless OIDC, no long-lived signing keys to rotate, leak, or lose. The certificate chains back to Fulcio, rooted in the Sigstore Public Good trust root.
3. **Attested with SLSA build provenance** (level 2) via `actions/attest-build-provenance`, recording the GitHub Actions workflow run, commit SHA, and inputs. Level 3 requires builds to take place in a reusable workflow shared across the org (so the build process is isolated from the calling workflow). Shadow does not meet that criterion today.
4. **Shipped with a CycloneDX 1.5 SBOM** per language surface:
   - `sbom-python.cdx.json`, the Python wheel's transitive deps
   - `sbom-typescript.cdx.json`, the npm package's tree
   - `sbom-rust.cdx.json`, the crate's cargo lockfile rendered

## Verifying a release

Assume you downloaded `shadow_diff-3.1.0-cp311-abi3-macosx_11_0_arm64.whl` from the GitHub release:

```bash
# 1. Install cosign (https://docs.sigstore.dev/cosign/installation)
brew install cosign

# 2. Verify the sigstore signature.
# The regex escapes every literal dot, anchors to $, and pins to release
# tags only, rejecting forged certificates from forked repos or non-tag
# workflow runs.
cosign verify-blob \
  --certificate shadow_diff-3.1.0-cp311-abi3-macosx_11_0_arm64.whl.crt \
  --signature   shadow_diff-3.1.0-cp311-abi3-macosx_11_0_arm64.whl.sig \
  --certificate-identity-regexp '^https://github\.com/manav8498/Shadow/\.github/workflows/release\.yml@refs/tags/v[0-9]+\.[0-9]+\.[0-9]+$' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  shadow_diff-3.1.0-cp311-abi3-macosx_11_0_arm64.whl

# 3. Verify the SLSA build provenance attestation.
gh attestation verify \
  shadow_diff-3.1.0-cp311-abi3-macosx_11_0_arm64.whl \
  --owner manav8498
```

Both commands must exit 0 for the artifact to be considered trusted.

## Generating SBOMs locally

```bash ./scripts/generate_sbom.sh
# outputs: sbom/shadow-{rust,python,typescript}.cdx.json
```

These are valid CycloneDX 1.5 JSON documents and can be ingested into any
supply-chain security tool (Snyk, Dependency-Track, Trivy, Grype, etc.).

## What's pinned

- **Rust toolchain**: `1.95.0` (rust-toolchain.toml).
- **Python**: 3.11 / 3.12 (pyproject.toml).
- **Node**: >=18 (typescript/package.json).
- **All dependencies** pinned to exact versions in `Cargo.toml` and
  `pyproject.toml`. Bumps require a commit with CHANGELOG entry.
- **GitHub Actions**: pinned to specific versions (no `@main`). Renovate
  should be configured to open PRs for bumps.

## Scope of supply-chain guarantees

We guarantee:
- Every released artifact was built from the tagged commit.
- Every dependency at build time was pinned at the version declared in the
  lockfile.
- No third-party script ran during the build other than what's declared.
- The SBOM reflects the actual dependency tree at build time.

We do NOT guarantee:
- That upstream dependencies don't have vulnerabilities, run `cargo audit`,
  `pip-audit`, `npm audit` on each release before deploying.
- That a determined attacker with admin GitHub access can't forge a release, this is inherent to any GitHub-hosted OSS project.

## Accepted advisories

`cargo audit` reads `.cargo/audit.toml` to suppress specific RustSec
advisories. Suppressions require a documented justification (dep chain,
build-time vs runtime, why we accept). Current ledger:

| Advisory | Crate | Reason |
|---|---|---|
| [RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436) | `paste` 1.0.15 (unmaintained) | Transitive proc-macro via `statrs → nalgebra → simba`. Build-time only; no runtime surface ships. Cleared when `nalgebra` drops `paste`. |

GitHub Actions in `.github/workflows/` are SHA-pinned (not version-tag
pinned) for the third-party action surface so a maintainer compromise
of an action's `v4` tag can't silently rewrite our CI. Bumps go through
Dependabot's `github-actions` ecosystem. The intended version is
preserved in a comment next to each SHA for reviewer readability:

```yaml
- uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5  # v4.3.1
```

`dtolnay/rust-toolchain@1.95.0` is the documented exception — its
publisher's point-release tags are immutable by convention.
