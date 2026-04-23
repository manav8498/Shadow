#!/usr/bin/env bash
# Generate CycloneDX SBOMs for all three language surfaces.
#
# Output files (repo-root relative):
#   sbom/shadow-rust.cdx.json        — Rust crate dependency tree
#   sbom/shadow-python.cdx.json      — Python venv (includes optional extras if installed)
#   sbom/shadow-typescript.cdx.json  — npm dep tree from typescript/node_modules
#
# Enterprise reviewers: verify every SBOM ships with the release artifact
# and matches the artifact's recorded hashes.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p sbom

echo "→ Rust SBOM (cargo-cyclonedx)"
if ! command -v cargo-cyclonedx >/dev/null 2>&1; then
    echo "  installing cargo-cyclonedx (~30s) ..."
    cargo install --locked cargo-cyclonedx --version 0.5.7
fi
cargo cyclonedx --format json --output-pattern package --manifest-path Cargo.toml
# cargo-cyclonedx writes next to Cargo.toml per crate; move into sbom/
find . -name "shadow-core.cdx.json" -not -path "./sbom/*" -exec cp {} sbom/shadow-rust.cdx.json \;

echo "→ Python SBOM (cyclonedx-bom)"
if ! .venv/bin/python -c "import cyclonedx_py" 2>/dev/null; then
    .venv/bin/pip install --quiet cyclonedx-bom==5.0.0
fi
.venv/bin/python -m cyclonedx_py environment -o sbom/shadow-python.cdx.json

echo "→ TypeScript SBOM (@cyclonedx/cyclonedx-npm)"
(
    cd typescript
    npx -y @cyclonedx/cyclonedx-npm --output-file ../sbom/shadow-typescript.cdx.json
)

echo
echo "SBOMs generated:"
ls -lh sbom/*.cdx.json
