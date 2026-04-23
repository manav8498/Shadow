#!/usr/bin/env bash
# Shadow end-to-end demo. Runs in <10s against committed fixtures.
#
#   $ bash examples/demo/demo.sh
#
# No network, no real LLM calls. Produces a nine-axis diff table in the
# terminal and writes intermediate artefacts under .shadow-demo/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

WORK="$(mktemp -d -t shadow-demo-XXXXXX)"
BASELINE="$SCRIPT_DIR/fixtures/baseline.agentlog"
CANDIDATE_REF="$SCRIPT_DIR/fixtures/candidate.agentlog"

# Prefer the in-repo venv's `shadow` when available, fall back to $PATH.
SHADOW_BIN="${REPO_ROOT}/.venv/bin/shadow"
if [[ ! -x "$SHADOW_BIN" ]]; then
  SHADOW_BIN="shadow"
fi

echo "→ shadow init"
"$SHADOW_BIN" init "$WORK" >/dev/null

echo "→ shadow replay (config_b against baseline, positional mock)"
"$SHADOW_BIN" replay "$SCRIPT_DIR/config_b.yaml" \
  --baseline "$BASELINE" \
  --backend positional \
  --reference "$CANDIDATE_REF" \
  --output "$WORK/replay.agentlog" >/dev/null

echo "→ shadow diff (baseline vs replay)"
"$SHADOW_BIN" diff "$BASELINE" "$WORK/replay.agentlog" \
  --seed 42 \
  --output-json "$WORK/report.json"

echo
echo "→ shadow bisect (config_a vs config_b)"
"$SHADOW_BIN" bisect "$SCRIPT_DIR/config_a.yaml" "$SCRIPT_DIR/config_b.yaml" \
  --traces "$BASELINE" \
  --output-json "$WORK/bisect.json" >/dev/null

echo "→ shadow report --format markdown (preview)"
"$SHADOW_BIN" report "$WORK/report.json" --format markdown | head -n 22

echo
echo "Artefacts in $WORK/"
echo "  - replay.agentlog  : the candidate trace"
echo "  - report.json      : nine-axis DiffReport"
echo "  - bisect.json      : attribution of deltas to axes"
