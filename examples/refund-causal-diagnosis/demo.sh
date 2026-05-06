#!/usr/bin/env bash
# Refund-agent causal diagnosis demo.
#
# This is the main demo for `shadow diagnose-pr`. The candidate
# config drops the "always confirm before refunding" instruction
# from the system prompt; against the policy
# `confirm-before-refund`, every refund request now skips the
# confirmation step. Shadow names the prompt change as the
# dominant cause and tells you the fix.
#
# Runs with --backend mock (deterministic, free, offline). Set
# OPENAI_API_KEY and pass --backend live to run against the real
# OpenAI API instead.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$DIR/.shadow/diagnose-pr"
mkdir -p "$OUT_DIR"

shadow diagnose-pr \
  --traces           "$DIR/baseline_traces" \
  --candidate-traces "$DIR/candidate_traces" \
  --baseline-config  "$DIR/baseline.yaml" \
  --candidate-config "$DIR/candidate.yaml" \
  --policy           "$DIR/policy.yaml" \
  --changed-files    "prompts/candidate.md" \
  --out              "$OUT_DIR/report.json" \
  --pr-comment       "$OUT_DIR/comment.md" \
  --backend          "${SHADOW_DEMO_BACKEND:-mock}" \
  --n-bootstrap      500

echo
echo "------------------- PR comment -------------------"
cat "$OUT_DIR/comment.md"
echo "--------------------------------------------------"
echo
echo "Full report:  $OUT_DIR/report.json"
echo "PR comment:   $OUT_DIR/comment.md"
echo
echo "To verify the fix (compliant candidate):"
echo "  shadow verify-fix \\"
echo "    --report '$OUT_DIR/report.json' \\"
echo "    --traces '$DIR/baseline_traces' \\"
echo "    --fixed-traces '$DIR/baseline_traces' \\"
echo "    --out '$OUT_DIR/verify.json'"
