# Shadow demo — vhs script (v3.1.1 — diagnose-pr wedge)
# ─────────────────────────────────────────────────────────────────
# Renders the README's hero GIF: ~60-second walkthrough of the
# v3.1 headline flow.
#
# Story arc:
#   Beat 1 — `shadow demo` (the wow moment, free, no API key)
#   Beat 2 — `shadow diagnose-pr` against the refund-agent fixtures
#            (the diagnose → verdict → dominant cause → fix loop)
#   Beat 3 — `shadow verify-fix` (closes the diagnose → fix → verify
#            loop)
#
# Render with:
#   vhs .github/assets/demo.tape
#
# Or via Docker (no local install required):
#   docker run --rm -v $PWD:/vhs ghcr.io/charmbracelet/vhs \
#       .github/assets/demo.tape
#
# Output: .github/assets/demo.gif + demo.mp4 + demo.webm

Output .github/assets/demo.gif
Output .github/assets/demo.mp4
Output .github/assets/demo.webm

# ─── Terminal config ─────────────────────────────────────────────
Set Shell    "bash"
Set FontSize 18
Set Width    1400
Set Height   900
Set Padding  28
Set Theme    "Catppuccin Mocha"
Set TypingSpeed     50ms
Set PlaybackSpeed   1.0
Set MarginFill "#1e1e2e"
Set Margin     14
Set BorderRadius 10

# ─── Setup (hidden from the GIF) ─────────────────────────────────
# Materialise the refund-causal-diagnosis fixtures into the working
# dir so the diagnose-pr beat reads files at relative paths
# (visually cleaner than absolute paths from /private/var/folders/…).
Hide
Type "cd $(mktemp -d)"
Enter
Type "cp -R $HOME/Downloads/Shadow/examples/refund-causal-diagnosis/* . 2>/dev/null || cp -R /Users/manavpatel/Downloads/Shadow/examples/refund-causal-diagnosis/* ."
Enter
Type "export PS1='\$ '"
Enter
Type "clear"
Enter
Sleep 200ms
Show

# ─── Beat 1: shadow demo — the free wow moment ────────────────────
Type@40ms "# Find the exact change that broke your AI agent."
Sleep 700ms
Enter
Type@40ms "# One command. No setup. No API key."
Sleep 800ms
Enter
Sleep 300ms
Type "shadow demo"
Sleep 350ms
Enter
Sleep 6000ms

# ─── Beat 2: shadow diagnose-pr — the headline command ───────────
Type "clear"
Enter
Type@40ms "# Real PR scenario: the refund agent's prompt loses 'always confirm'."
Sleep 700ms
Enter
Type@40ms "# diagnose-pr names the exact change that caused the regression."
Sleep 800ms
Enter
Sleep 300ms
Type "shadow diagnose-pr \"
Enter
Type "  --traces baseline_traces/ \"
Enter
Type "  --candidate-traces candidate_traces/ \"
Enter
Type "  --baseline-config baseline.yaml \"
Enter
Type "  --candidate-config candidate.yaml \"
Enter
Type "  --policy policy.yaml \"
Enter
Type "  --backend mock \"
Enter
Type "  --pr-comment comment.md \"
Enter
Type "  --out report.json"
Sleep 350ms
Enter
Sleep 4000ms

# Show the verdict + dominant cause as it would land on the PR.
Type "clear"
Enter
Type "cat comment.md | head -30"
Sleep 350ms
Enter
Sleep 6500ms

# ─── Beat 3: shadow verify-fix — closes the loop ─────────────────
Type "clear"
Enter
Type@40ms "# After landing the fix: re-run only the affected traces."
Sleep 700ms
Enter
Type@40ms "# Asserts the regression reversed AND no collateral damage."
Sleep 800ms
Enter
Sleep 300ms
Type "shadow verify-fix \"
Enter
Type "  --report report.json \"
Enter
Type "  --traces baseline_traces \"
Enter
Type "  --fixed-traces baseline_traces \"
Enter
Type "  --policy policy.yaml \"
Enter
Type "  --out verify.json"
Sleep 350ms
Enter
Sleep 4500ms

# ─── Final beat: the install-line ────────────────────────────────
Type "clear"
Enter
Type@40ms "# pip install shadow-diff"
Sleep 600ms
Enter
Type@40ms "# Apache-2.0. Runs entirely on your machine."
Sleep 600ms
Enter
Type@40ms "# https://github.com/manav8498/Shadow"
Sleep 1500ms

# ─── Final pause for the GIF ─────────────────────────────────────
Sleep 2000ms
