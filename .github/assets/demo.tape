# Shadow demo — vhs script
# ─────────────────────────────────────────────────────────────────
# Renders the README's hero GIF: ~50-second walkthrough of the
# Shadow workflow loop end-to-end. Six commands, all on bundled
# fixtures, no API key, no setup beyond `pip install shadow-diff`.
#
# Render with:
#   vhs .github/assets/demo.tape
#
# Or via Docker (no local install required):
#   docker run --rm -v $PWD:/vhs ghcr.io/charmbracelet/vhs \
#       .github/assets/demo.tape
#
# Output: .github/assets/demo.gif

Output .github/assets/demo.gif

# ─── Terminal config ─────────────────────────────────────────────
Set Shell    "bash"
Set FontSize 18
Set Width    1400
Set Height   900
Set Padding  28
Set Theme    "Catppuccin Mocha"
Set TypingSpeed     55ms
Set PlaybackSpeed   1.0
Set MarginFill "#1e1e2e"
Set Margin     14
Set BorderRadius 10

# ─── Setup (hidden from the GIF) ─────────────────────────────────
Hide
Type "cd $(mktemp -d)"
Enter
Type "export PS1='\$ '"
Enter
Type "clear"
Enter
Sleep 200ms
Show

# ─── Beat 1: shadow demo — wow moment ────────────────────────────
# One command. No setup. Real nine-axis diff on bundled fixtures.
Type@40ms "# Behavior testing for LLM agents, in the pull request."
Sleep 600ms
Enter
Type@40ms "# One command, no API key, no setup."
Sleep 800ms
Enter
Sleep 400ms
Type "shadow demo"
Sleep 350ms
Enter
Sleep 5500ms

# ─── Beat 2: shadow call — ship-readiness in one line ───────────
Type "clear"
Enter
Type@40ms "# Decide ship / hold / probe / stop in one line."
Sleep 600ms
Enter
Sleep 300ms
Hide
Type "shadow quickstart . > /dev/null 2>&1 && clear"
Enter
Sleep 400ms
Show
Type "shadow call fixtures/baseline.agentlog fixtures/candidate.agentlog"
Sleep 350ms
Enter
Sleep 5500ms

# ─── Beat 3: shadow autopr — pin the regression as a policy ──────
Type "clear"
Enter
Type@40ms "# Synthesise a policy that pins the regression."
Sleep 600ms
Enter
Sleep 300ms
Type "shadow autopr fixtures/baseline.agentlog fixtures/candidate.agentlog -o policy.yaml"
Sleep 350ms
Enter
Sleep 4500ms

# ─── Beat 4: shadow ledger + brief — daily glance + broadcast ───
Type "clear"
Enter
Type@40ms "# Recent state at a glance + a Slack-ready brief."
Sleep 600ms
Enter
Sleep 300ms
Type "shadow call fixtures/baseline.agentlog fixtures/candidate.agentlog --log > /dev/null"
Enter
Sleep 800ms
Type "shadow ledger"
Sleep 350ms
Enter
Sleep 4500ms

Type "clear"
Enter
Type "shadow brief --format markdown"
Sleep 350ms
Enter
Sleep 4500ms

# ─── Final pause ─────────────────────────────────────────────────
Sleep 1500ms
