# Shadow launch video — vhs script (terminal-only portion)
# ─────────────────────────────────────────────────────────────────
# Renders just the terminal content for the launch video. The
# build script (`scripts/build_launch_video.py`) wraps this output
# with intro / outro cards + ambient music + crossfades.
#
# Tighter than `demo-full.tape`: targets ~60-70s of terminal time
# so the final composited video lands at ~75-85s — within the
# X / LinkedIn sweet spot.

Output .github/assets/launch-raw.mp4
Output .github/assets/launch-raw.gif

# ─── Terminal config ─────────────────────────────────────────────
Set Shell    "bash"
Set FontSize 22
Set Width    1920
Set Height   1080
Set Padding  60
Set Theme    "Catppuccin Mocha"
Set TypingSpeed 50ms
Set Framerate   30
Set MarginFill  "#0a0a0f"
Set Margin      24
Set BorderRadius 14

# ─── Setup (hidden) ──────────────────────────────────────────────
Hide
Type "cd $(mktemp -d)"
Enter
Type "export PS1='\$ '"
Enter
Type "shadow quickstart . > /dev/null 2>&1"
Enter
Type "clear"
Enter
Sleep 200ms
Show

# ─── Beat 1: shadow demo ────────────────────────────────────────
Type@40ms "# 1. shadow demo — one command, no setup"
Sleep 700ms
Enter
Sleep 400ms
Type "shadow demo"
Sleep 350ms
Enter
Sleep 6500ms

# ─── Beat 2: shadow call ────────────────────────────────────────
Type "clear"
Enter
Type@40ms "# 2. shadow call — ship / hold / probe / stop in one panel"
Sleep 700ms
Enter
Sleep 400ms
Type "shadow call fixtures/baseline.agentlog fixtures/candidate.agentlog"
Sleep 350ms
Enter
Sleep 6500ms

# ─── Beat 3: shadow autopr ──────────────────────────────────────
Type "clear"
Enter
Type@40ms "# 3. shadow autopr — pin the regression as a policy"
Sleep 700ms
Enter
Sleep 400ms
Type "shadow autopr fixtures/baseline.agentlog fixtures/candidate.agentlog -o policy.yaml"
Sleep 350ms
Enter
Sleep 4500ms

# ─── Beat 4: shadow ledger ──────────────────────────────────────
Type "clear"
Enter
Type@40ms "# 4. shadow ledger — daily glance over recent artifacts"
Sleep 700ms
Enter
Hide
Type "shadow call fixtures/baseline.agentlog fixtures/candidate.agentlog --log > /dev/null 2>&1"
Enter
Sleep 600ms
Show
Type "shadow ledger"
Sleep 350ms
Enter
Sleep 5500ms

# ─── Beat 5: shadow trail ──────────────────────────────────────
Type "clear"
Enter
Type@40ms "# 5. shadow trail — walk back through the artifact graph"
Sleep 700ms
Enter
Sleep 400ms
Type "shadow trail a6569f69"
Sleep 350ms
Enter
Sleep 5500ms

# ─── Beat 6: shadow certify ────────────────────────────────────
Type "clear"
Enter
Type@40ms "# 6. shadow certify — signed release certificate"
Sleep 700ms
Enter
Sleep 400ms
Type "shadow certify fixtures/candidate.agentlog --agent-id refund-agent@2.3.0 -o release.cert.json"
Sleep 350ms
Enter
Sleep 4000ms
Type "shadow verify-cert release.cert.json"
Sleep 350ms
Enter
Sleep 3500ms

# ─── Hold final frame for transition ───────────────────────────
Sleep 1000ms
