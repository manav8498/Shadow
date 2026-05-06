# Shadow launch video — vhs script (terminal-only portion, v3.1.1)
# ─────────────────────────────────────────────────────────────────
# Renders just the terminal content for the launch video. The
# build script (`scripts/build_launch_video.py`) wraps this output
# with intro / outro cards + ambient music + crossfades.
#
# Story arc (~70-80s of terminal time):
#   Beat 1 — shadow demo (free, no API key, the wow moment)
#   Beat 2 — shadow diagnose-pr (verdict + dominant cause + fix)
#   Beat 3 — shadow verify-fix (closes the diagnose → fix → verify loop)
#   Beat 4 — shadow inspect (terminal trace viewer — daily-debug surface)
#   Beat 5 — shadow scan (secret leak gate — production hygiene)
#   Beat 6 — shadow certify (signed release certificate)

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
Type "cp -R /Users/manavpatel/Downloads/Shadow/examples/refund-causal-diagnosis/* ."
Enter
Type "shadow quickstart qs > /dev/null 2>&1 || true"
Enter
Type "export PS1='\$ '"
Enter
Type "clear"
Enter
Sleep 200ms
Show

# ─── Beat 1: shadow demo ────────────────────────────────────────
Type@40ms "# 1. shadow demo — find the exact change that broke your AI agent"
Sleep 800ms
Enter
Sleep 400ms
Type "shadow demo"
Sleep 350ms
Enter
Sleep 6500ms

# ─── Beat 2: shadow diagnose-pr ─────────────────────────────────
Type "clear"
Enter
Type@40ms "# 2. shadow diagnose-pr — name the exact change that broke the refund agent"
Sleep 800ms
Enter
Sleep 400ms
Type "shadow diagnose-pr --traces baseline_traces/ --candidate-traces candidate_traces/ --baseline-config baseline.yaml --candidate-config candidate.yaml --policy policy.yaml --backend mock --pr-comment comment.md --out report.json"
Sleep 350ms
Enter
Sleep 4500ms

# Show the rendered PR comment — this is what lands on the developer's PR.
Type "clear"
Enter
Type "cat comment.md"
Sleep 350ms
Enter
Sleep 8500ms

# ─── Beat 3: shadow verify-fix — closes the loop ───────────────
Type "clear"
Enter
Type@40ms "# 3. shadow verify-fix — confirm the fix without collateral damage"
Sleep 800ms
Enter
Sleep 400ms
Type "shadow verify-fix --report report.json --traces baseline_traces --fixed-traces baseline_traces --policy policy.yaml --out verify.json"
Sleep 350ms
Enter
Sleep 5000ms

# ─── Beat 4: shadow inspect — daily-debug surface ──────────────
Type "clear"
Enter
Type@40ms "# 4. shadow inspect — terminal trace viewer (pytest -v for agents)"
Sleep 800ms
Enter
Sleep 400ms
Type "shadow inspect qs/fixtures/baseline.agentlog qs/fixtures/candidate.agentlog"
Sleep 350ms
Enter
Sleep 6500ms

# ─── Beat 5: shadow scan — production hygiene ──────────────────
Type "clear"
Enter
Type@40ms "# 5. shadow scan — block secret leaks before they hit your repo"
Sleep 800ms
Enter
Sleep 400ms
Type "shadow scan baseline_traces/ candidate_traces/"
Sleep 350ms
Enter
Sleep 5500ms

# ─── Beat 6: shadow certify — signed release certificate ──────
Type "clear"
Enter
Type@40ms "# 6. shadow certify — signed release certificate"
Sleep 800ms
Enter
Sleep 400ms
Type "shadow certify baseline_traces/s1.agentlog --agent-id refund-agent@3.1.1 --output release.cert.json"
Sleep 350ms
Enter
Sleep 4000ms
Type "shadow verify-cert release.cert.json"
Sleep 350ms
Enter
Sleep 4000ms

# ─── Hold final frame for transition ───────────────────────────
Sleep 1500ms
