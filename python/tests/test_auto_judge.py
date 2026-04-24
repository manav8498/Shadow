"""Tests for `--judge auto` key-detection resolution.

Nailing `auto -> sanity on <backend>` deterministic behaviour matters
because this is what a new user hits the moment they run
`shadow diff` with an API key in their env. One off behaviour here
means either a surprise API call or axis 8 silently blank.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_diff(
    extra_env: dict[str, str],
    baseline: Path,
    candidate: Path,
    judge_flag: str = "auto",
) -> subprocess.CompletedProcess[str]:
    import os

    env = dict(os.environ)
    # Strip both keys first so tests don't inherit a real key from the
    # host shell.
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.update(extra_env)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "diff",
            str(baseline),
            str(candidate),
            "--judge",
            judge_flag,
        ],
        capture_output=True,
        text=True,
        # Windows' default cp1252 can't decode Shadow's ⚠/✓/✗ glyphs;
        # the child emits UTF-8 (see _force_utf8_io in cli/app.py) so
        # we must read it the same way or the reader thread crashes
        # and r.stdout/r.stderr come back as None.
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )


def _fixtures() -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[2]
    return (
        repo / "examples/demo/fixtures/baseline.agentlog",
        repo / "examples/demo/fixtures/candidate.agentlog",
    )


def test_auto_without_any_key_falls_through_to_none() -> None:
    """`--judge auto` with neither key must print a dim hint and
    NOT make a network call, NOT error out."""
    b, c = _fixtures()
    r = _run_diff({}, b, c)
    assert r.returncode == 0, r.stderr
    combined = r.stdout + r.stderr
    # The hint should mention the env-var names so the user knows
    # which to set.
    assert "ANTHROPIC_API_KEY" in combined
    assert "OPENAI_API_KEY" in combined
    assert "no ANTHROPIC_API_KEY" in combined.lower() or "no anthropic" in combined.lower()


def test_auto_with_anthropic_key_picks_anthropic() -> None:
    """`--judge auto` + ANTHROPIC_API_KEY should select anthropic.

    We use a dummy key that will make the real backend call fail —
    the test only checks the CLI's pre-flight message, not that the
    call succeeds. This avoids network dependency.
    """
    b, c = _fixtures()
    r = _run_diff({"ANTHROPIC_API_KEY": "sk-ant-test-DO-NOT-USE"}, b, c)
    # The pre-flight decision log line must appear regardless of
    # whether the backend call later fails.
    combined = r.stdout + r.stderr
    assert "auto -> sanity on anthropic" in combined.lower()


def test_auto_with_openai_key_picks_openai_when_anthropic_absent() -> None:
    b, c = _fixtures()
    r = _run_diff({"OPENAI_API_KEY": "sk-test-DO-NOT-USE"}, b, c)
    combined = r.stdout + r.stderr
    assert "auto -> sanity on openai" in combined.lower()


def test_auto_prefers_anthropic_when_both_keys_set() -> None:
    """Preference order: anthropic first (cheaper Haiku), openai second."""
    b, c = _fixtures()
    r = _run_diff(
        {
            "ANTHROPIC_API_KEY": "sk-ant-test-DO-NOT-USE",
            "OPENAI_API_KEY": "sk-test-DO-NOT-USE",
        },
        b,
        c,
    )
    combined = r.stdout + r.stderr
    assert "auto -> sanity on anthropic" in combined.lower()
    # Must NOT have fallen through to openai.
    assert "auto -> sanity on openai" not in combined.lower()


def test_explicit_judge_none_overrides_auto_ignored() -> None:
    """Passing an explicit judge (not auto) must bypass the
    resolver, even if API keys are set."""
    b, c = _fixtures()
    r = _run_diff(
        {"ANTHROPIC_API_KEY": "sk-ant-test"},
        b,
        c,
        judge_flag="none",
    )
    combined = r.stdout + r.stderr
    # None of the auto decision-log lines should appear.
    assert "auto -> sanity" not in combined.lower()
    assert r.returncode == 0
