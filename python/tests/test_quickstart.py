"""Tests for `shadow quickstart` and `shadow init --github-action`.

These are onboarding-critical: a new user running `pip install
shadow-diff && shadow quickstart` should produce a diff-able
scenario in 30 seconds.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _shadow(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "shadow.cli.app", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )


# ---- shadow quickstart ---------------------------------------------------


def test_quickstart_scaffolds_all_expected_files(tmp_path: Path) -> None:
    """Every file the QUICKSTART.md promises must land on disk."""
    target = tmp_path / "qs"
    result = _shadow(tmp_path, "quickstart", str(target))
    assert result.returncode == 0, result.stderr

    expected = [
        "agent.py",
        "config_a.yaml",
        "config_b.yaml",
        "QUICKSTART.md",
        "fixtures/baseline.agentlog",
        "fixtures/candidate.agentlog",
    ]
    for rel in expected:
        assert (target / rel).is_file(), f"missing {rel}"


def test_quickstart_agentlogs_are_valid(tmp_path: Path) -> None:
    """The scaffolded `.agentlog` files must parse as JSONL records."""
    target = tmp_path / "qs"
    _shadow(tmp_path, "quickstart", str(target))
    for name in ("baseline.agentlog", "candidate.agentlog"):
        records = [
            json.loads(line)
            for line in (target / "fixtures" / name).read_text().splitlines()
            if line
        ]
        assert len(records) >= 1
        assert records[0]["kind"] == "metadata"
        # Any chat_response record should have content + usage.
        responses = [r for r in records if r["kind"] == "chat_response"]
        assert responses, f"{name}: no chat_response records"
        assert "content" in responses[0]["payload"]
        assert "usage" in responses[0]["payload"]


def test_quickstart_output_diffs_successfully(tmp_path: Path) -> None:
    """Scaffolded fixtures must actually diff — the 'next steps' command
    in the quickstart output must work as advertised."""
    target = tmp_path / "qs"
    _shadow(tmp_path, "quickstart", str(target))
    diff = _shadow(
        target,
        "diff",
        "fixtures/baseline.agentlog",
        "fixtures/candidate.agentlog",
        "--output-json",
        "diff.json",
    )
    assert diff.returncode == 0, diff.stderr
    report = json.loads((target / "diff.json").read_text())
    assert len(report["rows"]) == 9  # nine axes
    assert "drill_down" in report
    assert "recommendations" in report
    assert "first_divergence" in report


def test_quickstart_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    """Running `quickstart` twice should skip existing files unless --force."""
    target = tmp_path / "qs"
    _shadow(tmp_path, "quickstart", str(target))

    # Tweak one of the files, then rerun without --force.
    (target / "agent.py").write_text("# user-edited\n")
    _shadow(tmp_path, "quickstart", str(target))
    assert (target / "agent.py").read_text() == "# user-edited\n"

    # With --force, the user's edit is overwritten.
    _shadow(tmp_path, "quickstart", str(target), "--force")
    assert (target / "agent.py").read_text() != "# user-edited\n"


def test_quickstart_prints_next_steps(tmp_path: Path) -> None:
    """The CLI should tell the user what to do next — adoption UX."""
    target = tmp_path / "qs"
    result = _shadow(tmp_path, "quickstart", str(target))
    combined = (result.stdout + result.stderr).lower()
    assert "next steps" in combined
    assert "shadow diff" in combined
    assert "baseline.agentlog" in combined


# ---- shadow init --github-action ----------------------------------------


def test_init_github_action_writes_diagnose_pr_workflow_by_default(tmp_path: Path) -> None:
    """`shadow init --github-action` defaults to scaffolding the
    diagnose-pr flow (the wedge), not the legacy raw `shadow diff`."""
    result = _shadow(tmp_path, "init", str(tmp_path), "--github-action")
    assert result.returncode == 0, result.stderr
    wf = tmp_path / ".github" / "workflows" / "shadow-diagnose-pr.yml"
    assert wf.is_file(), "default --github-action should write shadow-diagnose-pr.yml"
    # The legacy filename must NOT be present in the default path.
    assert not (tmp_path / ".github" / "workflows" / "shadow-diff.yml").exists()

    import yaml

    parsed = yaml.safe_load(wf.read_text())
    assert parsed["name"] == "shadow diagnose-pr"
    # PyYAML interprets the `on:` key as a Python bool True, not a string —
    # known YAML 1.1 quirk. Handle both.
    triggers = parsed.get("on") or parsed.get(True)
    assert triggers is not None
    assert "pull_request" in triggers
    job = parsed["jobs"]["diagnose"]
    assert job["runs-on"] == "ubuntu-latest"
    # At least one step must install shadow-diff and one must invoke
    # the diagnose-pr CLI surface (via gate-pr for verdict-mapped exit).
    text = wf.read_text()
    assert "shadow-diff" in text
    assert "shadow gate-pr" in text
    # The scaffold must wire --changed-files and --baseline-ref into
    # gate-pr so line-level prompt blame fires automatically in PRs.
    # Otherwise the new file:line attribution from P3 is dead code in
    # CI and only available to manual CLI users.
    assert "--changed-files" in text
    assert "--baseline-ref" in text
    # And the base SHA env var the action reads from must be defined.
    assert "BASE_SHA" in text


def test_init_github_action_legacy_diff_writes_old_workflow(tmp_path: Path) -> None:
    """`--legacy-diff` opts back into the older raw `shadow diff` flow."""
    result = _shadow(tmp_path, "init", str(tmp_path), "--github-action", "--legacy-diff")
    assert result.returncode == 0, result.stderr
    wf = tmp_path / ".github" / "workflows" / "shadow-diff.yml"
    assert wf.is_file()

    import yaml

    parsed = yaml.safe_load(wf.read_text())
    assert parsed["name"] == "shadow diff"
    job = parsed["jobs"]["diff"]
    assert job["runs-on"] == "ubuntu-latest"
    install_steps = [s for s in job["steps"] if "shadow-diff" in str(s)]
    assert install_steps, "no step installs shadow-diff"


def test_init_github_action_does_not_overwrite_existing(tmp_path: Path) -> None:
    """Existing workflow files must be preserved."""
    wf_dir = tmp_path / ".github" / "workflows"
    wf_dir.mkdir(parents=True)
    wf_path = wf_dir / "shadow-diagnose-pr.yml"
    wf_path.write_text("# user-crafted workflow\n")

    _shadow(tmp_path, "init", str(tmp_path), "--github-action")
    assert wf_path.read_text() == "# user-crafted workflow\n"


def test_init_without_flag_leaves_no_workflow(tmp_path: Path) -> None:
    """Baseline `init` must not create a workflow unless --github-action is set."""
    _shadow(tmp_path, "init", str(tmp_path))
    assert not (tmp_path / ".github").exists()


# ---- quickstart + init compose ------------------------------------------


def test_quickstart_then_init_github_action_produces_runnable_repo(
    tmp_path: Path,
) -> None:
    """Realistic adoption path: quickstart drops traces, init wires CI.

    Result should be a directory ready to `git add . && git commit`
    with all the pieces present.
    """
    # User runs `shadow quickstart .`
    _shadow(tmp_path, "quickstart", ".")
    # Then `shadow init . --github-action` to wire the CI.
    _shadow(tmp_path, "init", ".", "--github-action")

    assert (tmp_path / "fixtures" / "baseline.agentlog").is_file()
    assert (tmp_path / "fixtures" / "candidate.agentlog").is_file()
    assert (tmp_path / ".shadow" / "config.toml").is_file()
    assert (tmp_path / ".github" / "workflows" / "shadow-diagnose-pr.yml").is_file()


@pytest.mark.parametrize("extra_flag", [[], ["--force"]])
def test_quickstart_exits_zero(tmp_path: Path, extra_flag: list[str]) -> None:
    """Baseline sanity — returncode 0 in both fresh and re-run cases."""
    target = tmp_path / "qs"
    r1 = _shadow(tmp_path, "quickstart", str(target), *extra_flag)
    assert r1.returncode == 0
    r2 = _shadow(tmp_path, "quickstart", str(target), *extra_flag)
    assert r2.returncode == 0
