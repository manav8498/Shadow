"""Tests for the opt-in telemetry module.

Verifies the privacy-preserving defaults and the absence of any
network call without explicit consent.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from shadow import _telemetry


@pytest.fixture(autouse=True)
def _isolate_install_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect ~/.shadow/* to a tmp dir so tests don't touch real state."""
    monkeypatch.setattr(_telemetry, "_INSTALL_ID_PATH", tmp_path / "install_id")
    monkeypatch.setattr(_telemetry, "_OPT_IN_FLAG_PATH", tmp_path / "opt_in")


@pytest.fixture(autouse=True)
def _clear_ci_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip CI env vars so is_ci() doesn't auto-disable in our tests."""
    for var in (
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "BUILDKITE",
        "CIRCLECI",
        "JENKINS_URL",
        "TRAVIS",
        "DRONE",
        "APPVEYOR",
        "TEAMCITY_VERSION",
        "SHADOW_TELEMETRY",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Defaults: telemetry off
# ---------------------------------------------------------------------------


class TestTelemetryDefaultsOff:
    def test_disabled_when_no_opt_in_file(self) -> None:
        assert not _telemetry.is_telemetry_enabled()

    def test_disabled_in_ci(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        # Even if user opted in, CI suppresses.
        _telemetry._OPT_IN_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _telemetry._OPT_IN_FLAG_PATH.write_text("yes")
        assert not _telemetry.is_telemetry_enabled()

    def test_disabled_when_env_var_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHADOW_TELEMETRY", "off")
        _telemetry._OPT_IN_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _telemetry._OPT_IN_FLAG_PATH.write_text("yes")
        assert not _telemetry.is_telemetry_enabled()

    @pytest.mark.parametrize("val", ["off", "OFF", "0", "false", "False", "no"])
    def test_env_var_off_variants(self, val: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHADOW_TELEMETRY", val)
        _telemetry._OPT_IN_FLAG_PATH.write_text("yes")
        assert not _telemetry.is_telemetry_enabled()


# ---------------------------------------------------------------------------
# Opt-in must be explicit
# ---------------------------------------------------------------------------


class TestExplicitOptIn:
    def test_opt_in_via_file_does_not_emit_until_real_key(self) -> None:
        """Even with opt-in, the placeholder API key prevents network calls."""
        _telemetry._OPT_IN_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _telemetry._OPT_IN_FLAG_PATH.write_text("yes")
        # Placeholder key still in module — telemetry remains disabled.
        assert not _telemetry.is_telemetry_enabled()

    def test_opt_in_no_does_not_enable(self) -> None:
        _telemetry._OPT_IN_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _telemetry._OPT_IN_FLAG_PATH.write_text("no")
        assert not _telemetry.is_telemetry_enabled()


# ---------------------------------------------------------------------------
# emit() never raises
# ---------------------------------------------------------------------------


class TestEmitSilent:
    def test_emit_when_disabled_is_noop(self) -> None:
        # Should not call urllib at all.
        with mock.patch("urllib.request.urlopen") as urlopen:
            _telemetry.emit("test-event", {"foo": "bar"})
            urlopen.assert_not_called()

    def test_emit_does_not_propagate_network_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even if telemetry IS enabled (force it on), network errors
        must not propagate to the caller."""
        monkeypatch.setattr(_telemetry, "is_telemetry_enabled", lambda: True)
        with mock.patch("urllib.request.urlopen", side_effect=OSError("network down")):
            # Must not raise.
            _telemetry.emit("test-event", {"foo": "bar"})


# ---------------------------------------------------------------------------
# Install ID anonymity + persistence
# ---------------------------------------------------------------------------


class TestInstallId:
    def test_install_id_persists_across_calls(self) -> None:
        first = _telemetry._get_or_create_install_id()
        second = _telemetry._get_or_create_install_id()
        assert first == second
        assert len(first) == 36  # uuid4 string

    def test_install_id_is_uuid4_format(self) -> None:
        import re

        ident = _telemetry._get_or_create_install_id()
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            ident,
        )


# ---------------------------------------------------------------------------
# Prompt only in interactive TTY
# ---------------------------------------------------------------------------


class TestPromptBehaviour:
    def test_prompt_skipped_in_ci(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        # Should not write any flag file.
        _telemetry.prompt_opt_in_if_needed()
        assert not _telemetry._OPT_IN_FLAG_PATH.exists()

    def test_prompt_skipped_when_already_chosen(self) -> None:
        _telemetry._OPT_IN_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _telemetry._OPT_IN_FLAG_PATH.write_text("no")
        original = _telemetry._OPT_IN_FLAG_PATH.read_text()
        _telemetry.prompt_opt_in_if_needed()
        # Flag file content unchanged.
        assert _telemetry._OPT_IN_FLAG_PATH.read_text() == original

    def test_prompt_skipped_when_not_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # When stdin is not a TTY we don't ask, just stay opt-out.
        with (
            mock.patch("sys.stdin.isatty", return_value=False),
            mock.patch("sys.stdout.isatty", return_value=True),
        ):
            _telemetry.prompt_opt_in_if_needed()
        assert not _telemetry._OPT_IN_FLAG_PATH.exists()
