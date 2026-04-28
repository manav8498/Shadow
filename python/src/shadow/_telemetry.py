"""Opt-in anonymous telemetry for Shadow OSS.

Default-off. Records nothing without explicit consent. Designed so
maintainers can answer questions like:

  - How many unique installs / week?
  - Which Python versions are in the field?
  - Which OS / architecture combinations?
  - Which CLI commands get used?

Without this, an OSS project flies blind on adoption signals.

Privacy guarantees
------------------
- No code, payloads, agent traces, or user data ever leaves the
  user's machine. The only fields collected are:
    * SDK version (e.g. "2.5.0")
    * Python version (e.g. "3.11")
    * OS (e.g. "Linux", "Darwin", "Windows")
    * CPU architecture (e.g. "x86_64", "arm64")
    * Anonymous install ID — random UUID4 generated once on first
      run, stored at ~/.shadow/install_id, never tied to identity
- The user must explicitly opt in on first run via an interactive
  prompt. Default is OFF.
- CI environments (CI=true env var, GITHUB_ACTIONS, etc.) are
  detected and skipped — no prompt, no telemetry, no exception.
- ``SHADOW_TELEMETRY=off`` environment variable disables telemetry
  unconditionally for the lifetime of that shell.
- Events are sent best-effort: any network error is silently
  swallowed. Telemetry never blocks a CLI command and never errors
  the user's run.

Backend
-------
Events are sent to a PostHog free-tier project. The endpoint and
project key are baked into this module; nobody else can write to
them. PostHog's privacy posture: GDPR-compliant, EU data residency
available, IP addresses NOT stored.

Configuration is read on import; toggling telemetry mid-process
requires re-importing or restarting the process.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

# Public PostHog write-only project key for the Shadow OSS project.
# Can only ingest events; cannot read or modify project state.
_POSTHOG_API_KEY = "shadow-oss-public-write-key-placeholder"
_POSTHOG_ENDPOINT = "https://us.i.posthog.com/capture/"

_INSTALL_ID_PATH = Path.home() / ".shadow" / "install_id"
_OPT_IN_FLAG_PATH = Path.home() / ".shadow" / "telemetry_opt_in"


def _is_ci() -> bool:
    """Heuristic: are we running inside a CI environment?

    CI runs are noisy and not interesting for adoption metrics. We
    auto-skip the opt-in prompt and skip event emission in any
    environment with one of the standard CI signals.
    """
    return any(
        os.environ.get(var)
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
        )
    )


def _is_explicitly_disabled() -> bool:
    """SHADOW_TELEMETRY=off (or 0/false/no) disables for the run."""
    val = os.environ.get("SHADOW_TELEMETRY", "").lower()
    return val in {"off", "0", "false", "no"}


def _is_opted_in() -> bool:
    """User has explicitly opted in (the file exists and contains 'yes')."""
    try:
        return _OPT_IN_FLAG_PATH.is_file() and _OPT_IN_FLAG_PATH.read_text().strip() == "yes"
    except OSError:
        return False


def _get_or_create_install_id() -> str:
    """Anonymous install ID — random UUID4 stored once on first run.

    Cannot be tied to identity; survives across runs of Shadow but
    has no other usage and no relation to the user's GitHub / SSH /
    OS identity. Users who delete ~/.shadow/install_id get a fresh ID.
    """
    try:
        if _INSTALL_ID_PATH.is_file():
            return _INSTALL_ID_PATH.read_text().strip()
        _INSTALL_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_id = str(uuid.uuid4())
        _INSTALL_ID_PATH.write_text(new_id)
        return new_id
    except OSError:
        # If we can't read or write the file, emit a per-process random
        # ID. We get one event per process invocation — still useful.
        return str(uuid.uuid4())


def is_telemetry_enabled() -> bool:
    """True iff all preconditions are met for emitting events.

    Returns False in CI, when explicitly disabled, when the user
    has not opted in, or when the placeholder API key is still in
    place (the production deployment will replace the placeholder).
    """
    if _is_explicitly_disabled():
        return False
    if _is_ci():
        return False
    if not _is_opted_in():
        return False
    # Build hasn't been provisioned with a real key — stay silent.
    return not _POSTHOG_API_KEY.startswith("shadow-oss-public-write-key-placeholder")


def prompt_opt_in_if_needed() -> None:
    """Interactive opt-in prompt, shown once on first interactive run.

    Skipped in CI, when SHADOW_TELEMETRY=off, when stdin/stdout
    aren't a TTY, and when the user has already chosen.
    """
    if _is_ci() or _is_explicitly_disabled():
        return
    if _OPT_IN_FLAG_PATH.is_file():
        return  # already chose, never ask again
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return  # piped / scripted invocation — don't block on prompt

    print(
        "\nShadow can send anonymous usage telemetry (SDK version, OS,"
        " Python version, CLI command names — never code or trace data)."
        "\nThis helps maintainers understand adoption."
        "\n\nSee shadow._telemetry docstring for the full data list and "
        "privacy posture."
        "\n\nEnable telemetry? [y/N] ",
        end="",
        flush=True,
    )
    try:
        choice = sys.stdin.readline().strip().lower()
    except (KeyboardInterrupt, EOFError):
        choice = ""
    try:
        _OPT_IN_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _OPT_IN_FLAG_PATH.write_text("yes" if choice in {"y", "yes"} else "no")
    except OSError:
        pass


def _build_event(event_name: str, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    from shadow import __version__

    base = {
        "shadow_version": __version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "os": platform.system(),
        "machine": platform.machine(),
    }
    if properties:
        base.update(properties)
    return {
        "api_key": _POSTHOG_API_KEY,
        "event": event_name,
        "distinct_id": _get_or_create_install_id(),
        "properties": base,
    }


def emit(event_name: str, properties: dict[str, Any] | None = None) -> None:
    """Best-effort emit a telemetry event.

    No-op when telemetry isn't enabled. Network errors are silently
    swallowed — telemetry must never break a CLI run.
    """
    if not is_telemetry_enabled():
        return
    payload = _build_event(event_name, properties)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _POSTHOG_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    # Short timeout so a slow network never blocks the user.
    # Silent failure on any network/OS error by design — telemetry
    # must never break a CLI run.
    with contextlib.suppress(urllib.error.URLError, OSError, TimeoutError):
        urllib.request.urlopen(req, timeout=2.0).close()


__all__ = [
    "emit",
    "is_telemetry_enabled",
    "prompt_opt_in_if_needed",
]
