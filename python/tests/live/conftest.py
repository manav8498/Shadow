"""Pytest fixtures + global skip gate for live-provider tests.

All tests under ``python/tests/live/`` skip cleanly unless
``SHADOW_RUN_NETWORK_TESTS=1`` is set. Per-provider fixtures
additionally skip if the relevant API key env var is missing — so a
contributor with only ``OPENAI_API_KEY`` can run the OpenAI suite
without spurious Anthropic failures (and vice versa).

The fixtures here are intentionally small — they exist to keep the
gating logic out of each individual test file.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

RUN_LIVE = os.environ.get("SHADOW_RUN_NETWORK_TESTS") == "1"


_LIVE_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply the ``SHADOW_RUN_NETWORK_TESTS`` gate to every test in
    THIS directory (and no others). Tests still need API keys to
    actually run; the per-provider fixtures enforce that downstream.

    A conftest's ``pytest_collection_modifyitems`` hook is called by
    pytest's central collector and receives ``items`` from the whole
    session, not just items collected under this directory — so we
    must filter explicitly to avoid silently skipping the entire
    test suite.
    """
    if RUN_LIVE:
        return
    skip = pytest.mark.skip(reason="SHADOW_RUN_NETWORK_TESTS != 1 — live-provider tests are gated")
    for item in items:
        try:
            item_path = Path(str(item.fspath)).resolve()
        except (AttributeError, OSError):
            continue
        try:
            item_path.relative_to(_LIVE_DIR)
        except ValueError:
            continue
        item.add_marker(skip)


@pytest.fixture
def live_openai_client() -> Any:
    """Return a real ``openai.OpenAI()`` client or skip the test.

    Requires ``OPENAI_API_KEY`` in the environment. Skips cleanly if
    the SDK isn't importable so a partial install doesn't break the
    Anthropic half of the suite.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    try:
        import openai
    except ImportError:
        pytest.skip("openai SDK not installed")
    return openai.OpenAI()


@pytest.fixture
def live_anthropic_client() -> Any:
    """Return a real ``anthropic.Anthropic()`` client or skip the test.

    Requires ``ANTHROPIC_API_KEY`` in the environment. Skips cleanly
    if the SDK isn't importable.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    try:
        import anthropic
    except ImportError:
        pytest.skip("anthropic SDK not installed")
    return anthropic.Anthropic()


@pytest.fixture
def shadow_session(tmp_path: Path) -> Iterator[tuple[Any, Path]]:
    """Yield ``(Session, agentlog_path)`` with auto-instrumentation on.

    The session writes to a tmp ``.agentlog`` file. Tests read the
    file back after ``__exit__`` to assert what Shadow recorded — the
    invariant under test is the recorded-record shape, not just the
    raw SDK response.
    """
    from shadow.sdk import Session

    out = tmp_path / "live.agentlog"
    session = Session(output_path=out, auto_instrument=True)
    yield session, out
