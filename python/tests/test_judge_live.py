"""Live-LLM integration tests for every judge.

Gated by `SHADOW_RUN_NETWORK_TESTS=1` + the relevant API key env
var. Runs each judge against both Anthropic (Claude) and OpenAI
(GPT) backends with a carefully-chosen scenario per judge where the
correct verdict is unambiguous — so if the real LLM returns
anything other than the expected label, either the judge wiring
is broken or the model is behaving surprisingly.

These tests are *skipped by default in CI* (SHADOW_RUN_NETWORK_TESTS
is not set). Run locally with:

    SHADOW_RUN_NETWORK_TESTS=1 \\
      ANTHROPIC_API_KEY=... \\
      OPENAI_API_KEY=... \\
      pytest python/tests/test_judge_live.py -v

Set only one of the two API-key env vars to run against just that
backend; tests for the missing backend are auto-skipped. Token
budget per full run: ~2k input + ~400 output tokens across all
tests on each backend, roughly $0.01 per backend at current pricing.

## What "unambiguous" means per judge

Each test picks a scenario where the correct answer is self-
evident to any model smart enough to be useful as a judge:

- ProcedureAdherenceJudge: procedure requires `backup_database`
  before `run_migration`, candidate explicitly says "I skipped the
  backup." A competent judge calls this `violated`.
- SchemaConformanceJudge: schema requires `amount_usd: number`,
  candidate emits `amount_usd: "many"`. Competent judge:
  `violates`.
- FactualityJudge: known fact says 30-day refund window, candidate
  says 90 days. Competent judge: `contradicts`.
- RefusalAppropriateJudge: policy says "answer time-of-day
  questions," candidate refuses. Competent judge:
  `inappropriate`.
- ToneJudge: target is "≤ 3 sentences," candidate writes 8.
  Competent judge: `deviates`.
- LlmJudge (generic): three-tier rubric with an obviously-poor
  candidate. Competent judge: `poor`.
- SanityJudge: candidate is clearly worse than baseline on an
  arithmetic question. Competent judge: `worse`.
- PairwiseJudge: same scenario as SanityJudge, with both-order
  evaluation enforcing position-bias-free agreement.

If any backend returns an off verdict on these unambiguous cases,
that's either a signal the model has drifted or the wiring has
broken — both worth catching.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

RUN_LIVE = os.environ.get("SHADOW_RUN_NETWORK_TESTS") == "1"
HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not RUN_LIVE, reason="SHADOW_RUN_NETWORK_TESTS != 1"),
]


def _backends_to_test() -> list[tuple[str, str]]:
    """List of (backend_name, default_model) pairs to exercise."""
    out: list[tuple[str, str]] = []
    if HAS_ANTHROPIC:
        out.append(("anthropic", "claude-haiku-4-5-20251001"))
    if HAS_OPENAI:
        out.append(("openai", "gpt-4o-mini"))
    return out


BACKENDS = _backends_to_test()


def _resp(text: str) -> dict[str, Any]:
    return {
        "model": "x",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "latency_ms": 1,
        "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
    }


def _request(task: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": task}]}


def _skip_if_no_backends() -> None:
    if not BACKENDS:
        pytest.skip(
            "No live backends available. Set ANTHROPIC_API_KEY and/or "
            "OPENAI_API_KEY plus SHADOW_RUN_NETWORK_TESTS=1."
        )


# ---- per-judge live tests -----------------------------------------------


@pytest.mark.parametrize(("backend_name", "model"), BACKENDS)
def test_procedure_adherence_live(backend_name: str, model: str) -> None:
    _skip_if_no_backends()
    from shadow.judge import ProcedureAdherenceJudge
    from shadow.llm import get_backend

    backend = get_backend(backend_name)
    judge = ProcedureAdherenceJudge(
        backend,
        required_procedure=[
            "call `backup_database` before `run_migration`",
            "call `pause_replication` before any destructive op",
        ],
        model=model,
    )
    v = asyncio.run(
        judge.score_pair(
            _resp("I took a backup, then ran the migration."),
            _resp(
                "I skipped the backup and ran the migration directly. "
                "I also did not pause replication."
            ),
            _request("apply migration 42 to prod"),
        )
    )
    assert v["verdict"] == "violated", f"{backend_name}: {v}"
    assert v["score"] == 0.0
    assert v["confidence"] > 0.5, f"confidence too low: {v['confidence']}"


@pytest.mark.parametrize(("backend_name", "model"), BACKENDS)
def test_schema_conformance_live(backend_name: str, model: str) -> None:
    _skip_if_no_backends()
    from shadow.judge import SchemaConformanceJudge
    from shadow.llm import get_backend

    backend = get_backend(backend_name)
    judge = SchemaConformanceJudge(
        backend,
        expected_schema={
            "order_id": "string identifier",
            "amount_usd": "positive number in USD, never a word",
        },
        model=model,
    )
    v = asyncio.run(
        judge.score_pair(
            _resp('{"order_id": "A1", "amount_usd": 30}'),
            _resp('{"order_id": "A1", "amount_usd": "many"}'),
            _request("format this as JSON"),
        )
    )
    assert v["verdict"] == "violates", f"{backend_name}: {v}"
    assert v["score"] == 0.0


@pytest.mark.parametrize(("backend_name", "model"), BACKENDS)
def test_factuality_live(backend_name: str, model: str) -> None:
    _skip_if_no_backends()
    from shadow.judge import FactualityJudge
    from shadow.llm import get_backend

    backend = get_backend(backend_name)
    judge = FactualityJudge(
        backend,
        known_facts=["Acme Widgets offers refunds within 30 days of purchase."],
        model=model,
    )
    v = asyncio.run(
        judge.score_pair(
            _resp("You have 30 days."),
            _resp("Our refund window is 90 days — you have plenty of time to " "request a refund."),
            _request("how long do I have for a refund?"),
        )
    )
    assert v["verdict"] == "contradicts", f"{backend_name}: {v}"
    assert v["score"] == 0.0


@pytest.mark.parametrize(("backend_name", "model"), BACKENDS)
def test_refusal_appropriate_live(backend_name: str, model: str) -> None:
    _skip_if_no_backends()
    from shadow.judge import RefusalAppropriateJudge
    from shadow.llm import get_backend

    backend = get_backend(backend_name)
    judge = RefusalAppropriateJudge(
        backend,
        policy=(
            "REFUSE medical advice and legal advice.\n"
            "ANSWER any other well-formed request, including time-of-day "
            "questions."
        ),
        model=model,
    )
    v = asyncio.run(
        judge.score_pair(
            _resp("It's 3:04 pm."),
            _resp(
                "I'm sorry, but I can't help with that. Please consult a " "qualified professional."
            ),
            _request("what time is it?"),
        )
    )
    assert v["verdict"] == "inappropriate", f"{backend_name}: {v}"
    assert v["score"] == 0.0


@pytest.mark.parametrize(("backend_name", "model"), BACKENDS)
def test_tone_live(backend_name: str, model: str) -> None:
    _skip_if_no_backends()
    from shadow.judge import ToneJudge
    from shadow.llm import get_backend

    backend = get_backend(backend_name)
    judge = ToneJudge(
        backend,
        target_tone=(
            "Concise, under 3 sentences, no effusive apology or " "introductory pleasantries."
        ),
        model=model,
    )
    v = asyncio.run(
        judge.score_pair(
            _resp("Your order ships tomorrow."),
            _resp(
                "Thank you so much for reaching out — I'm genuinely sorry to "
                "hear about any frustration this may be causing you. Let me "
                "take a moment to look into this for you. I want to assure "
                "you that I'll do everything in my power to assist. After "
                "checking our system carefully, I can confirm that your "
                "order is on track. It will be shipped tomorrow. I hope "
                "this helps, and please don't hesitate to reach out if you "
                "have any further questions."
            ),
            _request("when does my order ship?"),
        )
    )
    assert v["verdict"] == "deviates", f"{backend_name}: {v}"
    assert v["score"] == 0.0


@pytest.mark.parametrize(("backend_name", "model"), BACKENDS)
def test_llm_generic_live(backend_name: str, model: str) -> None:
    _skip_if_no_backends()
    from shadow.judge import LlmJudge
    from shadow.llm import get_backend

    backend = get_backend(backend_name)
    judge = LlmJudge(
        backend,
        rubric=(
            "Rate whether the CANDIDATE answers the TASK at least as well "
            "as the BASELINE.\n\n"
            "TASK: {task}\n"
            "BASELINE: {baseline}\n"
            "CANDIDATE: {candidate}\n\n"
            "Reply with only this JSON: "
            '{{"verdict": "great" or "ok" or "poor", "confidence": 0-1, '
            '"reason": "..."}}'
        ),
        score_map={"great": 1.0, "ok": 0.5, "poor": 0.0},
        model=model,
    )
    v = asyncio.run(
        judge.score_pair(
            _resp("The answer is 4."),
            _resp("I think it's maybe 7 or something, not really sure."),
            _request("What is 2 + 2?"),
        )
    )
    assert v["verdict"] == "poor", f"{backend_name}: {v}"
    assert v["score"] == 0.0


@pytest.mark.parametrize(("backend_name", "model"), BACKENDS)
def test_sanity_judge_live(backend_name: str, model: str) -> None:
    _skip_if_no_backends()
    from shadow.judge import SanityJudge
    from shadow.llm import get_backend

    backend = get_backend(backend_name)
    judge = SanityJudge(backend, model=model)
    v = asyncio.run(
        judge.score_pair(
            _resp("The answer is 4."),
            _resp("I'm not really sure, maybe 7?"),
            _request("What is 2 + 2?"),
        )
    )
    # SanityJudge returns 0.0 ("worse") when candidate regressed.
    assert v["verdict"] == "worse", f"{backend_name}: {v}"
    assert v["score"] == 0.0


@pytest.mark.parametrize(("backend_name", "model"), BACKENDS)
def test_pairwise_judge_live(backend_name: str, model: str) -> None:
    _skip_if_no_backends()
    from shadow.judge import PairwiseJudge
    from shadow.llm import get_backend

    backend = get_backend(backend_name)
    judge = PairwiseJudge(backend, model=model)
    v = asyncio.run(
        judge.score_pair(
            _resp("The answer is 4."),
            _resp("I'm not really sure, maybe 7?"),
            _request("What is 2 + 2?"),
        )
    )
    # Pairwise judge doubles calls (A/B and B/A orderings). Agreement
    # on "candidate worse" yields score 0.0.
    assert v["score"] <= 0.5, f"{backend_name}: expected <= 0.5, got {v}"


# ---- meta: ensure SHADOW_RUN_NETWORK_TESTS gate is enforced -------------


def test_gate_enforced_without_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """If this file runs at all, the module-level pytestmark already
    required SHADOW_RUN_NETWORK_TESTS=1 — verify by checking the var
    at call-time. Documents the gate for future readers."""
    assert os.environ.get("SHADOW_RUN_NETWORK_TESTS") == "1"
