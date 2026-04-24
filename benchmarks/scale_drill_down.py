"""Scale benchmark: drill-down + full 9-axis diff at N=1000 paired responses.

All Shadow validations to date run on traces of 3–10 paired responses.
This harness confirms the differ, drill-down ranking, and serde bridge
hold up on production-sized traces (~1000 pairs is a plausible ceiling
for a week's worth of agent interactions in a mid-traffic product).

Asserts both correctness (drill-down length, sort invariant) and
performance (wall-time per pair count). Fails if any 1000-pair run
takes longer than 60 seconds end to end — a conservative cap that
catches accidental O(N^2) regressions.

Uses synthesized `chat_response` records only because the point is
load-testing the diff pipeline itself; real trace bytes would just be
expensive padding.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from typing import Any

from shadow import _core

# Trace-size ladder. The top end exercises production-scale behaviour
# without making the benchmark unpleasant to run locally.
SIZES = [100, 500, 1000]

# Conservative wall-time cap. Real numbers on a mid-spec laptop are
# much lower (see output); this threshold exists to catch accidental
# algorithmic regressions.
MAX_WALL_SECONDS_PER_1000 = 60.0


def _resp(i: int, latency_ms: int, out_tokens: int, text: str) -> dict[str, Any]:
    return {
        "version": "0.1",
        "id": f"sha256:{hashlib.sha256(f'r{i}:{text}'.encode()).hexdigest()}",
        "kind": "chat_response",
        "ts": f"2026-04-23T10:{(i // 60) % 60:02d}:{i % 60:02d}.000Z",
        "parent": f"sha256:{hashlib.sha256(f'req{i}'.encode()).hexdigest()}",
        "payload": {
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "latency_ms": latency_ms,
            "usage": {
                "input_tokens": 50,
                "output_tokens": out_tokens,
                "thinking_tokens": 0,
            },
        },
    }


def _req(i: int) -> dict[str, Any]:
    return {
        "version": "0.1",
        "id": f"sha256:{hashlib.sha256(f'req{i}'.encode()).hexdigest()}",
        "kind": "chat_request",
        "ts": f"2026-04-23T10:{(i // 60) % 60:02d}:{i % 60:02d}.000Z",
        "parent": None,
        "payload": {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": f"task {i}"}],
            "params": {"temperature": 0.2, "max_tokens": 512},
            "tools": [],
        },
    }


def _metadata(label: str) -> dict[str, Any]:
    payload = {"sdk": {"name": "shadow", "version": "bench"}, "label": label}
    return {
        "version": "0.1",
        "id": f"sha256:{hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}",
        "kind": "metadata",
        "ts": "2026-04-23T10:00:00.000Z",
        "parent": None,
        "payload": payload,
    }


def _synth_trace(n: int, baseline: bool) -> list[dict[str, Any]]:
    """Build a synthetic n-turn trace.

    Baseline emits short structured responses; candidate emits longer
    prose (deterministically divergent) so drill-down has real signal
    to rank across pairs. Every 5th pair is made intentionally more
    regressive so the top of the drill-down list is non-trivial.
    """
    out = [_metadata("baseline" if baseline else "candidate")]
    for i in range(n):
        out.append(_req(i))
        if baseline:
            out.append(_resp(i, 100 + (i % 50), 20 + (i % 10), f'{{"ok": {i}}}'))
        else:
            # Candidate: longer prose, higher latency. Every 5th pair
            # is intentionally ~5x more regressive so the top of
            # drill-down is deterministically identifiable.
            extreme = i % 5 == 0
            latency = 500 + (i % 100) + (2500 if extreme else 0)
            tokens = 80 + (i % 40) + (300 if extreme else 0)
            prose = "I'd be happy to help with that — let me look into it for you. " * (
                6 if extreme else 2
            )
            out.append(_resp(i, latency, tokens, prose))
    return out


def _ok(label: str, ok: bool) -> None:
    print(f"  {'✓' if ok else '✗'} {label}")
    if not ok:
        raise AssertionError(label)


def main() -> int:
    print("Scale benchmark: drill-down + full 9-axis diff on synthetic traces")
    print(f"  sizes: {SIZES}")
    print(f"  wall-time cap (N=1000): {MAX_WALL_SECONDS_PER_1000}s\n")

    for n in SIZES:
        baseline = _synth_trace(n, baseline=True)
        candidate = _synth_trace(n, baseline=False)

        t0 = time.perf_counter()
        report = _core.compute_diff_report(baseline, candidate, None, 42)
        dt = time.perf_counter() - t0

        drill = report.get("drill_down", [])
        scores = [row["regression_score"] for row in drill]

        ms_per_pair = (dt / n) * 1000.0 if n else 0.0
        print(
            f"N={n:5d}  wall={dt:6.2f}s  ms/pair={ms_per_pair:6.2f}  drill_rows={len(drill)}"
        )

        # Correctness invariants — these must hold at every scale.
        _ok(f"N={n}: drill_down returned exactly 5 rows (DEFAULT_K)", len(drill) == 5)
        _ok(
            f"N={n}: drill_down sorted by regression_score desc",
            scores == sorted(scores, reverse=True),
        )
        _ok(
            f"N={n}: top pair index is a multiple of 5 (our extreme-pair schedule)",
            drill[0]["pair_index"] % 5 == 0,
        )
        _ok(
            f"N={n}: every drill row has 8 axis_scores (Judge excluded)",
            all(len(r["axis_scores"]) == 8 for r in drill),
        )
        _ok(f"N={n}: pair_count agrees with input", report["pair_count"] == n)

        # Performance guard-rail (scaled to N).
        cap_at_n = MAX_WALL_SECONDS_PER_1000 * (n / 1000.0)
        _ok(
            f"N={n}: wall-time under cap ({dt:.2f}s < {cap_at_n:.2f}s)",
            dt < cap_at_n,
        )
        print()

    print("✅ Drill-down scales correctly and within perf budget up to N=1000.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
