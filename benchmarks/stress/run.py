"""Production-scale stress test — 10k chat pairs through the differ.

Goal: prove Shadow's Rust core scales to real trace volumes. Generates
10,000 synthetic baseline/candidate response pairs with realistic
variety (random latency, random output lengths, occasional refusals,
occasional tool calls), writes them to .agentlog files, and runs
`_core.compute_diff_report` end-to-end.

Pass criteria:
  - diff runtime < 30 seconds on a modern laptop
  - peak memory < 500 MB
  - produces a valid 9-axis DiffReport

Anything more than that means we broke the scale promise.
"""

from __future__ import annotations

import json
import random
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "src"))

from shadow import _core

N_PAIRS = 10_000
MAX_RUNTIME_SECONDS = 30.0
MAX_MEMORY_MB = 500.0

OUT = Path(__file__).parent / ".out"


def build_trace(n: int, *, is_candidate: bool, seed: int) -> list[dict]:
    """Generate a synthetic trace of `n` chat pairs with realistic variety."""
    rng = random.Random(seed)
    records: list[dict] = [
        {
            "version": "0.1",
            "id": f"sha256:{'0' if not is_candidate else '1'}{seed:063x}"[:71],
            "kind": "metadata",
            "ts": "2026-04-23T00:00:00.000Z",
            "parent": None,
            "payload": {"sdk": {"name": "shadow"}},
        }
    ]
    for i in range(n):
        # Request.
        req_id = f"sha256:r{i:04d}{'c' if is_candidate else 'b'}{'0' * 57}"[:71]
        records.append(
            {
                "version": "0.1",
                "id": req_id,
                "kind": "chat_request",
                "ts": f"2026-04-23T00:{i // 60 % 60:02d}:{i % 60:02d}.000Z",
                "parent": None,
                "payload": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": f"task {i}"}],
                    "params": {"temperature": 0.2, "max_tokens": 200},
                },
            }
        )
        # Response with realistic variety.
        latency = rng.randint(100, 2000)
        if is_candidate:
            latency = int(latency * 1.2)  # candidate is slower on average
        output_tokens = rng.randint(10, 80)
        if is_candidate:
            output_tokens = int(output_tokens * 1.15)  # slightly more verbose
        # Occasional tool call
        use_tool = rng.random() < 0.15
        content: list[dict]
        if use_tool:
            content = [
                {
                    "type": "tool_use",
                    "id": f"tc_{i}",
                    "name": ("lookup_order" if is_candidate else "check_order_status"),
                    "input": {"order_id": f"A-{i}"},
                }
            ]
            stop = "tool_use"
        else:
            text = "ok " * rng.randint(3, 15)
            content = [{"type": "text", "text": text}]
            stop = "end_turn"
        # Occasional refusal.
        if rng.random() < 0.02:
            stop = "content_filter"
            content = [{"type": "text", "text": "I can't help with that."}]
        resp_id = f"sha256:s{i:04d}{'c' if is_candidate else 'b'}{'0' * 57}"[:71]
        records.append(
            {
                "version": "0.1",
                "id": resp_id,
                "kind": "chat_response",
                "ts": f"2026-04-23T00:{i // 60 % 60:02d}:{i % 60:02d}.500Z",
                "parent": req_id,
                "payload": {
                    "model": "gpt-4o-mini",
                    "content": content,
                    "stop_reason": stop,
                    "latency_ms": latency,
                    "usage": {
                        "input_tokens": rng.randint(20, 200),
                        "output_tokens": output_tokens,
                        "thinking_tokens": 0,
                    },
                },
            }
        )
    return records


def rss_mb() -> float:
    """Current RSS in MB. macOS reports in bytes, Linux in KB."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / 1024 / 1024
    return ru / 1024


def main() -> int:
    OUT.mkdir(exist_ok=True)
    print(f"Generating {N_PAIRS} baseline + {N_PAIRS} candidate pairs ...")
    t0 = time.perf_counter()
    baseline = build_trace(N_PAIRS, is_candidate=False, seed=1)
    candidate = build_trace(N_PAIRS, is_candidate=True, seed=2)
    gen_s = time.perf_counter() - t0
    print(f"  generated in {gen_s:.2f}s")

    baseline_path = OUT / "baseline.agentlog"
    candidate_path = OUT / "candidate.agentlog"

    t0 = time.perf_counter()
    baseline_path.write_bytes(_core.write_agentlog(baseline))
    candidate_path.write_bytes(_core.write_agentlog(candidate))
    write_s = time.perf_counter() - t0
    print(
        f"  wrote {baseline_path.stat().st_size / 1024:.1f} KB + "
        f"{candidate_path.stat().st_size / 1024:.1f} KB in {write_s:.2f}s"
    )

    t0 = time.perf_counter()
    parsed_b = _core.parse_agentlog(baseline_path.read_bytes())
    parsed_c = _core.parse_agentlog(candidate_path.read_bytes())
    parse_s = time.perf_counter() - t0
    print(
        f"  parsed both back in {parse_s:.2f}s ({len(parsed_b)} + {len(parsed_c)} records)"
    )

    t0 = time.perf_counter()
    report = _core.compute_diff_report(parsed_b, parsed_c, None, 42)
    diff_s = time.perf_counter() - t0
    print(f"  diffed in {diff_s:.2f}s")

    mem = rss_mb()
    print(f"\nPeak RSS: {mem:.1f} MB")
    print(f"Total runtime (diff only): {diff_s:.2f}s vs {MAX_RUNTIME_SECONDS}s budget")

    axes = {r["axis"]: r for r in report["rows"]}
    assert len(axes) == 9, f"expected 9 axes, got {len(axes)}"
    latency = axes["latency"]
    assert latency["n"] >= N_PAIRS * 0.9, f"latency axis missing data: n={latency['n']}"

    (OUT / "report.json").write_text(json.dumps(report, indent=2))

    ok = True
    if diff_s > MAX_RUNTIME_SECONDS:
        print(f"FAIL: diff took {diff_s:.2f}s > budget {MAX_RUNTIME_SECONDS}s")
        ok = False
    if mem > MAX_MEMORY_MB:
        print(f"FAIL: RSS {mem:.1f} MB > budget {MAX_MEMORY_MB} MB")
        ok = False

    if ok:
        print("PASS: Shadow scales to 10k-pair production volumes")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
