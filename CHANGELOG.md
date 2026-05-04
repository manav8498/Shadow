# Changelog

All notable changes to Shadow are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Conventional Commits](https://www.conventionalcommits.org/).

## [unreleased] — Causal Regression Forensics + OTel bridge

Adds Phase 5 of the strategic-pivot roadmap: any OTel-GenAI-instrumented
trace can now be imported into Shadow and run through `diagnose-pr`.

### Added

* `shadow import --format otel-genai` and `shadow export --format otel-genai`
  alias the existing `--format otel` flags (per design spec §5 wording).
* `docs/features/otel-bridge.md` — user-facing OTel bridge documentation.

### Fixed (Phase 5 audit findings)

* OTel exporter now emits `gen_ai.user.message`, `gen_ai.system.message`,
  `gen_ai.assistant.message`, and `gen_ai.tool.message` events on chat
  spans. Previously, only attributes (model, usage, finish_reasons) were
  emitted, so the importer recovered empty messages + empty content
  blocks on round-trip — losing the signal the 9-axis differ needed.
* OTel importer now stamps the OTel `traceId` into envelope
  `meta.trace_id` so multiple OTel-imported traces with byte-identical
  metadata payloads stay distinct in the diagnose-pr mining filter.
  (Same class of fix as v3.0.5's content-hash collision; the import path
  needed it too.)
* OTel importer now prefers the `shadow.latency_ms` attribute over span
  duration for `latency_ms` in chat_response. Span duration is the
  fallback for third-party OTel data without the explicit attribute.

### Verified

* Round-trip preserves per-pair 9-axis diff outcome on the refund demo
  (per-axis severities + first_divergence identical native vs.
  roundtripped).
* `diagnose-pr` against an OTel-roundtripped corpus produces the same
  verdict + affected count + dominant cause as the native run
  (`STOP, 3/3 affected, prompt.system`).

---

## [unreleased] — Causal Regression Forensics

Strategic-pivot work landed across four weeks of plan-driven development.
The wedge product, `shadow diagnose-pr`, plus `verify-fix` and `gate-pr`,
ship the three-command loop:

  diagnose -> fix -> verify

### Added

* `shadow diagnose-pr` (new command) — names the exact change that broke
  the agent, with bootstrap CI + E-value when a backend is supplied.
  Produces a `diagnose-pr/v0.1` JSON report and a Markdown PR comment.
* `shadow verify-fix` (new command) — reads the diagnose report, re-diffs
  affected traces against a fixed candidate, asserts the regression is
  reversed without collateral damage. Pass criteria configurable via
  `--affected-threshold` / `--safe-ceiling`.
* `shadow gate-pr` (new command) — CI-friendly wrapper, verdict-mapped
  exit codes (0 ship / 1 hold|probe / 2 stop / 3 internal error).
* `--backend live` — real OpenAI replay anchored per baseline trace
  (corpus-mean divergence). `--max-cost USD` caps total spend; pricing
  table covers gpt-4o-mini / gpt-4o / gpt-4.1-mini / gpt-4.1 with a
  conservative-high fallback for unknown models.
* `--backend mock` — synthetic deterministic per-delta intervention
  for tests + offline demos. PR comment surfaces a "synthetic mock
  backend" disclosure so reviewers can't mistake it for real evidence.
* `examples/refund-causal-diagnosis/` — packaged wedge demo with
  baseline + candidate traces, policy, configs, and a `demo.sh`.
* `.github/actions/shadow-diagnose-pr/` — composite GitHub Action
  posting the diagnose-pr PR comment via dedup-by-marker.
* `docs/features/causal-pr-diagnosis.md` — feature page covering the
  flow, verdict matrix, three backends, performance, CI integration.
* README hero rewritten to lead with `shadow diagnose-pr`.

### Changed

* `shadow.mine` and `shadow.diagnose_pr.loaders` both prefer the
  envelope `meta.trace_id` (the per-trace UUID added in v3.0.5) over
  the metadata record's content hash, ending the silent collapse-to-
  one-cluster behavior on byte-identical metadata.
* Per-trace 9-axis diff + policy check now run in a thread pool
  above 16-pair corpora (Rust differ + regex policy both release
  the GIL).

### Internal

* Refactored: `shadow.diagnose_pr.runner.run_diagnose_pr(opts)` is the
  pure-Python entry point. The Typer commands are thin wrappers.
* Tests: 1795 passed, 15 expected skips. Live API e2e test
  (`test_diagnose_pr_live_api_e2e.py`) gated by
  `SHADOW_RUN_NETWORK_TESTS=1` + `OPENAI_API_KEY`. Snapshot test on
  the demo's PR comment (`test_diagnose_pr_demo_snapshot.py`).
* Performance (M-series, single Python 3.11): 1000 paired pairs +
  bootstrap + causal in 0.95s @ 85 MB; 5000 single-side traces with
  mining in 0.54s.

---

## [3.0.7](https://github.com/manav8498/Shadow/compare/v3.0.6...v3.0.7) (2026-05-03)

External customer retest of v3.0.6 surfaced three follow-up items.
This release fixes them; no functional changes outside the affected
extras.

### Fixed

* **`shadow-diff[sign]` works against sigstore 4.x.** `SigningContext.
  production()` and `SigningContext.staging()` were removed in sigstore
  4.0 in favour of going through `ClientTrustConfig` and
  `SigningContext.from_trust_config(...)`. Shadow's signing code still
  called the removed factories, so a fresh install of the `sign` extra
  picked up sigstore `4.2.0` and broke at runtime. `shadow.certify_sign`
  now detects the 4.x API at runtime via
  `hasattr(SigningContext, "from_trust_config")` and routes through
  `ClientTrustConfig.{production,staging}()` on 4.x while keeping the
  3.x path. The corrupt-bundle parse path also now catches
  `sigstore.errors.Error` (4.x raises a typed `InvalidBundle` here
  where 3.x raised plain `ValueError`). The `[sign]` extra continues
  to range over both majors via `sigstore>=3.0,<5`; tests verified
  green on both `3.6.7` and `4.2.0`.
* **`shadow-diff[langgraph]` resolves cleanly again.** The previous
  upper bound `langchain-core>=0.3,<1` excluded the 1.x line that
  `langgraph 1.x` and `langchain-openai 1.x` already require, so a
  fresh `pip install shadow-diff[langgraph]` failed dependency
  resolution. Lifted to `langchain-core>=0.3,<3` to track 1.x and an
  eventual 2.x without thrashing. Resolution now succeeds with
  `langchain-core 1.3.x`, `langgraph 1.1.x`, `langchain-openai 1.2.x`.

### Security

* **dev: `pytest==8.4.2` → `pytest==9.0.3`** — closes
  CVE-2025-71176 (the only fixed line is 9.0.3+). The plugin chain
  bumped alongside it: `pytest-asyncio==0.25.0 → 1.3.0` and
  `pytest-cov==6.0.0 → 7.1.0` so the dev environment stays internally
  consistent (both newer plugins advertise `pytest<10`). Suite ran
  green: 1701 passed, 14 expected network/live skips.

### Note

These three items are dependency-edge issues — Shadow's core code,
default install path, and CLI surface are unchanged from v3.0.6. The
fix is scoped to `python/src/shadow/certify_sign.py` (sigstore version
detection), `python/tests/test_certify_sign.py` (mirrored test patch),
and `python/pyproject.toml` (dev + langgraph-extra dependency ranges).

---

## [3.0.6](https://github.com/manav8498/Shadow/compare/v3.0.5...v3.0.6) (2026-05-02)

External customer audit found six security-strict-deployment items.
This release closes the cargo-audit advisories and refreshes the
optional / dev Python dependency ranges; no functional changes.

### Security

* **pyo3 0.22.6 → 0.24.2** — closes RUSTSEC-2025-0020
  (`PyString::from_object` use-after-free). Shadow's own code didn't
  call the affected API but the dependency still made `cargo audit`
  fail. Source migration was minor: `PyList::empty_bound` /
  `PyBytes::new_bound` / `PyList::new_bound` were renamed to
  `empty` / `new` and `PyList::new` is now fallible (allocation can
  in principle fail). One call site needed an explicit error path.
* **tokio 1.41.1 → 1.52.1** — closes RUSTSEC-2025-0023 (broadcast
  channel unsoundness warning). Shadow doesn't use `tokio::sync::
  broadcast` but the dependency was flagged anyway; the bump keeps
  scanners quiet.
* **`@anthropic-ai/sdk`** — already at `^0.92.0` from v3.0.5; flagged
  here for completeness.
* **Optional / dev Python dep ranges refreshed.** The customer audit
  saw vulnerable versions resolve in fresh installs of some optional
  extras and the dev environment because the lower bounds were too
  loose:
    - `sigstore>=3.0,<4` → `sigstore>=3.0,<5` (allows the patched
      4.x line)
    - `Pillow>=10,<12` → `Pillow>=10,<13` (allows 12.x with the
      patched chain)
    - `langchain-openai>=0.2,<2` → `langchain-openai>=0.3,<2`
      (lifts the lower bound past the vulnerable 0.2.x line)
    - `langgraph>=1,<2` → `langgraph>=1.0.2,<2` (lifts past the
      flagged 1.0.0 / 1.0.1 line)
    - dev: `pytest==8.3.4` → `pytest==8.4.2` (closes the pytest-
      8.3.x advisories; stays on 8.x to avoid plugin-compat
      breakage; 9.x bump can land in a separate dev refresh)

### Build

* `statrs 0.17.1 → 0.18.0` (lockfile alignment, no API changes
  affecting Shadow).
* `cargo audit` now reports clean except for one expected warning:
  `paste 1.0.15` is unmaintained, transitive through
  `statrs → nalgebra → simba`. The warning is well-known and
  upstream; we can't drop the path without losing Hotelling T²
  multivariate support.

### Note

The v3.0.5 CHANGELOG claimed the literal `sha256:aeaa25c8...` hash
the customer reproduced is in the regression test. That overstated
what the test actually does: the test reproduces the *collision
condition* (two byte-identical metadata payloads producing the same
content id, with envelope `meta.trace_id` distinguishing them) and
asserts the report uses the envelope value. It does not hard-code
the specific aeaa25c8 hash — that hash depends on environment and
isn't deterministic across runs. Correcting the record here.

## [3.0.5](https://github.com/manav8498/Shadow/compare/v3.0.4...v3.0.5) (2026-05-02)

External customer audit reported four production issues. All four
reproduced locally; all four fixed; two regression tests added.

### Fixed

* **Trace IDs collide across runs (HIGH).** Two `Session()` calls
  without distinguishing tags produced the same `baseline_trace_id`
  and `candidate_trace_id` in the diff report, even though each
  Session minted a unique 128-bit hex `trace_id` internally. Cause:
  the diff report populated trace-id fields from the first record's
  content hash, and the metadata payload `{"sdk": {"name": "shadow",
  ...}}` is byte-identical across tagless runs. Meanwhile the actual
  unique identifier the SDK mints lives in envelope `meta.trace_id`,
  which is intentionally not part of the content hash (SPEC §6).

  Fix: a new `trace_id_for(records)` helper in
  `crates/shadow-core/src/diff/mod.rs` prefers envelope
  `meta.trace_id` when present, falls back to the first record's
  content id otherwise. Backward compatible — third-party imports
  and hand-constructed fixtures without `meta.trace_id` keep their
  pre-fix behaviour. ([56be371](https://github.com/manav8498/Shadow/commit/56be371))

* **Generated GitHub Action installs old Shadow major.** `shadow init
  --github-action` scaffolded a workflow with `pip install --upgrade
  "shadow-diff>=2.4,<3"` — pinned two majors behind the running
  Shadow. New users got CI testing against v2.x while developing
  locally against v3. The scaffold now substitutes the major from
  `shadow.__version__` at write time, so the constraint always
  tracks whichever Shadow generated the workflow.

* **`@anthropic-ai/sdk` advisory GHSA-p7fg-763f-g4gf.** The TS
  package's peerDependency `@anthropic-ai/sdk: ^0.90.0` pinned a
  vulnerable range (insecure default file permissions in the local
  filesystem memory tool). Bumped to `^0.92.0`; `npm audit
  --omit=dev` now reports 0 vulnerabilities.

### Build

* `Cargo.lock` was stale at 3.0.1 while `Cargo.toml` was at 3.0.4;
  `cargo metadata --locked` errored. Lockfile regenerated; reproducible
  `--locked` builds restored.

## [3.0.4](https://github.com/manav8498/Shadow/compare/v3.0.3...v3.0.4) (2026-04-30)

### Fixed

* **`shadow mine` now accepts a directory path.** Previously,
  `shadow mine /path/to/agentlogs` (a directory of trace files —
  the natural shape of a production trace dump) leaked
  `[Errno 21] Is a directory: ...`. The command now recursively
  walks the directory and mines every `*.agentlog` file inside.
  An empty directory or one containing no `*.agentlog` files
  surfaces a clear error with a remediation hint instead of a
  Python traceback. Same UX polish as the v3.0.0 fix to
  `shadow holdout --base <directory>`. Two regression tests in
  `python/tests/test_mine.py` cover the directory-walk and
  empty-directory paths.

## [3.0.3](https://github.com/manav8498/Shadow/compare/v3.0.2...v3.0.3) (2026-04-30)

### Fixed

* **crates.io publish.** v3.0.2's crates.io step failed with
  `400 Bad Request: A verified email address is required to publish
  crates to crates.io` — a one-time account-level prerequisite the
  maintainer hadn't completed. Email is now verified; this release
  triggers the publish chain again to land `shadow-diff` on
  crates.io and align all three registries on the same version.
  No code or behaviour changes from 3.0.2.

## [3.0.2](https://github.com/manav8498/Shadow/compare/v3.0.1...v3.0.2) (2026-04-30)


### Fixed

* align package names across PyPI / npm / crates.io to shadow-diff ([40f3ace](https://github.com/manav8498/Shadow/commit/40f3aceaae13aed48d2a686a30714de7368271f8))


### Documentation

* add RELEASE.md with maintainer release / troubleshooting guide ([d82265d](https://github.com/manav8498/Shadow/commit/d82265dfdda3e0ba21c870a1dedce81d210bd1b0))


### Chores

* trigger 3.0.2 to publish to all three registries ([6646e6f](https://github.com/manav8498/Shadow/commit/6646e6fc845577bc48c0e3a08d018ac6ca2e81dc))

## [3.0.1](https://github.com/manav8498/Shadow/compare/v3.0.0...v3.0.1) (2026-04-30)

### Fixed

* **PyPI publish.** v3.0.0's wheel METADATA declared
  `License-File: LICENSE-APACHE` (auto-discovered by maturin from
  `python/LICENSE-APACHE`) but the wheel itself didn't include the
  file because the explicit `[tool.maturin] include` rule pointed
  at a non-existent `LICENSE` path. PyPI's PEP 639 validator (now
  strict) rejected the upload with `400 License-File LICENSE-APACHE
  does not exist in distribution`. Crates.io and npm published
  3.0.0 fine; PyPI never accepted it.

  The fix:
    - Added `license-files = ["LICENSE-APACHE"]` to `[project]` in
      `python/pyproject.toml` so maturin both packages the file
      AND declares it in METADATA, with no auto-detection drift.
    - Removed the stale `python/LICENSE-MIT` (project consolidated
      to Apache-2.0 only at v2.6.0; the MIT file lingered).
    - Removed the broken `[tool.maturin] include` entry that
      referenced a non-existent `LICENSE` file.

  No code or behaviour changes — purely a packaging metadata fix
  to make the v3.0.x line publishable on PyPI.

## [3.0.0](https://github.com/manav8498/Shadow/compare/v2.9.0...v3.0.0) (2026-04-30)

Workflow-loop release. Ships the full PR-to-runtime arc — `shadow demo` /
`call` / `autopr` / `ledger` / `trail` / `brief` / `listen` / `holdout` /
`heal` — alongside a switch to release-please for synchronised version
bumps, a corrected back-door causal estimator, and a complete
presentation overhaul of the diff output so non-expert reviewers can
read a Shadow PR comment top-to-bottom and act on it without consulting
the spec. The major-version bump is driven by the stricter
`causal_attribution` API; everything else is additive.

### ⚠ BREAKING CHANGES

* **`causal_attribution(confounders=[...])` requires `confounder_weights`.**
  Declaring confounders without supplying weights now raises `ValueError`.
  The previous silent fallback to uniform 1/n weighting biased the
  back-door estimate toward the simple average for any non-uniform
  P(C=c) — a quiet correctness gap that read as "the Pearl ATE" to
  callers who never supplied weights. Two opt-ins replace the silent
  default: `confounder_weights="uniform"` (sentinel acknowledging the
  uniform-P(C=c) assumption) or `confounder_weights={(combo,): w, ...}`
  (empirical weights). The bootstrap CI now honours the same weights
  as the point estimate; previously it always used 1/n regardless.
  ([38ea15c](https://github.com/manav8498/Shadow/commit/38ea15cdc0d17ca15593440a3b246be503b17aed))

### Added — workflow-loop CLI commands

* **`shadow demo`** — one-line trial against bundled fixtures. No
  arguments, no setup, no API key. ([cea859c](https://github.com/manav8498/Shadow/commit/cea859cf03bc0a6d3a86d2bbe33f9bbcd5bb44eb))
* **`shadow autopr`** — synthesise a policy YAML from a regression so
  the same diff pattern fails CI on the next PR. ([58693f9](https://github.com/manav8498/Shadow/commit/58693f9eb88b1b2027eda63e6e3023fec00b7651))
* **`shadow call`** — one-line ship / hold / probe / stop verdict on a
  baseline + candidate pair. ([201e32a](https://github.com/manav8498/Shadow/commit/201e32a7f47a85aef739ef27e1fbf99ac5cb2bcd))
* **`shadow ledger`** — daily glance over recent artifacts, plus the
  underlying opt-in `shadow.ledger` artifact record store.
  ([d59241c](https://github.com/manav8498/Shadow/commit/d59241cf26f1afaa7f8acaad5aabbeb3d5c9e7ef),
  [be310f6](https://github.com/manav8498/Shadow/commit/be310f6f9d5cca0ec9a93ac0d1a39dc1f9c3a09b))
* **`shadow trail`** — walk back through the artifact graph from any
  trace id. ([9913732](https://github.com/manav8498/Shadow/commit/9913732d83100e7e7c4c5db40e6c51b51c87f90b))
* **`shadow brief`** — share recent state in three formats (terminal /
  markdown / Slack). ([8339bca](https://github.com/manav8498/Shadow/commit/8339bca830db6d3a7a8a1cae6f4a66c5be7b7d57))
* **`shadow listen`** — stream calls as new `.agentlog` files land.
  ([2e47d7e](https://github.com/manav8498/Shadow/commit/2e47d7eed7eb066a72c81dec7e93f8ff3ca8b8be))
* **`shadow holdout`** — manage held-out trace ids with TTL + owner.
  ([320db0e](https://github.com/manav8498/Shadow/commit/320db0e3c2bbd4ead72cf2e2bbf1d44dd2ffabe7))
* **`shadow heal`** — causal classifier (audit-only, no actions yet).
  ([b4aa4c4](https://github.com/manav8498/Shadow/commit/b4aa4c44e90d1a14a89ee13deddca3ed1c4cb27c))

### Added — readability and ease-of-use

* **Plain-English diff output.** All three renderers (PR comment,
  markdown, terminal) now use display labels for the nine axes
  ("response meaning" / "tool calls" / "refusals" / "response length" /
  "response time" / "token cost" / "reasoning depth" / "LLM-judge
  score" / "output format") instead of the internal axis names
  ("semantic" / "trajectory" / "safety" / …). Internal names stay
  unchanged in the `.agentlog` format and the JSON `rows[].axis`
  field — only the rendered presentation switches.
  ([23d9886](https://github.com/manav8498/Shadow/commit/23d98868f8ec9983911e94c8335c8b2d334f0413))
* **PR comment ordering inverted.** Reviewers now read, top to bottom:
  a one-sentence verdict line ("Shadow recommends: hold this PR for
  review"), then "What probably broke" with the engine's plain-English
  recommendations as the headline, then "What changed at the turn
  level" as prose, then the nine-axis numbers in a `<details>` fold.
  Previously the math came first and the recommendations were buried.
* **Universal `low_power` flag suppressed** when it applies to every
  row — fires once in the banner above the table instead of cluttering
  every line.
* **`shadow --help` tiered into five panels** — Setup / Common /
  Replay & analysis / Reporting & history / Release & integrations.
  Same 27 commands, no longer a flat wall.
  ([a4dce9f](https://github.com/manav8498/Shadow/commit/a4dce9fdb3c69d128e50c363f0f8c013f12c9204))
* **README rewritten with a "first PR comment in 10 minutes"
  walkthrough.** Concrete four-step path from `pip install` to seeing
  a Shadow comment land on a real PR. Optional-extras table collapsed
  into a `<details>` fold so the install section stays scannable.

### Added — launch media + miscellaneous

* **Launch video** — 1080p, 84 s, six-beat product walkthrough with
  intro / outro cards and ambient music bed. Built by
  `scripts/build_launch_video.py`. ([19e1442](https://github.com/manav8498/Shadow/commit/19e1442d55256917a08ba71325b1430a4ab95d5d))
* **Workflow demo GIF + MP4 + WebM** — silent walkthrough alternative
  for embedded README playback. ([a2af843](https://github.com/manav8498/Shadow/commit/a2af843), [d34842e](https://github.com/manav8498/Shadow/commit/d34842e))
* `Recommendation.{baseline_turn, candidate_turn}` fields. ([902aefb](https://github.com/manav8498/Shadow/commit/902aefb))

### Fixed

* `shadow holdout add --base <existing-dir>` no longer leaks a Python
  traceback; raises typed `HoldoutPathError` instead. ([c960678](https://github.com/manav8498/Shadow/commit/c960678))
* Flaky holdout test on narrow CI terminals (Rich line-wrapping
  artifact). ([5166520](https://github.com/manav8498/Shadow/commit/5166520))

### Documentation

* Related sections in new command help + README CLI table. ([6f82520](https://github.com/manav8498/Shadow/commit/6f82520))
* Launch video links in README (inline play + MP4 / WebM). ([3aba1b4](https://github.com/manav8498/Shadow/commit/3aba1b4))
* License / causal / telemetry text tightened. ([ae5b81a](https://github.com/manav8498/Shadow/commit/ae5b81a83624025196b75e9fd0278b25ccc67d27),
  [f5f94ec](https://github.com/manav8498/Shadow/commit/f5f94ec0bd9bedf57d36a8a803f0241b79829595))

### Reverted

* Embedded `<video>` player in README — GitHub's markdown sanitizer
  strips `<video>` tags from non-user-attachment URLs. Reverted to GIF
  inline + MP4 / WebM links. ([4c89c25](https://github.com/manav8498/Shadow/commit/4c89c25))

### Build

* Adopted [release-please](https://github.com/googleapis/release-please)
  for synchronised version bumps across `Cargo.toml`,
  `python/pyproject.toml`, `python/src/shadow/__init__.py`,
  `typescript/package.json`, `typescript/package-lock.json`, and the
  README badge. The `release.yml` publish workflow now also fires on
  `release: published` so the chain completes when release-please
  creates the GitHub Release. Switched the README version badge from
  `img.shields.io/badge/...` to `img.shields.io/static/v1?...` so the
  release-please semver regex doesn't greedy-match the trailing
  colour suffix as a prerelease tag. ([8d85ec3](https://github.com/manav8498/Shadow/commit/8d85ec3))

## [Unreleased]

## [2.9.0] - 2026-04-28

Closes the last two A+ holdouts identified in the post-v2.8 third-
party review: causal-replay semantic divergence is now embedder-
aware (matches the Rust nine-axis cosine when an embedder is
provided), and the deterministic 10-pattern recommendations engine
gains an opt-in LLM-backed novel-pattern fallback for signatures
the curated rules don't catch. Both close real fidelity gaps with
clean opt-in paths and zero impact on backward compatibility.

### Added

- **`OpenAIReplayer(embedder=...)`** — when supplied, semantic
  divergence is computed as ``1 - cosine(embed(baseline), embed(candidate))``
  rather than the v2.8 Jaccard fallback. Cross-validated to match
  the Rust ``compute_semantic_axis_with_embedder`` output within
  1e-6 on identical inputs. Failure modes (misshapen output, dim
  mismatch, exception) all map to divergence = 1.0 (fail-loud) so
  the embedder bug surfaces as a regression rather than as silent
  zero-divergence.

  10 dedicated tests in
  `python/tests/test_causal_replayer_embedder.py` including a
  cross-validation against the Rust nine-axis path on the same
  text pair + embedder.

- **`shadow.diff_py.recommendations.enrich_with_llm`** — LLM-backed
  novel-pattern diagnosis fallback. Calls the deterministic Rust
  engine first; only invokes an LLM when:
    * No rule-based root-cause is present, AND
    * At least one axis is at severity "severe", AND
    * `OPENAI_API_KEY` is in the env or a custom `llm_caller=` is
      passed.

  Uses OpenAI structured output (JSON Schema) so the response shape
  is enforced server-side; malformed responses are dropped silently
  (returning the deterministic recommendations unchanged) rather
  than propagating noise. LLM-derived rows are tagged with
  `"source": "llm"` for renderer differentiation, and confidence is
  capped at 0.65 to honestly signal "operating beyond curated
  patterns."

  14 dedicated tests in
  `python/tests/test_recommendations_llm.py` covering
  gating logic (no-LLM-when-root-cause-present, no-LLM-when-no-
  severe-axis, no-key-no-fallback), happy path, malformed-output
  validation (5 distinct invalidation cases), confidence cap, and
  dataclass round-trip.

### Notes

- These are additive — no breaking changes vs 2.8.0. SemVer minor.
- Default behavior is unchanged: existing OpenAIReplayer callers
  who didn't pass `embedder=` keep getting the Jaccard path; users
  who don't call `enrich_with_llm` keep getting the deterministic
  recommendations only. Both new paths are opt-in.
- Cost: at most one LLM call per diff report, only when both gating
  conditions fire. Typical: $0.001-0.01 with gpt-4o-mini. Zero cost
  when no severe axis is present or a rule-based root-cause already
  fired.

## [2.8.0] - 2026-04-28

Closes the final five A+ gaps from the post-v2.7.0 review: live OpenAI
replayer for causal attribution, embedding-derived fingerprint
dimensions wired through the Embedder trait, PyO3 callback path for
`compute_with_embedder`, five additional cross-axis recommendation
patterns, and property-based equivalence between the Python and
TypeScript LTLf evaluators. Honest 10/10 on technical execution.

### Added

- **`shadow.causal.replay` subpackage** — production replay backends
  for the causal-attribution pipeline:
  - `OpenAIReplayer` calls the live OpenAI Chat Completions API with
    deterministic seeding (derived from a SHA-256 of the canonical
    config), exponential-backoff retry on rate-limit / transient
    errors, and per-config caching. Reads `OPENAI_API_KEY` from env
    only — never accepts the key as a constructor parameter.
  - `RecordedReplayer` plays back a pre-recorded results table for
    CI / unit tests; same hash function as `OpenAIReplayer` so cache
    files port across replayers.
  - `Replayer` Protocol + `ReplayResult` dataclass for clean typing.
  - 3 live OpenAI integration tests gated on
    `SHADOW_RUN_NETWORK_TESTS=1` + `OPENAI_API_KEY`. Validated end-
    to-end against the real API: real call works, cache hits
    correctly on repeat, causal attribution detects a system_prompt
    delta with non-zero ATE on the semantic axis.
- **`shadow._core.compute_semantic_axis_with_embedder`** — PyO3
  binding that exposes the Rust `Embedder` trait to Python. Accepts
  any callable `list[str] -> list[list[float]]` (e.g.
  `sentence-transformers`, an OpenAI embeddings client, an in-house
  service). The Rust side computes cosine in Rust on the returned
  vectors. Cross-validated against a Python reference within 1e-6
  relative tolerance.
- **`shadow.statistical.fingerprint.fingerprint_trace_extended`** —
  D=14 fingerprint with two embedding-derived dimensions
  (`embedding_norm_log`, `embedding_centroid_dist`) wired through
  any embedder. Base D=12 is byte-identical to the existing
  `fingerprint_trace` output; the embedder only adds two columns.
- **Five additional cross-axis recommendation patterns** in
  `shadow_core::diff::recommendations`:
  - `context-window-overflow` (cost severe + reasoning shifts)
  - `retry-loop` (trajectory severe + latency moved without
    reasoning)
  - `cost-explosion-cached-mismatch` (cost severe with latency AND
    semantic stable — SDK upgrade dropped `cache_control`)
  - `prompt-injection-on-tool-args` (trajectory severe + safety
    delta NEGATIVE — agent refusing less while tools diverge)
  - `latency-spike-without-cost` (latency severe alone — provider
    capacity / network event, not a code regression)
  Each pattern has dedicated tests asserting it fires on matching
  evidence and is suppressed when an upstream pattern subsumes it.
  Plus a property test that no two patterns claim the same single-
  axis evidence.
- **Property-based TS↔Python LTLf parity** — 500 random LTLf
  formulas (depth 1-6, all 10 operators) × random traces (length
  0-12). Both Python and TypeScript evaluators MUST produce byte-
  identical truth-vectors at every position. Plus a 1000-state
  smoke test for performance equivalence.
  - New `typescript/scripts/eval-ltlf.mjs` CLI exposing
    `evalAllPositions` over stdin/stdout JSON for the parity harness.

### Verified

- 1465 Python tests pass (was 1428, +37 across embedder integration,
  fingerprint extended, causal replay offline + live, TS parity).
- 240 Rust tests pass (was 230, +10 cross-axis recommendation
  patterns and the new `RootCause` action variant).
- 72 TypeScript tests pass (unchanged; parity now backed by the
  property-based suite).
- 3 live OpenAI tests passed against the real API (rotated key,
  revoked after the validation run; never appeared in any committed
  file or git history).

## [2.7.0] - 2026-04-28

Closes the four production-grade gaps identified after the v2.6.0
release: causal extension beyond foundation, public discoverability
of the LASSO bootstrap-CI APIs, TypeScript gating parity, and a
pluggable embedder interface for the semantic axis.

### Added

- **`shadow.causal` extensions — production-grade pipeline.**
  `causal_attribution` now accepts `n_bootstrap`, `confounders`, and
  `sensitivity` parameters:
  - `n_bootstrap > 0` produces percentile-bootstrap CIs on the ATE
    (Efron 1979). Resampling is stratum-aware so back-door-adjusted
    estimates carry honest CIs.
  - `confounders=[...]` triggers Pearl's back-door adjustment via
    uniform-weighted stratification over confounder-value combinations
    (Pearl 2009 §3.3). Verified on a synthetic interaction scenario
    where the naive estimator returns 0.4 (one stratum) and the
    adjusted estimator returns 0.5 (mean of {0.4, 0.6}).
  - `sensitivity=True` computes the VanderWeele-Ding (2017) E-value
    for continuous outcomes — the smallest unmeasured-confounder
    effect that could explain away the observed ATE.
  - `CausalAttribution` gains `ci_low`, `ci_high`, and `e_values`
    fields; backward-compat preserved (existing tests using the
    point-ATE-only API pass unchanged).
  - Coverage validated by Monte-Carlo: 30-trial repeat on noisy
    synthetic with known truth achieves >= 80% nominal coverage at
    95%.
- **TypeScript SDK gating modules.** The TS SDK ships an evaluator
  surface so TS-only teams can gate CI without invoking Python:
  - `src/ltl/formula.ts` — LTLf AST as a discriminated union, all 10
    operators, constructor helpers and a deterministic stringifier.
  - `src/ltl/checker.ts` — Bottom-up DP O(|π|×|φ|) checker, byte-for-
    byte mirror of `shadow.ltl.checker`.
  - `src/policy/rules.ts` + `check.ts` — Stateless rule eval for
    `no_call`, `must_call_before`, `must_call_once`, `forbidden_text`,
    `must_include_text`. Sorted output for stable CI diffs.
  - `src/gate/index.ts` — Compose rules + LTLf formulas into one
    `GateResult { passed, violations, ltlResults }`.
    `renderGateSummary()` produces a stable CI line.
  - Cross-language conformance verified: 9 parity tests in
    `python/tests/test_typescript_parity.py` exercise both
    implementations on identical fixtures and assert byte-equal
    decisions on `(ruleId, pairIndex, kind)` tuples and per-formula
    pass/fail. 35 new TS tests in `typescript/test/` (LTLf operators,
    policy rule kinds, gate composition).
- **`shadow_core::diff::embedder` — pluggable Embedder trait.** The
  Rust semantic axis no longer locks in TF-IDF as the only path:
  - New `Embedder` trait with `embed(texts)` + `id()`.
  - `BoxedEmbedder` adapter wraps any `Fn(&[&str]) -> Vec<Vec<f32>>`
    closure (use case: ONNX runtime, HF Inference API, OpenAI
    embeddings, in-house service, PyO3 callback into Python
    `sentence-transformers`).
  - `compute_with_embedder(pairs, embedder, seed)` — new entry point
    in `shadow_core::diff::semantic` that lets callers route the
    semantic axis through a custom `Embedder`. The shared cosine
    + median + paired-CI tail keeps embedder choice independent of
    the rest of the pipeline.
  - Default `compute()` path unchanged: smoothed sklearn-style TF-IDF
    over the corpus, no extra dependencies.
  - Tests verify swap-in works (paraphrase pair scores 0 under TF-IDF
    but ≈1 under a custom embedder), dim-mismatch returns empty axis
    safely, both-zero vectors return cosine 1.0.

### Changed

- **`shadow.bisect` re-exports.** The v2.5+ stability-selection +
  residual-bootstrap CI APIs (`rank_attributions_with_ci`,
  `rank_attributions_with_interactions`, `AXIS_NAMES`) are now
  exported from the package root so `from shadow.bisect import ...`
  surfaces the public CI surface. Already wired into
  `corner_scorer.py` internally; this change just makes them
  discoverable from documented import sites.
- **`shadow.causal/__init__.py`** — status updated from "Foundation"
  to "Production". Out-of-scope items (front-door adjustment,
  optimal experiment design, matched-pair Rosenbaum bounds) listed
  explicitly; replaced the prior multi-week TODO with the actual
  shipped surface.
- **`docs/theory/causal.md`** — full rewrite covering the new
  pipeline (point ATE, bootstrap CIs, back-door adjustment, E-value)
  with worked example and citation list.
- **`typescript/PARITY.md`** — boundary statement rewritten: the
  evaluator surface (LTLf, policy gate) is in TS now; the compiler /
  mining tooling and rich-rule engine stay Python.
- **`README.md`** — TS parity table refreshed with the new gating
  rows; embeddings-extra description acknowledges the trait-based
  Embedder; causal description acknowledges the production pipeline.

### Notes

- These are **additive** changes. No public API was removed,
  renamed, or behaviourally altered. SemVer minor bump applies.
- The wheel size and default install footprint are unchanged. The
  `Embedder` trait does not pull in any new heavy dependencies; ONNX
  / neural backends remain a downstream user choice.

## [2.6.0] - 2026-04-28

Comprehensive technical-debt cleanup pass on top of v2.5.0.
Closes 50+ specific items identified in the post-v2.5.0 audit and
external real-world stress evaluation.

### Added — new public modules

- **`shadow.diff_py`** — scenario-aware multi-case diff. Partition
  records by `meta.scenario_id` and run per-scenario diffs so
  multi-incident regression suites no longer collapse into spurious
  "dropped turns" messages. Backward-compatible: traces without
  scenario_ids fall into a single `__default__` bucket.
- **`shadow.causal`** — Pearl-style do-calculus attribution
  (foundation). Replaces the LASSO-based bisection with single-delta
  intervention ATE estimates. Ground-truth test verifies 3 real
  deltas + 5 noise → algorithm correctly attributes.
- **`shadow.policy_suggest`** — mine baseline traces for
  `must_call_before` patterns. Pure-derivable ordering invariants
  only; the operator approves suggestions before adding to YAML.
- **`shadow.storage`** — pluggable `Storage` Protocol (FileStore +
  InMemoryStore). Foundation for cloud-backed Postgres / S3 /
  ClickHouse stores; existing call sites continue to use the Rust
  core directly and migrate incrementally.
- **`shadow._telemetry`** — opt-in anonymous usage telemetry
  (default off, CI auto-skip, SHADOW_TELEMETRY=off override).
  18 tests verify privacy-preserving defaults.

### Added — new statistical / formal primitives

- **`shadow.statistical.MSPRTtDetector`** — variance-adaptive mixture
  SPRT (Welford running variance). Practical 't-mixture spirit'
  (Lai & Xing 2010) for the unknown-σ case where a long warmup
  isn't available. Asymptotic always-valid bound; exact unknown-σ
  variant deferred to a future release.
- **`shadow.conformal.ACIDetector`** — Adaptive Conformal Inference
  (Gibbs & Candès 2021). Online α adaptation that converges to the
  target miscoverage rate at O(1/(γT)) under arbitrary distribution
  shift — no exchangeability assumption.
- **`shadow.statistical.FingerprintConfig`** — configurable scales
  (token_scale, latency_scale_ms, max_tool_calls) for long-context
  / thinking-mode agents that would saturate the default scales.
- **Hotelling T² `permutations` argument** — Monte-Carlo p-value
  via label permutations (Phipson-Smyth 2010 corrected) for cases
  where the F-approximation under shrinkage is unreliable.

### Added — new examples

- **`examples/production-incident-suite/`** — canonical real-world
  stress eval. Five public-incident patterns (Air Canada, Avianca,
  NEDA/Tessa, McDonald's, Replit) encoded as scenario builders;
  audit pipeline must catch each one. 32 tests cover per-scenario
  coverage, multi-scenario diff sectioning, false-positive freedom,
  causal attribution accuracy, and CLI exit code.
- **`examples/harmful-content-judge/`** — domain-aware harm detector
  for content the narrow safety axis can't catch (medical
  misinformation, fake legal citations, eating-disorder advice).
  Reusable `build_harm_judge(backend)` helper.

### Added — verification and proof

- **`docs/theory/`** — primary-source references for every primitive
  (Hotelling, SPRT, conformal, LTL, causal). Designed so a reader
  with stats / formal-methods background can verify Shadow's claims
  in under five minutes per page.
- **`benchmarks/primitives_perf.py`** — wall-time budgets per
  primitive with 3× safety margin. Confirms current pure-Python LTL
  runs at 5K-turn DP in 2.5ms, well within agent-eval scale.
- **Statistical validation suite expanded**: power curves across
  8 (effect-size × n) cells, Wilson CI on Type-I tolerance, hand-
  derived OAS reference cross-validation (1e-6 agreement, 20 seeds),
  18 NaN/Inf edge-case tests, 4 Hypothesis property tests on the
  agentlog parser/writer roundtrip.
- **`@pytest.mark.slow` runs in CI** on Ubuntu/3.12 every push.

### Added — operator UX

- **Risk-scored PR comments** — `shadow report --format github-pr`
  prepends a tiered (CRITICAL / ERROR / WARNING / INFO) summary
  header with plain-language phrasing of axis severities and
  recommendations, ahead of the per-axis table.
- **Policy YAML `apiVersion` field** — `shadow.dev/v1alpha1`
  forward-compatibility marker. Files without an apiVersion load
  with a deprecation warning; unknown apiVersions raise with an
  explicit hint listing supported versions.
- **`shadow-diff[all]` meta-extra** — single-shot install of every
  optional integration. README `Install` section gains an
  "Optional extras" table documenting every gate.

### Changed

- **`shadow.conformal.build_conformal_coverage` → `build_parametric_estimate`**.
  The old name was misleading: the function synthesizes a Gaussian
  calibration set rather than computing a distribution-free conformal
  bound. The old name remains as a `DeprecationWarning` alias and
  will be removed in v3.0.
- **`shadow.statistical.fingerprint.tool_call_rate`** is now
  log-scaled (was a clipped count). 1 vs 4 vs 8 tool calls per turn
  now discriminate properly on this dimension.
- **`shadow.ltl` checker** rewritten as bottom-up dynamic programming
  with truth-vectors. Genuinely O(|π| × |φ|) — the previous memoized
  recursion was O(|π|³) on Until despite the docstring claim.
- **`must_call_before` rule kind** now compiles to weak-until
  `(¬B) W A` instead of strong-until `(¬B) U A`. A trace that fires
  neither A nor B is now correctly treated as vacuously safe.
- **Wald SPRT decisions are absorbing**. Once a boundary is crossed
  the detector stops accumulating; previously the log-LR continued
  past the decision and could flip back.
- **TypeScript SDK 2.2.0 → 2.5.0** to match Python wheel; npm
  `vitest` bumped 2.x → 3.2.4 to clear 5 moderate dev-dep advisories.
- **Rust crate `shadow-core` 2.4.3 → 2.5.0** to match Python wheel.
- **License consolidated to Apache-2.0** across all three components
  (was a mix of `MIT OR Apache-2.0`).
- **Versioning policy formalized**: all three components (Python
  wheel, Rust crate, TypeScript SDK) bump together on every release.
  Documented in `CONTRIBUTING.md`.
- **API stability committed**: SemVer 2.0.0 starting at v2.5.0.
  No breaking changes within v2.x. Specific list of what counts as
  "breaking" vs "additive" in `CONTRIBUTING.md`.

### Fixed

- **`shadow.judge.LlmJudge` rubrics accept literal `{}`** without
  crashing on `KeyError`. Switched the placeholder validator from
  `string.Formatter().parse` to a regex over identifier-only patterns.
  Reported by external real-world stress evaluation.
- **CLI pricing-table validation error** now names the offending key,
  the actual type, and both valid shapes (list[float, float] or
  dict with named keys). Previous message ("must be or a dict")
  truncated the list shape.
- **`shadow.conformal._quantile`** no longer returns NaN on
  +Inf scores. The naive linear-interpolation form `inf × 0` is
  NaN; we now branch to return the endpoint directly when frac == 0
  or lo == hi. Caught by NaN/Inf edge-case test.
- **Permutation Hotelling p-value** documented as Phipson-Smyth
  (2010) corrected `(b+1)/(B+1)` with regression tests for the
  1/(B+1) lower bound and the never-zero invariant.

### Documentation

- **CHANGELOG v2.5.0 entry rewritten** to accurately reflect what
  shipped (the original was written early in the development cycle
  and contained inaccurate citations).
- **Safety axis docstring** (`crates/shadow-core/src/diff/safety.rs`)
  explicitly directs users to the Judge axis + harmful-content-judge
  example for harmful semantic content the refusal-only safety axis
  can't catch.
- **mSPRT plug-in σ̂ caveat** documented in module docstring.
  Wald + mSPRT bounds are exact under known σ; with plug-in σ̂ from
  finite warmup, they are asymptotic with O(1/√warmup) slack.
- **LTL automaton-compilation deferral** documented with rationale
  + perf benchmark proving pure-Python is fine at current scale.
- **TypeScript SDK parity boundary** documented in
  `typescript/PARITY.md` — intentionally narrower than Python.
- **Mypy override block** comment now explains why
  `shadow.enterprise.*`, `shadow.serve.*`, `shadow.mcp_server`,
  `shadow.adapters.*`, `shadow.tools.sandbox`, `shadow.certify_sign`
  stay overridden (optional-extras gating, not an oversight).

### Operations

- **Pre-commit hooks** (`.pre-commit-config.yaml`) mirror CI gates
  so contributors catch lint/format issues before pushing.
- **DCO sign-off check** at `.github/workflows/dco.yml` rejects
  commits without a `Signed-off-by` trailer matching the author.
- **Branch protection** documented; admin bypass kept on for solo
  development with explicit re-tighten-with-collaborators path.
- **Apache 2.0 license** + trademark / CLA decisions documented.
- **`@shadow/sdk` npm publish** path gated on NPM_TOKEN secret;
  idempotent reruns via `npm view package@version` check.
- **Windows `cp1252` codec crash on Unicode** in CLI demos fixed
  via `sys.stdout.reconfigure(encoding="utf-8")`.

## [2.5.0] - 2026-04-27

Feature release: behavioral fingerprinting + Hotelling T², Wald and
mixture SPRT, finite-trace LTL policy verification with WeakUntil, and
distribution-free conformal coverage. Two real-world example scenarios
(refund-agent audit, canary monitor) plus a statistical-property
validation suite that empirically verifies the claims.

### Added

- **`shadow.statistical` — sequential testing and behavioral fingerprinting.**
  - `fingerprint.py`: D=8 feature vector per response turn
    (`tool_call_rate` log-scaled count of tool_use blocks,
    `distinct_tool_frac`, three one-hot stop-reason features,
    `output_len_log`, `latency_log`, `refusal_flag`). Bounded [0,1] so
    no axis dominates multivariate tests. New `FingerprintConfig`
    dataclass exposes `token_scale`, `latency_scale_ms`,
    `max_tool_calls` for long-context / thinking-mode agents.
  - `hotelling.py`: Two-sample Hotelling T² with Oracle Approximating
    Shrinkage (OAS, Chen et al. 2010) on the pooled covariance so the
    test does not blow up when n1+n2−2 ≤ D. Optional permutation
    p-value via `permutations=N` argument when the F-approximation is
    unreliable (small samples, shrinkage applied, non-normality).
  - `sprt.py`:
    - `SPRTDetector` — classic Wald SPRT with **absorbing decisions**.
      Once a boundary is crossed, subsequent updates return the same
      decision without further accumulation, preserving the (α, β)
      bounds.
    - `MSPRTDetector` — mixture SPRT (Robbins 1970, Johari–Pekelis–
      Walsh 2017). Non-negative martingale under H0 with the always-
      valid bound `P(sup_n Λ_n ≥ 1/α) ≤ α` simultaneously across all
      sample sizes. The recommended detector for production A/B
      monitoring with continuous peeking.
    - `MultiSPRT` — per-axis ensemble.
    - All detectors estimate (μ0, σ²) online from a warmup buffer.
      Caveat: with plug-in σ̂ from finite warmup, bounds are
      asymptotic; documented in module docstring.

- **`shadow.ltl` — finite-trace LTL model checker.**
  - `formula.py`: AST nodes `Atom`, `Not`, `And`, `Or`, `Implies`,
    `Next`, `Until`, `WeakUntil`, `Globally`, `Finally`. Convenience
    constructors `g`, `f`, `x`, `u`, `w`, `conj`, `disj`.
  - `checker.py`: bottom-up dynamic-programming checker with truth-
    vectors. Genuinely O(|π|×|φ|) — for each subformula, computes a
    length-|π| boolean array via the recurrences
      G(φ)[i] = φ[i] ∧ G(φ)[i+1]; G(φ)[n] = True
      F(φ)[i] = φ[i] ∨ F(φ)[i+1]; F(φ)[n] = False
      (φ U ψ)[i] = ψ[i] ∨ (φ[i] ∧ (φ U ψ)[i+1]); [n] = False
      (φ W ψ)[i] = ψ[i] ∨ (φ[i] ∧ (φ W ψ)[i+1]); [n] = True
    Public `eval_all_positions` exposes the full truth-vector for
    callers that want it.
  - `compiler.py`: rule kinds compile to formulas:
      `no_call(A)` → `G(¬tool_call:A)`
      `must_call_before(A, B)` → `(¬tool_call:B) W tool_call:A`
        (weak-until: a trace that calls neither A nor B is vacuously
        safe; the previous strong-until encoding misclassified those)
      `must_call_once(A)` → `F(A) ∧ G(A → X G ¬A)`
      `required_stop_reason(allowed)` → `F(disj over stop_reason atoms)`
      `forbidden_text(T)` → `G(¬text_contains:T)`
      `must_include_text(T)` → `F(text_contains:T)`
      `ltl_formula(formula=...)` → `parse_ltl(...)`
    Parser supports `G F X U W ! & | ->` with standard precedence.

- **`shadow.conformal` — distribution-free conformal prediction.**
  - `conformal_calibrate(per_axis_scores, ...)` — real split-conformal
    calibration from per-run nonconformity scores. Returns
    `ConformalCoverageReport` with `is_distribution_free=True` and the
    PAC-certified marginal claim `P(score_{n+1} ≤ q̂) ≥ 1-α`.
  - `build_conformal_coverage(axis_rows, ...)` — parametric fallback
    when only summary statistics are available. Synthesizes a Gaussian
    calibration set from `delta` and CI half-width; flagged
    `is_distribution_free=False`.
  - `ConformalCoverageReport.is_distribution_free` boolean discriminates
    real vs parametric.
  - `n_min = ⌈log(1−confidence) / log(coverage)⌉` and `pac_delta`
    binomial CDF for the PAC threshold.

- **`examples/refund-agent-audit/`** — real-world scenario auditing a
  customer-support agent upgrade (claude-haiku-4-5 → claude-opus-4-7).
  Catches the candidate processing a refund in the same turn as the
  lookup and announcing "processed successfully" without confirmation.
  Includes baseline / candidate `.agentlog` fixtures, an importable
  `run_audit()` function, and a CLI that exits 1 when unsafe. 65 tests.

- **`examples/canary-monitor/`** — production canary monitor combining
  mSPRT for always-valid latency monitoring, WeakUntil safety policies
  for transfer-funds verification, and conformal calibration with a
  Bonferroni-corrected family-wise error bound across all alarm
  channels. Continuous peeking does not inflate the false-positive rate
  above α regardless of dashboard refresh cadence. 28 tests.

- **`python/tests/test_statistical_validation.py`** — 11 simulation
  tests under `@pytest.mark.slow` that empirically verify Type-I rate,
  power, always-valid bound, and held-out conformal coverage
  (including heavy-tailed t-distributions). Ships proof that the math
  delivers the claims, not just that the code runs.

### Changed

- License consolidated to **Apache 2.0** (was dual MIT/Apache).
- Contributor flow now requires **DCO sign-off** on every commit
  (`git commit -s`), enforced by `.github/workflows/dco.yml`.
- **Pre-commit hooks** (`.pre-commit-config.yaml`) mirror CI gates so
  contributors catch lint/format issues before pushing.
- **Branch protection** on `main` configured in repo settings (admin
  bypass kept on for solo development; re-tighten with collaborators).

### Decisions

- **OAS shrinkage** chosen over Ledoit-Wolf and cross-validated
  shrinkage because it has a closed-form coefficient (no CV required)
  and handles n+m−2 ≤ D gracefully. Permutation p-values available
  for cases where the F-approximation under shrinkage is unreliable.
- **Finite-trace LTL semantics (LTLf)** chosen over infinite-trace.
  `G φ` is vacuously true at trace end (correct for safety properties)
  and `F φ` is false at trace end (requires evidence within the trace).
- **Bottom-up DP for the LTL checker** chosen over the original
  memoized recursion because the latter was actually O(|π|³) on
  `Until` despite the claimed O(|π|×|φ|). The DP variant is genuinely
  linear in trace length.
- **WeakUntil for `must_call_before`**. The previous strong-until
  encoding `(¬B) U A` incorrectly flagged traces that called neither
  A nor B as policy violations. Weak-until `(¬B) W A` correctly
  treats them as vacuously safe.
- **Plug-in σ̂ caveat for SPRT/mSPRT**. Robbins's bound is exact
  under known σ; with plug-in σ̂ from finite warmup, the bound is
  asymptotic. Documented in the module docstring; large warmup
  (≥100) recommended for accurate Type-I control.

### Fixed

- `Cargo.toml` workspace `version` is now `2.5.0` (was 2.4.3 — drift
  between Python wheel and Rust crate at v2.5.0 release).
- `typescript/package.json` `version` is now `2.5.0` (was 2.2.0).
- Windows `cp1252` codec crash on Unicode math symbols (`σ̂`, `q̂`,
  `±`) in CLI demos: `sys.stdout.reconfigure(encoding="utf-8")` at
  startup; subprocess test calls use `encoding="utf-8"`.
- `python/src/shadow/hierarchical.py` line-too-long lint error.
- Entire codebase reformatted via `ruff format` to match repo style.

## [2.4.3] - 2026-04-27

Patch on top of 2.4.2. No new features — only the sdist fix.

### Fixed

- **sdist upload to PyPI failed with `400 License-File LICENSE-APACHE does not exist`** because maturin auto-discovered `python/LICENSE-APACHE` for PKG-INFO metadata but did not bundle the file into the tarball archive at `shadow_diff-2.4.3/LICENSE-APACHE`. Added explicit `[tool.maturin] include` entries scoped to `format = "sdist"` so the files land at the expected path inside the archive.

## [2.4.2] - 2026-04-27

Distribution patch. Adds a Python source distribution (sdist) to the PyPI release so `pip install shadow-diff` works on platforms without a published wheel.

### Added

- **Python sdist on PyPI.** The release workflow now runs `maturin sdist` and uploads the resulting `.tar.gz` alongside the per-OS wheels. Pre-built wheels still cover Linux x86_64, macOS arm64, and Windows x86_64. On other platforms (Intel Mac, ARM Linux, older glibc, Alpine, FreeBSD) pip transparently falls back to the sdist and builds the Rust core locally — Rust must be on PATH.
- **README install section.** Two-step pip install instructions, plus a separate note for unsupported-platform users with the `rustup` one-liner.

### Fixed

- **`pip install shadow-diff` failing with "No matching distribution found"** on platforms outside the three published wheel targets. The sdist fallback above closes that gap.

## [2.4.1] - 2026-04-27

Patch release. All bug fixes from two external audit rounds. No new features, no API breaks (the auto-record default change in `wrap_tools` is backwards-compat-detected by the dedup logic).

### Fixed

- **`when:` policy clauses with list-index path segments returned None.** `request.messages.1.content` walked into the dict but stopped at the list. Numeric segments now resolve as list indices.
- **Bundled `pricing.json` crashed `--pricing` and the MCP `shadow_diff` handler** because the `_comment` and `_updated` documentation keys aren't model entries. The CLI parser, `shadow mine`, `shadow certify`, and the MCP server now share a single `load_pricing_file()` helper that skips underscore-prefixed metadata keys.
- **Session-scoped policy rules raised `IndexError`** when a candidate trace had different length from baseline (dropped tool turn, etc.). `_check_rule_per_session` now bounds-guards both indexed accesses.
- **Embeddings backend left stale BM25 recommendations.** When `--semantic embeddings` lowered the semantic axis from severe to minor, the report still carried the original "severe BM25" recommendation. New `_refresh_after_axis_swap` drops stale axis recommendations, recomputes drill-down `dominant_axis` / `regression_score`, and clears `first_divergence` when its axis was swapped. Same fix applied to `--judge` axis swaps.
- **Bisect terminal labels showed `?`** because the renderer read `row['label']` and `row['category']` but the unified attribution schema stores the delta name under `row['delta']`. Renderer now falls through `label → delta → category`.
- **`wrap_tools` didn't auto-record successful tool calls**, causing `must_call_before` rules to silently fail when the caller didn't manually invoke `s.record_tool_call(...)`. Auto-record is now the default; the wrapper detects when the tool function recorded itself and skips to avoid duplicate records (preserves backwards compat with framework adapters and existing user code).
- **Pricing data coverage gap.** Added 9 missing OpenAI 2025-2026 model entries (`gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-5-pro`, `gpt-5-nano`, `gpt-5-codex`, `o1-pro`, `o3`, `o3-mini`, `o4-mini`). Combined with the snapshot-tail fallback shipped in 2.4.0, dated snapshots like `gpt-4.1-mini-2025-04-14` now resolve cleanly to the bare alias.

### Added

- 7 new regression tests pinning the above fixes.
- `load_pricing_file()` public helper in `shadow.cli.app` so other consumers don't reinvent the parser.

## [2.4.0] - 2026-04-25

The two final roadmap items shipped — every entry in ROADMAP's "What's next" is now in "Shipping today." ROADMAP.md is deleted; remaining work is tracked as GitHub issues against this repo.

### Added

- **Cross-modal semantic diff axis** in `shadow.multimodal_diff` — compares `blob_ref` records across two traces. Two-tier comparison aligned with RAGAS / TruLens / DeepEval / LangSmith / Langfuse conventions:
  - **Cheap tier**: 64-bit dHash Hamming distance (always available when `phash` is on the records). Threshold ≤ 10/64 = "near-duplicate" (severity none), 10–16 = "minor visual drift," > 16 = "moderate." Cheap tier alone never escalates to severe — there isn't enough signal.
  - **Semantic tier**: cosine similarity over `embedding.vec` when both sides have an embedding of the same model. Threshold ≥ 0.85 = "same content" (none), ≥ 0.75 = "same subject" (minor), ≥ 0.5 = "moderate," < 0.5 = "severe." Per LangSmith / Langfuse defaults.
  - Severity decision: semantic wins when both tiers are present (embeddings are higher signal). Identical `blob_id` short-circuits to none (content-addressing means same id = same bytes). Unmatched blobs (one side has more than the other) flagged severe — the candidate either lost or introduced a blob the baseline didn't have.
  - Renderers: `render_terminal()` for CLI output, `render_markdown()` for PR comments. Both render unchanged blobs silently — only show what changed.

- **Harness-event diff renderer** in `shadow.harness_diff_render` — surfaces `harness_event_diff` output (regressions, fixes, count deltas, first-occurrence pair indices) as reviewer-friendly text:
  - `render_terminal()`: separates regressions from fixes, sorts regressions by severity desc then absolute delta desc, emits severity-coloured glyphs (`🔴 error`, `🟠 warning`, `🟡 info`).
  - `render_markdown()`: two-table PR-comment layout — regressions table with severity column + first-occurrence pair index, fixes table simpler. Empty input returns a one-line notice so callers can pipe unconditionally.

- **`shadow diff` gains two new flags**: `--harness-diff` surfaces the harness-event diff inline in the report, `--multimodal-diff` runs the cross-modal axis. Both default off; cost is zero when the trace has no relevant records.

### Tests

- 24 new tests at `python/tests/test_v24_renderers.py` — cosine identity / orthogonality / opposite / zero-norm / length-mismatch, dHash near-dup / far / no-signal severity classification, semantic-takes-precedence-over-phash, unmatched-blob severity, worst-severity aggregation, terminal + markdown rendering shape, severity ordering in the harness renderer (errors before warnings before info), CLI integration with the `--harness-diff` flag.

### Roadmap

- ROADMAP.md is deleted. Every entry that was in "What's next" has shipped: streaming replay (v2.3 chunk records), multimodal traces (v2.3 blob_ref + v2.4 cross-modal diff axis), harness-diff instrumentation (v2.3 harness_event records + v2.4 renderer), MCP-native replay (v2.3), TypeScript streaming parity (v2.2), auto-instrument-layer pre-dispatch (v2.2). Future work is tracked as GitHub issues against the repo.

810 pytest, 205 cargo, 34 vitest, ci-local green, mkdocs `--strict` green.

## [2.3.0] - 2026-04-25

`.agentlog` v0.2 + MCP-native replay. Four roadmap items shipped together. Each design choice researched against canonical conventions (OpenTelemetry GenAI semconv stable Jan 2026, OpenInference, Langfuse v3 media API, RAGAS / TruLens / DeepEval multimodal baselines, MCP SEP-1287 draft) before implementation.

### Added

- **`.agentlog` v0.2** spec (SPEC §4.8 / §4.9 / §4.10) adds three new record kinds. Backwards-compatible: every v0.1 record still validates. v0.2 readers MUST treat unknown kinds as passthrough so future spec adds don't break old tools.

  - **`chunk`** (§4.8) — single streaming-LLM chunk with `chunk_index`, absolute `time_unix_nano` (per OTel convention; relative offsets drift on long streams), provider-shape `delta` (Anthropic `text_delta` / `input_json_delta` / `thinking_delta`, OpenAI `{content?, tool_calls?[]}`), optional `is_final`. Logical-response identity remains the assembled `chat_response`'s content-id.
  - **`harness_event`** (§4.9) — single record kind with `category` discriminator over the closed taxonomy `{retry, rate_limit, model_switch, context_trim, cache, guardrail, budget, stream_interrupt, tool_lifecycle}` (matches OTel `gen_ai.cache.*`, `gen_ai.guardrail.*`, etc.). Each event carries `name`, `severity ∈ {info, warning, error, fatal}`, free-form `attributes`. Single-kind-with-discriminator beats kinds-per-event because new event types don't require code changes — same lesson Langfuse / Helicone / Phoenix all hit.
  - **`blob_ref`** (§4.10) — content-addressed binary reference. sha256 `blob_id`, `mime`, `size_bytes`, optional `agentlog-blob://` URI (mirrors OTel's `otel-blob://`), optional 64-bit dHash `phash` (RAGAS / TruLens / DeepEval no-LLM-judge baseline; Hamming ≤10/64 = near-dup, ≥16 = different), optional `embedding` for the semantic-tier diff. Inline base64 stays permitted under a 4 KiB cap; anything larger is a `blob_ref` to keep records parseable in line-buffered tools.

- **`shadow.v02_records` Python module** with full recording + diff support:
  - `record_harness_event(session, *, category, name, severity, attributes)` — validates category + severity at record time so typos surface up front instead of as silent diff misses.
  - `record_chunk(session, *, chunk_index, delta, is_final, time_unix_nano)` — `time_unix_nano` defaults to `time.time_ns()` at the call site.
  - `replay_chunks_async(chunks, yielder, speed=1.0)` — monotonic-deadline replay loop, NOT cumulative `sleep(delta)` (cumulative drifts on long streams; deadline-relative stays accurate). Handles non-monotonic timestamps without deadlocking. `speed` multiplier accepts `1e9` for effectively-instant replay.
  - `BlobStore(root)` — git-objects-style sharded sha256 blob store with atomic temp-file + rename for crash safety. Identical content collapses to one file across repeated puts.
  - `compute_phash_dhash64(image_bytes)` — optional `imagehash` dep; returns SPEC-shaped `{algo: dhash64, hex: ...}` or None when the lib is missing.
  - `phash_distance(a, b)` — Hamming distance over hex; returns None on algo mismatch so callers can branch.
  - `record_blob_ref(session, *, blob, mime, store)` — content-addresses + writes the blob, computes dHash for `image/*` mime types, appends a `blob_ref` record.
  - `harness_event_diff(baseline, candidate)` — returns `[HarnessEventDelta]` with `(category, name)` keying, count delta, first-occurrence pair index for both sides, sorted by absolute count delta descending.

- **`shadow.mcp_replay` Python module** — protocol-level MCP replay via the transport-stream shim pattern (research-recommended path; survives SDK upgrades, aligns with SEP-1287's `replay://` URI scheme):
  - `canonicalize_params(params)` — sorted-keys, no-whitespace, `ensure_ascii=False` JSON encoding so non-ASCII URIs in `resources/read` round-trip cleanly.
  - `RecordingIndex(calls)` — indexes `MCPCall` objects by `(method, canonicalize(params))`. Repeated calls return responses in recorded order then fall back to the last recorded response (preserves "the second `tools/list` returned one fewer tool" behaviour). `unconsumed_keys()` surfaces calls the candidate skipped — drift detection at the protocol layer.
  - `ReplayClientSession(index, strict=False)` — drop-in replacement for `mcp.ClientSession`. Implements `call_tool`, `read_resource`, `list_tools` / `list_resources` / `list_prompts`, `get_prompt`, `initialize` (with synthetic capability stub when not recorded). Sync + async variants. `strict=True` raises `MCPCallNotRecorded` on misses; non-strict returns None for null-check paths. Errors in recordings raise `MCPServerError`.
  - `index_from_imported_mcp_records(records)` — builds an index from an MCP-imported `.agentlog` (Shadow's existing `shadow import --format mcp` output). Recognises `tool_call` + paired `tool_result` records, plus `metadata.payload.mcp.calls` for non-tool methods.

- **Rust core**: `Kind::Chunk`, `Kind::HarnessEvent`, `Kind::BlobRef` added to the `record::Kind` enum so the parser accepts v0.2 records and the replay engine copy-throughs them.

### Tests

- 41 new tests across `python/tests/test_v02_records.py` (22) and `python/tests/test_mcp_replay.py` (19) — chunk record/replay round-trip, replay timing fidelity (deadline loop, non-monotonic timestamps, speed multiplier), harness event recording + diff at scale, BlobStore dedup + atomic replace + URI scheme, dHash distance correctness, MCP canonicalization (key-order independence, unicode, integer-vs-float distinction), 1000-call lookup performance, repeated-call ordering, error propagation, strict vs non-strict miss handling, drift detection via `unconsumed_keys`.
- Real-world adverse stress harness at `examples/stress/run_stress.py` — 20 assertions covering 10K-chunk session round-trip, 5 concurrent replays without state leakage, backward-timestamp non-deadlock, harness diff at scale (sub-100ms over thousands of records), 1000-puts dedup, atomic-replace crash simulation, 16 MiB blob round-trip, real PNG dHash, 1000-call MCP recording lookup in <2ms, canonicalize collision matrix (int vs float, key order, unicode). 20/20 passes in 0.32s wall-clock.

### Roadmap

- "What's next" loses streaming replay, multimodal traces, harness-diff instrumentation, MCP-native replay (all shipped). Remaining: cross-modal semantic diff axis (CLIP / Whisper-embed on top of the v0.2 `blob_ref.embedding` slot), harness-diff renderer for PR comments.

786 pytest, 205 cargo, 34 vitest, ci-local green, mkdocs `--strict` green.

## [2.2.0] - 2026-04-25

Two roadmap items shipped together. Both researched against canonical guardrail / auto-instrumentation patterns (NeMo Guardrails, Bedrock Guardrails, OpenTelemetry openai-instrumentation) before implementation — buffer-to-completion + replace-whole-response is the production norm, not strip-individual-blocks.

### Added

- **Auto-instrument-layer pre-dispatch enforcement.** When an `EnforcedSession` is active, the OpenAI / Anthropic `.create` wrapper now probes the enforcer with every `tool_use` block in the non-streaming response BEFORE returning to user code. Violating tool calls raise `PolicyViolationError` at the wrapped `.create` site — the user's tool dispatcher never sees the violating response, so dangerous tools (`issue_refund`, `send_email`, `execute_sql`, `delete_user`, `deploy_service`) can't fire. No code changes to user tool functions; works for any OpenAI / Anthropic-driven agent. Replace mode at this layer is approximated by raise (modifying SDK response objects across versions is fragile — use `wrap_tools` for finer control). Plain `Session` (no enforcer) is a complete no-op; users who never opted into runtime enforcement see zero behaviour change. 9 new tests at `python/tests/test_instrumentation_predispatch.py` cover the no-op, raise / replace / warn modes, allowed pass-through, repeated-block probe-state cleanliness, and translator-error graceful handling, plus an end-to-end test driving a fake OpenAI Completions class through the full Instrumentor pipeline.
- **TypeScript SDK streaming aggregation.** The TS SDK's auto-instrument wrapper now intercepts `stream: true` calls via an async-iterator proxy that yields each chunk through to the caller AND feeds it to a per-provider aggregator. On stream end (or caller-side break), a single `chat_response` record lands with the assembled content. Two production aggregators:
  - **OpenAI**: rebuilds text from `choice.delta.content` deltas, reconstructs interleaved `tool_calls` by index (each tool's id / name / arguments string assembled across chunks), captures `finish_reason` from the final chunk, and folds in `usage` if `stream_options: {include_usage: true}` was set.
  - **Anthropic**: tracks content blocks by index across `content_block_start` / `content_block_delta` / `content_block_stop` events, accumulates text deltas / `input_json_delta` partial JSON / thinking deltas, captures `stop_reason` from `message_delta`, finalises into the same Message shape `anthropicTranslators.resp()` consumes for non-streaming responses.
  - 4 new tests at `typescript/test/instrumentation_streaming.test.ts` covering OpenAI text aggregation, OpenAI tool-call argument-delta reassembly, Anthropic mixed text + `tool_use` block reassembly, and caller-side-break early termination.
- New exports from `typescript/src/instrumentation.ts`: `Translators`, `StreamAggregator`, `openaiTranslators`, `anthropicTranslators`. Lets integration code drive the production aggregators directly.

### Changed

- README TypeScript / Python parity matrix updated — streaming aggregation row now ✅ on both.
- ROADMAP "What's next" loses two entries (TS streaming, auto-instrument pre-dispatch). Remaining items: streaming replay (`.agentlog` v0.2 chunk records), multimodal traces, harness-diff instrumentation, MCP-native replay.

## [2.1.0] - 2026-04-25

### Added

- **Pre-tool-call (pre-dispatch) policy enforcement.** New public API in `shadow.policy_runtime`:
  - `wrap_tools(tools, enforcer, *, session=None, records_provider=None, blocked_replacement=None)` — wraps a `{name: callable}` tool registry. Each entry returns a `GuardedTool` that probes the enforcer with a synthesised candidate `tool_call` record BEFORE invoking the underlying function. On `allow`, the function runs. On deny: `raise` mode throws `PolicyViolationError`, `replace` mode returns a placeholder (configurable per-tool via `blocked_replacement=`), `warn` mode logs and runs anyway. Catches `no_call`, `must_call_before`, `must_call_once` at the dispatch site for dangerous tools (`issue_refund`, `send_email`, `execute_sql`, `delete_user`, `deploy_service`).
  - `Session.wrap_tools(tools)` convenience method on `EnforcedSession` that auto-binds the session.
  - `PolicyEnforcer.probe(records)` non-mutating evaluation. The probe asks "if these records were the trace, would any rule fire?" without remembering the violation in `_known` — repeatedly blocked tool calls don't pollute enforcer state, and a denied probe followed by a real dispatch correctly fires once on the next `evaluate`.
  - `GuardedTool` — the per-tool wrapper. Exposes `.name`, `.fn`, and a `__call__` that performs the probe + dispatch.

- **`_extract_tool_call_sequence` now reads standalone `tool_call` records**, not only `tool_use` blocks inside `chat_response` content. This is what makes pre-dispatch enforcement work — a synthesised candidate `tool_call` record is now visible to `no_call` / `must_call_before` / `must_call_once` rules. Side benefit: `Session.record_tool_call` calls are now first-class to the policy engine; previously they were invisible to those rules unless paired with an Anthropic-style `tool_use` content block.

- 11 new tests at `python/tests/test_policy_runtime_predispatch.py` covering: probe non-mutation, allowed dispatch passing through, blocked dispatch in all three modes (raise / replace / warn), `must_call_before` ordering enforcement, `wrap_tools` with explicit `records_provider`, `wrap_tools` requiring either session or records_provider, custom `blocked_replacement`, and repeated-block probe-state cleanliness.

### Docs

- `docs/features/runtime-enforcement.md` adds a "Pre-tool-call enforcement (v2.1)" section covering the new surface, the probe-vs-evaluate distinction, what rule kinds fire pre-dispatch vs response-side, and the `records_provider=` integration point for framework adapters.
- README runtime-enforcement section gains a runnable `s.wrap_tools(...)` example with the `delete_user` blocked case.
- ROADMAP moves "Pre-tool-call interception" out of "What's next." Remaining roadmap entry is auto-instrument-layer pre-dispatch (so OpenAI/Anthropic-driven agents get pre-dispatch enforcement automatically without wrapping their tool registry).

## [2.0.5] - 2026-04-25

All six items the reviewer raised were verified real and fixed.

### Fixed

- **`SPEC.md` §3.3 said "A trace MUST NOT contain more than one `metadata` record"** — directly contradicted shipping code. `Session.record_metadata()` has been writing additional metadata records to mark session boundaries since v1.4 (a docstring explicitly says "Shadow's session detector treats multiple metadata records in a trace as the canonical session boundary signal"). The spec rule is removed; replaced with an explicit clause documenting that non-root `metadata` records are valid as session-boundary markers, MUST have a non-null `parent`, and that consumers without session-boundary semantics MAY treat them as no-ops.
- **`SECURITY.md` "Supported versions"** still listed `1.x` and `0.x`. Updated to `2.x` (active) + `1.x` (security fixes only on the latest 1.7.x line) + `0.x` (unsupported).
- **`SECURITY.md` overclaim about "end-to-end" private advisories** — softened. GitHub's private advisory channel is access-restricted but not cryptographic E2E (GitHub holds the data at rest). The doc now says it's the "preferred private reporting transport" and offers a separate cryptographic channel on request.
- **`ROADMAP.md` duplicated runtime enforcement and richer behavior contracts in BOTH "Shipping today" and "What's next"** — these shipped in v2.0.0. Removed from the "What's next" section, leaving only the truly outstanding items (streaming replay, multimodal, harness-diff, MCP-native replay, TypeScript streaming parity, tool-call pre-dispatch interception).
- **`ROADMAP.md` said "Eight importers"** but listed nine (Langfuse, Braintrust, LangSmith, OpenAI Evals, OTLP, MCP, A2A, Vercel AI SDK, PydanticAI). Off-by-one fixed.
- **`ROADMAP.md` MCP server bullet** listed five tools (diff, policy check, token diff, schema watch, summary). v1.7.2 added `shadow_certify` and `shadow_verify_cert`. Now lists all seven.
- **`ROADMAP.md` claimed Python and TypeScript auto-instrumentation "including the OpenAI Responses API and streaming"** — TypeScript explicitly passes streaming through unrecorded (`typescript/src/instrumentation.ts:10`). Bullet now states the gap honestly: Python covers streaming aggregation, TypeScript currently passes streaming through unrecorded. New roadmap entry "TypeScript SDK parity for streaming" tracks closing it.
- **CI `python-full-extras` job was installing only six extras** (`dev`, `anthropic`, `openai`, `otel`, `serve`, `embeddings`) — missed `mcp`, `sign`, and the three framework adapters (`langgraph`, `crewai`, `ag2`). Local `ci-local-extras` was already more complete. The CI job now installs every optional extra including `[sign]` under `--prerelease=allow`. This closes the "local parity stronger than GitHub CI" inversion.
- **README claimed the TypeScript SDK "works the same way"** as Python. Replaced with an explicit feature parity matrix that names every gap: TS streaming passes through unrecorded; runtime enforcement / certify / sign / replay / diff / bisect / mine / MCP server are Python-CLI-only. The `.agentlog` format itself is the contract — TS-recorded traces feed into Python's tooling without translation.

## [2.0.4] - 2026-04-25

### Fixed

- **`shadow.certify_sign` was breaking mypy `--strict` on CI.** The module's lazy `sigstore` imports raise `import-not-found` when sigstore isn't installed (which is the default — sigstore is gated behind the optional `[sign]` extra and additionally requires `--prerelease=allow` at install time because its dependency tree pulls pre-release wheels). CI doesn't install the `[sign]` extra; my local venv had sigstore from a manual install during v1.8 development, which is why ci-local was green locally while v1.8.0–v2.0.3 silently failed mypy on every CI run.

  Fix: add `shadow.certify_sign` to the existing `ignore_errors` mypy override block in `pyproject.toml`, alongside the other optional-extra-only modules (`shadow.serve.*`, `shadow.mcp_server`, `shadow.adapters.*`, `shadow.tools.sandbox`). Verified: with sigstore uninstalled locally, mypy `--strict` now passes; with sigstore installed, the imports type-check normally.

  This is the same local/CI parity drift class that v1.6.5's `ci-local` recipe was meant to prevent. The recipe's `python-full-extras` job installs every extra EXCEPT `[sign]` (because the `--prerelease=allow` flag complicates the install command), so CI exposed a mismatch the local recipe didn't. Worth a follow-up to extend `ci-local-extras` with a sigstore-install step under `--prerelease=allow`.

## [2.0.3] - 2026-04-25

### Fixed

- `docs/features/runtime-enforcement.md` described the enforcer's dedup key as `(rule_id, pair_index, detail)`. The v2.0.1 fix already changed the code to `(rule_id, pair_index)` only, but the docs still showed the old shape. Now corrected with the same explanation as the inline code comment — whole-trace rules embed running counts in the detail string, so detail-keyed dedup let them respam.

### Docs

- README hedges `causal bisection` from "isolates which specific change caused which specific regression" to "estimates which specific change most likely explains each regression, then points you at the replay / counterfactual primitives to confirm it." Matches the hedged terminal renderer already in the bisect command (the `est.` prefix and `(stable, CI excludes 0)` qualifiers shipped in v1.5).
- README runtime-enforcement headline rewords "block a violating response *as it happens*" to "block or replace a violating model response at record time". More precise — `EnforcedSession` evaluates after the model returned, not before tool dispatch.
- README CLI reference table for `shadow certify` and `shadow verify-cert` now mentions the v1.8 signing flags (`--sign`, `--verify-signature`, `--cert-identity`) so a reader scanning the table doesn't miss that signing is shipped.
- README `must_be_grounded` mention in the rule list now flags it as "cheap lexical grounding gate, not NLI-backed faithfulness" with a pointer to the docs page that documents what it catches and what it doesn't. Same hedging that v2.0.2 added to `docs/features/policy.md`, surfaced inline in the README.

## [2.0.2] - 2026-04-25

### Fixed

- **Stale rule-count strings** in three docs surfaces still said "9 rule kinds" or "Nine kinds ship today" after v2.0 added three new kinds (`must_remain_consistent`, `must_followup`, `must_be_grounded`):
  - `docs/features/policy.md` header
  - `docs/quickstart/ci.md` next-section link
  - `docs/reference/cli.md` `shadow diff --policy` description
  - `shadow.mcp_server` `shadow_check_policy` tool description
  All four now say "twelve" / "12" and list every kind.

### Docs

- **`must_be_grounded` honest scope** added to `docs/features/policy.md`. The rule is lexical-overlap, not semantic faithfulness or NLI-backed grounding. Now explicitly documents what it catches (off-topic responses) and what it doesn't (semantic-equivalent paraphrase with different vocabulary, citations with unsupported conclusions, claims a chunk contradicts). For deeper grounding, pair with the `Judge` axis or an external faithfulness evaluator. Treat the rule as a cheap CI gate, not a hallucination guarantee.
- **Runtime enforcement scope** now explicit in the README: `EnforcedSession.record_chat` evaluates AFTER the model response, not before tool dispatch. The README points users at the `enforcer.evaluate(records_so_far)` pattern between model response and tool dispatch when pre-tool blocking matters. Pre-dispatch interception via the auto-instrument layer is documented as roadmap.
- **README comparison table softened.** Cells for Langfuse / Braintrust / LangSmith on policy rules and merge-blocking moved from "no" to "partial via evals" / "partial via webhooks" with a one-line clarification under the table: those platforms support evals + webhooks + custom CI a team can wire into a PR-comment / gate workflow. Shadow's claim is that it ships the workflow as a single command and ships the trace format / policy language / release certificate as primitives, not that competitors can't be made to work. Self-hostable cell on Braintrust softened to "partial."

## [2.0.1] - 2026-04-25

### Fixed

- **`PolicyEnforcer` was respamming whole-trace rules every turn after they crossed.** The dedup key was `(rule_id, pair_index, detail)`. Whole-trace rules like `max_turns` and `must_call_once` embed a running count in their detail string ("trace has 5 turns; max is 4", then "trace has 6 turns; max is 4", etc.), so each subsequent turn produced a new detail and the enforcer reported it as new. Now keyed on `(rule_id, pair_index)` only — detail is human-output, not identity. Caught by the v2.0 real-LLM stress harness; existing 15 runtime tests still pass and a new regression test (`test_enforcer_whole_trace_rule_with_growing_count_does_not_respam`) locks the fix.
- New committed real-LLM stress harness at `examples/stress/run_stress.py` — 13 assertions against real OpenAI gpt-4o-mini covering `must_remain_consistent` against live agent behavior, `must_be_grounded` against real RAG context (both grounded and off-topic prompts), all three `EnforcedSession` modes (replace/raise/warn) verifying the on-disk trace shape, incremental violation detection across a 6-turn live trace, the certify+verify-cert pipeline against an `EnforcedSession` output, and three concurrent `EnforcedSessions`. 13/13 passes against real OpenAI in ~17 seconds at well under $0.05.

## [2.0.0] - 2026-04-25

Major version bump because v2.0 grows the SDK's public surface (new `shadow.policy_runtime` module with `EnforcedSession` / `PolicyEnforcer`). All v1.x APIs remain backwards-compatible — existing `Session`, `policy_diff`, `shadow diff --policy`, certificate workflow are unchanged. The major bump reflects the new public module, not a breaking change to existing code.

### Added

- **Three new policy rule kinds** for stateful and RAG-aware contracts:
  - `must_remain_consistent` — once a value at `path` is observed, every later pair where the path resolves must equal it. Useful for "the agent must not change the refund amount after confirming it." Pairs where the path is absent are skipped (absence ≠ change).
  - `must_followup` — when `trigger` conditions hold in pair N, pair N+1 must satisfy `must` (a tool call by name, or a text-includes substring). A trigger on the final pair is itself a violation. Captures patterns like "after a quote, the next turn must call `confirm_with_user`."
  - `must_be_grounded` — every response must overlap meaningfully with retrieved chunks at `retrieval_path`. Default `min_unigram_precision: 0.5` matches the no-LLM-judge fallback baseline used by RAGAS, TruLens, DeepEval. Tokenisation drops punctuation and len-1 tokens so an attacker can't satisfy the rule by emitting `the , .`. Twelve rule kinds total now.
- **Runtime policy enforcement** in new module `shadow.policy_runtime`:
  - `PolicyEnforcer(rules, on_violation=...)` evaluates rules incrementally on a growing record list and reports only NEW violations since the last call. Three modes: `replace` (default — swap the offending response for a refusal payload while preserving structural fields), `raise` (throw `PolicyViolationError`), `warn` (log only).
  - `EnforcedSession(enforcer=..., output_path=...)` extends `Session` and runs the enforcer on every `record_chat`. The flushed `.agentlog` is structurally valid even when responses were replaced — every existing Shadow command (`diff`, `verify-cert`, `mine`, `mcp-serve`) reads it without modification.
  - `Verdict` dataclass carries `(allow, replacement, reason, violations)`. `default_replacement_response` builds a refusal payload that preserves `model`, `usage`, `latency_ms` so downstream renderers don't break. Custom builders accepted via `replacement_builder=`.
  - API shape mirrors NeMo Guardrails / Bedrock Guardrails / Guardrails AI conventions: callback/verdict pattern, return-replacement default, raising opt-in. Researched against canonical guardrails-API patterns to pick the most-canonical shape.
- 36 new tests across `python/tests/test_policy_stateful_rag.py` (rule semantics) and `python/tests/test_policy_runtime.py` (enforcer + EnforcedSession). Covers happy paths, anchor pinning, absence-isn't-change, final-pair triggers, replace/raise/warn modes, custom replacement builders, incremental violation detection, and round-trip via disk.
- New docs page `docs/features/runtime-enforcement.md` covering the three modes, custom replacements, the programmatic API for callers not using `EnforcedSession`, and what the surface explicitly does NOT do (no tool-call interception, no network-level guardrails, no cross-process state). Wired into mkdocs `Features` nav.
- `docs/features/policy.md` updated with a "Stateful and RAG-aware rules" section covering the three new kinds with runnable examples.

## [1.8.0] - 2026-04-25

### Added

- **Cosign / sigstore keyless signing for Agent Behavior Certificates.** `shadow certify --sign` writes a sidecar `<output>.sigstore` Bundle containing the signature, the Fulcio-issued signing certificate, and a Rekor transparency-log entry. The signed payload is the canonical certificate body bytes — the same bytes `cert_id` hashes — so tampering breaks both content-id and signature. Optional via the `[sign]` extra (`pip install 'shadow-diff[sign]'`); unsigned certificates from prior versions still verify content-addressing as before.
- **`shadow verify-cert --verify-signature --cert-identity <email-or-workflow-url>`** binds verification to a specific signer identity. A leaked Bundle signed by another identity fails this check even if its cryptography is otherwise valid — the keyless flow's whole value is identity binding. Defaults to GitHub Actions OIDC issuer (`https://token.actions.githubusercontent.com`); override with `--cert-oidc-issuer` for other providers.
- New `shadow.certify_sign` module wraps the sigstore-python `Signer` / `Verifier` API and handles canonicalisation + identity policy. Eight new tests at `python/tests/test_certify_sign.py` cover canonical-body determinism (dataclass and dict forms produce identical bytes), `cert_id` exclusion (the body fingerprint must equal the body part of `cert_id`), sidecar path convention, sign-writes-bundle (with the sigstore boundary mocked), no-OIDC-token error path, missing/corrupt bundle, and the verify boundary's input-bytes contract. `pytest.importorskip("sigstore")` gates the file so the default install path stays sigstore-free.
- README + `docs/features/certificate.md` document the signing flow with both CI (GitHub Actions OIDC) and local (interactive browser) examples. Comparison-table row in README is now "Cosign-signed release certificate."

### Changed

- ROADMAP entry for the v1.8 signing layer is now in "Shipping today (v1.8.x)." Next-up sections cover runtime policy enforcement (a major-version surface change tracked for v2.0.0) and stateful / RAG-aware contracts (v1.9).

## [1.7.6] - 2026-04-25

### Fixed

- **README + `docs/features/hierarchical.md` policy examples used `severity: critical`** — not a valid severity. The loader silently stored unknown values as the raw string, and the `shadow diff --fail-on` gate's rank lookup fell through to a default, so a rule the user wrote as "block hard" never tripped a severe gate. Examples updated to `severity: error` (the documented value for a hard-block rule).
- **`load_policy` now validates `severity`** against `{info, warning, error}` and raises `ShadowConfigError("policy rule #N has invalid severity 'X'")` at load time. Three new tests cover the rejection, the three valid values, and the default. The validation makes the v1.6.5 `--fail-on` gate actually trip on the rule the user wrote, instead of silently downgrading to info.
- **README comparison table said "Signed release certificate"** — overclaim. The certificate is content-addressed and self-verifying; PKI/cosign signing is on the roadmap, not shipped. Row corrected to "Content-addressed release certificate."

## [1.7.5] - 2026-04-25

### Fixed

- `docs/features/mcp.md` (the MCP importer page) now cross-links to `mcp-server.md`. v1.7.4 added the server→importer link but missed the symmetric one — readers landing on the importer page wouldn't discover the server page.
- `docs/quickstart/ci.md` was anchored to the v1.6 era CI workflow and never picked up the v1.6.5 `--fail-on` flag. Quickstart readers got the same non-gating workflow that v1.7.2 fixed in `shadow init --github-action`'s template. Now includes a "Gating the merge on regressions" section with the recommended `--fail-on severe` step.
- `docs/quickstart/ci.md` "Next" section now links to the policy and certificate feature pages added in v1.7.2.

## [1.7.4] - 2026-04-25

### Added

- **`docs/features/mcp-server.md`** — dedicated docs page for running Shadow as an MCP server (`shadow mcp-serve`). The existing `features/mcp.md` covers the MCP *importer* (ingesting MCP traces into `.agentlog`); the new page covers the reverse direction (Shadow exposing its analyses *as* MCP tools to clients like Claude Desktop, Cursor, Zed). Documents all seven tools with per-tool purposes and a typical agentic-CLI session. Wired into the `Features` mkdocs nav.

### Fixed

- README's `## Use Shadow from an agentic CLI (MCP server)` section listed only the original five tools — missing `shadow_certify` and `shadow_verify_cert` from v1.7.2. Same drift class as the docs/reference/cli.md fix in v1.7.3.
- README's policy and certificate sections now deep-link to `docs/features/policy.md` and `docs/features/certificate.md`. Without those links the new feature pages were unreachable from the README.

## [1.7.3] - 2026-04-25

### Fixed

- `docs/reference/cli.md` had drifted four releases behind the CLI: missing entries for `shadow mine`, `shadow mcp-serve`, `shadow certify`, `shadow verify-cert`. Added all four. Also corrected a stale "8 rule kinds" line under `shadow diff --policy` (now nine, including `must_match_json_schema`) and added the `--fail-on` flag with explanation.
- `shadow mcp-serve` reference section enumerates all seven MCP tools, including `shadow_certify` and `shadow_verify_cert` (added in v1.7.2).

## [1.7.2] - 2026-04-25

### Added

- **MCP server gains `shadow_certify` and `shadow_verify_cert`** so agentic CLIs (Claude Desktop, Claude Code, Cursor, Zed, Windsurf, any MCP-aware client) can generate and verify Agent Behavior Certificates over the protocol — same arguments and contract as the CLI commands. Tool-handler registry, tool descriptors, and module docstring all updated. Five new tests cover the round-trip.
- **`shadow init --github-action` template includes a commented-out merge-gate step** so freshly scaffolded workflows can opt into `--fail-on severe` without rewriting the YAML. Default behaviour is still non-blocking; uncommenting one step turns Shadow into a required check.
- **mkdocs site adds two new feature pages** — `docs/features/policy.md` (the nine rule kinds, conditional `when:` operators, structured-output assertions, severity → `--fail-on` mapping, scope) and `docs/features/certificate.md` (ABOM format, generate/verify workflow, what it proves vs. what it doesn't, MCP integration, format stability). Both wired into the `Features` nav.

### Fixed

- MCP server's `shadow_check_policy` description listed eight rule kinds; now lists nine, including `must_match_json_schema` (the rule landed in v1.7.0 but the MCP description didn't).

## [1.7.1] - 2026-04-25

### Fixed

- `examples/stress/run_stress.py` had 5 mypy `--strict` errors that ci-local missed because the harness was outside mypy scope. Same drift pattern caught for `stress` in v1.6.4 — applied here too. Type signatures fixed; the harness still passes 26/26 at runtime.
- `just ci-local` and `.github/workflows/ci.yml` mypy scope now covers `examples/stress/run_stress.py`. Future stress-harness changes are caught by both local and CI mypy.

### Docs

- README now documents `must_match_json_schema` (with example), `--fail-on` for `shadow diff`, and the `shadow certify` / `shadow verify-cert` workflow. Comparison table includes "Merge-blocking CI gate" and "Signed release certificate" rows. CLI reference table covers the new commands. The previous 1.7.0 README still listed only eight policy rule kinds — fixed to nine.

## [1.7.0] - 2026-04-25

### Added

- **`must_match_json_schema` policy rule kind.** Asserts that every response's text content parses as JSON and validates against a supplied JSON Schema. Accepts either an inline `schema:` dict or a `schema_path:` to a JSON Schema file. Mismatches surface with the offending dotted path (e.g. `json schema mismatch at properties.amount: ...`). This closes the most common gap in v1.6.x policies for agents that produce structured output. Uses `jsonschema>=4.0` (now a runtime dependency).
- **Agent Behavior Certificate (ABOM)** via `shadow certify` and `shadow verify-cert`. Generates a content-addressed JSON release artefact that captures `agent_id`, `released_at`, the trace's content-id, all distinct models observed, content-ids of all distinct system prompts, content-ids of every tool schema, optional `policy_hash` (sha256 of the policy file), and optional `regression_suite` (the nine-axis severity rollup vs a baseline trace). The certificate is self-verifying: `shadow verify-cert release.cert.json` recomputes the body's hash and exits 1 on mismatch, so it can run as a release gate. PKI / cosign signing lands in v1.8 — the format is stable today, signing layers on top.

### Fixed

- `must_match_json_schema` was accepting `NaN`, `Infinity`, and `-Infinity` because Python's `json.loads` accepts them as a CPython extension. Those literals are NOT valid JSON per RFC 8259 and downstream consumers (browsers, other-language parsers, strict JSON consumers) will choke on them. The rule now rejects them with a clear "non-standard JSON literal" violation. Caught by the new adverse-stress harness in `examples/stress/run_stress.py`.

### Tests

- 12 new tests for `must_match_json_schema` (valid JSON passes, malformed JSON / schema mismatch / empty text / both-or-neither schema params / external schema_path / policy_diff regressions / NaN-Infinity rejection).
- 13 new tests for the certificate module: build extracts models/prompts/tools, self-verifies, tampering breaks verification, unsupported `cert_version` rejected, optional policy hash + baseline regression suite, CLI `certify` writes JSON, CLI `verify-cert` exits 0 on valid / 1 on tampered.
- New `examples/stress/run_stress.py` adverse harness — 26 assertions covering 12 malformed-JSON variants, unicode/RTL/emoji payloads, 62 KB / 2000-item payload scaling, `oneOf`/`$ref` schemas, invalid-schema short-circuiting, pathological `schema_path` cases, 50-thread concurrent validation, deterministic certificate builds with fixed timestamp, 20-thread concurrent builds producing identical `cert_id`, all 9 per-field tamper detections, `cert_id`-only swap detection, round-trip via disk, forward-compat with unknown fields, version + format rejection, 100-turn trace certification scaling.

## [1.6.5] - 2026-04-25

### Added

- **`shadow diff --fail-on {minor,moderate,severe}`** — exits non-zero when the worst axis severity *or* a policy regression reaches the threshold. The diff report and policy summary are still printed (and the JSON output is still written) before the gate fires, so blocked PRs always see the explanation. Default remains `never` (post the report, exit 0). Use `--fail-on severe` in CI to convert Shadow from "shows you a diff" to "blocks the merge."
- **GitHub Action gains a `fail-on` input** plumbed through to `shadow diff`. The PR comment is posted first, *then* the gate runs as a separate step, so blocked PRs always have the comment that explains why. New optional `policy` and `shadow-version` inputs too. Action defaults remain non-blocking, so existing consumers don't suddenly fail.

### Fixed

- **GitHub Action install was broken.** The composite action attempted `pip install shadow==0.1.0` — wrong package name (Shadow ships as `shadow-diff` on PyPI) and a version that was never published. External consumers always silently fell through to the in-tree fallback, which only works when running the action *from this repo*. Install line now uses `shadow-diff` (current latest) with optional pinning via the new `shadow-version` input.
- **`ROADMAP.md` was anchored to v1.2.x** and listed sandboxed deterministic agent-loop replay under "What's next" even though it shipped in v1.6.0. Section header is now "Shipping today (v1.6.x)"; sandboxed replay, tool backends, novel-call policies, counterfactual primitives, conditional `when:` policies, framework adapters, importers, MCP server, trace mining, and PyPI Trusted Publisher are all in the shipping list. The "What's next" section now reflects the real outstanding work (streaming replay, multimodal traces, harness-diff instrumentation, MCP-native replay, runtime policy enforcement, richer behaviour contracts, ABOM). Added a "Not on the roadmap" entry making the sandbox's "best-effort isolation, not a security boundary" framing explicit.

## [1.6.4] - 2026-04-25

### Fixed

- `examples/stress/run_stress.py` — corrected `record_baseline`'s declared return type (was `list[dict[str, Any]]`, actually returned a `(records, summary)` tuple), removed two stale `# type: ignore[arg-type]` comments, and added explicit type annotations on baseline-construction dicts. The harness ran correctly at runtime but the type signatures lied; mypy can now actually help.

### Changed

- `just ci-local` and `.github/workflows/ci.yml` mypy scope now includes `examples/stress/run_stress.py`. Previously only `examples/demo/agent.py` and `examples/demo/generate_fixtures.py` were type-checked, so the committed stress harness drifted unchecked. Future stress-harness changes are now caught by both local and remote CI.

## [1.6.3] - 2026-04-25

### Fixed

- **`OpenAILLM` was dropping `tool_calls` and `tool_call_id` from messages on follow-up requests.** The agent-loop engine emits assistant messages of the form `{role:"assistant", content:"", tool_calls:[…]}` (the OpenAI wire shape). The converter's early-return path for string `content` was returning before forwarding `tool_calls`. The very next request — carrying the `role:"tool"` follow-up — was rejected by the API with HTTP 400 *"messages with role 'tool' must be a response to a preceding message with 'tool_calls'"*. This blocked every real-world OpenAI agent-loop replay past the first tool round-trip. The converter now forwards both fields regardless of `content` shape.
- Found by an end-to-end stress test against real `gpt-4o-mini` (`examples/stress/run_stress.py`) — 25 adverse-condition assertions covering branch_at_turn mid-trajectory, replace_tool_result re-drive with hostile output, replace_tool_args under sandbox redispatch, hostile-tool sandbox (socket / subprocess / write), max_turns truncation under runaway, four novel-call policies, five concurrent branches, long-trace truncation, empty seed, and past-end branch. The harness went from 20/24 (broken at OpenAI handoff) to 25/25 once the converter was fixed.

### Added

- `python/tests/test_openai_backend.py` — five focused unit tests covering the converter's tool-calls forwarding, including the exact shape the agent-loop engine produces. Locks the regression.
- `examples/stress/run_stress.py` — runnable real-LLM adverse stress harness. Gated behind `SHADOW_RUN_NETWORK_TESTS=1` and `OPENAI_API_KEY`; skips otherwise. Costs well under $0.05 per run against gpt-4o-mini.

## [1.6.2] - 2026-04-25

### Fixed

- `drive_loop_forward` now returns `AgentLoopSummary` (a public type) instead of the private `_SessionStats`. The function was added to `__all__` in 1.6.1 but leaked an internal struct, which made it impossible to type-annotate user code that consumed it. The internal `_accumulate(summary, stats)` helper in `shadow.counterfactual_loop` is replaced by `_merge(summary, addend)` which sums two public summaries.
- `CHANGELOG.md` v1.6.1 was incorrectly dated 2026-04-24 (predating v1.6.0 on the same day). The 1.6.1 section is unchanged in content.

### Added

- Direct contract tests for `drive_loop_forward`: `test_drive_loop_forward_returns_public_summary_type` (verifies the public-type return, parent chaining, and content-addressing) and `test_drive_loop_forward_truncation_surfaces_in_summary` (verifies `sessions_truncated` propagates from inner stats to the public summary).
- `just ci-local` now also runs the `python-full-extras` job locally — installs every optional extra (`anthropic`, `openai`, `otel`, `serve`, `mcp`, `embeddings`) and re-runs pytest with no `--ignore` filter so optional-extras gating bugs (the kind that bit v1.4.1 with `mcp`) get caught before pushing.

## [1.6.1] - 2026-04-25

### Fixed

- `branch_at_turn(turn=N)` for N≥1 now preserves the baseline prefix verbatim (with content-addressed ids intact through end of turn N, including any trailing `tool_call`/`tool_result` records) and then drives the agent loop forward from turn N+1's seed messages. Earlier behaviour stopped after the prefix and never produced the candidate continuation, so the docstring's promise of "branch and replay forward" was a no-op.
- `replace_tool_result` re-drive mode (when `llm_backend` is supplied) preserves the prefix through the patched `tool_result`, then continues forward from that point. Previously it re-drove from turn 0, which broke `MockLLM` lookups (the patched tool message changed the next request's content-id) and produced a trace whose prefix did not match the baseline.
- `branch_at_turn(turn=K)` where K exceeds the baseline's turn count now raises `ShadowConfigError("baseline has fewer than K …")` instead of silently emitting a stub trace.
- New `drive_loop_forward` primitive in `shadow.replay_loop` is the shared driver these counterfactual helpers now use; existing `run_agent_loop_replay` is unchanged.

### Added

- **Local CI parity script.** `just ci-local` runs the exact command set from `.github/workflows/ci.yml` in the same order — including the `python/ examples/` ruff/mypy scope and the demo step — so drift between local and CI lint scope is caught before pushing. Catches the three classes of failure that bit prior releases: ruff/mypy scope, optional-extras gating, and demo wall-clock regressions. The recipe is portable across macOS (uses `gtimeout` if installed, falls back to plain bash) and Linux.
- `just lint-python` was widened to match CI scope (`python/ examples/`, plus the demo entry points for mypy). Previously narrower than CI, so `just ci` could pass locally and still fail on push.
- New tests:
  - `test_branch_at_turn_one_replays_prefix_then_drives_forward` — verifies the prefix is preserved with content-addressed ids and that forward-drive emits at least one additional chat pair.
  - `test_branch_at_turn_past_end_raises` — verifies the new bounds-check error.
  - `test_replace_tool_result_redrive_preserves_prefix_then_drives_forward` — verifies prefix `chat_request` ids carry through verbatim.
  - `test_delegate_policy_can_bridge_to_sandboxed_backend` — pinpoints the documented composition pattern: novel calls flow through `DelegatePolicy` into `SandboxedToolBackend.execute`.
  - `test_engine_handles_multi_session_baseline` — multi-session baselines (two `metadata` records) now have an explicit assertion that both sessions replay end-to-end.
  - `test_replay_loop_live.py` — real-LLM end-to-end test against `gpt-4o-mini`, gated behind `SHADOW_RUN_NETWORK_TESTS=1` so CI never opts in. Asserts the agent-loop engine produces a structurally valid, content-addressed trace from a live API call.

637 pytest tests pass, 205 cargo tests pass, full CI parity script (`just ci-local`) is green on macOS.

## [1.6.0] - 2026-04-25

### Added

**Sandboxed deterministic agent-loop replay.** A new replay mode that drives the candidate's agent loop forward against a baseline — same shape as a real agent run, no real network calls, no real database writes, no real charges. The output is an ordinary `.agentlog` so every existing Shadow command (`diff`, `check-policy`, `mine`, `mcp-serve`, `bisect`) reads it without modification.

- New `shadow.tools` package mirrors `shadow.llm`: `ToolBackend` Protocol with three implementations.
  - `ReplayToolBackend` indexes baseline `tool_result` records by `(tool_name, canonical_args_hash)` and serves them back. Default for `shadow replay --agent-loop`.
  - `SandboxedToolBackend` wraps user tool functions; blocks `socket.connect`, `subprocess.run` / `Popen` / `os.system` / `os.execvp`, and write-mode `open()` calls (redirected to a tempdir). Optional `freeze_time` pins `time.time` and `datetime.utcnow`. Best-effort isolation for replay determinism, not a security boundary.
  - `StubToolBackend` returns deterministic placeholders. For tests and the `stub` novel-call policy.
- New `shadow.replay_loop` module: `run_agent_loop_replay(baseline, llm_backend, tool_backend)` drives the loop forward with a `max_turns` safety cap and a structured `AgentLoopSummary` of stats. Errors from a tool backend become `is_error=True` `tool_result` records by default; runaway loops emit an `error` record with `code=loop_max_exceeded`.
- New `shadow.tools.novel` module: four configurable policies for tool calls the baseline never recorded — `StrictPolicy` (raise), `StubPolicy` (placeholder), `FuzzyMatchPolicy` (Jaccard distance over arg keys), `DelegatePolicy` (defer to a user-supplied async callable).
- New `shadow.counterfactual_loop` module: `replace_tool_result`, `replace_tool_args`, `branch_at_turn`. Each produces a `CounterfactualLoopResult` carrying the new trace, summary, and an `override` dict describing the substitution. These are the rails the bisect renderer's "confirm with `shadow replay`" caveat references.
- `shadow replay` gains `--agent-loop`, `--tool-backend {replay|stub|sandbox}`, `--novel-tool-policy {strict|stub|fuzzy}`, and `--max-turns` flags.
- New docs page (`docs/features/sandboxed-replay.md`) and a runnable worked example at `examples/sandboxed-replay/run.py` (no API keys needed).

48 new tests across the new modules. 632 pytest, 205 cargo, mypy --strict / ruff / clippy / fmt all clean. Coverage 85.56%.

## [1.5.0] - 2026-04-25

### Added

- **Conditional policy rules.** Every rule can now carry a `when:` clause that gates it on a list of field-path conditions; the rule only fires on pairs whose request/response context satisfies every condition. Path is a dotted expression rooted at a per-pair context (`request.*`, `response.*`, plus `model` and `stop_reason` aliases). Operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not_in`, `contains`, `not_contains`. Multiple conditions AND together. Missing paths silently don't match instead of crashing. This unlocks rules like "high-value refunds must call confirm first" without forking the policy file by amount bucket.
- **`shadow bisect` terminal renderer with hedged language.** The CLI now renders attribution output through a dedicated formatter that prefixes percentages with `est.`, replaces the bare `✓` with explicit `(stable, CI excludes 0)` / `(screening only)` / `(weak signal)` qualifiers, labels brackets as 95% bootstrap CIs, and leads every output with a one-line caveat noting attribution is correlational, not causally proven (confirm with `shadow replay`). New `--format terminal|markdown|json` flag; previous JSON shape is preserved under `--format json`.

### Changed

- **README**: framework users are now pointed at the matching adapter as the more stable instrumentation surface than auto-instrument's monkey-patching of provider SDK `.create` methods.
- **`docs/features/bisect.md`**: example output updated to match the new hedged renderer; reading-the-signal section explains each row's qualifier instead of claiming `✓` means "proven."

### Notes

- The provider SDK upper bounds (`anthropic<1`, `openai<3`) are intentionally aligned with the next major because that's where `.create` class paths can move and our auto-instrument patches can silently break. Users on a major above the cap can lift the pin in their own pyproject and report breakage.

## [1.4.1] - 2026-04-24

### Changed

- `shadow-diff[langgraph]` now pulls `langchain-openai>=0.2,<2` alongside `langchain-core` and `langgraph`. Most LangGraph users pick ChatOpenAI as their chat provider and were hitting `ModuleNotFoundError` on first run. The adapter is still provider-neutral; users on Anthropic or Bedrock add `langchain-anthropic` / `langchain-aws` alongside without conflicts.

### Fixed

- CI coverage gate was failing at 83.97% on the default extras-less matrix because `shadow.mcp_server` counted against the denominator while its tests skipped. Added it to the `[tool.coverage.run].omit` list (same pattern as adapters, enterprise, serve, embeddings). Local run now reports 85.74%.
- `test_mcp_server.py` now skips cleanly via `pytest.importorskip("mcp")` when the `[mcp]` extra is absent, instead of raising `ModuleNotFoundError` at import time and failing the whole job.
- `ShadowCrewAIListener(quiet_internal_listeners=True)` detaches CrewAI's built-in `TraceCollectionListener.on_crew_started` and sibling handlers that raise `'str' object has no attribute 'id'` when synthetic events drive the bus. Production `Crew.kickoff()` paths are untouched.

### Docs

- New `docs/features/adapters.md` covering LangGraph, CrewAI, and AG2 adapters in depth. Wired into the mkdocs nav.
- README has a "Record from agent frameworks" section with runnable snippets for each adapter and an "Import traces from any OpenTelemetry backend" section noting GenAI semconv v1.40 support.

## [1.4.0] - 2026-04-24

### Added

**OTel GenAI semconv v1.40 compliance.** The `shadow import --format otel` importer now parses the full v1.40 attribute surface: structured `gen_ai.input.messages` / `gen_ai.output.messages` (whether carried as span attributes or inside the `gen_ai.client.inference.operation.details` event), `gen_ai.provider.name`, cache-token attributes (`gen_ai.usage.cache_read.input_tokens` and `cache_creation.input_tokens`), `gen_ai.response.id`, `gen_ai.output.type`, `gen_ai.conversation.id`, `gen_ai.tool.definitions`, agent spans (`create_agent`/`invoke_agent` land in metadata with id/name/description/version), and evaluation events. The deprecated v1.28-v1.36 flat-indexed `gen_ai.prompt.N.*` / `gen_ai.completion.N.*` shape still parses, so traces from OpenLLMetry and other implementers that haven't tracked the v1.37 restructure round-trip cleanly. Spans are sorted by `startTimeUnixNano` so content IDs are deterministic.

**Framework adapters: LangGraph, CrewAI, AG2.** New `shadow.adapters` package exposes first-class tracing hooks for the three dominant agent frameworks as of April 2026. Each routes its framework's native instrumentation surface through `Session.record_chat` / `record_tool_call` / `record_tool_result`.

- `shadow.adapters.langgraph.ShadowLangChainHandler` — an `AsyncCallbackHandler` subclass. Hooks: `on_chat_model_start`/`end`/`error`, `on_tool_start`/`end`/`error`. Pair-buffers by LangChain's `run_id` so concurrent graph branches don't cross-contaminate. Session-groups by the config's `configurable.thread_id`.
- `shadow.adapters.crewai.ShadowCrewAIListener` — a `BaseEventListener` subclass wired to `LLMCall{Started,Completed,Failed}` and `ToolUsage{Started,Finished,Error}`. Pairs via CrewAI's `call_id` (LLM) and `started_event_id` (tools) conventions.
- `shadow.adapters.ag2.ShadowAG2Adapter` — uses AG2's `register_hook` surface on `safeguard_llm_inputs`/`safeguard_llm_outputs`. Captures message bodies that the default `autogen.opentelemetry` spans redact. Supports per-agent and `install_all(agents)` registration.

New extras: `pip install 'shadow-diff[langgraph]'`, `[crewai]`, `[ag2]`.

### Fixed

Three correctness gaps surfaced by a real-world MCP adversarial test on 10 support-triage tickets.

- **Session-scoped policy rules.** `must_call_before` and the other rule kinds now accept `scope: session` in the policy YAML. Under session scope the rule is evaluated independently within each user-initiated session — inferred from `messages[-1].role == "user"` on the request — so a correct ordering in one ticket can no longer mask violations in later tickets of the same multi-ticket trace. Default stays `scope: trace` for back-compat; the MCP tool description and policy loader both document the new field. Real-world test: the adversarial candidate trace went from 0 reported violations (bug) to 6 (correct — one per offending ticket).
- **Cost axis `no_pricing` flag.** When no pricing table is supplied, or the traced models aren't in the table, the cost axis previously reported `delta=0, severity=none` — indistinguishable from "both sides priced equally." The axis now emits `flags: ["no_pricing"]` when fewer than half the pairs can be priced. The summariser surfaces this as "per-call cost not comparable (no pricing table supplied)" rather than omitting cost silently.
- **Mining metadata field name.** `MiningResult.to_agentlog()` now writes `cases_selected` (was `selected_cases`), aligning with the sibling keys `total_input_pairs` and `clusters_found`.
- **Explicit session markers for adapter traces.** `Session.record_metadata(payload)` appends an authoritative session-boundary marker. When a trace contains two or more metadata records, Shadow's session detector uses them exclusively instead of falling back to the stop-reason heuristic. The CrewAI adapter now emits a marker on every `CrewKickoffStartedEvent`, so one kickoff equals one session even when every `LLMCallCompleted` ends with `end_turn`. Verified end-to-end on real GPT-4o-mini traces: 3 topics run through CrewAI produce 3 sessions (previously fragmented).

## [1.3.0] - 2026-04-24

### Added

Three features aimed at the April 2026 agent ecosystem.

#### `shadow mcp-serve`: run Shadow as a Model Context Protocol server

Shadow now speaks MCP. Any MCP-aware client (Claude Desktop, Claude Code, Cursor, Zed, Windsurf, and others) can connect over stdio and invoke Shadow's capabilities as tools. Five tools exposed:

- `shadow_diff`: nine-axis diff between two `.agentlog` files, with optional policy enforcement
- `shadow_check_policy`: run a YAML/JSON policy against two traces
- `shadow_token_diff`: per-dimension token distribution summary
- `shadow_schema_watch`: classify tool-schema changes before replay
- `shadow_summarise`: plain-English summary from a saved DiffReport

Wire it into a client's config:

```json
{ "shadow": { "command": "shadow", "args": ["mcp-serve"] } }
```

Install the extra: `pip install 'shadow-diff[mcp]'`.

#### `shadow import --format a2a`: Agent-to-Agent session logs

The A2A protocol (Linux Foundation, used in production at Microsoft, AWS, Salesforce, SAP, ServiceNow) is the agent-to-agent companion to MCP. Shadow's A2A importer:

- Reads JSONL, JSON-array, and wrapped session shapes
- Maps `tasks/send` and `tasks/result` pairs into `chat_request` and `chat_response`
- Extracts Signed Agent Cards (A2A v1.0) into the metadata payload
- Captures error responses as `stop_reason=error`

The resulting `.agentlog` plugs into every existing Shadow command (`diff`, `bisect`, `policy`, `suggest-fixes`).

#### `shadow mine`: turn production traces into regression suites

```bash
shadow mine production.agentlog --output suite.agentlog --max-cases 50
```

Clusters turn-pairs by tool sequence, stop reason, verbosity, and latency. Picks the most interesting representative from each cluster (errors, refusals, high cost, heavy reasoning, empty or very long responses). Writes a new `.agentlog` suitable as a committed baseline for CI. Optional `--pricing pricing.json` biases the selection toward expensive pairs.

### Tests

- 31 new tests covering MCP server, A2A importer, and mine
- Full pytest suite: 519 passed (up from 488)
- MCP server verified live over stdio: initialize, notifications/initialized, tools/list return 5 tools cleanly
- cargo test, mypy --strict, ruff, ruff format --check, clippy -D warnings, cargo fmt --check all clean

## [1.2.4] - 2026-04-24

### Fixed, fourth discrepancy-sweep pass (deepest)

A last deep audit turned up real user-facing drift in docs + metadata:

#### Docs-site was missing every v1.2 feature

- **`docs/reference/cli.md`** didn't document any v1.2 flag: no
  `--token-diff`, `--policy`, `--suggest-fixes`, `--partial`,
  `--branch-at`, `vercel-ai`, `pydantic-ai`. Users reading the
  published CLI reference couldn't find half the v1.2 release's
  features. Rewrote with full v1.2 coverage.
- **`docs/features/hierarchical.md`** claimed **"token-level: deferred
  to v1.1+"**, but we shipped token-level in v1.2.0. The table now
  documents all six real levels (trace/session/turn/span/token/policy)
  with actual shipped-in-version labels, plus dedicated sections for
  the two v1.2 levels covering CLI flags, rule kinds, and scale.
- **`docs/index.md` Highlights** was pre-v1.2: no mention of token/
  policy diff, partial replay, Vercel AI + PydanticAI importers,
  LLM-assisted fixes, or Python 3.13 support. Updated.

#### SPEC.md drift from code

- `replay_summary` §4.7 documented a `candidate_config_hash` field
  that the implementation has **never** emitted; the actual field is
  `backend_id`. Also missing the v1.2 `branch_at` + `prefix_turn_count`
  optional fields for partial replay. SPEC now documents the real
  shape with both required and optional fields in a proper table.

#### Platform-claim alignment

- `pyproject.toml` classifiers claimed **Linux + macOS** only but
  Shadow has shipped a working **Windows CI matrix since v1.1**. Added
  `Operating System :: Microsoft :: Windows`.
- `pyproject.toml` classifies **Python 3.13** support but CI didn't
  test it. Added `python: "3.13"` for all three OSs in the CI matrix
  (9 pytest jobs now: 3 OS × 3 Python versions). Verified Shadow
  1.2.x imports and runs cleanly under a fresh Python 3.13 venv.

#### TypeScript license inconsistency

- `typescript/package.json` was licensed `"MIT"` only while Rust +
  Python packages are dual-licensed `"MIT OR Apache-2.0"`. SPDX form
  `"(MIT OR Apache-2.0)"` now matches.

#### Stale version references

- `.github/ISSUE_TEMPLATE/bug_report.yml` placeholder said
  `"shadow 0.1.0"` → now `"1.2.4"`.
- `docs/PYPI-PUBLISHING.md` example used `v0.2.1` as the tag to push
  → now `v1.2.4`.

### Added, crates.io publish (gated, auto-skip if token absent)

Noticed during this audit: `shadow-core` is published on crates.io at
**v0.1.0** and has never been updated, the release workflow builds a
Rust source tarball but doesn't push to crates.io. Added a new
`publish-crates` job to `.github/workflows/release.yml` that:

- Runs on every `v*` tag push after `sign-and-release`.
- Emits a `::notice::` and exits clean if `CARGO_REGISTRY_TOKEN` is
  not configured as a repo secret (so the release pipeline still
  succeeds).
- Publishes `shadow-core` via `cargo publish` when the token is set.

Maintainers can enable crates.io auto-publish by adding the
`CARGO_REGISTRY_TOKEN` secret to the repo, no workflow change
needed.

### Verified

- 488 pytest green
- Python 3.13 compat confirmed against the PyPI 1.2.2 wheel in a
  fresh venv
- `cargo test` + clippy + fmt clean
- mypy `--strict`, ruff check + format clean
- Docs auto-sync (`CHANGELOG.md` / `SECURITY.md` → `docs/`) continues
  to run on every docs build

## [1.2.3] - 2026-04-24

### Fixed, third discrepancy-sweep pass (deep dependency + docs audit)

A deeper cross-repo audit turned up two more real issues:

#### Dependency pins were dangerously stale

The runtime + optional-extra dependency pins were exact-pinned to
April-2025-era versions:

- `anthropic==0.40.0` → current PyPI is `0.97.0` (57 minor versions
  behind, **incompatible with users on modern `anthropic`**)
- `openai==1.58.1` → current PyPI is `2.32.0` (**one major version
  behind**, users with `openai>=2` couldn't install `shadow[openai]`)
- `pydantic==2.10.3` → current `2.13.3`
- `httpx==0.28.1`, `rich==13.9.4`, `scikit-learn==1.6.0`,
  `numpy==2.2.0`, `pyyaml==6.0.2`, all exact pins

**Verified** Shadow works correctly against `anthropic 0.97` +
`openai 2.32` + `pydantic 2.13` (all module paths Shadow monkey-patches
still exist: `anthropic.resources.messages.Messages`,
`openai.resources.chat.completions.Completions`,
`openai.resources.responses.Responses`). Loosened to permissive ranges:

| Dependency | Old | New |
|---|---|---|
| `anthropic` | `==0.40.0` | `>=0.40,<1` |
| `openai` | `==1.58.1` | `>=1.58,<3` |
| `pydantic` | `==2.10.3` | `>=2.10,<3` |
| `httpx` | `==0.28.1` | `>=0.27,<1` |
| `rich` | `==13.9.4` | `>=13.9,<15` |
| `scikit-learn` | `==1.6.0` | `>=1.6,<2` |
| `numpy` | `==2.2.0` | `>=2.2,<3` |
| `pyyaml` | `==6.0.2` | `>=6.0,<7` |
| `sentence-transformers` | `==3.3.1` | `>=3.3,<6` |
| `opentelemetry-sdk` | `>=1.27.0` | `>=1.27,<2` |
| `fastapi` | `>=0.115.0` | `>=0.115,<1` |
| `uvicorn` | `>=0.32.0` | `>=0.32,<1` |
| `websockets` | `>=13.1` | `>=13.1,<16` |

Dev deps (`hypothesis`, `mypy`, `ruff`, `pytest`, `pytest-asyncio`,
`pytest-cov`, `maturin`, `types-PyYAML`) stay exact-pinned for CI
reproducibility.

#### Docs-site drift

- **`docs/changelog.md` was a stale copy** of root `CHANGELOG.md`,
  missing every v1.2.x entry. Published docs site at
  [manav8498.github.io/Shadow](https://manav8498.github.io/Shadow/)
  was showing a changelog stuck at v1.1.0. Now re-synced and the docs
  workflow mirrors root → `docs/` on every build so this won't drift
  again (`.github/workflows/docs.yml`).
- **`ROADMAP.md` was written from a pre-1.0 perspective**, calling
  things already shipped (live backends, LASSO bisection,
  auto-instrumentation, OTel, 10 judges, Windows CI, PyPI pipeline)
  "planned for v0.2" and ending with *"Shadow is a v0.1.0 project
  with early adopters welcome"*. Rewritten to reflect actual v1.2.x state
  with accurate "shipped" vs "next up" sections.
- **`SECURITY.md`**: version-reference bumped to `v1.2.x`.
- **`examples/README.md`** falsely claimed every subdirectory ships a
  `WALKTHROUGH.md` and uses a `generate_fixtures.py` recipe. Only 4 of
  9 fit that shape. Prose rewritten to match reality; "the other four
  directories" section added for `edge-cases/`, `acme-extreme/`,
  `integrations/`, `judges/`.

### Verified

- 488 pytest green against the loosened dep set
- 30/30 TypeScript tests green
- `cargo test` + clippy + fmt clean
- mypy `--strict`, ruff check + format clean
- Smoke-tested a fresh venv install with the latest `anthropic 0.97`
  + `openai 2.32` + `pydantic 2.13`, all Shadow imports and backend
  instantiations work, instrumentation module paths resolve.

## [1.2.2] - 2026-04-24

### Fixed, second discrepancy-sweep pass

Caught during a deeper post-1.2.1 audit across every version string,
command-reference, and doc in the repo:

- **`CITATION.cff`**, version `0.1.0` → `1.2.2` (in both top-level
  and `preferred-citation`). Academic citations produced via GitHub's
  "Cite this repository" button will now pin the actual released
  version. Release date also updated.
- **TypeScript SDK (`typescript/src/session.ts`)**, hardcoded
  `version: '0.1.0'` in emitted metadata records replaced with a
  module-load-time read from the shipped `package.json`. Every
  `.agentlog` the TS SDK writes now carries accurate SDK provenance.
- **`examples/README.md`**, four example directories existed on
  disk but weren't listed: `acme-extreme/`, `judges/`,
  `mcp-session/`, `integrations/`. All four added with their real
  scope descriptions.
- All package versions (Cargo, pyproject, TS package, README badge,
  CITATION.cff) bumped to `1.2.2` in lockstep.

No functional changes to CLI or library APIs.

## [1.2.1] - 2026-04-24

### Fixed, maturity + documentation discrepancies

A pass against the repo audit for discrepancies between what the
package *is* and what it *says about itself*:

- **PyPI classifier** `Development Status :: 3 - Alpha` → `4 - Beta`.
  A 1.x release with 90 days of feature work behind it isn't Alpha;
  calling it Alpha undersells maturity for package discovery.
- **GitHub Action template fix** *(functional bug).* The workflow
  that `shadow init --github-action` scaffolds pinned
  `shadow-diff>=0.2,<0.3`, which would install a pre-1.0 CLI into
  every user's CI. Now pins `shadow-diff>=1.2,<2`. Users who already
  ran `shadow init --github-action` before 1.2.1 should update their
  generated `.github/workflows/shadow-diff.yml` pin.
- **Embedded SDK version tracks package version.** Every emitter
  (langfuse/braintrust/langsmith/openai-evals/otel importers, OTel
  exporter, FastAPI `shadow serve`) hard-coded
  `"sdk": {"name": "shadow", "version": "0.1.0"}` in the metadata
  record's SDK-provenance field. That's a lie: a v1.2 install was
  stamping records "written by v0.1". Now reads `shadow.__version__`
  at import time. Metadata content-IDs change between versions
  (expected, that's what provenance is for); diff results are
  unaffected.
- **TypeScript SDK** `@shadow/sdk` `0.1.0` → `1.2.1` to match the
  Python + Rust packages.
- **Stale "Phase-N stub" comments** removed from `crates/shadow-core/
  src/lib.rs` and `agentlog/mod.rs`. Replaced with real submodule
  documentation. Similar prose fixes in `error.rs`, `parser.rs`,
  and `tests/test_bisect.py`.

### Not changed (and why)

- **No third-party security audit.** Noted in
  `SECURITY.md`, Shadow is self-hostable and processes traces
  locally by default, but has not been penetration-tested. Users who
  need formal assurance should assume that gap and treat `.agentlog`
  files as potentially-sensitive (they are, by default).
- **SPEC-example record versions stay `"0.1.0"`.** The example
  payloads in `SPEC.md` and `test_core.py` hashing vectors are
  illustrative and pin specific SHA-256 values; changing the version
  string there would break the known-vector tests that lock the
  canonical JSON algorithm.

## [1.2.0] - 2026-04-24

### Added, partial-completion close-out

Five partial items from the strategic roadmap closed out this release,
each shipped with implementation + unit tests + CLI wiring.

#### 1. Vercel AI SDK importer

New `shadow.importers.vercel_ai` + `shadow import --format vercel-ai`.
Accepts both OTLP-style `{spans: [...]}` (the AI SDK telemetry exporter)
and dashboard-style `{events: [...]}` (the Vercel AI Observability JSON
export). Maps the full `ai.*` attribute namespace: `ai.prompt.messages`,
`ai.response.text`, `ai.response.toolCalls`, `ai.tools`, `ai.settings.*`,
`ai.usage.*`, `ai.finishReason`. Tool invocations surface as
Anthropic-shape `tool_use` content blocks so the rest of the differ
works unchanged. Error spans map to a synthetic `error` stop-reason.
16 unit tests + CLI integration.

#### 2. PydanticAI importer

New `shadow.importers.pydantic_ai` + `shadow import --format pydantic-ai`.
Accepts the native `all_messages_json()` output, wrapped
`{messages: [...]}` dumps, and Logfire span exports that carry the
message history under `attributes.all_messages_json`. Handles the full
part-kind set: `system-prompt`, `user-prompt`, `text`, `tool-call`,
`tool-return`, `retry-prompt`, both snake-case and CamelCase variants
across PydanticAI versions. Tool schemas from `model_request_parameters`
propagate to every downstream request. 11 unit tests + CLI integration.

#### 3. Token-level + policy-level hierarchical diff

Two new layers in `shadow.hierarchical` complete the strategic-plan
hierarchy (trace → session → turn → span → **token** → **policy**):

- **Token-level**, `token_diff(baseline, candidate)` produces
  per-dimension distribution summaries (median, p25, p75, p95, max,
  total) for `input_tokens` / `output_tokens` / `thinking_tokens`,
  plus a per-pair delta list ranked by absolute shift. Surfaces via
  `shadow diff --token-diff`. Handles zero-median baseline without
  blowing up (returns `+inf` normalised shift).
- **Policy-level**, declarative YAML overlay with eight rule kinds:
  `must_call_before`, `must_call_once`, `no_call`, `max_turns`,
  `required_stop_reason`, `max_total_tokens`, `must_include_text`,
  `forbidden_text`. `policy_diff(baseline, candidate, rules)` classifies
  each violation as pre-existing, a regression, or a fix. Surfaces via
  `shadow diff --policy path/to/policy.yaml`.

21 new unit tests covering both layers + CLI wiring.

#### 4. Partial replay

New `shadow.replay.run_partial_replay(baseline, branch_at, backend)`
, the 2nd of the replay-as-science slices (after counterfactual in
v1.0). Locks the baseline prefix verbatim (turns `0..branch_at-1`),
then switches to live replay at the branch point. Isolates behaviour
change to a specific turn so reviewers can ask "if we diverged at turn
3, what happens from turn 4 onwards under config B?" without
confounding earlier turns. Surfaces via
`shadow replay --partial --branch-at <idx>`. Clamps gracefully when
`branch_at` exceeds the trace length (full-baseline copy). Preserves
parent DAG consistency under all three modes (zero = fully live,
mid = split, end = pure copy). 10 unit tests.

#### 5. LLM-assisted prescriptive fixes

New `shadow.suggest_fixes` module + `shadow diff --suggest-fixes` flag.
Layers an LLM pass on top of the deterministic recommendation engine to
produce concrete code-level fix proposals. The module:

- Collects up to 6 anchors from the deterministic `Recommendation` list,
  prioritised by severity.
- Builds a bounded evidence window (top axes + first-divergence +
  flagged-turn request/response payloads, truncated to
  `MAX_EVIDENCE_CHARS = 1800` per record).
- Calls the configured LLM backend with a strict JSON schema.
- **Rejects ungrounded suggestions**, if the model invents an anchor
  id not in the deterministic set, that suggestion is dropped. This
  keeps the LLM from inventing fixes for problems that don't exist.
- Tolerates markdown fences, trailing chatter, malformed JSON (returns
  empty gracefully), and out-of-range confidence values.
- Flags suggestions with `confidence < 0.3` as `[speculative]` rather
  than dropping them silently.

Opt-in only (~1-2k output tokens per diff, same backend selection rules
as `--judge` / `--explain`). 13 unit tests covering anchor-grounding
enforcement, JSON robustness, and evidence-truncation safety.

### Upgraded

- Rust workspace version `1.1.0 → 1.2.0`.
- Python package `1.1.0 → 1.2.0` (ABI-compatible with 1.1 consumers).
- `shadow.importers` now exports 8 formats:
  Braintrust, Langfuse, LangSmith, MCP, OpenAI Evals, OTel,
  **Vercel AI**, **PydanticAI**.

### Testing

- **70 new unit tests** across the five modules.
- **Full suite: 468 passed** (up from 398), no regressions.
- `cargo test --workspace`: 201 passed.
- `ruff check`, `ruff format --check`, `mypy --strict`, `cargo clippy
  -- -D warnings`, `cargo fmt --check`: all clean.

## [1.1.0] - 2026-04-24

### Added, scale, correctness, ops hardening

A six-item hardening pass against the honest gaps called out in the
v1.0 postmortem. Each item ships with concrete scope and explicit
documentation of what's **not** done.

#### 1. Scale verified to N=10k (item 6)

Extended `benchmarks/scale_drill_down.py` with
`SHADOW_SCALE_BIG=1` and `SHADOW_SCALE_HUGE=1` tiers. Running at
N=5k surfaced a **real super-linear blow-up** (17.92s at N=1k →
484.82s at N=5k, 27× wall-time for 5× pairs). Root-caused to the
O(N²) Needleman-Wunsch matrix allocation in
`crates/shadow-core/src/diff/alignment.rs`.

Fix: **banded Needleman-Wunsch**. Above
`SCALE_BAND_THRESHOLD = 1000` pairs, the DP is restricted to a band
of `max(|N-M| + 100, sqrt(max(N,M)))` cells around the diagonal.
standard technique from the sequence-alignment literature (SWAT,
Hirschberg). Below the threshold the full-matrix variant stays
exact for all existing tests. At N=5k the new numbers: 19.43s
(3.89 ms/pair). At N=10k: 40.10s (4.01 ms/pair). Per-pair cost
stays flat at big N, confirmed linear.

Added per-pair ms budget `MAX_MS_PER_PAIR = 50` so accidental
algorithmic regressions fail loudly at any N, not just at the
scale tier that happens to be running.

#### 2. Property-based tests via Hypothesis (item 5)

New `python/tests/test_properties.py`, **8 property tests
exercising ~600 generated inputs each**. Properties:

- Canonical-JSON roundtrip is byte-deterministic.
- `compute_diff_report` never crashes, always emits 9 axes, finite
  CI bounds, recognised severity enum values.
- Self-diff produces `|delta| < 1e-6` on every axis for any trace.
- Cost-attribution identity: `model_swap + token_movement +
  mix_residual == total_delta` to f64 precision, for any session
  pair and any pricing table.
- Schema-watch is monotone on no-op inputs for any config.
- `canonical_bytes` and `content_id` are deterministic on
  arbitrary nested JSON.

Catches regressions the example-based tests miss.

#### 3. Needleman-Wunsch span alignment (item 4)

`shadow.hierarchical.span_diff` previously used greedy per-index
alignment. On long tool-heavy responses (the real case as agents
accumulate 20+ tool calls per turn), a single inserted tool_use
block would cascade into every downstream block being reported as
`block_type_changed`.

Now: two-path dispatch by size. `≤ 5 blocks either side` uses the
greedy fast path (optimal and cheap). `> 5 blocks` uses Needleman-
Wunsch alignment with a cost model that nudges the aligner toward
reporting `add + remove` over `block_type_changed` when block
types differ. Verified with two new tests that drop / insert a
block in position 10 of a 20-block response, NW correctly reports
exactly 1 add/remove and zero cascaded type changes.

Token-level 5th hierarchy deferred to v1.2+.

#### 4. Security hardening pass (item 8), NOT a formal audit

Concrete hardening pass across four attack surfaces. Explicitly
documented as "hardening pass, not a formal third-party audit" in
SECURITY.md.

- **Parser resource bounds**: new `DEFAULT_MAX_LINE_BYTES` (16
  MiB per record) and `DEFAULT_MAX_TOTAL_BYTES` (1 GiB per trace)
  with typed `LineTooLarge` / `TraceTooLarge` errors. Tunable per
  `Parser` via `with_max_line_bytes` / `with_max_total_bytes`.
  The per-line cap uses `Read::take` so a newline-free stream
  errors out at the cap rather than growing the buffer unbounded.
- **Path-traversal on `shadow quickstart`**: refuses system
  directories (`/etc`, `/usr`, `/bin`, `/sbin`, `/boot`, `/proc`,
  `/sys`, `/dev`).
- **SECURITY.md updated** with an honest threat-model section,
  hardening-pass summary, and explicit list of what was NOT
  hardened (JSON depth, reproducible builds, formal audit).
- 2 new Rust tests (`rejects_a_line_longer_than_the_configured_limit`,
  `rejects_total_trace_exceeding_byte_cap`).

#### 5. Published docs site (item 7)

New `mkdocs.yml` + `docs/` tree + `.github/workflows/docs.yml`
GitHub Pages deploy. Complete navigation:

- Quickstart: Install, Record, Wire into CI
- Features: Nine-axis diff, Judges, Bisect, Schema-watch, MCP,
  Cost attribution, Hierarchical diff
- Reference: CLI, .agentlog format, Pricing table
- Security, Changelog

Built locally with `mkdocs build --strict` (zero warnings).
Deploys automatically from `main`.

#### 6. Counterfactual replay (item 3, one slice)

New `shadow.counterfactual` module, the first of five replay-as-
science slices the strategic analysis called out. Isolates a
single config delta (model swap, temperature change, system-prompt
override, tools-list replacement, etc.) and re-runs the trace
through a live backend with only that one thing changed.

Composes with `shadow bisect`: bisect gives statistical attribution
("we think the model swap is 78% of the latency regression"); a
counterfactual replay confirms it with a direct experiment that
holds everything else constant.

14 new unit tests. Explicitly documented deferred slices in the
module docstring: partial replay, sandboxed replay, streaming
replay, multimodal replay.

### Test totals

- **201 Rust tests** (was 199, +2 parser-bound tests)
- **398 Python tests** (was 374, +14 counterfactual, +2 hierarchical NW, +8 Hypothesis properties)
- **79 hero end-to-end assertions** (unchanged)
- **17 live-LLM judge tests** (unchanged)

## [1.0.0] - 2026-04-24

### Added, hierarchical diff (Phase D)

New `shadow.hierarchical` module closing the last remaining gap from
the four-phase plan. Shadow's reports previously sat at two levels.
`trace` (nine-axis table) and `turn` (drill-down). Two real-world
questions those couldn't answer:

1. *Which session in a multi-conversation trace regressed?*
2. *Within a regressed turn, which content block actually changed?*

Phase D adds both layers:

- **`diff_by_session(baseline, candidate, ...)`**, partitions both
  traces by `metadata` record, runs `compute_diff_report` on each
  session pair, returns one `SessionDiff` per session with its own
  `DiffReport` and `worst_severity`. Mismatched session counts pad
  the shorter side with empty sessions (pair_count 0 rows).

- **`span_diff(baseline_response, candidate_response)`**.
  content-block-level classifier. Surfaces: `text_block_changed`
  (with char-Jaccard similarity + previews), `tool_use_added/
  _removed`, `tool_use_args_changed` (arg-level deltas including
  rename-as-remove-plus-add), `tool_result_changed` (is_error flip
  + content differ), `stop_reason_changed`, `block_type_changed`.
  Uses greedy index alignment, per-turn block counts are small
  enough that Needleman-Wunsch's extra alignment artefacts
  outweigh the benefits.

- **CLI**: new `--hierarchical` flag on `shadow diff` prints a
  worst-severity-per-session rollup after the nine-axis table.
  Defaults off, on single-session traces it's redundant with the
  top-level severity.

### 1.0 milestone

v1.0.0 marks feature-complete for the four-phase strategic plan:

- **Phase A** (first-real-diff experience), auto-judge, low-n
  guidance, "what this means" deterministic summary, `--explain`
  LLM narrative (shipped in v0.4.0).
- **Phase B** (MCP server importer), `shadow import --format mcp`
  ingests Anthropic's Model Context Protocol session logs
  (shipped in v0.5.0).
- **Phase C** (session-cost attribution), per-session cost delta
  decomposition into model_swap + token_movement + mix_residual
  (shipped in v0.6.0).
- **Phase D** (hierarchical diff), session-level + span-level
  breakdowns (this release).

Test state at v1.0.0:
- **374 Python unit tests** (all green, no skips outside live-API)
- **199 Rust tests**
- **17 live-LLM judge tests** (verified against real Claude Haiku
  4.5 and GPT-4o-mini)
- **79 hero end-to-end assertions** across 10 stages on real
  committed `.agentlog` fixtures
- `cargo fmt/clippy -D warnings`, `ruff check/format --check`,
  `mypy --strict` all clean across 68 source files

### Tests (Phase D)

- 18 new `test_hierarchical.py` tests: session partitioning,
  worst-severity propagation, mismatched-session padding, span-level
  type-swap / tool-use arg changes / tool-result flips /
  stop_reason changes / identity-mapped block indices, both
  renderers.
- Hero harness: 73 → **79 assertions**. New
  `stage_hierarchical_diff` asserts exactly 1 session detected in
  the devops-agent trace, worst_severity = severe, 5 paired
  responses, and span-level detects ≥ 1 tool_use change on turn #0.

## [0.6.0] - 2026-04-24

### Added, session-cost attribution (Phase C)

New `shadow.cost_attribution` module + CLI integration. Shadow's
existing per-response `cost` axis says *whether* a PR moved cost.
This answers the follow-up question a CFO or eng lead always asks:
**why, and by how much per user-facing session?**

- **Session partitioning.** A "session" is the span between two
  `metadata` records in an `.agentlog`, one user-facing
  conversation including all follow-up tool calls. Shadow rolls
  up per-session input / output / cached / reasoning token counts
  and USD spend.

- **Attribution decomposition.** Cost delta between a baseline
  session and its candidate decomposes into three independent
  sources:

    total_delta = model_swap + token_movement + mix_residual

  - `model_swap`: how much of the delta is the candidate model's
    price-per-token vs the baseline's, holding tokens constant at
    candidate levels.
  - `token_movement`: how much is the token-count change, holding
    price at baseline.
  - `mix_residual`: non-additive interaction (simultaneous model
    swap + token movement). When `|residual| > 10% of |total_delta|`
    the decomposition is flagged as "less trustworthy" so the user
    knows the simple two-factor story is incomplete.

- **Pricing table compatibility.** Uses the same rich-dict pricing
  shape the Rust `cost` axis accepts, input / output /
  cached_input / cached_write_5m / cached_write_1h / reasoning
  rates. Unknown models contribute $0 (no crash).

- **Rendering.** `shadow diff` prints the attribution section
  after the nine-axis table when the cost delta is non-zero:

  ```
  cost attribution (per session):
    session #0: $0.0870 → $0.0174 (Δ $-0.0696, -80.0%)
      model swap claude-opus-4-7→claude-sonnet-4-6: $-0.0696 (+100%)
      token movement:            $+0.0000 (-0%)
    total: $0.0870 → $0.0174 (Δ $-0.0696)
  ```

  Markdown renderer emits a GitHub-flavoured table when callers
  attach a `cost_attribution` key to the `DiffReport` dict they
  pass to `render_markdown` / `render_github_pr`.

### Tests

- 21 new `test_cost_attribution.py` tests: session partitioning,
  per-session roll-up, pricing-shape tolerance (rich dict + legacy
  tuple), cached-input / reasoning token rates, fundamental
  identity (`swap + move + residual == delta` across every
  scenario), noisy flag, multi-session alignment, mismatched
  session counts, both renderers.
- Hero harness: 64 → **73 assertions.** New stage
  (`stage_session_cost_attribution`) synthesises pure-swap and
  mixed scenarios to lock the attribution arithmetic.

## [0.5.0] - 2026-04-24

### Added, MCP (Model Context Protocol) importer (Phase B)

Shadow now ingests MCP server session logs. MCP is Anthropic's
open JSON-RPC-2.0 protocol (spec 2025-06-18, v1.0) for
agents-to-tools communication, adopted by Claude Desktop, Cursor,
Windsurf, Zed, VS Code, and every Anthropic-flavoured IDE by
early 2026.

- **`shadow import --format mcp <log-file>`**, new importer
  (`shadow.importers.mcp`) converts an MCP session into a partial
  `.agentlog`. Accepts three in-the-wild input shapes:
  (a) JSONL, one JSON-RPC message per line (`mcp-server --log`
  output), (b) JSON array, what MCP Inspector's export produces,
  (c) wrapped object, `{"messages": [...], "metadata": {...}}`.
  Auto-detects between JSONL and wrapped-object by trying a
  whole-file parse before falling back to line-by-line.

- **Perfectly captured**: tool-call trajectory (arg rename,
  sequence change, omitted calls) and tool-schema (everything
  `tools/list` advertised). Shadow's trajectory axis, first-
  divergence detector, and schema-watch all work on imported
  traces with no further wiring.

- **Partial**: tool results (when present in the log). Mapped to
  Anthropic-style `tool_result` content blocks.

- **Not captured**: LLM completions. MCP is the *tool* protocol,
  not the LLM protocol; semantic / verbosity / safety axes show
  zero on MCP imports by design.

- **MCP error responses** are mapped to
  `{"type": "tool_result", "is_error": true, ...}` blocks so they
  surface on the conformance axis.

- **Orphan requests** (no response, client crash, disconnect)
  still produce a `chat_response` record with just the tool_use
  block, preserving the trajectory signal.

### Added, real-world MCP example

- `examples/mcp-session/`, committed baseline + candidate JSONL
  MCP logs for an Acme customer-support scenario. Baseline agent
  uses `search_orders` → `refund_order` in the correct order with
  original arg names; candidate silently renames `customer_id`
  → `cid` and skips the confirmation step. Shadow's trajectory
  axis fires end-to-end.

### Tests

- 13 new `test_mcp_importer.py` tests (shape round-trip, metadata
  tools-list capture, tool_use + tool_result block shape, parent-
  chain connectivity, error responses, orphan requests, all three
  input shapes, JSONL / JSON array / wrapped object, and full
  CLI integration including a trajectory-axis assertion on two
  imported MCP sessions).
- Hero harness extended from 54 to **64 end-to-end assertions**.
  a new stage (`stage_mcp_importer`) imports both committed MCP
  logs through the CLI and proves the trajectory axis detects the
  arg rename.

## [0.4.1] - 2026-04-24

### Fixed

- **Release pipeline**, the `sign-and-release` job in `release.yml`
  failed uploading wheels because all three OS matrix jobs were
  producing a file named `sbom-python.cdx.json`, and the GitHub
  release API rejects a second attempt to attach a same-named
  asset. SBOM generation is now gated to the `ubuntu-latest` matrix
  entry only, Python deps are identical across OSes, so a single
  SBOM is authoritative.

## [0.4.0] - 2026-04-24

### Added, first-real-diff experience (Phase A)

Turns "cool diff" into "actionable PR review", the natural follow-up to
v0.3.x's adoption focus. A new user who just `pip install shadow-diff`'d
now sees a useful diff on first run, not a noisy blank-judge table.

- **`--judge auto`**, new value for `--judge` that resolves to `sanity`
  against whichever API-key env var is present (ANTHROPIC_API_KEY
  preferred for Claude Haiku 4.5's cost, then OPENAI_API_KEY for
  gpt-4o-mini). Falls through to `none` cleanly with no key. A
  one-cent judge signal that doesn't require flag archaeology.

- **Low-n guidance banner**, terminal and markdown renderers both
  warn loudly when `pair_count < 5`: bootstrap CIs at that sample
  size are unreliable, and the user needs to know before reading
  severities.

- **"What this means" deterministic summary**, new
  `shadow.report.summary.summarise_report`. Turns the nine-axis
  table into a 2–4 line paragraph a PR reviewer can read in one
  breath: leads with structural axes (trajectory / conformance /
  safety), cites verbatim deltas with axis-appropriate units (ms,
  tokens, USD), embeds the first-divergence line, calls out the
  worst drill-down pair when its regression_score > 1.0, and
  surfaces the top error-severity recommendation. Rendered above
  the axis table in both terminal and markdown output. No LLM
  involved, fully reproducible.

- **`shadow diff --explain`**, opt-in flag that pipes the
  deterministic summary through the judge backend to produce a
  ~60-word prose narrative. Verified end-to-end against live Claude
  Haiku 4.5 on the real devops-agent fixture; produces a tight,
  accurate rundown ("tool set shrunk 4→1, format compliance failed
  (-1.0), root cause: structural drift at turn 0"). ~$0.0003 per
  run. Never fires without explicit opt-in, respects zero-friction
  defaults.

### Fixed

- **`AnthropicLLM` / `OpenAILLM` default model**. Real-API verification
  surfaced this: when neither `--judge-model` nor the request's model
  field was set, the backends forwarded an empty string, and the API
  rejected with `invalid_request_error: model: String should have at
  least 1 character`. Judges returned `error` → neutral 0.5 scores
  silently. Added `DEFAULT_MODEL` class constants
  (`claude-haiku-4-5-20251001` and `gpt-4o-mini`) used when both the
  override and the request payload are empty. Every live judge call
  from the CLI now works with no model override, fixing the
  accidental 0.5-axis-8 behaviour for users running
  `--judge sanity` or `--judge auto`.

### Tests

- 17 new `test_summary.py` tests (low-n caveats, axis priority,
  delta unit formatting, first-divergence embedding, worst-pair
  threshold, recommendation ranking, byte-budget discipline).
- 5 new `test_auto_judge.py` tests (no keys → fall-through, only
  anthropic → anthropic, only openai → openai, both → anthropic,
  explicit `none` bypasses auto).
- Hero harness extended from 47 to **54 end-to-end assertions**
  covering the new UX on committed fixtures.
- Live-API end-to-end: `--judge auto` + `--explain` against real
  Claude Haiku 4.5 produces a correct narrative; 17/17 judge-live
  tests still pass after the default-model fix.

## [0.3.1] - 2026-04-24

### Fixed

- **`shadow record` now fails fast on an unwritable output path.**
  An in-depth real-world verification of v0.3.0 caught one
  behavioural bug: if `-o` pointed at a read-only directory,
  `shadow record` would launch the wrapped agent anyway (burning
  real LLM tokens), then quietly emit a warning on atexit when the
  flush failed. The recording was silently lost. `shadow record`
  now probes the output path's writability before spawning the
  child and exits with code 2 + a human-actionable error if the
  write would fail. New ground-truth test
  `test_shadow_record_fails_fast_on_unwritable_output_path`
  guards the invariant.

## [0.3.0] - 2026-04-24

### Added, zero-friction adoption

A full week on the one thing that was harder than it should have been:
making Shadow trivially usable for a new user who just ran
`pip install shadow-diff`.

- **Auto-instrumentation via PYTHONPATH + sitecustomize.**
  `shadow record -- python your_agent.py` now records an agent's
  LLM calls **with zero code changes** to the agent itself. A shim
  `sitecustomize.py` prepended to the child's PYTHONPATH fires at
  interpreter startup, checks for `SHADOW_SESSION_OUTPUT`, and if
  set, constructs and enters a `Session` whose atexit handler
  flushes the trace on process shutdown. All existing Session
  features (anthropic/openai monkey-patching, redaction, trace
  propagation) apply automatically. New `--tags` flag on
  `shadow record` propagates to the metadata record. New
  `--no-auto-instrument` opts out cleanly if the agent already
  manages its own Session.

- **`shadow quickstart` scaffolder.** Drops a working demo
  (`agent.py`, `config_a.yaml`, `config_b.yaml`, two pre-recorded
  `.agentlog` fixtures, `QUICKSTART.md`) into a directory in one
  command. A brand-new user goes from `pip install shadow-diff` to
  seeing a real nine-axis diff in under 60 seconds, no API keys
  required. `--force` overwrites existing files; otherwise existing
  content is preserved.

- **`shadow init --github-action`.** Extends the existing `init`
  command with a flag that drops a ready-to-commit
  `.github/workflows/shadow-diff.yml` into the user's repo. Wires
  up `pip install shadow-diff`, runs `shadow diff` on every PR,
  renders the report as PR-comment markdown, and posts via the
  `gh` CLI. Edit two env vars (BASELINE / CANDIDATE paths),
  commit, and every PR gets a behavioural-diff comment with no
  further setup.

- **README rewrite.** The "Try it" section is now "5-minute
  adoption" and leads with the three-command adoption path:
  `pip install shadow-diff` → `shadow quickstart` → `shadow diff`.
  The "Instrument your own agent" section now documents the
  zero-config `shadow record` path as the recommended option, with
  the explicit `Session` pattern as the secondary choice.

### Fixed

- **Typer extra-args forwarding on `shadow record`.** Typer 0.24
  (bumped for the `--help` fix in 0.2.x) changed how the `--`
  separator is handled at the command level; `shadow record`
  needed its own `context_settings={"allow_extra_args": True,
  "ignore_unknown_options": True}` for child args after `--` to
  be forwarded to `ctx.args`. Previously only set at the app
  level, which no longer sufficed.

### Tests

- 10 new `test_autostart.py` tests covering env-var handling,
  tag parsing, empty-command rejection, exit-code propagation,
  `--no-auto-instrument` behaviour, and the PYTHONPATH shim contents.
- 12 new `test_quickstart.py` tests covering file scaffolding, valid
  agentlog output, `shadow diff` on the scaffolded fixtures,
  `--force` behaviour, `--github-action` workflow generation, and
  the composed quickstart + init flow.
- Hero harness extended from 34 to **47 end-to-end assertions**
  covering the new adoption path on committed fixtures.

## [0.2.2] - 2026-04-23

### Changed

- **PyPI distribution name renamed from `shadow` to `shadow-diff`.**
  The short `shadow` name was already registered on PyPI by an
  unrelated 2015 btrfs-snapshot utility, so the project is now
  published as `pip install shadow-diff`. The Python import path
  (`import shadow`), the installed CLI command (`shadow`), and the
  GitHub repo slug (`manav8498/Shadow`) are unchanged, only the
  PyPI distribution name differs.

### Fixed

- **Release pipeline**, `cargo cyclonedx --output-pattern package`
  rejected the `--output-pattern` flag on `cargo-cyclonedx` 0.5.7
  (upstream removed it). Dropped the flag and rely on the default
  per-crate output path; the schema-valid minimal-SBOM fallback
  still catches any future upstream path drift.

## [0.2.1] - 2026-04-23

### Fixed

- **Release pipeline**, `cargo package -p shadow-core` failed in
  the v0.2.0 release run because the crate's `include` allowlist
  matched only `src/**/*.rs`; `src/store/schema.sql`, which is read
  via `include_str!()`, was excluded and the verify-build inside the
  published tarball broke. Added `src/**/*.sql` to the allowlist.
- **Release pipeline**, the Python SBOM step wrote to `dist/` at
  the repo root (which didn't exist) while wheels landed in
  `python/dist/`; redirected SBOM output to match.

### Added

- **PyPI publish job** (`publish-pypi` in `release.yml`) using OIDC
  Trusted Publisher, no API token required. Bound to a `pypi`
  GitHub Environment so the trust link is (repo, workflow,
  environment)-scoped. See `docs/PYPI-PUBLISHING.md` for the
  one-time setup (pypi.org pending publisher + GitHub Environment
  creation).

## [0.2.0] - 2026-04-23

### Fixed

- **`shadow <cmd> --help` now renders**. Bumped `typer` from pinned
  `0.13.0` to `>=0.15,<1.0`; the old pin was incompatible with
  `click 8.2+` (breaking `TyperArgument.make_metavar()` signature).
  All twelve subcommands print their help pages again.

### Added

- **Live-LLM judge tests**, new `python/tests/test_judge_live.py`
  exercises every judge against real Anthropic and OpenAI backends.
  Gated by `SHADOW_RUN_NETWORK_TESTS=1` plus `ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY`; auto-skips otherwise. Each test picks a scenario
  where the correct verdict is unambiguous so a real LLM's behaviour
  can be asserted directly. ~$0.01 token budget per backend per full
  run.

- **Scale benchmark for drill-down**, new
  `benchmarks/scale_drill_down.py` runs the full nine-axis diff
  plus drill-down ranking on synthetic traces at `N ∈ {100, 500,
  1000}`. Asserts correctness (ranking invariants, dominant axis,
  `pair_count` match) and a 60-second wall-time cap at N=1000.
  Current numbers on a local laptop: 0.18 s @ N=100, 4.5 s @ N=500,
  18.1 s @ N=1000.

- **Every optional extra now exercised in CI.** New `serve` and
  `otel` extras in `pyproject.toml`, plus a new CI job
  `python-full-extras` that installs `dev`, `anthropic`, `openai`,
  `otel`, `serve`, and `embeddings` and runs the full test suite.
  no more silently-skipped tests. Test count went from 260 passed /
  18 skipped to 278 passed / 0 skipped (excluding the network-gated
  `test_judge_live.py` module, which correctly stays skipped without
  API keys).

- **Per-pair drill-down**, `DiffReport` gains a `drill_down` field
  ranking the top-K most-regressive response pairs in a trace set
  by an aggregate regression score, with a per-axis breakdown for
  each. Surfaces *which* specific turns drove each aggregate axis
  delta, so a reviewer scanning a PR with many paired traces can
  click-in to the single worst pair instead of hand-auditing them
  all.

  Each row carries `pair_index`, `baseline_turn`, `candidate_turn`,
  `regression_score`, `dominant_axis`, and an `axis_scores` list of
  8 `PairAxisScore` entries (Judge excluded, the Rust core never
  populates it). Per-axis `normalized_delta` is `|delta| /
  axis_scale` clamped to `[0, 4]`; scales are calibrated against the
  existing `Severity::Severe` thresholds so a value of 1.0
  corresponds to one severity-severe-sized movement on that axis
  and scores sum coherently across axes.

  On the real devops-agent fixture: top pair is pair #1 with
  regression score 9.32, verbosity collapsed (402→128 output
  tokens) and trajectory flipped (baseline had 0 tool-divergence,
  candidate had 100%). A reviewer spots the regression without
  opening any raw `.agentlog` bytes.

  Rendered by every report format: terminal (one line per pair with
  the top 2 contributing axes indented), markdown / github-pr
  (`### Top regressive pairs` section with the top 3 inline and the
  rest under a collapsible `<details>` block).

  Tests: 8 new Rust unit tests in `diff::drill_down::tests` + 7
  Python renderer/integration tests. Hero harness extended to
  assert `drill_down` surfaces ≥ 3 regressive pairs on the real
  devops-agent scenario (now 34/34 end-to-end assertions).

  Closes the last open v0.2 ROADMAP item ("Per-pair drill-down in
  the diff report").

- **Hero end-to-end real-world scenario.** New harness
  `benchmarks/hero_devops_scenario.py` that exercises every Shadow
  feature (schema-watch, nine-axis diff, first-divergence,
  top-K divergences, recommendations, hardened bisection, and all
  ten judges) end-to-end against **committed `.agentlog` trace
  fixtures**, not hand-crafted verdicts. Inputs are the real YAML
  configs and real recorded agent traces under
  `examples/devops-agent/` and `examples/customer-support/`. Every
  feature must independently surface a symptom of the PR's regression
  for the run to pass: 30/30 assertions currently green across two
  domains. This answers the integration question the per-feature
  `validate_*.py` harnesses can't: do the features compose on real
  trace data?

- **Judge axis defaults, 10 ready-made judges.** Previously axis 8
  was effectively a no-op without a user-written rubric. The package
  now ships six new Judge classes on top of the existing four
  (`SanityJudge`, `PairwiseJudge`, `CorrectnessJudge`, `FormatJudge`):

  - **`LlmJudge`**, generic, user-configurable LLM-as-judge. Caller
    supplies a rubric string (may reference `{task}`, `{baseline}`,
    `{candidate}`) plus a `score_map` of verdict strings to
    `[0, 1]` scores. Construction-time validation rejects unknown
    placeholders so mistakes surface before the first judge call.
    Defaults (`temperature=0`, `max_tokens=512`) match published
    LLM-as-judge best practice (Zheng et al. 2024).
  - **`ProcedureAdherenceJudge`**, flags candidates that skip steps
    from a required procedure. Catches the devops-agent pattern
    where a prompt rewrite silently drops
    `backup_database → run_migration` ordering.
  - **`SchemaConformanceJudge`**, semantic schema review (shape +
    meaning). Complements `FormatJudge`'s mechanical JSON-schema
    validation.
  - **`FactualityJudge`**, flags candidates whose claims contradict
    a known-fact set.
  - **`RefusalAppropriateJudge`**, catches over- AND under-refusals
    against an explicit policy.
  - **`ToneJudge`**, tone / persona drift against a target.

  Shared judge helpers consolidated in `shadow.judge._common`
  (response-text extraction, JSON-object extraction from prose, NaN-
  safe clamping, uniform error verdicts) so breaking changes to
  judge-response parsing now land in one file.

  CLI: `shadow diff --judge <kind>` extended from `none|sanity` to
  nine values (`none`, `sanity`, `pairwise`, `llm`, `procedure`,
  `schema`, `factuality`, `refusal`, `tone`). New
  `--judge-config <file.yaml>` option loads rubric data for the
  domain judges. Six example config templates shipped under
  `examples/judges/` covering the devops-agent procedure, the
  customer-support schema, Acme factuality, a domain-restriction
  refusal policy, a concise-persona tone target, and a generic
  three-tier `LlmJudge`.

  Tests: 23 new ground-truth unit tests in `test_llm_judge.py`
  (placeholder validation, custom score maps, error paths,
  out-of-range confidence clamping, each of the five domain
  judges). Real-world validation harness
  (`benchmarks/alignment/validate_judges.py`) passes 14/14
  assertions exercising the full rubric → render → parse → score
  pipeline via a deterministic backend.

- **`shadow schema-watch`**, proactive tool-schema change detection
  that runs *before* replay. Classifies each change between two
  configs into one of four severity tiers (breaking / risky /
  additive / neutral) across eleven change kinds (tool added/removed,
  param added/removed/renamed, type changed, required flipped,
  enum narrowed/broadened, description edited). Rename detection
  pairs a removed and added param on the same tool by matching type
  and required-status, catching the most common breaking tool-schema
  edit in practice (see the `devops-agent` example where every tool
  silently renames `database` to `db`).

  Exposed as a new CLI command with three output formats, terminal
  (rich markup), markdown (GitHub table + expandable rationale), and
  json, and a `--fail-on` flag that controls exit-code behaviour in
  CI. Example output on the committed `customer-support` fixture:

  ```
  ✖ BREAKING  lookup_order: parameter renamed `order_id` → `id`
  ! RISKY     refund_order: description rewritten, imperative verbs removed
  + ADDITIVE  lookup_order: parameter `include_shipping` added (optional)
  · NEUTRAL   lookup_order: description rewritten
  ```

  Intended to run first in CI so PRs get a fast schema-breakage
  signal before the full nine-axis diff runs. 24 ground-truth unit
  tests plus a real-world validation harness
  (`benchmarks/alignment/validate_schema_watch.py`) that passes
  14/14 assertions against the `devops-agent` (8-tool database→db
  rename) and `customer-support` (order_id→id rename + optional
  param + risky description edit) fixtures.

- **Hardened causal bisection**, new
  `shadow.bisect.attribution.rank_attributions_with_interactions`.
  Fits **pairwise interaction effects** (delta A x delta B) in addition
  to main effects, and emits honest **bootstrap 95% confidence
  intervals** on every attribution weight. Example output on a
  realistic 4-delta config PR:

  ```
  semantic:
    prompt.system            74.9% [71.0%, 89.2%]  sel_freq=1.00  ✓
    model_id x prompt.system 13.8% [3.8%, 17.5%]   sel_freq=0.89  ✓

  latency:
    model_id          61.3% [59.2%, 68.0%]  sel_freq=1.00  ✓
    tools             19.7% [15.3%, 22.4%]  sel_freq=0.94  ✓
    model_id x tools  16.6% [12.0%, 19.4%]  sel_freq=0.96  ✓
  ```

  Implementation follows the research brief's explicit guidance:
  `PolynomialFeatures(interaction_only=True)` for the augmented
  design; **residual bootstrap** (Chatterjee & Lahiri 2011) instead
  of pairs bootstrap to avoid LASSO's point-mass-at-zero pathology;
  `alpha` fixed via outer `LassoCV` once on the original data (not
  re-tuned per resample, Efron 2014 shows that inflates CI width);
  per-resample normalisation **before** percentile (not normalise-
  then-bootstrap, which breaks CI independence); and a strong-
  hierarchy post-filter that drops A x B interactions where neither
  main effect survived stability selection (Lim & Hastie 2015
  *glinternet*).

  The `significant` flag is now a conjunction, selection frequency
  ≥ 0.6 AND CI excludes zero, which is described as
  screening + magnitude, not a multiplicity-adjusted p-value. Lex-
  sorted interaction pair labels eliminate the `(A,B)` vs `(B,A)`
  ambiguity. Output available via `run_bisect(... backend=...)`
  under the new `attributions_with_interactions` key. 7 new Rust/
  Python ground-truth tests plus a real-world validation harness
  (8/8 passing) in `benchmarks/alignment/validate_hardened_bisect.py`.

- **Prescriptive fix recommendations** (`shadow.diff.recommendations`).
  Transforms the divergence list from "what changed" into "what to do
  about it": a ranked list of specific, imperative actions a PR
  reviewer can act on in under 30 seconds. Examples from the real-
  world 10-turn scenario:

  - `[error]   RESTORE`  turn 9, *"Restore `send_confirmation_email`
    at turn 9."*
  - `[error]   REMOVE`   turn 8, *"Remove duplicate invocation of
    `lookup_order(order_id)` at turn 8."*
  - `[error]   REVIEW`   turn 4, *"Review refusal behaviour at
    turn 4: candidate may be over-refusing."*
  - `[warning] REVERT`   turn 6, *"Revert `refund(amount)` at turn 6
    to the baseline value."*

  Three-tier severity (Error / Warning / Info) matching ESLint /
  SonarQube / Rustc conventions. Five action kinds (Restore / Remove
  / Revert / Review / Verify) covering the expected regression
  patterns. Pure deterministic rule engine, no LLM dependency;
  LLM-enriched recommendations can layer on later.

  Each recommendation carries `severity`, `action`, `turn`, a one-line
  `message`, a `rationale` line citing the signal that triggered it,
  and the primary `axis`. Sorted by severity × confidence, capped at
  8 entries so PR comments don't bloat.

  Exposed on `DiffReport` as the new `recommendations` field
  (documented by the `Recommendation` TypedDict in `_core.pyi`).
  Rendered in all three report formats: markdown / github-pr
  (bulleted `### Recommendations` section with severity icons),
  terminal (colour-coded by severity). 14 new Rust unit tests + 8
  new Python renderer tests; real-world benchmark harness passes
  18/18 including action-correctness assertions.

- **Top-K divergence ranking** on top of first-divergence detection.
  The diff report now carries a `divergences` list (up to
  `DEFAULT_K=5` entries) in addition to the backward-compatible
  `first_divergence` field. Divergences are sorted by importance:
  Structural > Decision > Style (by class), then by confidence
  within a class, with walk order as the stable tiebreaker.
  Renderers show the top 3 inline and collapse any extras (#4 and
  beyond) into a `<details>` section in markdown / github-pr, or a
  "+N more" line in terminal. `first_divergence` remains walk-order-
  first for back-compat; `divergences[0]` is severity-rank-first.
  Validated end-to-end with the benchmark harness at 27/27 cases
  (real committed fixtures + adversarial stress + top-K-specific
  coverage).
- **First-divergence detection** (`shadow.diff.alignment`). Given two
  traces, identifies the first turn at which the candidate diverged
  from the baseline and classifies the divergence as
  `style_drift` / `decision_drift` / `structural_drift`. Uses a
  Needleman-Wunsch global alignment with Gotoh affine gap penalties
  over the chat-response sequence; per-cell cost combines Jaccard
  distance on tool-shape, character-shingle text similarity, arg-
  value diff, and stop-reason mismatch. Surfaces in every report
  renderer (terminal / markdown / github-pr) as a one-line root-
  cause summary like
  *"tool set changed: removed `search(query)`, added `search(limit,query)`"*.
  Exposed on the `DiffReport` dict as the new `first_divergence` key
  (documented by the `FirstDivergence` TypedDict in `_core.pyi`);
  `None` when the traces agree end-to-end. 14 new Rust tests + 6 new
  Python renderer tests.

- Live LLM backends: `shadow.llm.AnthropicLLM` wraps
  `anthropic.AsyncAnthropic`, `shadow.llm.OpenAILLM` wraps
  `openai.AsyncOpenAI`. Both implement the `LlmBackend` Protocol and
  lazy-import their SDK so `shadow` still runs without the extras.
  `shadow.llm.get_backend(name, **kwargs)` factory dispatches by name.
- LASSO-over-corners bisection scorer (`shadow.bisect.corner_scorer`).
  Given a live `LlmBackend`, builds a 2^k full-factorial over the
  differing config categories (`{model, prompt, params, tools}`),
  replays the baseline through the backend at each corner, computes
  the nine-axis divergence per corner, and fits LASSO per axis.
  Ground-truth test recovers `latency → model` and `verbosity → prompt`
  with > 70% weight.
- CLI: `shadow bisect --backend {anthropic,openai,positional}` wires
  the live-replay scorer. Without `--backend`, falls back to the
  heuristic kind-based allocator when `--candidate-traces` is supplied,
  or zero-placeholder otherwise.
- `run_bisect` three-mode dispatch: `lasso_over_corners` (with backend)
  → `heuristic_kind_allocator` (only a candidate trace) →
  `lasso_placeholder_zero` (neither). The `mode` field names which ran.
- 11 new tests covering the backends (fake SDK stubs, no network) and
  the corner scorer (deterministic `FakeBackend`).
- OSS governance / community files: `SUPPORT.md`, `GOVERNANCE.md`,
  `MAINTAINERS.md`, `TRADEMARK.md`, `CITATION.cff`,
  `.github/FUNDING.yml`, `.github/dependabot.yml`.
- `AxisStat` and `DiffReport` `TypedDict`s in `shadow/_core.pyi` so
  downstream Python consumers get real types for the Rust-extension
  return shapes.

### Changed

- Dual-license the implementation under **MIT OR Apache-2.0** (Rust
  community default). `SPEC.md` stays Apache-2.0 only so re-implementers
  get an explicit patent grant for the format.
- README rewritten in plain English (~200 lines), one quickstart,
  sample output matches real `just demo` output.
- Project metadata on `Cargo.toml` and `pyproject.toml`:
  keywords, categories, URLs, `Typing :: Typed`, explicit `include`
  allowlist for the Rust crate.
- `.gitignore` hardened: added `node_modules/`, JS tool caches,
  `.env.*.local` variants, agent-state dirs, `*.tsbuildinfo`,
  Jupyter checkpoints, SBOM outputs.
- `shadow/__init__.py` wraps the abi3-mismatch ImportError with a
  "requires Python 3.11+" hint. `AnthropicLLM.__init__` fails fast
  with a branded error if no API key is set, instead of deferring to
  the opaque HTTP-layer error from the SDK.
- Removed the `CLAUDE.md` internal coding-agent instruction file from
  the repository (not appropriate in a public OSS project).

### Fixed

- Severity classifier false negative on rate-bounded axes: a CI like
  `[0.0, 1.0]` was treated as straddling zero and downgraded Severe to
  Minor. Now `ci_straddles_zero` is strict (`ci_low < -epsilon &&
  ci_high > +epsilon`). A unanimous `+1.0` trajectory delta now
  correctly classifies as Severe.
- Conformance axis was dead for tool-use-only agents (`n=0`). It now
  also fires on `tool_use`-intent and scores by top-level key-set match.
- Delta extractor exploded a single tool-schema edit into dozens of
  leaf-level deltas, which over-determined the LASSO fit. Default
  `diff_configs(coalesce=True)` collapses each tool to a single delta
  keyed by tool name. Legacy leaf-level output still available via
  `coalesce=False`.
- Attribution row schema was inconsistent across bisect modes. All
  three modes now emit the same six keys (`delta`, `weight`,
  `ci95_low`, `ci95_high`, `significant`, `selection_frequency`).
- `DELTA_KIND_AFFECTS["tools"]` was missing `conformance`, added, so
  tool-schema edits can be attributed to the conformance axis.
- 11 pre-existing `ruff` lints (all pattern/idiom nits such as
  `class X(str, Enum)` → `class X(StrEnum)` for Python 3.11+).

### Notes

- This release has **no production users yet**. All "real-world"
  validation references the project's own example scenarios, not
  external deployments. Claims about behaviour should be read as
  "what the code does," not "what teams have confirmed in prod."

## [0.1.0], 2026-04-22

First tagged release. Ships the Rust core, Python SDK + CLI, bisection
module, GitHub Action, end-to-end demo, and CI, see the per-phase
sections below for specifics.

### Summary

- 13 commits across 7 phases.
- **164 tests** total: 125 Rust (`cargo test -p shadow-core`) + 47 Python
  (`pytest python/tests`).
- **Rust coverage: 97.63% line / 98.93% function** on `shadow-core`.
- **Python coverage: 88.07%** across `shadow.*` packages.
- `cargo clippy --all-targets --all-features -- -D warnings` clean.
- `cargo fmt --check` clean.
- `mypy --strict` clean across 26 Python source files.
- `ruff check` + `ruff format --check` clean.
- `bash examples/demo/demo.sh` runs in **1.14 s** on an M-series laptop
  (target ≤ 10 s).

### Phase 0, Scaffold

#### Decisions

- **PyO3 build system: maturin** (D1 in the plan). Using `abi3-py311` so one wheel
  supports Python 3.11+. Alternative considered: setuptools-rust, rejected
  because maturin's `develop` loop is the same-day ergonomics we want for TDD.
- **Workspace-level Rust pinned to 1.83.0** (stable at planning time). Individual
  dep versions pinned exactly in `Cargo.toml` per the user's "no bleeding-edge
  churn" constraint.
- **Clippy `unwrap_used`/`expect_used`/`panic` denied in non-test code** via
  inner attributes in `lib.rs` (not workspace-wide) so tests can still use
  `unwrap()`.
- **Python deps pinned to exact versions** in `python/pyproject.toml`. Real LLM
  SDKs (`anthropic`, `openai`) are optional extras, Shadow can run without
  them via `MockLLM`.
- **Cargo `crate-type = ["cdylib", "rlib"]`** on shadow-core so the same crate
  builds both as a normal Rust library (for `cargo test`) and as a PyO3
  extension (for `maturin develop`). The `python` feature gates the PyO3
  module; non-Python consumers get a clean rlib.
- **CLI entrypoint: `shadow = "shadow.cli.app:main"`** (Python `console_script`),
  not a Rust binary. This keeps the CLI logic testable with `typer.testing.CliRunner`
  and lets the Rust core remain pure-library.

#### Dead ends

- _(none yet, Phase 0 ran cleanly through file scaffolding)_

#### Blockers surfaced and resolved

- **Rust toolchain install**, `cargo` / `rustc` were absent; user-local
  rustup curl install authorized post-Phase-1 per `<interaction_policy>`
  checkpoint 1, installed into `~/.cargo` / `~/.rustup` (no sudo, no brew,
  no system-wide changes).
- **Rust toolchain version**, initially pinned to `1.83.0` (stable at
  original-plan time, late 2024); the release drop to `v0.1.0` then
  bumped to **Rust 1.95.0** (stable 2026-04-14) because the ecosystem had
  already moved past 1.83, `indexmap 2.14+`, `proptest 1.11`, latest
  `just` / `maturin` all require edition2024 (Rust ≥ 1.85). The interim
  1.83 workaround (direct-pin `indexmap = "=2.6.0"` so `serde_json`'s
  transitive dep couldn't resolve to edition2024) was removed on the
  bump; Cargo.lock regenerated; all direct deps updated to the versions
  Cargo picks cleanly on 1.95.
- **Companion tool versions**, `just 1.50.0`, `maturin 1.13.1`,
  `cargo-llvm-cov 0.8.5`. All installed via `cargo install --locked`.
- **PyO3 feature split**, The original single `python` feature used
  `pyo3/extension-module`, which omits libpython link directives. That is
  correct for a maturin-built `.so` but makes `cargo test --features python`
  fail to link. Split into two features: `python` (pyo3 types available,
  libpython linked, `abi3-py311`) and `extension` (adds extension-module,
  used only by maturin). `cargo test --workspace` runs pure-Rust tests
  without pulling pyo3 at all; the PyO3 bindings are tested from Python
  via pytest after `maturin develop`.
- **Rust 1.95 clippy tightening**, the toolchain bump surfaced three new
  lints: `doc_overindented_list_items` (fixed by de-indenting continuation
  lines in the replay engine's doc comment), `useless_conversion` firing
  on PyO3's idiomatic `?`-in-`PyResult` patterns (addressed with a
  module-level `#![allow]` in `src/python.rs` with a comment explaining
  why), and a `clone`→`slice::from_ref` suggestion in a parser test.

### Phase 1, SPEC.md

#### Decisions

- **Content-address payload only, not the envelope.** `id = sha256(canonical_json(payload))`
  so two identical requests dedupe to the same blob. Envelope (`ts`, `parent`) is
  not hashed. Alternative considered: hash the whole envelope, rejected because
  it defeats dedup and makes MockLLM replay lookups harder (you'd need to reconstruct
  the envelope to look up a response).
- **RFC 8785 (JCS) for canonical JSON**, with two application clarifications
  (§5.2, Unicode NFC normalization on strings and keys; §5.4, no
  `Decimal`/`NaN`/`Infinity`). Picking an existing RFC instead of inventing
  our own rules means any JCS library is most of the way there; the NFC
  addition covers a gap in JCS where visually-identical strings encoded
  differently would hash differently.
- **Known-vector lives in §5.6 as a "Conformance test case"** (moved on
  review from §6.2, which now points back to §5.6). The vector covers both
  canonicalization bytes and the resulting content id, so a fresh
  implementer can verify both at once.
- **Known-vector hash pinned in §6.2:** `{"hello":"world"}` →
  `sha256:93a23971a914e5eacbf0a8d25154cda309c3c1c72fbb9914d47c60f3cb681588`.
  Verified locally with `python3 -c 'import hashlib; print(hashlib.sha256(b"{\"hello\":\"world\"}").hexdigest())'`.
  Phase 2's `agentlog::hash` test suite pins this vector.
- **One trace = one file.** Concatenating two `.agentlog` files does NOT
  produce a valid `.agentlog`, simpler invariant than allowing multi-trace
  files, and forces the SQLite index to be the "set" abstraction.
- **First record is always `metadata` with `parent: null`.** A trace is
  identified by its root's content id, so we don't need a separate trace_id
  field in the envelope.
- **Redaction before canonicalization.** The hash reflects redacted content,
  not raw. Means a post-hoc audit can't trivially reconstruct the original
 , that's intentional.
- **Streaming responses = one record** with an optional `stream_timings`
  array, not a record per token. Per-token records would explode storage
  and make `shadow diff` much slower. Timing preserved, token stream
  content aggregated.

#### Dead ends

- _(none yet, spec came together in one pass)_

### Phase 2, shadow-core (Rust)

Nine commits land the full Rust core. Final tree: 125 unit tests, `cargo
clippy --all-features -- -D warnings` clean, `cargo fmt --check` clean,
**97.63% line coverage** on `shadow-core` (target ≥85%) measured via
`cargo llvm-cov --workspace` (98.93% function coverage).

#### Decisions

- **Payload-only hashing (SPEC §6.1).** Record envelope (`ts`, `parent`)
  is not in the hash; only `canonical_json(payload)`. Two identical
  requests dedupe to the same id.
- **Records hold payloads as `serde_json::Value`, not typed structs.**
  Typed payload structs come later alongside the consumers that need
  them (Python SDK). Keeps the core storage path provider-agnostic.
- **Atomic writes for the FS store** via write-tmp + rename. A crash
  mid-write leaves a `.tmp` orphan, not a corrupt real trace.
- **`include_str!` SQLite schema.** Schema lives in `store/schema.sql`;
  bundled into the binary so there's no runtime file resolution and the
  schema can be versioned alongside the code that consumes it.
- **`PyO3` feature split: `python` vs `extension`.** The base `python`
  feature pulls pyo3 with `abi3-py311` but NOT `extension-module`, so
  `cargo test --features python` links libpython normally. `extension`
  adds `extension-module` for maturin. This avoids the "cargo test
  link-fails because libpython isn't linked" footgun.
- **`cargo test --workspace` runs NO features by default.** PyO3
  bindings are tested from Python after `maturin develop`. Keeps the
  Rust test loop zero-config (no need for python3.11 on PATH).
- **Backend trait takes `&Value`, returns `Value`.** Envelope ownership
  stays with the engine; backends are tiny adapters (a live Anthropic
  backend is ~30 lines). Matches SPEC §10's replay algorithm cleanly.
- **Replay errors → `Error` records, not engine panics.** A baseline
  with some-missing responses still produces a complete candidate
  trace, with error counts in the `replay_summary`.
- **Clock abstraction** in `replay::engine` so tests can pin
  timestamps. Ships a `FixedClock` for fixtures; production uses the
  system clock (Phase 3).
- **Semantic axis: hash-surrogate embedding in Rust.** Clearly labelled
  test-only in the module docstring. The production embedding
  (`sentence-transformers/all-MiniLM-L6-v2`) is plugged in by the
  Python layer per CONTRIBUTING.md. This kept an ML dep out of the Rust
  crate.
- **Axis nodes take `pairs: &[(&Record, &Record)]`.** Pair extraction
  lives in `diff/mod.rs::extract_response_pairs`. Uneven counts (e.g.
  candidate missed some) truncate to the shorter side; callers that
  want to flag count-mismatch consult the replay_summary directly.
- **Severity thresholds:** `None` (CI crosses zero and
  delta is tiny), `Minor` (<10% relative), `Moderate` (<30%), `Severe`
  (≥30% or CI clearly excludes zero with large magnitude).
- **Bootstrap defaults to 1000 iterations** with seeded RNG for
  reproducibility; callers can override both.
- **Test-only `assert_eq!` / `unwrap()`** allowed; clippy's
  unwrap_used/panic lints are denied only in non-test code via
  `#![cfg_attr(not(test), ...)]`.

#### Dead ends

- Attempted to write `assert_eq!` in `paired_ci` for length-mismatch
  precondition, tripped clippy's `panic` lint which applies to
  `panic!()` expansions of `assert!` macros. Resolved with a narrow
  `#[allow(clippy::panic)]` block around an explicit `panic!()`.
  cleaner than restructuring to return a Result for a programmer-error
  guard.
- First pass at the `meta` omit-when-None test used `!wire.contains("meta")`,
  which accidentally matched the kind string `"metadata"`. Fix: match the
  quoted field name `"\"meta\""` instead. Small lesson: substring
  assertions on JSON are fragile; prefer explicit field-level checks.

### Phase 3, Python SDK + CLI

#### Decisions

- **PyO3 bindings take/return dicts, not typed pyclass wrappers.** Simpler
  surface for Python users and no second type system to maintain. The
  serde_json::Value ↔ PyObject conversion goes through the `pythonize`
  crate (pinned `=0.22.0` to match `pyo3=0.22.6`).
- **Type stubs shipped in `python/src/shadow/_core.pyi`.** mypy --strict
  users see the PyO3 surface without importing the compiled extension.
- **Session is a manual recorder in v0.1.** Monkey-patch-based
  auto-instrumentation of `anthropic` / `openai` Python clients is
  deferred to v0.2, their streaming surfaces are too divergent to
  unify cleanly in a first cut, and forcing users into
  `record_chat(req, resp)` is a small enough overhead that it's not
  blocking adoption.
- **Python-side `run_replay`.** The Rust replay engine exists but isn't
  exposed through PyO3, the `LlmBackend` trait is async and calling
  back into Python from a Rust trait object needs PyO3 ceremony that
  wasn't worth the code for v0.1. The Python replay mirrors SPEC §10
  semantics exactly.
- **CLI uses typer.** Every subcommand has an end-to-end integration
  test via `typer.testing.CliRunner`. Machine-consumed JSON outputs go
  through `sys.stdout.write` (unstyled) to avoid Rich's ANSI escapes
  breaking `jq` pipelines.
- **`PositionalMockLLM` added** alongside `MockLLM`. Positional replay
  is the only sensible demo backend when baseline and reference traces
  were recorded with different configs (different request payloads →
  different content ids → MockLLM strict would miss every request).
  Clearly labelled as "for demos and integration tests, not production."

#### Dead ends

- First pass at `cargo test --features python` failed to link on
  macOS because the `extension-module` pyo3 feature omits libpython
  link directives. Resolved by splitting the feature into `python`
  (abi3-py311 only, links libpython) and `extension` (adds
  extension-module, used by maturin). `cargo test` takes neither by
  default, PyO3 bindings are tested from Python after `maturin
  develop` builds the `.so`.
- Initial pass at the `meta` omission test used a substring check
  against `"meta"` which accidentally matched `"metadata"` (the kind
  name). Fix: check for the quoted field name `"\"meta\""`.

### Phase 4, Bisection (LASSO + Plackett-Burman)

#### Decisions

- **Per-axis LASSO with `alpha=0.01`.** scikit-learn handles the coord
  descent. Normalization: `|coef| / sum(|coef|)` so each axis's
  attributions sum to 1 (or 0 when the axis is invariant across
  corners).
- **Hadamard/Paley PB matrices tabulated** for runs ∈ {8, 12, 16, 20,
  24}. Runs > 24 error out in v0.1 (k ≤ 23). Full factorial capped at
  k=6 (64 runs). `choose_design` picks the right one automatically.
- **v0.1 runner emits placeholder-zero divergence.** Real per-corner
  replay scoring lands in v0.2 with live-LLM support. The plumbing
  (delta extraction, design, LASSO, attribution ranking) is correct
  and tested against a synthetic ground-truth recovery case
  (≥0.9 attribution to delta #2 on the trajectory axis; ≤0.05 on
  every other delta; zero attribution on every other axis since there
  is no signal).

### Phase 5, GitHub Action

#### Decisions

- **Composite action, not a JavaScript action.** No Node build
  pipeline; the logic lives in `action.yml` shell steps plus a
  stdlib-only `comment.py`.
- **Hidden HTML marker** lets subsequent runs update the existing PR
  comment in place. One comment per PR, not a running log.
- **Step-summary write** means even fork PRs (where posting a comment
  is blocked) still surface the diff in the GitHub Actions log view.

### Phase 6, Demo + README

#### Decisions

- **Demo fixtures are committed.** `examples/demo/fixtures/{baseline,
  candidate}.agentlog` are deterministic outputs of
  `generate_fixtures.py` (also committed). A fresh clone runs
  `just demo` in ≈1 s without touching a network. Regenerating the
  fixtures is reproducible.
- **`PositionalMockLLM` is the demo backend.** `MockLLM` (content-id
  lookup) wouldn't work because baseline/candidate differ in
  system-prompt wording, their request payloads have different ids.
- **README opens with "Why?"** (per review), then a four-column
  competitive-landscape table (Langfuse / Braintrust / LangSmith /
  Shadow). Gives readers a reason to keep scrolling before they hit
  install instructions.

### Phase 7, CI + release

#### Decisions

- **Matrix: Ubuntu + macOS × Python 3.11 + 3.12.** Windows deferred
  (SDK is POSIX-path biased; `.venv/bin/python` assumptions are
  Windows-unfriendly without more plumbing than the v0.1 budget
  allows).
- **`taiki-e/install-action` for `cargo-llvm-cov`** so CI doesn't pay
  the ~45 s compile-from-source cost on every run.
- **Rust cache via `Swatinem/rust-cache`** shaves ~5 min off matrix
  legs.
- **Python tests gate at 85% coverage** (`--cov-fail-under=85`).
  Current: 88.07%.
- **Three jobs**: `rust` (fmt/clippy/test/coverage), `python` (ruff,
  mypy, pytest+coverage), `demo` (end-to-end). `python` depends on
  `rust`, `demo` on `python`.

---

## Conventions

- Each phase boundary lands a `### Phase N, <title>` section.
- `#### Decisions` for design choices, with the alternative that was
  considered and why it was rejected.
- `#### Dead ends` for approaches that were tried and backed out of, with
  the reason.
- `#### Blockers surfaced` for things the user needs to act on.
