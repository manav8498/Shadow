"""Python-side diff helpers that compose on top of the Rust core.

The Rust core (`shadow._core.compute_diff_report`) operates on a single
baseline / candidate trace pair. For multi-scenario regression suites
(many distinct test cases concatenated into one trace), the alignment
layer doesn't know where one scenario ends and the next begins, so it
spuriously reports "dropped turns" whenever scenarios have different
tool sequences.

The fix lives at the Python layer: partition the records by
``meta.scenario_id`` and run the Rust diff once per scenario, then
return a structured ``MultiScenarioReport`` that the CLI / PR comment /
report renderer can consume.

Records without a ``scenario_id`` fall into a synthetic ``__default__``
bucket so the function is backward-compatible with single-scenario
traces (which produce a one-element report).
"""

from shadow.diff_py.scenarios import (
    DEFAULT_SCENARIO_ID,
    MultiScenarioReport,
    ScenarioDiff,
    compute_multi_scenario_report,
    partition_by_scenario,
)

__all__ = [
    "DEFAULT_SCENARIO_ID",
    "MultiScenarioReport",
    "ScenarioDiff",
    "compute_multi_scenario_report",
    "partition_by_scenario",
]
