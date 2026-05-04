# shadow-align

Reusable trace-comparison primitives for AI agents. Standalone Rust port of
`shadow.align` (Python) and `@shadow-diff/align` (TypeScript).

## Use cases

- **Eval frameworks** that want a stable trace-alignment primitive without
  pulling in all of Shadow.
- **Observability platforms** surfacing "the FIRST thing that diverged
  between this run and last week's golden trace."
- **Regression test harnesses** with stored tool-call sequences per
  scenario.
- **Tool-schema linters** detecting added/removed/changed argument fields.

## Surface (v0.1)

```rust
use shadow_align::{trajectory_distance, tool_arg_delta};
use serde_json::json;

// Levenshtein on flat sequences
let d = trajectory_distance(&["search", "summarize"], &["search", "edit"]);
assert!((d - 0.5).abs() < 1e-9);

// Structural diff of two JSON values
let deltas = tool_arg_delta(&json!({"x": 1}), &json!({"x": 2}));
assert_eq!(deltas[0].path, "/x");
```

Five functions total:

| Function | v0.1 status |
|---|---|
| `trajectory_distance` | ✓ pure Rust |
| `tool_arg_delta` | ✓ pure Rust |
| `align_traces` | stub — depend on `shadow-core` for full impl |
| `first_divergence` | stub |
| `top_k_divergences` | stub |

## Cross-language parity

Same algorithms, byte-identical results on shared inputs to
[`shadow.align`](https://pypi.org/project/shadow-diff/) (Python) and
`@shadow-diff/align` (TypeScript, in the `shadow-diff` npm package).

The `shadow-diff` npm package can also load this crate directly via a
napi-rs binding (built from `typescript/native/`), so TS workloads on
long traces get native-speed `trajectory_distance` and `tool_arg_delta`
without leaving the JS runtime.

## License

Apache-2.0.
