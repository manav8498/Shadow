//! First-divergence detection over paired chat responses.
//!
//! Given two traces, this module identifies the **first turn at which
//! the candidate meaningfully diverged from the baseline** and classifies
//! the divergence as one of three kinds:
//!
//! - **Structural** — the tool-call sequence differs (missing, extra, or
//!   reordered calls). Surfaces as a gap in the alignment, OR a same-
//!   position pair with different `tool_name`s.
//! - **Decision** — the tool sequence matches but the *decision* changed:
//!   same tool, different arg values; final-answer semantic cosine < 0.8;
//!   `stop_reason` flipped; refusal where there wasn't one.
//! - **Style** — cosmetic wording differences only; semantic cosine ≥ 0.9,
//!   identical tool shape, identical stop_reason.
//!
//! ## Algorithm
//!
//! A Needleman-Wunsch global alignment with Gotoh affine gap penalties
//! pairs baseline and candidate chat_response records. The cost for
//! aligning pair `(a, b)` is:
//!
//! ```text
//!   cost(a, b) = w_struct * (1 - jaccard(tool_shape_a, tool_shape_b))
//!              + w_sem    * (1 - text_similarity(a, b))
//!              + w_stop   * stop_reason_mismatch(a, b)
//! ```
//!
//! After alignment, we walk the alignment path left-to-right and emit
//! the first cell whose per-cell divergence exceeds the noise floor.
//!
//! ## Why NW and not position-match
//!
//! Position-match fails when one side inserts or drops a turn (common
//! when one config retries where the other doesn't). NW pays a
//! controlled gap cost instead of mis-pairing every subsequent turn.
//! Cost is O(n·m) in DP cells; traces rarely exceed ~100 turns, so
//! runtime is trivial in practice.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::agentlog::{Kind, Record};
use crate::diff::axes::Axis;

/// Classification of the first divergence between two traces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DivergenceKind {
    /// Cosmetic wording only: semantic similarity high, tool shape
    /// identical, stop reason identical. Safe-to-merge signal, usually.
    Style,
    /// Same structure, different decision: arg values differ, refusal
    /// flipped, or final-answer semantics shifted meaningfully.
    Decision,
    /// Tool-call sequence differs: insertion, deletion, or reorder.
    /// This is almost always a real behavioural regression.
    Structural,
}

impl DivergenceKind {
    /// Short machine-readable label used in terminal / markdown / JSON output.
    pub fn label(&self) -> &'static str {
        match self {
            DivergenceKind::Style => "style_drift",
            DivergenceKind::Decision => "decision_drift",
            DivergenceKind::Structural => "structural_drift",
        }
    }
}

/// First meaningful divergence between two traces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FirstDivergence {
    /// 0-based index in the baseline's chat_response sequence. For an
    /// insertion on the candidate side, this is the baseline index
    /// where the insertion appeared (effectively a "between" marker).
    pub baseline_turn: usize,
    /// Same for the candidate side. May differ from baseline_turn when
    /// gaps are present on the alignment path.
    pub candidate_turn: usize,
    /// Classification (Style / Decision / Structural).
    pub kind: DivergenceKind,
    /// Primary axis the divergence surfaces on (semantic, trajectory,
    /// safety, conformance). Provides a machine-readable hint for
    /// grouping regressions by root cause.
    pub primary_axis: Axis,
    /// One-line human-readable explanation. Designed to be embeddable
    /// in a PR comment without additional context.
    pub explanation: String,
    /// Confidence in 0..1. Higher means "the signal exceeds the noise
    /// floor by a wide margin". Callers can gate display on >= 0.5.
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Detect the first meaningful divergence between two traces.
///
/// Returns `None` when the traces agree on every compared turn up to the
/// length of the shorter one (and the longer tail is empty or also
/// matches). Returns `Some` at the first pair whose combined per-cell
/// cost exceeds the noise floor.
pub fn detect(baseline: &[Record], candidate: &[Record]) -> Option<FirstDivergence> {
    let baseline_responses: Vec<&Record> = baseline
        .iter()
        .filter(|r| r.kind == Kind::ChatResponse)
        .collect();
    let candidate_responses: Vec<&Record> = candidate
        .iter()
        .filter(|r| r.kind == Kind::ChatResponse)
        .collect();
    if baseline_responses.is_empty() || candidate_responses.is_empty() {
        return None;
    }

    let alignment = align(&baseline_responses, &candidate_responses);
    walk_for_first_divergence(&alignment, &baseline_responses, &candidate_responses)
}

// ---------------------------------------------------------------------------
// Alignment: Needleman-Wunsch with Gotoh affine gap penalties
// ---------------------------------------------------------------------------

/// Weights for the per-cell cost function. Tuned against the real
/// demo fixtures; exported as constants so tests and callers can see
/// the exact numbers without grepping source.
const W_STRUCT: f64 = 0.40; // Jaccard distance on tool_shape
const W_SEM: f64 = 0.25; // 1 - text_similarity
const W_STOP: f64 = 0.15; // stop_reason mismatch
const W_ARGS: f64 = 0.20; // tool_use input VALUE differences (same keys, different values)

/// Gotoh affine gap penalty: opening a gap is more expensive than
/// extending one. Prevents the aligner from fragmenting a multi-turn
/// insertion into many single-turn insertions.
const GAP_OPEN: f64 = 0.60;
const GAP_EXTEND: f64 = 0.15;

/// Noise floor for per-cell divergence. Cells below this are treated
/// as "no divergence" — covers bootstrap non-determinism, minor
/// token-count drift from prompt caching, etc.
const NOISE_FLOOR: f64 = 0.12;

/// Style-drift upper bound on the per-cell cost. Above this we call
/// it Decision or Structural. Calibrated for semantic cosine ≥ 0.9
/// with identical tool shape.
const STYLE_MAX_COST: f64 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Diagonal: pair baseline[i-1] with candidate[j-1].
    Match(usize, usize),
    /// Horizontal: gap on baseline side (candidate inserted a turn).
    InsertCandidate(usize),
    /// Vertical: gap on candidate side (candidate dropped a turn).
    DeleteBaseline(usize),
}

/// Alignment result: a path of steps from (0,0) to (n,m).
struct Alignment {
    steps: Vec<Step>,
}

fn align(baseline: &[&Record], candidate: &[&Record]) -> Alignment {
    let n = baseline.len();
    let m = candidate.len();
    // DP table: cost of aligning baseline[0..i] with candidate[0..j].
    // We carry three matrices (M, X, Y) per Gotoh to track whether the
    // previous op was a match, a horizontal gap (in baseline), or a
    // vertical gap (in candidate). INF sentinel: 1e18 (cannot realistically
    // be reached in practice).
    const INF: f64 = 1e18;
    let mut mat = vec![vec![INF; m + 1]; n + 1];
    let mut xg = vec![vec![INF; m + 1]; n + 1]; // gap in baseline (insertion)
    let mut yg = vec![vec![INF; m + 1]; n + 1]; // gap in candidate (deletion)
    let mut back = vec![vec![Step::Match(0, 0); m + 1]; n + 1];

    mat[0][0] = 0.0;
    for i in 1..=n {
        yg[i][0] = GAP_OPEN + (i as f64 - 1.0) * GAP_EXTEND;
        mat[i][0] = yg[i][0];
        back[i][0] = Step::DeleteBaseline(i - 1);
    }
    for j in 1..=m {
        xg[0][j] = GAP_OPEN + (j as f64 - 1.0) * GAP_EXTEND;
        mat[0][j] = xg[0][j];
        back[0][j] = Step::InsertCandidate(j - 1);
    }

    for i in 1..=n {
        for j in 1..=m {
            let c = pair_cost(baseline[i - 1], candidate[j - 1]);
            // Match path: best of (prev-match, prev-xgap, prev-ygap) + pair cost.
            let m_cost = mat[i - 1][j - 1]
                .min(xg[i - 1][j - 1])
                .min(yg[i - 1][j - 1])
                + c;
            // Open a horizontal gap (insertion on candidate) or extend one.
            let xg_cost = (mat[i][j - 1] + GAP_OPEN).min(xg[i][j - 1] + GAP_EXTEND);
            // Open a vertical gap (deletion on baseline) or extend one.
            let yg_cost = (mat[i - 1][j] + GAP_OPEN).min(yg[i - 1][j] + GAP_EXTEND);
            mat[i][j] = m_cost;
            xg[i][j] = xg_cost;
            yg[i][j] = yg_cost;
            // Record the back-pointer for the cell's *minimum* overall
            // reachable cost — we walk the cheapest path.
            let best = m_cost.min(xg_cost).min(yg_cost);
            back[i][j] = if (best - m_cost).abs() < 1e-12 {
                Step::Match(i - 1, j - 1)
            } else if (best - xg_cost).abs() < 1e-12 {
                Step::InsertCandidate(j - 1)
            } else {
                Step::DeleteBaseline(i - 1)
            };
        }
    }

    // Traceback from (n, m) to (0, 0).
    let mut steps = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        let s = back[i][j];
        steps.push(s);
        match s {
            Step::Match(_, _) => {
                i -= 1;
                j -= 1;
            }
            Step::InsertCandidate(_) => {
                j -= 1;
            }
            Step::DeleteBaseline(_) => {
                i -= 1;
            }
        }
    }
    steps.reverse();
    Alignment { steps }
}

// ---------------------------------------------------------------------------
// Per-cell cost
// ---------------------------------------------------------------------------

/// Cost of aligning one baseline response with one candidate response.
/// Always returns a value in [0, 1].
fn pair_cost(a: &Record, b: &Record) -> f64 {
    let tool_shape_a = tool_shape(a);
    let tool_shape_b = tool_shape(b);
    let structural = 1.0 - jaccard(&tool_shape_a, &tool_shape_b);

    let text_a = response_text(a);
    let text_b = response_text(b);
    let semantic = 1.0 - text_similarity(&text_a, &text_b);

    let stop_a = stop_reason(a);
    let stop_b = stop_reason(b);
    let stop = if stop_a != stop_b { 1.0 } else { 0.0 };

    // Arg-value divergence: same tool name AND same arg-key set on
    // both sides, but different arg VALUES. Without this component
    // we'd miss the "`search(limit=10)` → `search(limit=50)`" case
    // because structural + stop + (empty-text) semantic are all 0.
    let args = if tool_shape_a == tool_shape_b && !tool_shape_a.is_empty() {
        if arg_value_diff(a, b).is_some() {
            1.0
        } else {
            0.0
        }
    } else {
        0.0
    };

    W_STRUCT * structural + W_SEM * semantic + W_STOP * stop + W_ARGS * args
}

/// Extract a canonical tool-shape token per tool_use block. The token
/// is `"<tool_name>(<sorted-comma-arg-keys>)"` — captures both the
/// tool called and the KEYS (not values) of its input. Values are
/// compared separately by `arg_values_differ`.
fn tool_shape(r: &Record) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let Some(arr) = r.payload.get("content").and_then(|c| c.as_array()) else {
        return out;
    };
    for part in arr {
        if part.get("type").and_then(|t| t.as_str()) != Some("tool_use") {
            continue;
        }
        let name = part.get("name").and_then(|n| n.as_str()).unwrap_or("_");
        let mut keys: Vec<String> = part
            .get("input")
            .and_then(|i| i.as_object())
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default();
        keys.sort();
        out.insert(format!("{name}({})", keys.join(",")));
    }
    out
}

fn response_text(r: &Record) -> String {
    let Some(arr) = r.payload.get("content").and_then(|c| c.as_array()) else {
        return String::new();
    };
    arr.iter()
        .filter_map(|p| {
            if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                p.get("text")
                    .and_then(|t| t.as_str())
                    .map(ToString::to_string)
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn stop_reason(r: &Record) -> String {
    r.payload
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Jaccard similarity on two string sets. Returns 1.0 for two empty
/// sets (both sides produced no tool calls — they agree structurally).
fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f64;
    let uni = a.union(b).count() as f64;
    if uni == 0.0 {
        1.0
    } else {
        inter / uni
    }
}

/// Lightweight text-similarity proxy: character-shingle Jaccard. We
/// avoid bringing in embeddings here because this module is in the
/// Rust core and must not take a heavy ML dep. The Python layer can
/// upgrade this via a similarity callback in v0.2.
///
/// For empty strings, returns 1.0 (both empty → identical).
fn text_similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a == b {
        return 1.0;
    }
    let sa = shingles(a, 4);
    let sb = shingles(b, 4);
    jaccard(&sa, &sb)
}

fn shingles(s: &str, k: usize) -> BTreeSet<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = BTreeSet::new();
    if chars.len() < k {
        if !s.is_empty() {
            out.insert(s.to_string());
        }
        return out;
    }
    for w in chars.windows(k) {
        out.insert(w.iter().collect());
    }
    out
}

// ---------------------------------------------------------------------------
// Walk the alignment for the first divergence
// ---------------------------------------------------------------------------

fn walk_for_first_divergence(
    alignment: &Alignment,
    baseline: &[&Record],
    candidate: &[&Record],
) -> Option<FirstDivergence> {
    // Track cursors through the walk so that gap steps can report the
    // baseline / candidate position correctly — without this, an
    // insertion on the candidate side can't tell which baseline turn
    // it lived BETWEEN.
    let mut b_cursor: usize = 0;
    let mut c_cursor: usize = 0;
    for step in &alignment.steps {
        match *step {
            Step::InsertCandidate(j) => {
                // Candidate inserted a turn the baseline didn't have —
                // structural by definition. The insertion sits between
                // `b_cursor - 1` and `b_cursor` on the baseline side.
                let cand = candidate[j];
                let insertion_point = b_cursor;
                return Some(FirstDivergence {
                    baseline_turn: insertion_point,
                    candidate_turn: j,
                    kind: DivergenceKind::Structural,
                    primary_axis: Axis::Trajectory,
                    explanation: format!(
                        "candidate turn {j} has no baseline counterpart — inserted {} tool call(s) / response",
                        tool_shape(cand).len()
                    ),
                    confidence: 1.0,
                });
            }
            Step::DeleteBaseline(i) => {
                let b = baseline[i];
                let deletion_point = c_cursor;
                return Some(FirstDivergence {
                    baseline_turn: i,
                    candidate_turn: deletion_point,
                    kind: DivergenceKind::Structural,
                    primary_axis: Axis::Trajectory,
                    explanation: format!(
                        "baseline turn {i} has no candidate counterpart — candidate dropped {} tool call(s) / response",
                        tool_shape(b).len()
                    ),
                    confidence: 1.0,
                });
            }
            Step::Match(i, j) => {
                let b = baseline[i];
                let c = candidate[j];
                let cost = pair_cost(b, c);
                b_cursor = i.saturating_add(1);
                c_cursor = j.saturating_add(1);
                if cost <= NOISE_FLOOR {
                    continue;
                }
                // Above noise floor — classify.
                let (kind, axis, explanation) = classify(b, c, cost);
                let confidence = ((cost - NOISE_FLOOR) / (1.0 - NOISE_FLOOR)).clamp(0.0, 1.0);
                return Some(FirstDivergence {
                    baseline_turn: i,
                    candidate_turn: j,
                    kind,
                    primary_axis: axis,
                    explanation,
                    confidence,
                });
            }
        }
    }
    None
}

/// Classify a significant (above-noise-floor) matched pair.
fn classify(b: &Record, c: &Record, cost: f64) -> (DivergenceKind, Axis, String) {
    let shape_b = tool_shape(b);
    let shape_c = tool_shape(c);
    let text_b = response_text(b);
    let text_c = response_text(c);
    let stop_b = stop_reason(b);
    let stop_c = stop_reason(c);
    let sem_sim = text_similarity(&text_b, &text_c);

    // Structural: tool shapes differ (name or arg-key set).
    if shape_b != shape_c {
        let explanation = describe_tool_diff(&shape_b, &shape_c);
        return (DivergenceKind::Structural, Axis::Trajectory, explanation);
    }

    // Stop reason flipped — often signals refusal / filter.
    if stop_b != stop_c {
        return (
            DivergenceKind::Decision,
            Axis::Safety,
            format!("stop_reason changed: `{stop_b}` → `{stop_c}`"),
        );
    }

    // Tool shape matches and stop_reason matches but something still
    // drives a divergence. Two sub-cases:
    //   - tool_use.input values differ (same keys, different values)
    //   - response text diverged
    if let Some(arg_diff) = arg_value_diff(b, c) {
        return (
            DivergenceKind::Decision,
            Axis::Trajectory,
            format!("tool arg value changed: {arg_diff}"),
        );
    }

    // Pure text divergence. Style vs decision depends on similarity.
    if sem_sim >= 0.90 && cost <= STYLE_MAX_COST {
        (
            DivergenceKind::Style,
            Axis::Semantic,
            "cosmetic wording change — tool sequence and semantics preserved".to_string(),
        )
    } else {
        (
            DivergenceKind::Decision,
            Axis::Semantic,
            format!(
                "response text diverged (text similarity {:.2}); same tool sequence",
                sem_sim
            ),
        )
    }
}

fn describe_tool_diff(a: &BTreeSet<String>, b: &BTreeSet<String>) -> String {
    let only_a: Vec<&String> = a.difference(b).collect();
    let only_b: Vec<&String> = b.difference(a).collect();
    if !only_a.is_empty() && only_b.is_empty() {
        format!("candidate dropped tool call(s): {}", list(&only_a))
    } else if !only_b.is_empty() && only_a.is_empty() {
        format!("candidate added tool call(s): {}", list(&only_b))
    } else if !only_a.is_empty() && !only_b.is_empty() {
        format!(
            "tool set changed: removed {}, added {}",
            list(&only_a),
            list(&only_b)
        )
    } else {
        "tool ordering differs".to_string()
    }
}

fn list(items: &[&String]) -> String {
    items
        .iter()
        .map(|s| format!("`{s}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Compare arg values for tools that have the same name and arg keys.
/// Returns `Some(summary)` if any tool's values differ, `None` if every
/// tool's values match.
fn arg_value_diff(a: &Record, b: &Record) -> Option<String> {
    let ta = tool_use_inputs(a);
    let tb = tool_use_inputs(b);
    for (name, va) in &ta {
        if let Some(vb) = tb.get(name) {
            if va != vb {
                // Find the first differing key.
                if let (Some(oa), Some(ob)) = (va.as_object(), vb.as_object()) {
                    for (k, v) in oa {
                        if ob.get(k) != Some(v) {
                            let other = ob
                                .get(k)
                                .map(|x| x.to_string())
                                .unwrap_or("<missing>".to_string());
                            return Some(format!("`{name}({k})`: `{v}` → `{other}`"));
                        }
                    }
                }
                return Some(format!("`{name}`: input changed"));
            }
        }
    }
    None
}

/// Index a chat_response's tool_use blocks by tool_name → input value.
/// First occurrence wins if a tool is called twice in the same turn.
fn tool_use_inputs(r: &Record) -> std::collections::BTreeMap<String, serde_json::Value> {
    let mut out = std::collections::BTreeMap::new();
    let Some(arr) = r.payload.get("content").and_then(|c| c.as_array()) else {
        return out;
    };
    for part in arr {
        if part.get("type").and_then(|t| t.as_str()) != Some("tool_use") {
            continue;
        }
        let name = part
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("_")
            .to_string();
        let input = part
            .get("input")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        out.entry(name).or_insert(input);
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::Kind;
    use serde_json::json;

    fn response_text_only(text: &str, stop: &str) -> Record {
        Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": [{"type": "text", "text": text}],
                "stop_reason": stop,
                "latency_ms": 0,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }),
            "2026-04-23T00:00:00Z",
            None,
        )
    }

    fn response_with_tool(name: &str, input: serde_json::Value, stop: &str) -> Record {
        Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": [{
                    "type": "tool_use",
                    "id": "t1",
                    "name": name,
                    "input": input,
                }],
                "stop_reason": stop,
                "latency_ms": 0,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }),
            "2026-04-23T00:00:00Z",
            None,
        )
    }

    fn meta() -> Record {
        Record::new(
            Kind::Metadata,
            json!({"sdk": {"name": "shadow"}}),
            "2026-04-23T00:00:00Z",
            None,
        )
    }

    #[test]
    fn identical_traces_return_none() {
        let r = response_text_only("Paris is the capital of France.", "end_turn");
        let baseline = vec![meta(), r.clone(), r.clone()];
        let candidate = vec![meta(), r.clone(), r.clone()];
        assert_eq!(detect(&baseline, &candidate), None);
    }

    #[test]
    fn whitespace_only_diff_is_style() {
        let b = response_text_only("Paris is the capital of France.", "end_turn");
        let c = response_text_only("Paris is  the capital of France.", "end_turn");
        let baseline = vec![meta(), b];
        let candidate = vec![meta(), c];
        // At shingle-level, whitespace variance is tiny but nonzero.
        // Classification depends on cost; this case is typically below
        // NOISE_FLOOR in practice. The key assertion is that if ANY
        // divergence is reported, it's Style, not Structural/Decision.
        if let Some(d) = detect(&baseline, &candidate) {
            assert_eq!(d.kind, DivergenceKind::Style);
            assert_eq!(d.primary_axis, Axis::Semantic);
        }
    }

    #[test]
    fn different_tool_name_is_structural_on_trajectory_axis() {
        let b = response_with_tool("search", json!({"q": "cats"}), "tool_use");
        let c = response_with_tool("lookup", json!({"q": "cats"}), "tool_use");
        let baseline = vec![meta(), b];
        let candidate = vec![meta(), c];
        let d = detect(&baseline, &candidate).expect("divergence expected");
        assert_eq!(d.kind, DivergenceKind::Structural);
        assert_eq!(d.primary_axis, Axis::Trajectory);
        assert_eq!(d.baseline_turn, 0);
        assert_eq!(d.candidate_turn, 0);
        assert!(d.explanation.contains("search") || d.explanation.contains("lookup"));
    }

    #[test]
    fn same_tool_different_arg_value_is_decision() {
        let b = response_with_tool("search", json!({"q": "cats", "limit": 10}), "tool_use");
        let c = response_with_tool("search", json!({"q": "cats", "limit": 50}), "tool_use");
        let baseline = vec![meta(), b];
        let candidate = vec![meta(), c];
        let d = detect(&baseline, &candidate).expect("divergence expected");
        assert_eq!(d.kind, DivergenceKind::Decision);
        assert_eq!(d.primary_axis, Axis::Trajectory);
        assert!(d.explanation.contains("limit"));
    }

    #[test]
    fn stop_reason_flip_is_decision_on_safety() {
        let b = response_text_only("Here is the answer.", "end_turn");
        let c = response_text_only("I can't help with that.", "content_filter");
        let baseline = vec![meta(), b];
        let candidate = vec![meta(), c];
        let d = detect(&baseline, &candidate).expect("divergence expected");
        assert_eq!(d.kind, DivergenceKind::Decision);
        assert_eq!(d.primary_axis, Axis::Safety);
        assert!(d.explanation.contains("end_turn"));
        assert!(d.explanation.contains("content_filter"));
    }

    #[test]
    fn candidate_drops_a_turn_is_structural() {
        let r1 = response_text_only("first turn", "end_turn");
        let r2 = response_text_only("second turn", "end_turn");
        let baseline = vec![meta(), r1.clone(), r2];
        let candidate = vec![meta(), r1]; // dropped the second
        let d = detect(&baseline, &candidate).expect("divergence expected");
        assert_eq!(d.kind, DivergenceKind::Structural);
        assert_eq!(d.primary_axis, Axis::Trajectory);
    }

    #[test]
    fn candidate_inserts_a_turn_is_structural() {
        let r1 = response_text_only("turn one", "end_turn");
        let r2 = response_text_only("inserted", "end_turn");
        let r3 = response_text_only("turn two", "end_turn");
        let baseline = vec![meta(), r1.clone(), r3.clone()];
        let candidate = vec![meta(), r1, r2, r3];
        let d = detect(&baseline, &candidate).expect("divergence expected");
        assert_eq!(d.kind, DivergenceKind::Structural);
    }

    #[test]
    fn significant_text_shift_is_decision_on_semantic() {
        // Different topics entirely — semantic similarity low, no tools.
        let b = response_text_only(
            "Photosynthesis is the process by which plants convert sunlight.",
            "end_turn",
        );
        let c = response_text_only(
            "The stock market closed higher on Thursday after strong earnings.",
            "end_turn",
        );
        let baseline = vec![meta(), b];
        let candidate = vec![meta(), c];
        let d = detect(&baseline, &candidate).expect("divergence expected");
        assert_eq!(d.kind, DivergenceKind::Decision);
        assert_eq!(d.primary_axis, Axis::Semantic);
    }

    #[test]
    fn empty_traces_return_none() {
        assert_eq!(detect(&[meta()], &[meta()]), None);
        assert_eq!(detect(&[], &[]), None);
    }

    #[test]
    fn first_divergence_is_truly_first() {
        // Three turns; only the SECOND differs. Detector must locate
        // turn index 1, not turn 2.
        let r1 = response_text_only("same", "end_turn");
        let r2b = response_text_only("baseline version of turn two with lots of text", "end_turn");
        let r2c = response_text_only(
            "CANDIDATE SAID SOMETHING COMPLETELY DIFFERENT HERE",
            "end_turn",
        );
        let r3 = response_text_only("also same", "end_turn");
        let baseline = vec![meta(), r1.clone(), r2b, r3.clone()];
        let candidate = vec![meta(), r1, r2c, r3];
        let d = detect(&baseline, &candidate).expect("divergence expected");
        assert_eq!(d.baseline_turn, 1);
        assert_eq!(d.candidate_turn, 1);
    }

    #[test]
    fn confidence_is_in_valid_range() {
        let b = response_with_tool("search", json!({"q": "a"}), "tool_use");
        let c = response_with_tool("other", json!({"q": "a"}), "tool_use");
        let baseline = vec![meta(), b];
        let candidate = vec![meta(), c];
        let d = detect(&baseline, &candidate).unwrap();
        assert!((0.0..=1.0).contains(&d.confidence));
    }

    #[test]
    fn tool_shape_captures_name_and_arg_keys() {
        let r = response_with_tool("search", json!({"q": "a", "limit": 10}), "tool_use");
        let shape = tool_shape(&r);
        assert_eq!(shape.len(), 1);
        let entry = shape.iter().next().unwrap();
        assert!(entry.starts_with("search("));
        assert!(entry.contains("limit"));
        assert!(entry.contains("q"));
    }

    #[test]
    fn jaccard_on_empty_sets_is_one() {
        let empty = BTreeSet::new();
        assert_eq!(jaccard(&empty, &empty), 1.0);
    }

    #[test]
    fn alignment_prefers_matches_over_gaps_when_both_cheap() {
        // Two identical turns. NW should produce two Match steps and
        // no gaps.
        let r = response_text_only("same", "end_turn");
        let alignment = align(&[&r, &r], &[&r, &r]);
        let matches = alignment
            .steps
            .iter()
            .filter(|s| matches!(s, Step::Match(..)))
            .count();
        assert_eq!(matches, 2);
        let gaps = alignment.steps.len() - matches;
        assert_eq!(gaps, 0);
    }
}
