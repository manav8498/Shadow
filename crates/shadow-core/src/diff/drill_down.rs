//! Per-pair drill-down: surfaces which specific turn in the paired
//! trace set drove each aggregate axis regression.
//!
//! The nine-axis report is informative in aggregate but hides *where*
//! the regression happened. A reviewer looking at a PR with 50 trace
//! pairs sees `trajectory: delta +0.42, severe` but has to hand-audit
//! every pair to find which ones actually regressed. This module
//! computes per-pair, per-axis deltas and returns the top-K
//! most-regressive pairs ranked by a normalised aggregate score.
//!
//! Design choices:
//!
//! 1. **No bootstrap per pair.** Bootstrap CIs are an aggregate-level
//!    stat; per-pair we need only the raw deltas. This keeps drill-down
//!    cheap (O(N) extraction per axis, no resampling).
//!
//! 2. **Self-contained extractors.** Rather than refactor the nine axis
//!    modules to expose their per-pair internals, we re-implement the
//!    (small) extractors here. Each is ≤ 20 lines; duplicating them
//!    buys independence — drill-down's value function can evolve
//!    without touching the statistical axis implementations.
//!
//! 3. **Normalised ranking.** Raw deltas have wildly different scales
//!    (0-1 for semantic, 0-10000 for latency_ms). Each axis has a
//!    per-axis `scale` used to normalise deltas into [0, ~1] so they
//!    can be summed into a single regression score. Scales are
//!    calibrated against the axis's Severity::Severe threshold so a
//!    per-pair normalised delta of 1.0 corresponds roughly to one
//!    severity-severe-sized movement.
//!
//! 4. **Text similarity via character-shingle Jaccard.** The aggregate
//!    semantic axis uses corpus-level TF-IDF, which is meaningless per
//!    singleton-pair. Character-shingle Jaccard (what alignment.rs uses)
//!    is the same proxy first-divergence detection uses, so drill-down
//!    results are internally consistent with the first-divergence row.
//!
//! 5. **Judge is skipped.** The Rust core never populates the Judge
//!    axis (it's Python-side). Including it here would produce spurious
//!    zeros for every pair.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::agentlog::{Kind, Record};
use crate::diff::axes::Axis;
use crate::diff::cost::Pricing;

/// Default number of pairs to surface in a drill-down list. Matches
/// `alignment::DEFAULT_K` so downstream renderers can share the same
/// "show 3 inline, collapse the rest" heuristic.
pub const DEFAULT_K: usize = 5;

/// One axis's contribution to a single pair's drill-down row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairAxisScore {
    /// Which axis this score describes.
    pub axis: Axis,
    /// Axis-specific baseline value in raw units (ms, tokens, USD,
    /// similarity ratio, …).
    pub baseline_value: f64,
    /// Axis-specific candidate value in the same units as
    /// `baseline_value`.
    pub candidate_value: f64,
    /// `candidate_value - baseline_value`. Sign is direction; magnitude
    /// is raw axis units (ms, tokens, USD, …).
    pub delta: f64,
    /// `|delta| / axis_scale`, clamped to `[0, 4]`. Used as the per-axis
    /// contribution to the pair's `regression_score`. 0 means "no
    /// movement"; ~1 means "one severity-severe-sized movement on this
    /// axis"; ≥ 2 is unambiguous regression.
    pub normalized_delta: f64,
}

/// One pair's per-axis breakdown plus an aggregate regression score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairDrilldown {
    /// 0-based index into the paired-responses list.
    pub pair_index: usize,
    /// The turn number in the baseline trace (counting only chat_responses).
    pub baseline_turn: usize,
    /// The turn number in the candidate trace.
    pub candidate_turn: usize,
    /// Per-axis scores, in `Axis::all()` order (minus Judge).
    pub axis_scores: Vec<PairAxisScore>,
    /// Sum of `normalized_delta` across all included axes. Ranking key.
    pub regression_score: f64,
    /// The single axis that contributed the most to `regression_score`.
    /// Useful for "the regression at turn 4 was a trajectory change" one-
    /// liners in renderers.
    pub dominant_axis: Axis,
}

/// Compute drill-down rows for every pair, return the top-`top_k`
/// sorted by `regression_score` descending.
///
/// `top_k = 0` or `top_k >= pairs.len()` returns every pair.
pub fn compute(
    pairs: &[(&Record, &Record)],
    pricing: &Pricing,
    top_k: usize,
) -> Vec<PairDrilldown> {
    let mut rows: Vec<PairDrilldown> = pairs
        .iter()
        .enumerate()
        .map(|(i, (b, c))| compute_pair(i, b, c, pricing))
        .collect();
    // Stable sort by regression_score descending; ties broken by
    // pair_index ascending so output is deterministic.
    rows.sort_by(|a, b| {
        b.regression_score
            .partial_cmp(&a.regression_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.pair_index.cmp(&b.pair_index))
    });
    if top_k > 0 && rows.len() > top_k {
        rows.truncate(top_k);
    }
    rows
}

fn compute_pair(index: usize, b: &Record, c: &Record, pricing: &Pricing) -> PairDrilldown {
    let scores: Vec<PairAxisScore> = vec![
        axis_semantic(b, c),
        axis_trajectory(b, c),
        axis_safety(b, c),
        axis_verbosity(b, c),
        axis_latency(b, c),
        axis_cost(b, c, pricing),
        axis_reasoning(b, c),
        axis_conformance(b, c),
    ];
    let regression_score: f64 = scores.iter().map(|s| s.normalized_delta).sum();
    let dominant_axis = scores
        .iter()
        .max_by(|a, b| {
            a.normalized_delta
                .partial_cmp(&b.normalized_delta)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|s| s.axis)
        .unwrap_or(Axis::Semantic);
    PairDrilldown {
        pair_index: index,
        baseline_turn: index,
        candidate_turn: index,
        axis_scores: scores,
        regression_score,
        dominant_axis,
    }
}

// ---- per-axis extractors -------------------------------------------------
//
// Each returns a `PairAxisScore` with a normalised-delta scaled so that
// 1.0 corresponds to one severity-severe-sized movement on that axis.
// Scales are chosen to match the thresholds in `axes.rs::Severity::from_*`.

/// Semantic axis: 1 − character-shingle-4 Jaccard similarity.
/// Returns 0 for identical responses, 1 for totally disjoint.
fn axis_semantic(b: &Record, c: &Record) -> PairAxisScore {
    let sim = text_jaccard(&response_text(b), &response_text(c));
    let delta = (1.0 - sim) - 0.0; // baseline "similarity to self" = 1
    PairAxisScore {
        axis: Axis::Semantic,
        baseline_value: 1.0,
        candidate_value: sim,
        delta: sim - 1.0, // delta as "how far candidate sim drifted from 1"
        normalized_delta: clamp_norm(delta / 0.5),
    }
}

/// Trajectory axis: normalised Levenshtein over tool-shape sequence.
fn axis_trajectory(b: &Record, c: &Record) -> PairAxisScore {
    let bs = tool_shape_seq(b);
    let cs = tool_shape_seq(c);
    let div = normalised_edit_distance(&bs, &cs);
    PairAxisScore {
        axis: Axis::Trajectory,
        baseline_value: 0.0,
        candidate_value: div,
        delta: div,
        normalized_delta: clamp_norm(div / 0.5),
    }
}

/// Safety axis: binary refusal indicator per side.
fn axis_safety(b: &Record, c: &Record) -> PairAxisScore {
    let br = is_refusal(b) as i32 as f64;
    let cr = is_refusal(c) as i32 as f64;
    PairAxisScore {
        axis: Axis::Safety,
        baseline_value: br,
        candidate_value: cr,
        delta: cr - br,
        normalized_delta: clamp_norm((cr - br).abs()),
    }
}

/// Verbosity axis: output_tokens.
fn axis_verbosity(b: &Record, c: &Record) -> PairAxisScore {
    let bv = output_tokens(b).unwrap_or(0.0);
    let cv = output_tokens(c).unwrap_or(0.0);
    PairAxisScore {
        axis: Axis::Verbosity,
        baseline_value: bv,
        candidate_value: cv,
        delta: cv - bv,
        // One severe verbosity shift ≈ 100-token delta (calibrated
        // against the severity thresholds in axes.rs).
        normalized_delta: clamp_norm((cv - bv).abs() / 100.0),
    }
}

/// Latency axis: latency_ms.
fn axis_latency(b: &Record, c: &Record) -> PairAxisScore {
    let bv = latency_ms(b).unwrap_or(0.0);
    let cv = latency_ms(c).unwrap_or(0.0);
    PairAxisScore {
        axis: Axis::Latency,
        baseline_value: bv,
        candidate_value: cv,
        delta: cv - bv,
        // One severe latency shift ≈ 1000ms delta.
        normalized_delta: clamp_norm((cv - bv).abs() / 1000.0),
    }
}

/// Cost axis: tokens × pricing for this model.
fn axis_cost(b: &Record, c: &Record, pricing: &Pricing) -> PairAxisScore {
    let bc = cost_of(b, pricing);
    let cc = cost_of(c, pricing);
    PairAxisScore {
        axis: Axis::Cost,
        baseline_value: bc,
        candidate_value: cc,
        delta: cc - bc,
        // One severe cost shift ≈ $0.01 delta per pair.
        normalized_delta: clamp_norm((cc - bc).abs() / 0.01),
    }
}

/// Reasoning axis: thinking_tokens from usage.
fn axis_reasoning(b: &Record, c: &Record) -> PairAxisScore {
    let bv = thinking_tokens(b).unwrap_or(0.0);
    let cv = thinking_tokens(c).unwrap_or(0.0);
    PairAxisScore {
        axis: Axis::Reasoning,
        baseline_value: bv,
        candidate_value: cv,
        delta: cv - bv,
        normalized_delta: clamp_norm((cv - bv).abs() / 100.0),
    }
}

/// Conformance axis: does the response body parse as JSON? Binary.
fn axis_conformance(b: &Record, c: &Record) -> PairAxisScore {
    let bp = parses_as_json(&response_text(b)) as i32 as f64;
    let cp = parses_as_json(&response_text(c)) as i32 as f64;
    PairAxisScore {
        axis: Axis::Conformance,
        baseline_value: bp,
        candidate_value: cp,
        delta: cp - bp,
        // A loss of JSON parseability is an outright severe signal.
        normalized_delta: clamp_norm((cp - bp).abs()),
    }
}

// ---- small extractors (self-contained so drill-down owns its logic) ------

fn response_text(r: &Record) -> String {
    if r.kind != Kind::ChatResponse {
        return String::new();
    }
    let arr = match r.payload.get("content").and_then(|c| c.as_array()) {
        Some(a) => a,
        None => return String::new(),
    };
    let mut out = String::new();
    for part in arr {
        if part.get("type").and_then(|t| t.as_str()) == Some("text") {
            if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(t);
            }
        }
    }
    out
}

fn tool_shape_seq(r: &Record) -> Vec<String> {
    let arr = match r.payload.get("content").and_then(|c| c.as_array()) {
        Some(a) => a,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    for part in arr {
        if part.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
            let name = part
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("_")
                .to_string();
            let mut keys: Vec<String> = part
                .get("input")
                .and_then(|i| i.as_object())
                .map(|o| o.keys().cloned().collect())
                .unwrap_or_default();
            keys.sort();
            out.push(format!("{name}({})", keys.join(",")));
        }
    }
    out
}

fn latency_ms(r: &Record) -> Option<f64> {
    r.payload.get("latency_ms").and_then(|v| v.as_f64())
}

fn output_tokens(r: &Record) -> Option<f64> {
    r.payload
        .get("usage")
        .and_then(|u| u.get("output_tokens"))
        .and_then(|v| v.as_f64())
}

fn thinking_tokens(r: &Record) -> Option<f64> {
    r.payload
        .get("usage")
        .and_then(|u| u.get("thinking_tokens"))
        .and_then(|v| v.as_f64())
}

fn is_refusal(r: &Record) -> bool {
    match r.payload.get("stop_reason").and_then(|s| s.as_str()) {
        Some("content_filter") | Some("refusal") => return true,
        _ => {}
    }
    let text = response_text(r).to_lowercase();
    // Conservative refusal indicators (matching safety axis heuristics).
    text.contains("i can't help")
        || text.contains("i cannot help")
        || text.contains("i'm unable")
        || text.contains("i am unable")
        || text.contains("i won't")
        || text.contains("i will not")
}

fn parses_as_json(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    // Accept values wrapped in code fences too (mirrors conformance
    // axis's tolerance).
    let unfenced = if let Some(s) = trimmed.strip_prefix("```json") {
        s.trim().trim_end_matches("```").trim()
    } else if let Some(s) = trimmed.strip_prefix("```") {
        s.trim().trim_end_matches("```").trim()
    } else {
        trimmed
    };
    serde_json::from_str::<serde_json::Value>(unfenced).is_ok()
}

fn cost_of(r: &Record, pricing: &Pricing) -> f64 {
    crate::diff::cost::cost_of(r, pricing).unwrap_or(0.0)
}

// ---- shared helpers ------------------------------------------------------

fn clamp_norm(v: f64) -> f64 {
    if v.is_nan() {
        return 0.0;
    }
    v.abs().min(4.0)
}

fn text_jaccard(a: &str, b: &str) -> f64 {
    let sa = shingles(a, 4);
    let sb = shingles(b, 4);
    if sa.is_empty() && sb.is_empty() {
        return 1.0;
    }
    let inter = sa.intersection(&sb).count() as f64;
    let uni = sa.union(&sb).count() as f64;
    if uni == 0.0 {
        1.0
    } else {
        inter / uni
    }
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

fn normalised_edit_distance(a: &[String], b: &[String]) -> f64 {
    let denom = a.len().max(b.len());
    if denom == 0 {
        return 0.0;
    }
    levenshtein(a, b) as f64 / denom as f64
}

fn levenshtein(a: &[String], b: &[String]) -> usize {
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::Kind;
    use serde_json::json;

    fn resp(latency: u64, out_tokens: u64, text: &str) -> Record {
        Record::new(
            Kind::ChatResponse,
            json!({
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "latency_ms": latency,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": out_tokens,
                    "thinking_tokens": 0,
                },
            }),
            "2026-04-21T10:00:00Z",
            None,
        )
    }

    #[test]
    fn identical_responses_have_zero_regression() {
        let r = resp(100, 20, "hello world");
        let pairs = vec![(&r, &r)];
        let out = compute(&pairs, &Pricing::new(), 0);
        assert_eq!(out.len(), 1);
        assert!(
            out[0].regression_score < 0.01,
            "expected near-zero, got {}",
            out[0].regression_score
        );
    }

    #[test]
    fn divergent_pair_scores_higher_than_matched_pair() {
        let match_a = resp(100, 20, "hello world");
        let match_b = resp(100, 20, "hello world");
        let diverge_a = resp(100, 20, "hello world");
        let diverge_b = resp(2500, 200, "totally different output");
        let pairs = vec![(&match_a, &match_b), (&diverge_a, &diverge_b)];
        let out = compute(&pairs, &Pricing::new(), 0);
        assert_eq!(out.len(), 2);
        // First in the sorted output is the divergent pair.
        assert_eq!(out[0].pair_index, 1);
        assert!(out[0].regression_score > out[1].regression_score);
    }

    #[test]
    fn top_k_truncates_result_list() {
        let rs: Vec<Record> = (0..10)
            .map(|i| resp(100 + i * 50, 20, &format!("response {}", i)))
            .collect();
        let pairs: Vec<(&Record, &Record)> = rs.iter().zip(rs.iter().rev()).collect();
        let out = compute(&pairs, &Pricing::new(), 3);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn ranking_is_deterministic_on_ties() {
        // Two pairs with identical regression: tie-break by pair_index asc.
        let a = resp(100, 20, "hello");
        let b = resp(200, 30, "hello");
        let pairs = vec![(&a, &b), (&a, &b), (&a, &b)];
        let out1 = compute(&pairs, &Pricing::new(), 0);
        let out2 = compute(&pairs, &Pricing::new(), 0);
        assert_eq!(out1, out2);
        assert_eq!(
            out1.iter().map(|r| r.pair_index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn tool_shape_change_surfaces_trajectory_as_dominant() {
        let baseline = Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": [
                    {"type": "tool_use", "name": "search", "input": {"query": "x"}},
                ],
                "stop_reason": "end_turn",
                "latency_ms": 100,
                "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0},
            }),
            "ts",
            None,
        );
        let candidate = Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": [
                    {"type": "tool_use", "name": "fetch", "input": {"url": "x"}},
                ],
                "stop_reason": "end_turn",
                "latency_ms": 100,
                "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0},
            }),
            "ts",
            None,
        );
        let pairs = vec![(&baseline, &candidate)];
        let out = compute(&pairs, &Pricing::new(), 0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].dominant_axis, Axis::Trajectory);
    }

    #[test]
    fn refusal_surfaces_safety_axis() {
        let b = resp(100, 20, "Here you go.");
        let c = resp(100, 20, "I can't help with that.");
        let pairs = vec![(&b, &c)];
        let out = compute(&pairs, &Pricing::new(), 0);
        let safety = out[0]
            .axis_scores
            .iter()
            .find(|s| s.axis == Axis::Safety)
            .unwrap();
        assert!(
            safety.normalized_delta > 0.5,
            "expected safety axis to fire, got {}",
            safety.normalized_delta
        );
    }

    #[test]
    fn json_loss_surfaces_conformance_axis() {
        let b = resp(100, 20, r#"{"ok": true}"#);
        let c = resp(100, 20, "sure thing");
        let pairs = vec![(&b, &c)];
        let out = compute(&pairs, &Pricing::new(), 0);
        let conf = out[0]
            .axis_scores
            .iter()
            .find(|s| s.axis == Axis::Conformance)
            .unwrap();
        assert_eq!(conf.baseline_value, 1.0);
        assert_eq!(conf.candidate_value, 0.0);
        assert!(conf.normalized_delta > 0.5);
    }

    #[test]
    fn empty_pairs_returns_empty_vec() {
        let pairs: Vec<(&Record, &Record)> = Vec::new();
        let out = compute(&pairs, &Pricing::new(), 5);
        assert!(out.is_empty());
    }
}
