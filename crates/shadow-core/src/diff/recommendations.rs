//! Prescriptive fix recommendations derived from a [`DiffReport`].
//!
//! Once Shadow has detected divergences and ranked them (see
//! [`crate::diff::alignment`]), this module turns that raw signal into
//! a **list of actionable next-steps a reviewer can act on in under
//! 30 seconds** — the "what do I do about this?" layer. It replaces
//! the generic "investigate turn 9" instinct with specific moves like
//! *"Restore `send_confirmation_email`"* or *"Revert refund amount
//! 9.99 → 99.99"*.
//!
//! ## Design decisions
//!
//! - **Deterministic, rule-based.** No LLM dependency in the core crate.
//!   The recommendation engine maps each divergence shape to one or
//!   more specific recommendations via pure pattern matching. An LLM-
//!   assisted "enriched" suggestion layer can live in the Python
//!   optional-extras path later, but the rule-based core must always
//!   work offline.
//! - **Actionable phrasing.** Every recommendation starts with a verb
//!   (Restore / Remove / Revert / Review / Verify). No "consider
//!   investigating" or "you might want to". If the rule can't decide
//!   whether action is safe, it uses `Review` + rationale; it doesn't
//!   recommend a specific action it can't justify.
//! - **Hedged for non-obvious cases.** Style drift → Info severity.
//!   Low-confidence signals are tagged with `Verify` action rather
//!   than directive `Restore` / `Revert`. The rule engine errs on the
//!   side of under-prescribing: false positives waste reviewer time
//!   more than false negatives waste engineer attention.
//! - **Rationale included.** Every recommendation has both a short
//!   action and a one-line rationale explaining the signal that
//!   triggered it. Reviewers should never have to re-read the raw
//!   trace to understand why a recommendation appeared.
//!
//! ## Severity scheme (mirrors ESLint / SonarQube / Rustc)
//!
//! - `Error` — structural regression with high confidence (dropped or
//!   reordered tool, refusal flip). Block-merge signal.
//! - `Warning` — decision drift (arg value change, semantic shift).
//!   Needs review before merge; may be intended.
//! - `Info` — style drift, below-noise variance, low-confidence
//!   signals. FYI, not action-required.

use serde::{Deserialize, Serialize};

use crate::diff::alignment::{DivergenceKind, FirstDivergence};
use crate::diff::axes::{Axis, Severity as AxisSeverity};
use crate::diff::report::DiffReport;

/// Severity of a recommendation. Uses the same three-tier scheme as
/// mainstream linters / static analyzers (ESLint, SonarQube, Rustc).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendationSeverity {
    /// Likely a real regression that should block merge (or at least
    /// demand explicit review). Produced for high-confidence
    /// structural drift and stop_reason flips to `content_filter`.
    Error,
    /// Probably a behaviour change worth reviewing before merge. Decision
    /// drift with meaningful confidence, axis-level severity of
    /// `moderate`/`severe`.
    Warning,
    /// Informational — style drift, below-noise divergence, or
    /// low-confidence signals. Won't gate merge; callers may hide it
    /// in summary views.
    Info,
}

impl RecommendationSeverity {
    /// Short lowercase label for terminal / markdown / JSON rendering.
    pub fn label(&self) -> &'static str {
        match self {
            RecommendationSeverity::Error => "error",
            RecommendationSeverity::Warning => "warning",
            RecommendationSeverity::Info => "info",
        }
    }

    /// Numeric weight so callers can sort recommendations by severity
    /// without a case match. Higher = more urgent.
    pub fn rank(&self) -> u8 {
        match self {
            RecommendationSeverity::Error => 3,
            RecommendationSeverity::Warning => 2,
            RecommendationSeverity::Info => 1,
        }
    }
}

/// The action category. Informs rendering (icon/color) and tells the
/// reviewer what KIND of move is being suggested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    /// Bring back something the candidate dropped (tool call, required
    /// field, turn).
    Restore,
    /// Remove something the candidate added without justification
    /// (duplicate tool call, extra unneeded turn).
    Remove,
    /// Change a value back to the baseline (arg value, temperature).
    Revert,
    /// Human judgment needed — the candidate change may be intentional
    /// or context-dependent (prompt wording, refusal behaviour).
    Review,
    /// Low-signal event that might be noise; verify before acting.
    Verify,
    /// A higher-level root-cause recommendation inferred from a
    /// cross-axis correlation pattern (e.g. "looks like a model swap
    /// because cost + latency + semantic all moved together"). Subsumes
    /// the individual per-axis recommendations the same signature
    /// would have produced; the renderer should prefer the RootCause
    /// over the individual ones when both are present.
    RootCause,
}

impl ActionKind {
    /// Short lowercase label used in terminal / markdown / JSON rendering.
    pub fn label(&self) -> &'static str {
        match self {
            ActionKind::Restore => "restore",
            ActionKind::Remove => "remove",
            ActionKind::Revert => "revert",
            ActionKind::Review => "review",
            ActionKind::Verify => "verify",
            ActionKind::RootCause => "root_cause",
        }
    }
}

/// One prescriptive recommendation for a reviewer.
///
/// Wire format is verbose-but-diff-friendly (all fields named); the
/// renderers condense it into one-line "severity · action · target"
/// sentences for display.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recommendation {
    /// Severity/priority; sort key for display.
    pub severity: RecommendationSeverity,
    /// What kind of move is being suggested.
    pub action: ActionKind,
    /// 0-based turn index the recommendation refers to. 0 when the
    /// recommendation is trace-wide (not tied to a specific turn) —
    /// in that case `source` should say "overall" or similar.
    pub turn: usize,
    /// One-line action statement, starts with an imperative verb.
    /// Example: "Restore `send_confirmation_email` tool call at turn 9."
    pub message: String,
    /// One-line explanation of the signal that triggered this
    /// recommendation. Sourced from the underlying `FirstDivergence`'s
    /// `explanation` field plus any rule-specific context.
    pub rationale: String,
    /// Which diff axis the signal came from. Lets callers filter
    /// recommendations by concern (e.g. only safety-related).
    pub axis: Axis,
    /// Confidence from the underlying divergence, 0..1. Low-confidence
    /// recommendations may be suppressed in compact views.
    pub confidence: f64,
}

/// Generate recommendations from a complete [`DiffReport`].
///
/// Rules:
///
/// 1. For every divergence in `report.divergences` (top-K ranked),
///    apply the rule that matches its `(kind, primary_axis, explanation)`
///    shape.
/// 2. If the overall worst-severity axis is `Severe` and no divergence
///    produced an Error-level recommendation, add a single trace-wide
///    recommendation pointing the reviewer at the strongest axis.
/// 3. Sort by (severity desc, confidence desc, turn asc). Cap at 8 to
///    avoid overwhelming PR comments (callers that want all can still
///    read the raw list — but the canonical ordering is stable here).
///
/// Output is stable and deterministic given the same input `DiffReport`.
pub fn generate(report: &DiffReport) -> Vec<Recommendation> {
    let mut out: Vec<Recommendation> = Vec::new();

    // Cross-axis pattern detection runs FIRST so root-cause
    // recommendations sort to the top and the per-divergence
    // recommendations land underneath as supporting evidence.
    out.extend(detect_cross_axis_patterns(report));

    for dv in &report.divergences {
        if let Some(rec) = rule_for_divergence(dv) {
            out.push(rec);
        }
    }

    // Trace-wide fallback: if the worst axis severity is Severe and we
    // haven't produced any Error-level recommendation, surface the
    // strongest axis as a top-level concern. Protects against the
    // edge case where axis severity is loud but first-divergence
    // couldn't attribute it to a specific turn.
    // The `Severity` enum derives `Ord` (declared in the None < Minor <
    // Moderate < Severe order). We filter to Severe rows and take any —
    // the axis itself tiebreaks naturally via the iteration order.
    let worst_axis_row = report
        .rows
        .iter()
        .filter(|r| r.severity == AxisSeverity::Severe)
        .max_by(|a, b| a.severity.cmp(&b.severity));
    if let Some(worst) = worst_axis_row {
        let has_error = out
            .iter()
            .any(|r| r.severity == RecommendationSeverity::Error);
        if !has_error {
            out.push(Recommendation {
                severity: RecommendationSeverity::Error,
                action: ActionKind::Review,
                turn: 0,
                message: format!(
                    "Review the candidate: {} axis shifted with severity {}.",
                    worst.axis.label(),
                    worst.severity.label(),
                ),
                rationale: format!(
                    "Aggregate signal crosses the `severe` threshold \
                    ({}: delta {:+.3}, CI [{:+.3}, {:+.3}]).",
                    worst.axis.label(),
                    worst.delta,
                    worst.ci95_low,
                    worst.ci95_high,
                ),
                axis: worst.axis,
                confidence: 0.8,
            });
        }
    }

    // Sort: severity desc, confidence desc, turn asc.
    out.sort_by(|a, b| {
        b.severity
            .rank()
            .cmp(&a.severity.rank())
            .then_with(|| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.turn.cmp(&b.turn))
    });
    out.truncate(8);
    out
}

/// Map a single divergence to a recommendation. Returns `None` if the
/// divergence is noise below an actionable threshold.
fn rule_for_divergence(dv: &FirstDivergence) -> Option<Recommendation> {
    let exp = dv.explanation.to_lowercase();
    match dv.kind {
        // ---------------------------------------------------------
        // Structural drift — Error severity, specific actions
        // ---------------------------------------------------------
        DivergenceKind::Structural => {
            // Pattern 1: candidate dropped tool call(s) → Restore
            if exp.contains("dropped tool")
                || exp.contains("dropped a response turn")
                || exp.contains("dropped a turn")
            {
                let tool_ref = extract_backticked(&dv.explanation).unwrap_or("missing element");
                Some(Recommendation {
                    severity: RecommendationSeverity::Error,
                    action: ActionKind::Restore,
                    turn: dv.baseline_turn,
                    message: format!("Restore {tool_ref} at turn {}.", dv.baseline_turn),
                    rationale: dv.explanation.clone(),
                    axis: dv.primary_axis,
                    confidence: dv.confidence,
                })
            }
            // Pattern 2: candidate added tool call(s) → Remove (if it
            // looks like a duplicate) OR Review (unclear intent).
            else if exp.contains("added tool") || exp.contains("inserted an extra") {
                let tool_ref = extract_backticked(&dv.explanation).unwrap_or("extra element");
                Some(Recommendation {
                    severity: RecommendationSeverity::Error,
                    action: ActionKind::Review,
                    turn: dv.baseline_turn,
                    message: format!(
                        "Review unexpected addition at turn {}: {tool_ref}.",
                        dv.baseline_turn
                    ),
                    rationale: dv.explanation.clone(),
                    axis: dv.primary_axis,
                    confidence: dv.confidence,
                })
            }
            // Pattern 3: duplicate tool invocation → Remove
            else if exp.contains("duplicate tool") {
                let tool_ref = extract_backticked(&dv.explanation).unwrap_or("the duplicated tool");
                Some(Recommendation {
                    severity: RecommendationSeverity::Error,
                    action: ActionKind::Remove,
                    turn: dv.baseline_turn,
                    message: format!(
                        "Remove duplicate invocation of {tool_ref} at turn {}.",
                        dv.baseline_turn
                    ),
                    rationale: dv.explanation.clone(),
                    axis: dv.primary_axis,
                    confidence: dv.confidence,
                })
            }
            // Pattern 4: tool set changed (renamed / swapped / reordered) → Review
            else if exp.contains("tool set changed") || exp.contains("tool ordering differs") {
                Some(Recommendation {
                    severity: RecommendationSeverity::Error,
                    action: ActionKind::Review,
                    turn: dv.baseline_turn,
                    message: format!(
                        "Review tool-schema change at turn {}: call shape diverged.",
                        dv.baseline_turn
                    ),
                    rationale: dv.explanation.clone(),
                    axis: dv.primary_axis,
                    confidence: dv.confidence,
                })
            }
            // Catch-all for unmatched Structural — still Error, generic Review.
            else {
                Some(Recommendation {
                    severity: RecommendationSeverity::Error,
                    action: ActionKind::Review,
                    turn: dv.baseline_turn,
                    message: format!("Review structural change at turn {}.", dv.baseline_turn),
                    rationale: dv.explanation.clone(),
                    axis: dv.primary_axis,
                    confidence: dv.confidence,
                })
            }
        }
        // ---------------------------------------------------------
        // Decision drift — Warning severity, rule picks action
        // ---------------------------------------------------------
        DivergenceKind::Decision => {
            // Safety-axis decision: refusal flip / stop_reason change.
            if dv.primary_axis == Axis::Safety && exp.contains("stop_reason") {
                // Upgrade to Error when candidate introduced a refusal.
                let is_new_refusal = exp.contains("content_filter");
                let severity = if is_new_refusal {
                    RecommendationSeverity::Error
                } else {
                    RecommendationSeverity::Warning
                };
                Some(Recommendation {
                    severity,
                    action: ActionKind::Review,
                    turn: dv.baseline_turn,
                    message: format!(
                        "Review refusal behaviour at turn {}: candidate may be over-refusing.",
                        dv.baseline_turn
                    ),
                    rationale: dv.explanation.clone(),
                    axis: dv.primary_axis,
                    confidence: dv.confidence,
                })
            }
            // Trajectory-axis decision: arg value changed → Revert.
            else if dv.primary_axis == Axis::Trajectory && exp.contains("arg value") {
                let arg_ref = extract_backticked(&dv.explanation).unwrap_or("arg value");
                Some(Recommendation {
                    severity: RecommendationSeverity::Warning,
                    action: ActionKind::Revert,
                    turn: dv.baseline_turn,
                    message: format!(
                        "Revert {arg_ref} at turn {} to the baseline value.",
                        dv.baseline_turn
                    ),
                    rationale: dv.explanation.clone(),
                    axis: dv.primary_axis,
                    confidence: dv.confidence,
                })
            }
            // Semantic decision drift: content meaning shifted → Review.
            else if dv.primary_axis == Axis::Semantic {
                Some(Recommendation {
                    severity: RecommendationSeverity::Warning,
                    action: ActionKind::Review,
                    turn: dv.baseline_turn,
                    message: format!(
                        "Review response text at turn {}: semantic content shifted.",
                        dv.baseline_turn
                    ),
                    rationale: dv.explanation.clone(),
                    axis: dv.primary_axis,
                    confidence: dv.confidence,
                })
            }
            // Catch-all for unmatched Decision.
            else {
                Some(Recommendation {
                    severity: RecommendationSeverity::Warning,
                    action: ActionKind::Review,
                    turn: dv.baseline_turn,
                    message: format!("Review decision change at turn {}.", dv.baseline_turn),
                    rationale: dv.explanation.clone(),
                    axis: dv.primary_axis,
                    confidence: dv.confidence,
                })
            }
        }
        // ---------------------------------------------------------
        // Style drift — Info severity, Verify (might not need action)
        // ---------------------------------------------------------
        DivergenceKind::Style => Some(Recommendation {
            severity: RecommendationSeverity::Info,
            action: ActionKind::Verify,
            turn: dv.baseline_turn,
            message: format!(
                "Cosmetic wording change at turn {} — verify intended.",
                dv.baseline_turn
            ),
            rationale: dv.explanation.clone(),
            axis: dv.primary_axis,
            confidence: dv.confidence,
        }),
    }
}

/// Extract the first backtick-delimited token from a string, e.g.
/// pull `"search(limit,query)"` out of
/// `"tool set changed: removed `search(query)`, added `search(limit,query)`"`.
/// Returns `None` when there is no backticked span.
fn extract_backticked(s: &str) -> Option<&str> {
    let first = s.find('`')?;
    let rest = &s[first + 1..];
    let end = rest.find('`')?;
    Some(&rest[..end])
}

// ---------------------------------------------------------------------------
// Cross-axis correlation pattern detection
// ---------------------------------------------------------------------------

/// Threshold above which an axis is considered "moved" for cross-axis
/// pattern detection. Severity must be Moderate or Severe.
fn axis_moved(report: &DiffReport, axis: Axis) -> bool {
    report
        .rows
        .iter()
        .find(|r| r.axis == axis)
        .map(|r| matches!(r.severity, AxisSeverity::Moderate | AxisSeverity::Severe))
        .unwrap_or(false)
}

/// True iff the axis severity is at least Severe (the strongest tier).
fn axis_severe(report: &DiffReport, axis: Axis) -> bool {
    report
        .rows
        .iter()
        .find(|r| r.axis == axis)
        .map(|r| r.severity == AxisSeverity::Severe)
        .unwrap_or(false)
}

/// Detect cross-axis correlation patterns and emit RootCause
/// recommendations naming the inferred underlying change.
///
/// Patterns encoded (each fires only when the per-axis severity
/// signals all clear the noise floor — Moderate or Severe — so
/// noise on one axis can't trigger a root-cause claim alone):
///
///   1. **Model swap** — cost + latency + semantic all moved together.
///      Frontier-vs-haiku swaps and provider switches show this exact
///      tri-axis signature; reverting the model_id usually neutralises
///      all three at once. Pattern reference: GPT-4 → GPT-4o, Sonnet
///      → Haiku, Anthropic → OpenAI provider migrations.
///
///   2. **Prompt drift** — semantic + verbosity moved together,
///      typically with safety joining when the prompt changed
///      refusal-style instructions. System-prompt edits classically
///      produce this two-or-three-axis signature.
///
///   3. **Refusal escalation** — safety axis severe + verbosity often
///      down. Stricter system instructions or tighter content
///      policies. Often paired with stop_reason flips to
///      content_filter (which the safety axis already encodes).
///
///   4. **Tool schema migration** — trajectory severe + reasoning
///      moves + sometimes cost slightly up. Adding/removing tool args
///      shows up as a same-set-of-tools-different-arg-keys divergence
///      that touches the trajectory axis directly and the reasoning
///      axis indirectly (the model thinks longer about the new schema).
///
///   5. **Hallucination cluster** — semantic moderate-or-severe +
///      judge moderate-or-severe + verbosity often up. The classic
///      "agent talks confidently and incorrectly" signature: the
///      semantic axis catches the divergence, the judge axis catches
///      the wrongness, and verbosity often spikes because the
///      hallucinated content is longer than the correct answer.
fn detect_cross_axis_patterns(report: &DiffReport) -> Vec<Recommendation> {
    let mut out = Vec::new();

    // 1. Model swap signature
    if axis_moved(report, Axis::Cost)
        && axis_moved(report, Axis::Latency)
        && axis_moved(report, Axis::Semantic)
    {
        let cost_delta = axis_delta(report, Axis::Cost);
        let lat_delta = axis_delta(report, Axis::Latency);
        let sem_delta = axis_delta(report, Axis::Semantic);
        out.push(Recommendation {
            severity: RecommendationSeverity::Error,
            action: ActionKind::RootCause,
            turn: 0,
            message:
                "Looks like a model change. Cost, latency, and semantic axes all shifted together."
                    .to_string(),
            rationale: format!(
                "Cross-axis signature: cost Δ {cost_delta:+.3}, latency Δ {lat_delta:+.3}, \
                 semantic Δ {sem_delta:+.3}. Three axes moving together is the canonical \
                 model-swap signature (provider change, frontier→haiku, etc.). Diff the \
                 `model` field across configs first."
            ),
            axis: Axis::Cost,
            confidence: 0.85,
        });
    }

    // 2. Prompt drift signature
    if axis_moved(report, Axis::Semantic) && axis_moved(report, Axis::Verbosity) {
        // Don't double-fire when we already attributed to model swap.
        let already_model_swap = out.iter().any(|r| {
            matches!(r.action, ActionKind::RootCause)
                && r.message.starts_with("Looks like a model change")
        });
        if !already_model_swap {
            let sem_delta = axis_delta(report, Axis::Semantic);
            let vrb_delta = axis_delta(report, Axis::Verbosity);
            let safety_part = if axis_moved(report, Axis::Safety) {
                " plus safety axis (refusal-style instruction change)"
            } else {
                ""
            };
            out.push(Recommendation {
                severity: RecommendationSeverity::Warning,
                action: ActionKind::RootCause,
                turn: 0,
                message: format!(
                    "Looks like a system-prompt edit. Semantic + verbosity moved together{safety_part}."
                ),
                rationale: format!(
                    "Cross-axis signature: semantic Δ {sem_delta:+.3}, verbosity Δ {vrb_delta:+.3}. \
                     Diff the `system` field of the request across configs."
                ),
                axis: Axis::Semantic,
                confidence: 0.70,
            });
        }
    }

    // 3. Refusal escalation signature
    if axis_severe(report, Axis::Safety) {
        let safety_delta = axis_delta(report, Axis::Safety);
        if safety_delta > 0.0 {
            out.push(Recommendation {
                severity: RecommendationSeverity::Error,
                action: ActionKind::RootCause,
                turn: 0,
                message: "Refusal rate is up severely. Check for stricter system instructions \
                          or tighter content policies."
                    .to_string(),
                rationale: format!(
                    "Safety axis severe with positive delta {safety_delta:+.3} — the candidate \
                     refused or was content-filtered more often than baseline. Common causes: \
                     added safety preamble in system prompt, model upgrade with stricter RLHF, \
                     provider-side content-filter tightening."
                ),
                axis: Axis::Safety,
                confidence: 0.80,
            });
        }
    }

    // 4. Tool schema migration signature
    if axis_severe(report, Axis::Trajectory) && axis_moved(report, Axis::Reasoning) {
        let traj_delta = axis_delta(report, Axis::Trajectory);
        let reason_delta = axis_delta(report, Axis::Reasoning);
        out.push(Recommendation {
            severity: RecommendationSeverity::Error,
            action: ActionKind::RootCause,
            turn: 0,
            message: "Looks like a tool-schema migration. Trajectory + reasoning both moved."
                .to_string(),
            rationale: format!(
                "Cross-axis signature: trajectory Δ {traj_delta:+.3} (tool sequence/args \
                 changed), reasoning Δ {reason_delta:+.3} (the model is thinking through a \
                 different schema). Diff the `tools` array across configs and check whether \
                 arg keys were added or removed."
            ),
            axis: Axis::Trajectory,
            confidence: 0.78,
        });
    }

    // 5. Hallucination cluster signature
    if axis_moved(report, Axis::Semantic) && axis_moved(report, Axis::Judge) {
        let sem_delta = axis_delta(report, Axis::Semantic);
        let judge_delta = axis_delta(report, Axis::Judge);
        let verbosity_part = if axis_moved(report, Axis::Verbosity) {
            ", with verbosity also up"
        } else {
            ""
        };
        out.push(Recommendation {
            severity: RecommendationSeverity::Error,
            action: ActionKind::RootCause,
            turn: 0,
            message: format!(
                "Possible hallucination regression. Semantic and judge axes both moved{verbosity_part}."
            ),
            rationale: format!(
                "Cross-axis signature: semantic Δ {sem_delta:+.3}, judge Δ {judge_delta:+.3}. \
                 The classic 'confident-and-wrong' signature — the response diverged \
                 semantically AND was scored lower by the rubric. Sample 3-5 candidate \
                 outputs and verify factual claims against ground truth before merging."
            ),
            axis: Axis::Judge,
            confidence: 0.82,
        });
    }

    out
}

/// Look up the per-axis delta value for cross-axis pattern formatting.
/// Returns 0.0 when the axis is missing from the report.
fn axis_delta(report: &DiffReport, axis: Axis) -> f64 {
    report
        .rows
        .iter()
        .find(|r| r.axis == axis)
        .map(|r| r.delta)
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::axes::{Axis, AxisStat, Severity};

    fn empty_report() -> DiffReport {
        let rows = Axis::all().iter().map(|a| AxisStat::empty(*a)).collect();
        DiffReport {
            rows,
            baseline_trace_id: String::new(),
            candidate_trace_id: String::new(),
            pair_count: 0,
            first_divergence: None,
            divergences: Vec::new(),
            recommendations: Vec::new(),
            drill_down: Vec::new(),
        }
    }

    fn divergence(
        kind: DivergenceKind,
        axis: Axis,
        explanation: &str,
        confidence: f64,
    ) -> FirstDivergence {
        FirstDivergence {
            baseline_turn: 3,
            candidate_turn: 3,
            kind,
            primary_axis: axis,
            explanation: explanation.to_string(),
            confidence,
        }
    }

    #[test]
    fn no_divergences_produces_no_recommendations() {
        let out = generate(&empty_report());
        assert!(out.is_empty());
    }

    #[test]
    fn dropped_tool_becomes_restore_error() {
        let mut r = empty_report();
        r.divergences.push(divergence(
            DivergenceKind::Structural,
            Axis::Trajectory,
            "candidate dropped tool call(s): `send_confirmation_email(order_id,to)`",
            0.9,
        ));
        let recs = generate(&r);
        assert_eq!(recs.len(), 1);
        let rec = &recs[0];
        assert_eq!(rec.severity, RecommendationSeverity::Error);
        assert_eq!(rec.action, ActionKind::Restore);
        assert!(rec.message.contains("Restore"));
        assert!(rec.message.contains("send_confirmation_email"));
        assert_eq!(rec.turn, 3);
    }

    #[test]
    fn duplicate_tool_becomes_remove_error() {
        let mut r = empty_report();
        r.divergences.push(divergence(
            DivergenceKind::Structural,
            Axis::Trajectory,
            "candidate called `lookup_order(order_id)` 2 time(s) vs baseline's 1 — duplicate tool invocation",
            0.5,
        ));
        let recs = generate(&r);
        assert_eq!(recs.len(), 1);
        let rec = &recs[0];
        assert_eq!(rec.severity, RecommendationSeverity::Error);
        assert_eq!(rec.action, ActionKind::Remove);
        assert!(rec.message.contains("Remove duplicate"));
        assert!(rec.message.contains("lookup_order"));
    }

    #[test]
    fn added_tool_becomes_review_error() {
        let mut r = empty_report();
        r.divergences.push(divergence(
            DivergenceKind::Structural,
            Axis::Trajectory,
            "candidate added tool call(s): `new_tool(arg)`",
            0.7,
        ));
        let recs = generate(&r);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].action, ActionKind::Review);
        assert_eq!(recs[0].severity, RecommendationSeverity::Error);
    }

    #[test]
    fn refusal_flip_to_content_filter_is_error() {
        let mut r = empty_report();
        r.divergences.push(divergence(
            DivergenceKind::Decision,
            Axis::Safety,
            "stop_reason changed: `end_turn` → `content_filter`",
            0.8,
        ));
        let recs = generate(&r);
        assert_eq!(recs.len(), 1);
        // content_filter → stricter treatment → Error, not Warning
        assert_eq!(recs[0].severity, RecommendationSeverity::Error);
        assert_eq!(recs[0].action, ActionKind::Review);
        assert!(recs[0].message.to_lowercase().contains("refusal"));
    }

    #[test]
    fn arg_value_change_becomes_revert_warning() {
        let mut r = empty_report();
        r.divergences.push(divergence(
            DivergenceKind::Decision,
            Axis::Trajectory,
            "tool arg value changed: `refund(amount)`: `99.99` → `9.99`",
            0.6,
        ));
        let recs = generate(&r);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].severity, RecommendationSeverity::Warning);
        assert_eq!(recs[0].action, ActionKind::Revert);
        assert!(recs[0].message.contains("Revert"));
        assert!(recs[0].message.contains("refund(amount)"));
    }

    #[test]
    fn semantic_decision_drift_becomes_review_warning() {
        let mut r = empty_report();
        r.divergences.push(divergence(
            DivergenceKind::Decision,
            Axis::Semantic,
            "response text diverged (text similarity 0.10); same tool sequence",
            0.6,
        ));
        let recs = generate(&r);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].severity, RecommendationSeverity::Warning);
        assert_eq!(recs[0].action, ActionKind::Review);
    }

    #[test]
    fn style_drift_becomes_verify_info() {
        let mut r = empty_report();
        r.divergences.push(divergence(
            DivergenceKind::Style,
            Axis::Semantic,
            "cosmetic wording change — tool sequence preserved",
            0.3,
        ));
        let recs = generate(&r);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].severity, RecommendationSeverity::Info);
        assert_eq!(recs[0].action, ActionKind::Verify);
    }

    #[test]
    fn sort_puts_errors_before_warnings_before_info() {
        let mut r = empty_report();
        r.divergences.push(divergence(
            DivergenceKind::Style,
            Axis::Semantic,
            "cosmetic wording change",
            0.9, // high confidence, but still Info
        ));
        r.divergences.push(divergence(
            DivergenceKind::Structural,
            Axis::Trajectory,
            "candidate dropped tool call(s): `x(y)`",
            0.2, // low confidence, but Error
        ));
        r.divergences.push(divergence(
            DivergenceKind::Decision,
            Axis::Trajectory,
            "tool arg value changed: `f(a)`: `1` → `2`",
            0.5,
        ));
        let recs = generate(&r);
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[0].severity, RecommendationSeverity::Error);
        assert_eq!(recs[1].severity, RecommendationSeverity::Warning);
        assert_eq!(recs[2].severity, RecommendationSeverity::Info);
    }

    #[test]
    fn trace_wide_severe_axis_adds_fallback_recommendation() {
        // No divergences but one axis is severe — should produce a
        // trace-wide Review recommendation.
        let mut r = empty_report();
        let row = r
            .rows
            .iter_mut()
            .find(|a| a.axis == Axis::Semantic)
            .unwrap();
        row.delta = -0.6;
        row.baseline_median = 1.0;
        row.candidate_median = 0.4;
        row.ci95_low = -0.7;
        row.ci95_high = -0.5;
        row.severity = Severity::Severe;
        row.n = 20;
        let recs = generate(&r);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].severity, RecommendationSeverity::Error);
        assert_eq!(recs[0].action, ActionKind::Review);
        assert_eq!(recs[0].turn, 0);
        assert!(recs[0].message.contains("semantic"));
        assert!(recs[0].rationale.contains("severe"));
    }

    #[test]
    fn trace_wide_fallback_skipped_when_error_already_exists() {
        let mut r = empty_report();
        r.divergences.push(divergence(
            DivergenceKind::Structural,
            Axis::Trajectory,
            "candidate dropped tool call(s): `x(y)`",
            0.8,
        ));
        let row = r
            .rows
            .iter_mut()
            .find(|a| a.axis == Axis::Semantic)
            .unwrap();
        row.delta = -0.6;
        row.severity = Severity::Severe;
        row.n = 20;
        let recs = generate(&r);
        // Expect exactly ONE recommendation (from the Structural), not two.
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].severity, RecommendationSeverity::Error);
    }

    #[test]
    fn output_capped_at_8() {
        let mut r = empty_report();
        for i in 0..15 {
            r.divergences.push(divergence(
                DivergenceKind::Decision,
                Axis::Trajectory,
                &format!("tool arg value changed: `f(a)`: `{i}` → `{}`", i + 1),
                0.5,
            ));
        }
        let recs = generate(&r);
        assert_eq!(recs.len(), 8);
    }

    #[test]
    fn extract_backticked_pulls_first_token() {
        assert_eq!(
            extract_backticked("before `first(token)` middle `second`"),
            Some("first(token)")
        );
        assert_eq!(extract_backticked("no backticks here"), None);
        assert_eq!(extract_backticked("`only-one`"), Some("only-one"));
    }

    #[test]
    fn severity_rank_ordering_is_error_above_warning_above_info() {
        assert!(RecommendationSeverity::Error.rank() > RecommendationSeverity::Warning.rank());
        assert!(RecommendationSeverity::Warning.rank() > RecommendationSeverity::Info.rank());
    }

    // ----------------------------------------------------------------
    // Cross-axis correlation pattern detection
    // ----------------------------------------------------------------

    fn force_axis_severe(report: &mut DiffReport, axis: Axis, delta: f64) {
        let row = report.rows.iter_mut().find(|a| a.axis == axis).unwrap();
        row.delta = delta;
        row.baseline_median = if delta < 0.0 { 1.0 } else { 0.0 };
        row.candidate_median = row.baseline_median + delta;
        row.ci95_low = delta - 0.05;
        row.ci95_high = delta + 0.05;
        row.severity = Severity::Severe;
        row.n = 20;
    }

    fn force_axis_moderate(report: &mut DiffReport, axis: Axis, delta: f64) {
        let row = report.rows.iter_mut().find(|a| a.axis == axis).unwrap();
        row.delta = delta;
        row.baseline_median = if delta < 0.0 { 1.0 } else { 0.0 };
        row.candidate_median = row.baseline_median + delta;
        row.ci95_low = delta - 0.05;
        row.ci95_high = delta + 0.05;
        row.severity = Severity::Moderate;
        row.n = 20;
    }

    #[test]
    fn model_swap_signature_emits_root_cause() {
        let mut r = empty_report();
        force_axis_moderate(&mut r, Axis::Cost, 0.6);
        force_axis_moderate(&mut r, Axis::Latency, 0.8);
        force_axis_moderate(&mut r, Axis::Semantic, -0.3);
        let recs = generate(&r);
        let model_swap_rec = recs
            .iter()
            .find(|r| r.action == ActionKind::RootCause && r.message.contains("model change"));
        assert!(
            model_swap_rec.is_some(),
            "model-swap signature should produce a root-cause recommendation; got {:#?}",
            recs
        );
        let rec = model_swap_rec.unwrap();
        assert_eq!(rec.severity, RecommendationSeverity::Error);
        assert!(rec.rationale.contains("cost"));
        assert!(rec.rationale.contains("latency"));
        assert!(rec.rationale.contains("semantic"));
    }

    #[test]
    fn prompt_drift_signature_fires_when_only_two_axes_move() {
        let mut r = empty_report();
        force_axis_moderate(&mut r, Axis::Semantic, -0.2);
        force_axis_moderate(&mut r, Axis::Verbosity, 0.4);
        // Cost and latency NOT moved — should NOT fire model_swap.
        let recs = generate(&r);
        let prompt_rec = recs
            .iter()
            .find(|r| r.action == ActionKind::RootCause && r.message.contains("prompt"));
        assert!(prompt_rec.is_some());
        let no_model = recs
            .iter()
            .all(|r| !(r.action == ActionKind::RootCause && r.message.contains("model change")));
        assert!(no_model, "prompt-drift should not also fire model_swap");
    }

    #[test]
    fn prompt_drift_suppressed_when_model_swap_already_fires() {
        let mut r = empty_report();
        force_axis_moderate(&mut r, Axis::Cost, 0.5);
        force_axis_moderate(&mut r, Axis::Latency, 0.7);
        force_axis_moderate(&mut r, Axis::Semantic, -0.3);
        force_axis_moderate(&mut r, Axis::Verbosity, 0.4);
        let recs = generate(&r);
        let n_root_cause = recs
            .iter()
            .filter(|r| r.action == ActionKind::RootCause)
            .count();
        // Exactly one root-cause should fire (model swap subsumes prompt
        // drift when both signatures match — the model swap is the
        // upstream explanation).
        let n_model = recs
            .iter()
            .filter(|r| r.action == ActionKind::RootCause && r.message.contains("model change"))
            .count();
        let n_prompt = recs
            .iter()
            .filter(|r| r.action == ActionKind::RootCause && r.message.contains("prompt"))
            .count();
        assert_eq!(n_model, 1);
        assert_eq!(n_prompt, 0, "prompt drift should be suppressed; got {n_root_cause} root-causes");
    }

    #[test]
    fn refusal_escalation_fires_on_severe_safety_with_positive_delta() {
        let mut r = empty_report();
        force_axis_severe(&mut r, Axis::Safety, 0.4); // refusal rate up 40%
        let recs = generate(&r);
        let refusal_rec = recs
            .iter()
            .find(|r| r.action == ActionKind::RootCause && r.message.contains("Refusal rate"));
        assert!(refusal_rec.is_some(), "got {:#?}", recs);
        assert_eq!(refusal_rec.unwrap().severity, RecommendationSeverity::Error);
    }

    #[test]
    fn refusal_escalation_does_not_fire_on_negative_safety_delta() {
        // Refusal rate DROPPED — that's the candidate refusing less,
        // which isn't a "stricter instructions" signature.
        let mut r = empty_report();
        force_axis_severe(&mut r, Axis::Safety, -0.4);
        let recs = generate(&r);
        let refusal_rec = recs
            .iter()
            .find(|r| r.action == ActionKind::RootCause && r.message.contains("Refusal rate"));
        assert!(refusal_rec.is_none());
    }

    #[test]
    fn tool_schema_migration_fires_on_severe_trajectory_plus_reasoning() {
        let mut r = empty_report();
        force_axis_severe(&mut r, Axis::Trajectory, 0.5);
        force_axis_moderate(&mut r, Axis::Reasoning, 0.3);
        let recs = generate(&r);
        let tool_rec = recs
            .iter()
            .find(|r| r.action == ActionKind::RootCause && r.message.contains("tool-schema"));
        assert!(tool_rec.is_some(), "got {:#?}", recs);
    }

    #[test]
    fn hallucination_cluster_fires_on_semantic_plus_judge() {
        let mut r = empty_report();
        force_axis_moderate(&mut r, Axis::Semantic, -0.3);
        force_axis_moderate(&mut r, Axis::Judge, -0.4);
        let recs = generate(&r);
        let halluc_rec = recs
            .iter()
            .find(|r| r.action == ActionKind::RootCause && r.message.contains("hallucination"));
        assert!(halluc_rec.is_some(), "got {:#?}", recs);
        assert_eq!(halluc_rec.unwrap().severity, RecommendationSeverity::Error);
    }

    #[test]
    fn no_cross_axis_signature_when_only_one_axis_moves() {
        let mut r = empty_report();
        force_axis_severe(&mut r, Axis::Latency, 0.7);
        let recs = generate(&r);
        let any_root_cause = recs.iter().any(|r| r.action == ActionKind::RootCause);
        assert!(
            !any_root_cause,
            "single-axis movement should NOT trigger any root-cause: got {:#?}",
            recs
        );
    }

    #[test]
    fn root_cause_action_label_is_root_cause() {
        assert_eq!(ActionKind::RootCause.label(), "root_cause");
    }
}
