//! Axis 8: LLM-judge (user-supplied rubric).
//!
//! This module defines the [`Judge`] trait that users implement (usually
//! in Python — see `python/src/shadow/llm/`). The Rust side only provides
//! the trait and the aggregation logic; no Rust-side default evaluator
//! is included, because calling an LLM from Rust is out of scope for v0.1
//! (SDKs are Python-first per CLAUDE.md D6).

use async_trait::async_trait;

use crate::agentlog::Record;
use crate::diff::axes::{Axis, AxisStat};
use crate::diff::bootstrap::{median, paired_ci};

/// User-supplied evaluator that scores a single (baseline, candidate)
/// response pair. Scores are in `[0.0, 1.0]` where 1.0 means "candidate
/// is at least as good as baseline."
#[async_trait]
pub trait Judge: Send + Sync {
    /// Return a score in `[0.0, 1.0]` for the given pair.
    async fn score(&self, baseline: &Record, candidate: &Record) -> f64;
}

/// Aggregate scores from a user-supplied judge into an [`AxisStat`].
pub async fn compute<J: Judge + ?Sized>(
    pairs: &[(&Record, &Record)],
    judge: &J,
    seed: Option<u64>,
) -> AxisStat {
    if pairs.is_empty() {
        return AxisStat::empty(Axis::Judge);
    }
    let mut scores = Vec::with_capacity(pairs.len());
    for (b, c) in pairs {
        scores.push(judge.score(b, c).await);
    }
    let baseline_ones: Vec<f64> = (0..scores.len()).map(|_| 1.0).collect();
    let bm = 1.0;
    let cm = median(&scores);
    let delta = cm - bm;
    let ci = paired_ci(
        &baseline_ones,
        &scores,
        |bs, cs| median(cs) - median(bs),
        0,
        seed,
    );
    AxisStat::new_value(Axis::Judge, bm, cm, delta, ci.low, ci.high, pairs.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::Kind;
    use crate::diff::axes::Severity;
    use async_trait::async_trait;
    use serde_json::json;

    /// A Judge that returns a fixed score regardless of input — good for
    /// testing the aggregation plumbing without needing an actual LLM.
    struct ConstantJudge(f64);

    #[async_trait]
    impl Judge for ConstantJudge {
        async fn score(&self, _baseline: &Record, _candidate: &Record) -> f64 {
            self.0
        }
    }

    fn response() -> Record {
        Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": [],
                "stop_reason": "end_turn",
                "latency_ms": 0,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        )
    }

    #[tokio::test]
    async fn perfect_score_is_no_regression() {
        let r = response();
        let pairs = vec![(&r, &r); 10];
        let stat = compute(&pairs, &ConstantJudge(1.0), Some(1)).await;
        assert!((stat.candidate_median - 1.0).abs() < 1e-9);
        assert_eq!(stat.severity, Severity::None);
    }

    #[tokio::test]
    async fn low_score_is_flagged_severe() {
        let r = response();
        let pairs = vec![(&r, &r); 10];
        let stat = compute(&pairs, &ConstantJudge(0.4), Some(1)).await;
        assert!(stat.candidate_median < 1.0);
        assert!(matches!(
            stat.severity,
            Severity::Severe | Severity::Moderate
        ));
    }
}
