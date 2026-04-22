//! Axis 1: final-output semantic similarity.
//!
//! The production version uses `sentence-transformers/all-MiniLM-L6-v2`
//! from Python (see CLAUDE.md D5). Rust-only callers get a
//! **test-only hash-surrogate embedding**: a deterministic 128-d vector
//! derived from SHA-256 of the normalized text. That surrogate is
//! explicitly NOT suitable for production semantic scoring — it's
//! just stable enough to exercise the plumbing (pairs extraction, CI,
//! severity classification) without pulling an ML dependency into the
//! Rust crate. The Python layer overrides this axis with real embeddings.

use sha2::{Digest, Sha256};

use crate::agentlog::Record;
use crate::diff::axes::{Axis, AxisStat, Severity};
use crate::diff::bootstrap::{median, paired_ci};

/// Lightweight, reproducible, *non-semantic* text embedding for tests.
///
/// Do NOT expose this to users as a production signal.
fn surrogate_embedding(text: &str) -> Vec<f64> {
    let normalized = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let digest = Sha256::digest(normalized.as_bytes());
    // Fan out the 32-byte digest into 128 dimensions by taking each byte's
    // value divided by 255 (stable, deterministic).
    let mut v = Vec::with_capacity(128);
    for chunk in 0..4 {
        for byte in digest.iter() {
            v.push(f64::from(*byte ^ chunk as u8) / 255.0);
        }
    }
    v
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        0.0
    } else {
        dot / (na * nb)
    }
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

/// Compute the semantic-similarity axis using the hash-surrogate embedding.
///
/// The Python layer should override this with a real sentence-transformer.
pub fn compute(pairs: &[(&Record, &Record)], seed: Option<u64>) -> AxisStat {
    if pairs.is_empty() {
        return AxisStat::empty(Axis::Semantic);
    }
    let similarities: Vec<f64> = pairs
        .iter()
        .map(|(b, c)| {
            let eb = surrogate_embedding(&response_text(b));
            let ec = surrogate_embedding(&response_text(c));
            cosine_similarity(&eb, &ec)
        })
        .collect();
    let baseline_ones: Vec<f64> = (0..similarities.len()).map(|_| 1.0).collect();
    let bm = 1.0;
    let cm = median(&similarities);
    let delta = cm - bm;
    let ci = paired_ci(
        &baseline_ones,
        &similarities,
        |bs, cs| median(cs) - median(bs),
        0,
        seed,
    );
    AxisStat {
        axis: Axis::Semantic,
        baseline_median: bm,
        candidate_median: cm,
        delta,
        ci95_low: ci.low,
        ci95_high: ci.high,
        severity: Severity::classify(delta, 1.0, ci.low, ci.high),
        n: pairs.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::Kind;
    use serde_json::json;

    fn response(text: &str) -> Record {
        Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "latency_ms": 0,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        )
    }

    #[test]
    fn identical_text_has_similarity_1() {
        let r = response("hello world");
        let pairs = [(&r, &r)];
        let stat = compute(&pairs, Some(1));
        assert!((stat.candidate_median - 1.0).abs() < 1e-9);
        assert_eq!(stat.severity, Severity::None);
    }

    #[test]
    fn very_different_text_has_lower_similarity() {
        let baseline: Vec<Record> = (0..10)
            .map(|i| response(&format!("response {i}")))
            .collect();
        let candidate: Vec<Record> = (0..10)
            .map(|i| response(&format!("totally unrelated {i}")))
            .collect();
        let pairs: Vec<(&Record, &Record)> = baseline.iter().zip(candidate.iter()).collect();
        let stat = compute(&pairs, Some(1));
        // Surrogate embedding is not real semantics — just assert
        // similarity is measurably < 1.
        assert!(stat.candidate_median < 1.0);
    }
}
