//! Axis 1: final-output semantic similarity.
//!
//! The production-grade path uses `sentence-transformers/all-MiniLM-L6-v2`
//! from Python (see CONTRIBUTING.md). The Rust-side default is a **TF-IDF
//! cosine similarity** over the corpus of response texts being compared.
//! This is genuinely lexical — it reflects word-level overlap weighted by
//! how common each token is in the corpus, so two responses that share
//! the same rare content score high while boilerplate token overlap
//! contributes little.
//!
//! TF-IDF cosine is not a semantic embedding: "I agree" and "Yes" still
//! score 0. That's a real limitation; users who need paraphrase-robust
//! similarity should install the `[embeddings]` extra. The TF-IDF axis
//! is honest about what it measures (word overlap) rather than faking
//! signal (the old hash-surrogate axis did the latter).

use std::collections::HashMap;

use unicode_normalization::UnicodeNormalization;

use crate::agentlog::Record;
use crate::diff::axes::{Axis, AxisStat};
use crate::diff::bootstrap::{median, paired_ci};

/// Lowercase, NFC-normalize, split on non-alphanumeric. Empty tokens
/// dropped. Tokens shorter than 2 chars are kept (e.g. "ok", "no").
fn tokenize(text: &str) -> Vec<String> {
    text.nfc()
        .flat_map(|c| c.to_lowercase())
        .collect::<String>()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Term frequency: count of each token in `tokens`.
fn term_frequency(tokens: &[String]) -> HashMap<String, f64> {
    let mut out: HashMap<String, f64> = HashMap::new();
    for tok in tokens {
        *out.entry(tok.clone()).or_insert(0.0) += 1.0;
    }
    out
}

/// Document frequency: number of docs in which each term appears (at
/// least once). Input is already-tokenized text per doc.
fn document_frequency(corpus: &[Vec<String>]) -> HashMap<String, usize> {
    let mut df: HashMap<String, usize> = HashMap::new();
    for doc in corpus {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for tok in doc {
            if seen.insert(tok.as_str()) {
                *df.entry(tok.clone()).or_insert(0) += 1;
            }
        }
    }
    df
}

/// Sparse TF-IDF vector: `tfidf[t] = (1 + log(tf)) · log((N + 1) / (df + 1))`.
/// This is the smoothed variant used by scikit-learn's `TfidfVectorizer`.
fn tfidf_vector(
    tokens: &[String],
    df: &HashMap<String, usize>,
    n_docs: usize,
) -> HashMap<String, f64> {
    let tf = term_frequency(tokens);
    let mut out: HashMap<String, f64> = HashMap::with_capacity(tf.len());
    let n = n_docs as f64;
    for (tok, tf_v) in tf {
        let df_v = *df.get(&tok).unwrap_or(&0) as f64;
        let idf = ((n + 1.0) / (df_v + 1.0)).ln() + 1.0;
        let tf_weight = 1.0 + tf_v.ln();
        out.insert(tok, tf_weight * idf);
    }
    out
}

fn sparse_cosine(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
    let na: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
    let nb: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return if na < 1e-12 && nb < 1e-12 { 1.0 } else { 0.0 };
    }
    // Walk the smaller of the two.
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let mut dot = 0.0;
    for (k, v) in small {
        if let Some(w) = large.get(k) {
            dot += v * w;
        }
    }
    dot / (na * nb)
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

/// Compute the semantic-similarity axis using TF-IDF cosine.
pub fn compute(pairs: &[(&Record, &Record)], seed: Option<u64>) -> AxisStat {
    if pairs.is_empty() {
        return AxisStat::empty(Axis::Semantic);
    }
    // Build a shared corpus DF over all responses on both sides so
    // pairwise similarities are comparable.
    let baseline_tokens: Vec<Vec<String>> = pairs
        .iter()
        .map(|(b, _)| tokenize(&response_text(b)))
        .collect();
    let candidate_tokens: Vec<Vec<String>> = pairs
        .iter()
        .map(|(_, c)| tokenize(&response_text(c)))
        .collect();
    let mut corpus: Vec<Vec<String>> = Vec::with_capacity(pairs.len() * 2);
    corpus.extend(baseline_tokens.clone());
    corpus.extend(candidate_tokens.clone());
    let df = document_frequency(&corpus);
    let n_docs = corpus.len();

    let similarities: Vec<f64> = baseline_tokens
        .iter()
        .zip(candidate_tokens.iter())
        .map(|(bt, ct)| {
            let bv = tfidf_vector(bt, &df, n_docs);
            let cv = tfidf_vector(ct, &df, n_docs);
            sparse_cosine(&bv, &cv).clamp(0.0, 1.0)
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
    AxisStat::new_value(Axis::Semantic, bm, cm, delta, ci.low, ci.high, pairs.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::Kind;
    use crate::diff::axes::Severity;
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
        let r = response("the quick brown fox jumps over the lazy dog");
        let pairs = [(&r, &r)];
        let stat = compute(&pairs, Some(1));
        assert!((stat.candidate_median - 1.0).abs() < 1e-9);
        assert_eq!(stat.severity, Severity::None);
    }

    #[test]
    fn very_different_text_has_lower_similarity() {
        let baseline: Vec<Record> = (0..10)
            .map(|i| response(&format!("refund issued for order {i}")))
            .collect();
        let candidate: Vec<Record> = (0..10)
            .map(|i| {
                response(&format!(
                    "unable to process request {i}, please contact support"
                ))
            })
            .collect();
        let pairs: Vec<(&Record, &Record)> = baseline.iter().zip(candidate.iter()).collect();
        let stat = compute(&pairs, Some(1));
        assert!(stat.candidate_median < 0.5);
    }

    #[test]
    fn tokenize_splits_on_punctuation_and_lowercases() {
        assert_eq!(
            tokenize("Hello, world!  It's nice"),
            vec![
                "hello".to_string(),
                "world".to_string(),
                "it".to_string(),
                "s".to_string(),
                "nice".to_string(),
            ]
        );
    }

    #[test]
    fn tokenize_handles_unicode_nfc() {
        // "café" in NFD vs NFC — tokenize should normalize.
        let nfd = "cafe\u{0301}";
        let nfc = "café";
        assert_eq!(tokenize(nfd), tokenize(nfc));
    }

    #[test]
    fn empty_text_has_zero_similarity_to_nonempty() {
        let empty = response("");
        let full = response("some content here");
        let pairs = [(&empty, &full); 3];
        let stat = compute(&pairs, Some(1));
        assert!(stat.candidate_median < 0.1);
    }

    #[test]
    fn identical_content_scores_higher_than_partial_overlap() {
        // One pair is identical; the other shares only boilerplate.
        let identical_b = response("refund issued for order abc123");
        let identical_c = response("refund issued for order abc123");
        let partial_b = response("refund issued for order abc123");
        let partial_c = response("unable to process please contact support");
        let baseline = [identical_b, partial_b];
        let candidate = [identical_c, partial_c];
        let pairs: Vec<(&Record, &Record)> = baseline.iter().zip(candidate.iter()).collect();
        let bt: Vec<Vec<String>> = pairs
            .iter()
            .map(|(b, _)| tokenize(&response_text(b)))
            .collect();
        let ct: Vec<Vec<String>> = pairs
            .iter()
            .map(|(_, c)| tokenize(&response_text(c)))
            .collect();
        let mut corpus = bt.clone();
        corpus.extend(ct.clone());
        let df = document_frequency(&corpus);
        let n = corpus.len();
        let score_identical =
            sparse_cosine(&tfidf_vector(&bt[0], &df, n), &tfidf_vector(&ct[0], &df, n));
        let score_partial =
            sparse_cosine(&tfidf_vector(&bt[1], &df, n), &tfidf_vector(&ct[1], &df, n));
        assert!(
            score_identical > score_partial + 0.3,
            "identical={score_identical} partial={score_partial}"
        );
    }
}
