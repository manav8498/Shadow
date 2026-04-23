//! Axis 2: tool-call trajectory divergence.
//!
//! For each response, extract the sequence of `(tool_name, arg_shape)`
//! tuples where `arg_shape` is the sorted list of top-level argument keys.
//! Compare baseline vs candidate sequences with Levenshtein edit distance.
//! Normalize by max(len(baseline_seq), len(candidate_seq)) so the metric
//! is in [0, 1] — per-response divergence score.

use crate::agentlog::Record;
use crate::diff::axes::{Axis, AxisStat};
use crate::diff::bootstrap::{median, paired_ci};

fn tool_shape(r: &Record) -> Vec<String> {
    let content = match r.payload.get("content").and_then(|c| c.as_array()) {
        Some(arr) => arr,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    for part in content {
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

fn normalized_divergence(b: &[String], c: &[String]) -> f64 {
    let denom = b.len().max(c.len());
    if denom == 0 {
        0.0
    } else {
        levenshtein(b, c) as f64 / denom as f64
    }
}

/// Compute the tool-trajectory axis.
pub fn compute(pairs: &[(&Record, &Record)], seed: Option<u64>) -> AxisStat {
    if pairs.is_empty() {
        return AxisStat::empty(Axis::Trajectory);
    }
    let baseline_zero: Vec<f64> = (0..pairs.len()).map(|_| 0.0).collect();
    let divergence: Vec<f64> = pairs
        .iter()
        .map(|(b, c)| normalized_divergence(&tool_shape(b), &tool_shape(c)))
        .collect();
    let bm = median(&baseline_zero);
    let cm = median(&divergence);
    let delta = cm - bm;
    let ci = paired_ci(
        &baseline_zero,
        &divergence,
        |bs, cs| median(cs) - median(bs),
        0,
        seed,
    );
    // Trajectory is a rate in [0, 1] measured *from* zero (identical
    // sequences → 0 divergence). The relative-delta severity used by
    // `new_value` divides by baseline_median=0.0 and always returns
    // Minor, regardless of magnitude. `new_rate` uses absolute-delta
    // thresholds, which is the honest classification for this axis.
    AxisStat::new_rate(
        Axis::Trajectory,
        bm,
        cm,
        delta,
        ci.low,
        ci.high,
        pairs.len(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::Kind;
    use serde_json::json;

    fn response_with_tools(tools: &[(&str, &[&str])]) -> Record {
        let content: Vec<serde_json::Value> = tools
            .iter()
            .map(|(name, keys)| {
                let input: serde_json::Map<String, serde_json::Value> = keys
                    .iter()
                    .map(|k| ((*k).to_string(), json!("v")))
                    .collect();
                json!({
                    "type": "tool_use",
                    "id": format!("t_{name}"),
                    "name": name,
                    "input": input,
                })
            })
            .collect();
        Record::new(
            Kind::ChatResponse,
            json!({
                "model": "x",
                "content": content,
                "stop_reason": "tool_use",
                "latency_ms": 0,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        )
    }

    #[test]
    fn identical_tool_shapes_produce_zero_divergence() {
        let r = response_with_tools(&[("search_files", &["query"])]);
        let pairs = [(&r, &r)];
        let stat = compute(&pairs, Some(1));
        assert_eq!(stat.candidate_median, 0.0);
    }

    #[test]
    fn schema_change_on_same_tool_is_divergence() {
        let baseline = response_with_tools(&[("search_files", &["query"])]);
        // Candidate adds a `limit` key.
        let candidate = response_with_tools(&[("search_files", &["query", "limit"])]);
        let pairs = [(&baseline, &candidate); 10];
        let stat = compute(&pairs, Some(2));
        assert!(stat.candidate_median > 0.0);
    }

    #[test]
    fn levenshtein_basic() {
        let a = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let b = vec!["a".to_string(), "x".to_string(), "c".to_string()];
        assert_eq!(levenshtein(&a, &b), 1);
    }
}
