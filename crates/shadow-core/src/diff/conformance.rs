//! Axis 9: schema / format conformance rate.
//!
//! Intent-gated on the baseline side. A pair is counted (and scored) if
//! the baseline response has EITHER:
//!
//!   1. JSON-text intent: its text starts with `{` or `[` (after fence-
//!      strip). Both sides are scored on whether their text parses.
//!
//!   2. Tool-use intent: it emits at least one `tool_use` block with a
//!      dict-shaped `input`. Both sides are scored on whether their
//!      (first) tool_use's input keys match the baseline's. This covers
//!      the common agent pattern where the "structured final answer"
//!      is a tool call (e.g. `submit_answer(...)`) rather than JSON
//!      text — the same conformance question in a different syntax.
//!
//! Pairs where baseline has neither JSON intent nor tool_use intent are
//! excluded (we don't penalise the candidate for not structuring when
//! nothing asked for structure).

use crate::agentlog::Record;
use crate::diff::axes::{Axis, AxisStat};
use crate::diff::bootstrap::paired_ci;

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

/// Strip leading ```json / ``` and trailing ``` markdown fences, if present.
/// Frontier models (Claude, GPT) routinely wrap JSON output in fences even
/// when instructed not to — treating the fence as "not JSON intent" would
/// silently disable this axis on real traffic.
///
/// Handles four fence shapes:
///   1. Multi-line with closer: "```json\n{...}\n```"     (the common case)
///   2. Multi-line no closer:   "```json\n{...}"          (truncated streams)
///   3. Single-line with closer: "```json{...}```"        (rare, but seen)
///   4. Single-line no closer:  "```json{...}"            (pathological)
///
/// In every case the fence plus optional language tag is stripped, leading
/// whitespace on the body is trimmed (so `has_json_intent` can rely on the
/// first char), and any trailing ``` closer is removed.
fn strip_markdown_fences(text: &str) -> &str {
    let t = text.trim();
    if !t.starts_with("```") {
        return t;
    }
    // Multi-line: strip everything up to and including the first newline.
    let after_fence = if let Some(i) = t.find('\n') {
        &t[i + 1..]
    } else {
        // Single-line: strip the leading ``` plus an optional language tag
        // like ```json / ```sql up to either `{`, `[`, or another `.
        let no_backticks = &t[3..];
        // Skip over an alphanumeric language tag.
        let after_lang = no_backticks.trim_start_matches(|c: char| c.is_ascii_alphanumeric());
        after_lang
    };
    // Trim a trailing closing fence (whether or not it was present).
    let trimmed_end = after_fence.trim_end();
    let body = trimmed_end.strip_suffix("```").unwrap_or(trimmed_end);
    body.trim()
}

/// True if `text` (after trim + fence-strip) parses as JSON.
fn is_json_parseable(text: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(strip_markdown_fences(text)).is_ok()
}

/// True if `text` (after fence-strip + trim) starts with `{` or `[` — our
/// heuristic for "this response intends to be JSON."
fn has_json_intent(text: &str) -> bool {
    let body = strip_markdown_fences(text);
    body.starts_with('{') || body.starts_with('[')
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Return the top-level key set of the first `tool_use` block whose
/// `input` is a dict. Returns None if no such block exists.
fn first_tool_use_keys(r: &Record) -> Option<std::collections::BTreeSet<String>> {
    let arr = r.payload.get("content").and_then(|c| c.as_array())?;
    for part in arr {
        if part.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
            if let Some(input) = part.get("input").and_then(|i| i.as_object()) {
                return Some(input.keys().cloned().collect());
            }
        }
    }
    None
}

/// Compute the schema-conformance axis.
pub fn compute(pairs: &[(&Record, &Record)], seed: Option<u64>) -> AxisStat {
    let mut b = Vec::new();
    let mut c = Vec::new();
    for (br, cr) in pairs {
        let baseline_text = response_text(br);
        let baseline_tool_keys = first_tool_use_keys(br);
        let baseline_json_intent = has_json_intent(&baseline_text);
        let baseline_tool_intent = baseline_tool_keys.is_some();
        if !baseline_json_intent && !baseline_tool_intent {
            // Baseline wasn't producing structure of either flavour —
            // this pair isn't part of the conformance population.
            continue;
        }
        // Score each side. Conformance is "did the structured contract
        // survive". We use the union of signals: a pair scores 1.0 if
        // EITHER the JSON intent was honoured OR (when applicable) the
        // tool_use key set matched. A candidate that abandons both
        // scores 0.0 — the full regression signal.
        let b_score;
        let c_score;
        if baseline_json_intent {
            b_score = f64::from(is_json_parseable(&baseline_text));
            c_score = f64::from(is_json_parseable(&response_text(cr)));
        } else {
            // Tool-use intent only. Score by key-set match.
            // Baseline always 1.0 (by construction — it has the keys).
            b_score = 1.0;
            let candidate_tool_keys = first_tool_use_keys(cr);
            c_score = match (&baseline_tool_keys, &candidate_tool_keys) {
                (Some(bk), Some(ck)) if bk == ck => 1.0,
                _ => 0.0,
            };
        }
        b.push(b_score);
        c.push(c_score);
    }
    if b.is_empty() {
        return AxisStat::empty(Axis::Conformance);
    }
    let bm = mean(&b);
    let cm = mean(&c);
    let delta = cm - bm;
    let ci = paired_ci(&b, &c, |bs, cs| mean(cs) - mean(bs), 0, seed);
    // Use absolute-scale severity: a rate axis whose baseline is 1.0
    // and candidate is 0.5 is a 50% regression, not "within noise."
    AxisStat::new_rate(Axis::Conformance, bm, cm, delta, ci.low, ci.high, b.len())
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
    fn baseline_json_intent_candidate_prose_flags_severe() {
        // Headline fix: the case where baseline produces valid JSON and
        // candidate regresses to prose MUST surface as a severe drop.
        let baseline = response(r#"[{"a": 1}]"#);
        let candidate = response("Here are your results: ...");
        let pairs = [(&baseline, &candidate); 3];
        let stat = compute(&pairs, Some(1));
        assert!((stat.baseline_median - 1.0).abs() < 1e-9);
        assert!((stat.candidate_median - 0.0).abs() < 1e-9);
        assert_eq!(stat.severity, Severity::Severe);
        assert_eq!(stat.n, 3);
    }

    #[test]
    fn both_sides_valid_json_is_no_regression() {
        let r = response(r#"{"a": 1}"#);
        let pairs = [(&r, &r); 5];
        let stat = compute(&pairs, Some(1));
        assert_eq!(stat.severity, Severity::None);
    }

    #[test]
    fn baseline_without_json_intent_is_excluded_from_population() {
        // Both responses are prose — no JSON intent on the baseline side,
        // so the pair doesn't count toward the conformance axis.
        let baseline = response("hello");
        let candidate = response("world");
        let pairs = [(&baseline, &candidate); 3];
        let stat = compute(&pairs, Some(1));
        assert_eq!(stat.n, 0);
    }

    #[test]
    fn baseline_json_candidate_broken_json_is_counted() {
        // Candidate attempted JSON but produced invalid output.
        let baseline = response(r#"{"ok": true}"#);
        let candidate = response("{broken");
        let pairs = [(&baseline, &candidate); 4];
        let stat = compute(&pairs, Some(1));
        assert!((stat.baseline_median - 1.0).abs() < 1e-9);
        assert!((stat.candidate_median - 0.0).abs() < 1e-9);
        assert_eq!(stat.severity, Severity::Severe);
    }

    #[test]
    fn partial_regression_is_moderate() {
        use crate::diff::axes::Flag;
        // Baseline always JSON, candidate mostly JSON with 1/3 regressions
        // — at n=30 the CI is tight and severity reaches Moderate/Severe.
        let baseline = response(r#"{"ok": true}"#);
        let good = response(r#"{"ok": true}"#);
        let bad = response("plain text response");
        let mut pairs: Vec<(&Record, &Record)> = Vec::new();
        for i in 0..30 {
            pairs.push(if i % 3 == 0 {
                (&baseline, &bad)
            } else {
                (&baseline, &good)
            });
        }
        let stat = compute(&pairs, Some(1));
        assert!(matches!(
            stat.severity,
            Severity::Moderate | Severity::Severe
        ));
        assert!(!stat.flags.contains(&Flag::LowPower));
    }

    #[test]
    fn markdown_fenced_json_is_still_json_intent() {
        // Real-world: Claude/GPT wrap JSON in ```json ... ``` fences. The
        // intent-gate must strip fences before checking for JSON intent.
        let baseline = response("```json\n{\"ok\": true}\n```");
        let candidate = response("```json\n{\"ok\": true}\n```");
        let pairs = [(&baseline, &candidate); 5];
        let stat = compute(&pairs, Some(1));
        // At least some pairs should be scored (n > 0) — the bug was n=0.
        assert!(stat.n > 0, "fenced JSON should count toward conformance");
    }

    /// Build a chat_response with the specified tool_use input keys.
    fn tool_use_response(name: &str, keys: &[&str]) -> Record {
        let input: serde_json::Map<String, serde_json::Value> = keys
            .iter()
            .map(|k| ((*k).to_string(), json!("v")))
            .collect();
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
                "stop_reason": "tool_use",
                "latency_ms": 0,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        )
    }

    #[test]
    fn tool_use_intent_counts_toward_conformance_axis() {
        // REGRESSION TEST: the v0.1 conformance axis only fired on JSON
        // text responses, so agents that output their final answer via a
        // tool_use call (the common real-world pattern, e.g.
        // `submit_answer(...)`) got n=0 and the axis was dead weight.
        // Now tool_use blocks with dict input count.
        let baseline = tool_use_response("submit", &["ticker", "revenue", "net_income"]);
        let pairs = [(&baseline, &baseline); 5];
        let stat = compute(&pairs, Some(1));
        assert_eq!(stat.n, 5, "tool_use intent must produce n>0");
        assert_eq!(stat.severity, Severity::None); // same on both sides
    }

    #[test]
    fn tool_use_key_regression_flags_severe() {
        // Baseline submits all 3 required keys; candidate drops 2.
        let baseline = tool_use_response("submit", &["ticker", "revenue", "net_income"]);
        let candidate = tool_use_response("submit", &["ticker"]);
        let pairs = [(&baseline, &candidate); 5];
        let stat = compute(&pairs, Some(1));
        assert_eq!(stat.n, 5);
        assert!((stat.baseline_median - 1.0).abs() < 1e-9);
        assert!((stat.candidate_median - 0.0).abs() < 1e-9);
        assert_eq!(stat.severity, Severity::Severe);
    }

    #[test]
    fn tool_use_candidate_no_tool_call_flags_severe() {
        // Candidate didn't even call the tool (returned prose instead).
        let baseline = tool_use_response("submit", &["ticker", "revenue"]);
        let candidate = response("I think the answer is 42.");
        let pairs = [(&baseline, &candidate); 4];
        let stat = compute(&pairs, Some(1));
        assert_eq!(stat.n, 4);
        assert_eq!(stat.severity, Severity::Severe);
    }

    #[test]
    fn strip_fences_handles_single_line_fence() {
        // Pathological but seen in the wild: fence + body + closing fence on
        // one line. The old code short-circuited on no-newline and returned
        // the input verbatim.
        assert_eq!(strip_markdown_fences("```json{\"a\":1}```"), "{\"a\":1}");
        assert_eq!(strip_markdown_fences("```{\"a\":1}```"), "{\"a\":1}");
    }

    #[test]
    fn strip_fences_handles_unclosed_fence_with_indent() {
        // Truncated-stream case: opening fence, newline, indented body,
        // no closing fence. Old else-branch leaked leading whitespace.
        let input = "```json\n   {\"a\":1}";
        assert_eq!(strip_markdown_fences(input), "{\"a\":1}");
    }

    #[test]
    fn single_line_fenced_json_intent_is_recognised() {
        let baseline = response("```json{\"ok\":true}```");
        let candidate = response("plain prose");
        let pairs = [(&baseline, &candidate); 5];
        let stat = compute(&pairs, Some(1));
        assert!(stat.n > 0, "single-line fenced JSON should count");
    }

    #[test]
    fn fenced_json_vs_fenced_sql_is_severe_regression() {
        let baseline = response("```json\n{\"sql\": \"SELECT 1\"}\n```");
        let candidate = response("```sql\nSELECT 1\n```");
        let pairs = [(&baseline, &candidate); 30];
        let stat = compute(&pairs, Some(1));
        assert!(stat.n > 0);
        assert!(stat.delta < 0.0);
        assert!(matches!(
            stat.severity,
            Severity::Moderate | Severity::Severe
        ));
    }
}
