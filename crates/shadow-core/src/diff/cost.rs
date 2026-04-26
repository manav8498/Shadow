//! Axis 6: cost (input+output tokens × per-model pricing).
//!
//! Pricing is richer than just (input, output). Modern frontier models
//! bill differently for:
//!
//! - **Cached input tokens**: Anthropic prompt caching, OpenAI prompt
//!   caching — typically 10% of the uncached rate.
//! - **Reasoning tokens**: GPT-5+ reasoning / o1-style thinking tokens
//!   exposed via `usage.completion_tokens_details.reasoning_tokens` —
//!   usually billed at the output rate.
//! - **Batch API**: OpenAI and Anthropic batch APIs are ~50% off.
//!
//! Unknown models still contribute 0 cost rather than phantom infinities.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::agentlog::Record;
use crate::diff::axes::{Axis, AxisStat, Flag};
use crate::diff::bootstrap::{median, paired_ci};

/// Per-model pricing in USD per token.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelPricing {
    /// USD per uncached input token.
    pub input: f64,
    /// USD per output token.
    pub output: f64,
    /// USD per cached input READ token (cache hit). For Anthropic this is
    /// ~10% of `input`; for OpenAI ~50%. If 0.0, cached reads are billed
    /// at the full input rate.
    #[serde(default)]
    pub cached_input: f64,
    /// USD per cached input WRITE token, 5-minute TTL tier. Anthropic
    /// charges ~1.25× input for 5m-ephemeral cache creation. If 0.0,
    /// cache writes are billed at the uncached input rate.
    #[serde(default)]
    pub cached_write_5m: f64,
    /// USD per cached input WRITE token, 1-hour TTL tier. Anthropic
    /// charges ~2.0× input for 1h-ephemeral cache creation.
    #[serde(default)]
    pub cached_write_1h: f64,
    /// USD per reasoning / thinking token. If 0.0, reasoning tokens
    /// are billed at the `output` rate.
    #[serde(default)]
    pub reasoning: f64,
    /// Multiplier applied to the final per-call cost when
    /// `meta.batch == true` — e.g. 0.5 for a 50% batch discount.
    /// 1.0 (or 0.0 as a sentinel) means no discount.
    #[serde(default)]
    pub batch_discount: f64,
}

impl ModelPricing {
    /// Simple constructor that assumes no caching, no reasoning, no batch.
    pub fn simple(input: f64, output: f64) -> Self {
        Self {
            input,
            output,
            cached_input: 0.0,
            cached_write_5m: 0.0,
            cached_write_1h: 0.0,
            reasoning: 0.0,
            batch_discount: 0.0,
        }
    }
}

/// Pricing table keyed by `chat_response.payload.model`.
pub type Pricing = HashMap<String, ModelPricing>;

/// Look up pricing for a model name with a fallback that strips a
/// trailing dated snapshot suffix (`gpt-5-2025-08-07` →`gpt-5`,
/// `claude-opus-4-7-20250219` → `claude-opus-4-7`). Provider SDKs
/// almost always return the dated snapshot in chat-response bodies,
/// while pricing tables are usually keyed by the bare alias.
fn lookup_with_snapshot_fallback<'a>(
    pricing: &'a Pricing,
    model: &str,
) -> Option<&'a ModelPricing> {
    if let Some(p) = pricing.get(model) {
        return Some(p);
    }
    if let Some(base) = strip_snapshot_tail(model) {
        return pricing.get(base);
    }
    None
}

/// If `model` ends with a dated snapshot tail (`-YYYY-MM-DD` or `-YYYYMMDD`),
/// return the prefix without it. Otherwise None.
fn strip_snapshot_tail(model: &str) -> Option<&str> {
    let bytes = model.as_bytes();
    // Try -YYYY-MM-DD (length 11).
    if bytes.len() > 11 && bytes[bytes.len() - 11] == b'-' {
        let tail = &bytes[bytes.len() - 10..];
        if tail.len() == 10
            && tail[..4].iter().all(u8::is_ascii_digit)
            && tail[4] == b'-'
            && tail[5..7].iter().all(u8::is_ascii_digit)
            && tail[7] == b'-'
            && tail[8..10].iter().all(u8::is_ascii_digit)
        {
            return Some(&model[..model.len() - 11]);
        }
    }
    // Try -YYYYMMDD (length 9).
    if bytes.len() > 9 && bytes[bytes.len() - 9] == b'-' {
        let tail = &bytes[bytes.len() - 8..];
        if tail.len() == 8 && tail.iter().all(u8::is_ascii_digit) {
            return Some(&model[..model.len() - 9]);
        }
    }
    None
}

pub(crate) fn cost_of(r: &Record, pricing: &Pricing) -> Option<f64> {
    let model = r.payload.get("model")?.as_str()?;
    let usage = r.payload.get("usage")?;
    let input = usage.get("input_tokens")?.as_f64()?;
    let output = usage.get("output_tokens")?.as_f64()?;
    let cached_input = usage
        .get("cached_input_tokens")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let cached_write_5m = usage
        .get("cached_write_5m_tokens")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let cached_write_1h = usage
        .get("cached_write_1h_tokens")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let thinking = usage
        .get("thinking_tokens")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    // NaN-guard every token count. Malformed trace data must not produce
    // phantom costs that poison the diff report (NaN propagates through
    // the whole bootstrap and would silently land on the cost axis).
    if !(input.is_finite()
        && output.is_finite()
        && cached_input.is_finite()
        && cached_write_5m.is_finite()
        && cached_write_1h.is_finite()
        && thinking.is_finite())
    {
        return Some(0.0);
    }
    let Some(p) = lookup_with_snapshot_fallback(pricing, model) else {
        return Some(0.0);
    };
    let cached_rate = if p.cached_input > 0.0 {
        p.cached_input
    } else {
        p.input
    };
    let reasoning_rate = if p.reasoning > 0.0 {
        p.reasoning
    } else {
        p.output
    };
    // Cache-write rates fall back to input rate if not set.
    let write_5m_rate = if p.cached_write_5m > 0.0 {
        p.cached_write_5m
    } else {
        p.input
    };
    let write_1h_rate = if p.cached_write_1h > 0.0 {
        p.cached_write_1h
    } else {
        p.input
    };
    // input_tokens in the envelope is assumed to mean "uncached input".
    // If the provider emits a combined count, callers should split by
    // populating `cached_input_tokens` / `cached_write_5m_tokens` /
    // `cached_write_1h_tokens` separately.
    let mut cost = input * p.input
        + cached_input * cached_rate
        + cached_write_5m * write_5m_rate
        + cached_write_1h * write_1h_rate
        + output * p.output
        + thinking * reasoning_rate;
    // Batch API flag lives at the envelope-meta level, but to avoid
    // coupling the cost axis to the Envelope type we also honor a
    // `payload.meta_batch` convention. (Default: no discount.)
    let batch = r
        .payload
        .get("batch")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if batch && p.batch_discount > 0.0 {
        cost *= p.batch_discount;
    }
    Some(cost)
}

/// True iff both records in the pair reference models that appear in
/// the pricing table. Unknown models make `cost_of` fall back to 0.0
/// (so the bootstrap stays numerically clean), but for the purpose of
/// deciding whether the axis has real signal we need to distinguish
/// "priced at zero" from "model not in table."
fn pair_is_priced(br: &Record, cr: &Record, pricing: &Pricing) -> bool {
    fn model_in_table(r: &Record, pricing: &Pricing) -> bool {
        r.payload
            .get("model")
            .and_then(|m| m.as_str())
            .is_some_and(|m| lookup_with_snapshot_fallback(pricing, m).is_some())
    }
    model_in_table(br, pricing) && model_in_table(cr, pricing)
}

/// Compute the cost axis.
///
/// Silent zeros are a footgun: if no pricing table is supplied, or the
/// table has no entries for the traced models, every pair prices at
/// `0.0` and the naive implementation reports `delta=0,
/// severity=None` — indistinguishable from "both sides are genuinely
/// free." To prevent that, we track how many pairs had both models
/// present in the pricing table. When that count is zero (or below
/// half of `pairs.len()`), we attach [`Flag::NoPricing`] so downstream
/// renderers and reviewers see a caveat instead of a spurious clean
/// bill of health.
pub fn compute(pairs: &[(&Record, &Record)], pricing: &Pricing, seed: Option<u64>) -> AxisStat {
    let mut b = Vec::with_capacity(pairs.len());
    let mut c = Vec::with_capacity(pairs.len());
    let mut priced_pairs = 0usize;
    for (br, cr) in pairs {
        if let (Some(bv), Some(cv)) = (cost_of(br, pricing), cost_of(cr, pricing)) {
            b.push(bv);
            c.push(cv);
            if pair_is_priced(br, cr, pricing) {
                priced_pairs += 1;
            }
        }
    }
    if b.is_empty() {
        let mut stat = AxisStat::empty(Axis::Cost);
        // An empty pair list means "nothing to compare" (n=0 already
        // surfaces that). A non-empty list with no costable entries
        // means malformed records — still worth flagging.
        if !pairs.is_empty() {
            stat.flags.push(Flag::NoPricing);
        }
        return stat;
    }
    let bm = median(&b);
    let cm = median(&c);
    let delta = cm - bm;
    let ci = paired_ci(&b, &c, |bs, cs| median(cs) - median(bs), 0, seed);
    let mut stat = AxisStat::new_value(Axis::Cost, bm, cm, delta, ci.low, ci.high, b.len());
    // Flag whenever fewer than half of the input pairs had both models
    // priced — the median is unreliable and the user likely forgot
    // pricing for a subset of models. `priced_pairs == 0` means no
    // pricing at all; the flag covers both cases.
    if priced_pairs * 2 < pairs.len() {
        stat.flags.push(Flag::NoPricing);
    }
    stat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentlog::Kind;
    use crate::diff::axes::Severity;
    use serde_json::json;

    fn response(model: &str, input: u64, output: u64) -> Record {
        Record::new(
            Kind::ChatResponse,
            json!({
                "model": model,
                "content": [],
                "stop_reason": "end_turn",
                "latency_ms": 0,
                "usage": {"input_tokens": input, "output_tokens": output, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        )
    }

    fn response_with_usage(model: &str, usage: serde_json::Value) -> Record {
        Record::new(
            Kind::ChatResponse,
            json!({
                "model": model,
                "content": [],
                "stop_reason": "end_turn",
                "latency_ms": 0,
                "usage": usage,
            }),
            "2026-04-21T10:00:00Z",
            None,
        )
    }

    #[test]
    fn pricing_lookup_drives_cost() {
        let mut pricing = Pricing::new();
        pricing.insert("opus".to_string(), ModelPricing::simple(0.000015, 0.000075));
        pricing.insert(
            "haiku".to_string(),
            ModelPricing::simple(0.0000008, 0.000004),
        );
        let baseline: Vec<Record> = (0..10).map(|_| response("opus", 1000, 500)).collect();
        let candidate: Vec<Record> = (0..10).map(|_| response("haiku", 1000, 500)).collect();
        let pairs: Vec<(&Record, &Record)> = baseline.iter().zip(candidate.iter()).collect();
        let stat = compute(&pairs, &pricing, Some(1));
        assert!(stat.delta < 0.0);
        assert_eq!(stat.severity, Severity::Severe);
    }

    #[test]
    fn unknown_model_costs_zero() {
        let pricing = Pricing::new();
        let r = response("mystery", 1000, 500);
        let pairs = [(&r, &r)];
        let stat = compute(&pairs, &pricing, Some(1));
        assert_eq!(stat.baseline_median, 0.0);
    }

    #[test]
    fn no_pricing_flag_when_table_is_empty_but_pairs_exist() {
        // Pairs present, pricing absent → flag so reviewers know the
        // delta=0 is "unknown" and not "equal cost".
        let pricing = Pricing::new();
        let r = response("mystery", 1000, 500);
        let pairs = [(&r, &r), (&r, &r), (&r, &r)];
        let stat = compute(&pairs, &pricing, Some(1));
        assert!(stat.flags.contains(&Flag::NoPricing));
        assert_eq!(stat.delta, 0.0);
    }

    #[test]
    fn no_pricing_flag_when_most_models_unpriced() {
        // Pricing provided for one of three traced models → too many
        // pairs unpriced, flag still fires (median is unreliable).
        let mut pricing = Pricing::new();
        pricing.insert("opus".to_string(), ModelPricing::simple(0.000015, 0.000075));
        let priced = response("opus", 1000, 500);
        let unpriced1 = response("sonnet-unlisted", 1000, 500);
        let unpriced2 = response("gpt-x-unlisted", 1000, 500);
        let pairs = [
            (&priced, &priced),
            (&unpriced1, &unpriced1),
            (&unpriced2, &unpriced2),
        ];
        let stat = compute(&pairs, &pricing, Some(1));
        // Only 1 of 3 pairs priced; half-or-more unpriced → flag fires.
        assert!(stat.flags.contains(&Flag::NoPricing));
    }

    #[test]
    fn no_pricing_flag_absent_when_all_pairs_priced() {
        let mut pricing = Pricing::new();
        pricing.insert("opus".to_string(), ModelPricing::simple(0.000015, 0.000075));
        let r = response("opus", 1000, 500);
        let pairs = [(&r, &r), (&r, &r)];
        let stat = compute(&pairs, &pricing, Some(1));
        assert!(!stat.flags.contains(&Flag::NoPricing));
    }

    #[test]
    fn no_pricing_flag_absent_when_pairs_empty() {
        // Empty pairs is a different story (nothing to compare), and the
        // NoPricing flag would be misleading there — n=0 already says
        // "no data." Assert we don't spuriously flag that case.
        let pricing = Pricing::new();
        let pairs: Vec<(&Record, &Record)> = Vec::new();
        let stat = compute(&pairs, &pricing, Some(1));
        assert!(!stat.flags.contains(&Flag::NoPricing));
        assert_eq!(stat.n, 0);
    }

    #[test]
    fn cached_input_tokens_billed_at_cheaper_rate() {
        // 1000 uncached input @ $15/Mtok + 1000 cached @ $1.50/Mtok + 500 out @ $75/Mtok
        let mut pricing = Pricing::new();
        pricing.insert(
            "opus".to_string(),
            ModelPricing {
                input: 0.000015,
                output: 0.000075,
                cached_input: 0.0000015, // 10% of input
                cached_write_5m: 0.0,
                cached_write_1h: 0.0,
                reasoning: 0.0,
                batch_discount: 0.0,
            },
        );
        let r = response_with_usage(
            "opus",
            json!({
                "input_tokens": 1000,
                "output_tokens": 500,
                "thinking_tokens": 0,
                "cached_input_tokens": 1000,
            }),
        );
        let pairs = [(&r, &r)];
        let stat = compute(&pairs, &pricing, Some(1));
        // 1000*0.000015 + 1000*0.0000015 + 500*0.000075 = 0.015 + 0.0015 + 0.0375 = 0.054
        assert!((stat.baseline_median - 0.054).abs() < 1e-9);
    }

    #[test]
    fn reasoning_tokens_billed_at_reasoning_rate() {
        let mut pricing = Pricing::new();
        pricing.insert(
            "gpt-5".to_string(),
            ModelPricing {
                input: 0.000010,
                output: 0.000040,
                cached_input: 0.0,
                cached_write_5m: 0.0,
                cached_write_1h: 0.0,
                reasoning: 0.000060, // reasoning costs more than output
                batch_discount: 0.0,
            },
        );
        let r = response_with_usage(
            "gpt-5",
            json!({
                "input_tokens": 100,
                "output_tokens": 100,
                "thinking_tokens": 500,
            }),
        );
        let pairs = [(&r, &r)];
        let stat = compute(&pairs, &pricing, Some(1));
        // 100*10e-6 + 100*40e-6 + 500*60e-6 = 0.001 + 0.004 + 0.030 = 0.035
        assert!((stat.baseline_median - 0.035).abs() < 1e-6);
    }

    #[test]
    fn anthropic_cache_write_tiers_are_billed_separately() {
        // Opus: input=$15/Mtok, 5m-write=$18.75/Mtok (1.25x), 1h-write=$30/Mtok (2x)
        let mut pricing = Pricing::new();
        pricing.insert(
            "opus".to_string(),
            ModelPricing {
                input: 0.000015,
                output: 0.000075,
                cached_input: 0.0000015,
                cached_write_5m: 0.00001875,
                cached_write_1h: 0.00003,
                reasoning: 0.0,
                batch_discount: 0.0,
            },
        );
        let r = Record::new(
            Kind::ChatResponse,
            json!({
                "model": "opus",
                "content": [],
                "stop_reason": "end_turn",
                "latency_ms": 0,
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "thinking_tokens": 0,
                    "cached_input_tokens": 500,
                    "cached_write_5m_tokens": 200,
                    "cached_write_1h_tokens": 100,
                },
            }),
            "2026-04-21T10:00:00Z",
            None,
        );
        let pairs = [(&r, &r)];
        let stat = compute(&pairs, &pricing, Some(1));
        // True cost:
        //   uncached input: 1000 * 15e-6    = 0.015
        //   cached read:    500  * 1.5e-6   = 0.00075
        //   5m write:       200  * 18.75e-6 = 0.00375
        //   1h write:       100  * 30e-6    = 0.003
        //   output:         200  * 75e-6    = 0.015
        // Total: 0.0375
        assert!(
            (stat.baseline_median - 0.0375).abs() < 1e-6,
            "got {}",
            stat.baseline_median
        );
    }

    #[test]
    fn nan_usage_values_produce_zero_cost_not_phantom_inf() {
        let mut pricing = Pricing::new();
        pricing.insert("m".to_string(), ModelPricing::simple(0.001, 0.002));
        let r = Record::new(
            Kind::ChatResponse,
            json!({
                "model": "m",
                "content": [],
                "stop_reason": "end_turn",
                "latency_ms": 0,
                "usage": {
                    "input_tokens": 100.0,
                    "output_tokens": 100.0,
                    "thinking_tokens": 0,
                    "cached_input_tokens": f64::NAN,
                },
            }),
            "2026-04-21T10:00:00Z",
            None,
        );
        let pairs = [(&r, &r)];
        let stat = compute(&pairs, &pricing, Some(1));
        // Cost must be finite (0.0 by our NaN-guard policy), not NaN.
        assert!(stat.baseline_median.is_finite());
        assert_eq!(stat.severity, Severity::None);
    }

    #[test]
    fn batch_flag_applies_discount() {
        let mut pricing = Pricing::new();
        pricing.insert(
            "opus".to_string(),
            ModelPricing {
                input: 0.000015,
                output: 0.000075,
                cached_input: 0.0,
                cached_write_5m: 0.0,
                cached_write_1h: 0.0,
                reasoning: 0.0,
                batch_discount: 0.5, // 50% off for batch API
            },
        );
        let batched = Record::new(
            Kind::ChatResponse,
            json!({
                "model": "opus",
                "content": [],
                "stop_reason": "end_turn",
                "latency_ms": 0,
                "batch": true,
                "usage": {"input_tokens": 1000, "output_tokens": 500, "thinking_tokens": 0},
            }),
            "2026-04-21T10:00:00Z",
            None,
        );
        let non_batched = response("opus", 1000, 500);
        let pairs_batched = [(&batched, &batched)];
        let pairs_normal = [(&non_batched, &non_batched)];
        let stat_b = compute(&pairs_batched, &pricing, Some(1));
        let stat_n = compute(&pairs_normal, &pricing, Some(1));
        assert!((stat_b.baseline_median - stat_n.baseline_median * 0.5).abs() < 1e-9);
    }

    #[test]
    fn snapshot_tail_strips_iso_dates() {
        assert_eq!(strip_snapshot_tail("gpt-5-2025-08-07"), Some("gpt-5"));
        assert_eq!(
            strip_snapshot_tail("gpt-4o-mini-2024-07-18"),
            Some("gpt-4o-mini"),
        );
        // Anthropic-style packed YYYYMMDD.
        assert_eq!(
            strip_snapshot_tail("claude-opus-4-7-20250219"),
            Some("claude-opus-4-7"),
        );
        // No tail.
        assert_eq!(strip_snapshot_tail("gpt-5"), None);
        assert_eq!(strip_snapshot_tail("gpt-4o-mini"), None);
        // False positives not stripped (model name happens to end in digits).
        assert_eq!(strip_snapshot_tail("o1"), None);
    }

    #[test]
    fn cost_resolves_dated_snapshot_to_bare_alias() {
        // Real-world bug: chat_response carries `gpt-5-2025-08-07` but
        // pricing.json is keyed by `gpt-5`. Without the fallback the cost
        // axis silently flagged `no_pricing` and reported zero delta.
        let mut pricing = Pricing::new();
        pricing.insert(
            "gpt-5".to_string(),
            ModelPricing {
                input: 0.000010,
                output: 0.000040,
                cached_input: 0.0,
                cached_write_5m: 0.0,
                cached_write_1h: 0.0,
                reasoning: 0.0,
                batch_discount: 0.0,
            },
        );
        let r = response("gpt-5-2025-08-07", 100, 50);
        let cost = cost_of(&r, &pricing).unwrap();
        // 100 * 1e-5 + 50 * 4e-5 = 0.001 + 0.002 = 0.003
        assert!((cost - 0.003).abs() < 1e-9, "got {}", cost);
        // pair_is_priced must also see the snapshot as priced.
        let pairs = [(&r, &r)];
        let stat = compute(&pairs, &pricing, Some(42));
        assert!(
            !stat.flags.contains(&Flag::NoPricing),
            "pair_is_priced should accept dated snapshots"
        );
    }
}
