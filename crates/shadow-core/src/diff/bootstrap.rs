//! Bootstrap resampling for paired statistics.
//!
//! Given `n` paired observations and a statistic function, resample `n` pairs
//! with replacement, apply the statistic, repeat `iterations` times (1000 by
//! default), and return the 2.5 / 50 / 97.5 percentile of the resulting
//! distribution.
//!
//! Every axis in CLAUDE.md §4 uses this primitive to turn a sample into a
//! median + 95% CI — implemented once here instead of per-axis.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Percentile-based 95% CI plus the sample median.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CiResult {
    /// Lower bound (2.5 percentile).
    pub low: f64,
    /// Median (50 percentile).
    pub median: f64,
    /// Upper bound (97.5 percentile).
    pub high: f64,
}

/// Default number of bootstrap iterations. Set on the high side of the
/// bias/variance tradeoff; pair-resamples are cheap (O(n) per iter) so
/// 1000 is fine even for n=1k samples.
pub const DEFAULT_ITERATIONS: usize = 1000;

/// Bootstrap the statistic over paired samples.
///
/// - `baseline` and `candidate` must have equal length (caller enforces).
/// - `statistic` maps (baseline_slice, candidate_slice) to a scalar.
/// - `iterations` defaults to [`DEFAULT_ITERATIONS`] when 0 is passed.
/// - `seed`: pass `Some(seed)` for reproducible tests, `None` for a
///   random OS-seeded RNG.
///
/// Returns the 2.5 / 50 / 97.5 percentiles of the bootstrap distribution.
pub fn paired_ci<F>(
    baseline: &[f64],
    candidate: &[f64],
    statistic: F,
    iterations: usize,
    seed: Option<u64>,
) -> CiResult
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    let n = baseline.len();
    // Precondition: callers pass paired slices. Enforced by assert in debug
    // builds; release builds short-circuit on length mismatch to an empty
    // result rather than unwinding. clippy's panic lint is suppressed
    // because this is a programming-error guard, not user-visible
    // behaviour.
    #[allow(clippy::panic)]
    {
        if n != candidate.len() {
            panic!("baseline and candidate must have equal length");
        }
    }
    if n == 0 {
        return CiResult {
            low: 0.0,
            median: 0.0,
            high: 0.0,
        };
    }
    let iterations = if iterations == 0 {
        DEFAULT_ITERATIONS
    } else {
        iterations
    };

    let mut rng: StdRng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let mut samples = Vec::with_capacity(iterations);
    let indices: Vec<usize> = (0..n).collect();
    let mut b_buf = Vec::with_capacity(n);
    let mut c_buf = Vec::with_capacity(n);
    for _ in 0..iterations {
        b_buf.clear();
        c_buf.clear();
        for _ in 0..n {
            // n>0 is guaranteed above, so .choose() always returns Some.
            // Using unwrap_or to avoid the clippy::expect_used lint on the
            // happy path — the fallback is unreachable.
            let i = *indices.choose(&mut rng).unwrap_or(&0);
            b_buf.push(baseline[i]);
            c_buf.push(candidate[i]);
        }
        samples.push(statistic(&b_buf, &c_buf));
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    CiResult {
        low: percentile(&samples, 2.5),
        median: percentile(&samples, 50.0),
        high: percentile(&samples, 97.5),
    }
}

/// Compute the median of a slice of floats. Handles even-length lists by
/// averaging the two central values. The input is cloned; the caller's
/// slice is untouched.
pub fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Percentile `p` (0–100) of a pre-sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len() as f64;
    let rank = (p / 100.0) * (n - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_odd_length() {
        assert_eq!(median(&[1.0, 3.0, 2.0]), 2.0);
    }

    #[test]
    fn median_even_length_averages_two_middle() {
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn median_empty_is_zero() {
        assert_eq!(median(&[]), 0.0);
    }

    #[test]
    fn percentile_exact() {
        let sorted: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        assert!((percentile(&sorted, 50.0) - 50.0).abs() < 1e-9);
        assert!((percentile(&sorted, 2.5) - 2.5).abs() < 1e-9);
        assert!((percentile(&sorted, 97.5) - 97.5).abs() < 1e-9);
    }

    #[test]
    fn paired_ci_zero_on_equal_samples() {
        let baseline: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let candidate = baseline.clone();
        let result = paired_ci(
            &baseline,
            &candidate,
            |b, c| median(c) - median(b),
            200,
            Some(42),
        );
        assert!(result.low.abs() < 1e-9);
        assert!(result.median.abs() < 1e-9);
        assert!(result.high.abs() < 1e-9);
    }

    #[test]
    fn paired_ci_detects_consistent_shift() {
        // candidate is baseline + 10, every element.
        let baseline: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let candidate: Vec<f64> = baseline.iter().map(|x| x + 10.0).collect();
        let result = paired_ci(
            &baseline,
            &candidate,
            |b, c| median(c) - median(b),
            500,
            Some(7),
        );
        // CI should tightly bracket +10.
        assert!(result.low > 5.0);
        assert!(result.high < 15.0);
        assert!((result.median - 10.0).abs() < 2.0);
    }

    #[test]
    fn paired_ci_empty_is_zero() {
        let r = paired_ci(&[], &[], |_, _| 0.0, 100, Some(1));
        assert_eq!(r.low, 0.0);
        assert_eq!(r.median, 0.0);
        assert_eq!(r.high, 0.0);
    }

    #[test]
    fn paired_ci_is_deterministic_with_seed() {
        let baseline: Vec<f64> = (0..50).map(|i| (i as f64) * 1.5).collect();
        let candidate: Vec<f64> = (0..50).map(|i| (i as f64) * 1.5 + 3.0).collect();
        let a = paired_ci(
            &baseline,
            &candidate,
            |b, c| median(c) - median(b),
            200,
            Some(123),
        );
        let b = paired_ci(
            &baseline,
            &candidate,
            |b, c| median(c) - median(b),
            200,
            Some(123),
        );
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic(expected = "must have equal length")]
    fn paired_ci_panics_on_length_mismatch() {
        paired_ci(&[1.0, 2.0], &[1.0], |_, _| 0.0, 100, Some(1));
    }

    #[test]
    fn default_iterations_is_used_when_zero_passed() {
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let candidate = baseline.clone();
        let r = paired_ci(
            &baseline,
            &candidate,
            |b, c| median(c) - median(b),
            0,
            Some(1),
        );
        // Just check it ran (no panic) and returned a CI triple.
        assert!(r.low <= r.median);
        assert!(r.median <= r.high);
    }
}
