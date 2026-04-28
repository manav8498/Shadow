//! Pluggable embedding backend for the semantic axis.
//!
//! The default `TfIdfEmbedder` implements smoothed TF-IDF cosine over
//! the corpus of texts being compared — production-quality for lexical
//! similarity but blind to paraphrase ("yes" vs "I agree" score 0).
//!
//! For paraphrase-robust similarity, callers supply an [`Embedder`]
//! that produces dense vectors per text. The crate stays free of heavy
//! ML dependencies; users bring their own embedding source via:
//!
//!   * `BoxedEmbedder::new(|texts| { ... })` — a closure returning
//!     `Vec<Vec<f32>>` for any external source (ONNX runtime, HF
//!     Inference API, OpenAI embeddings, in-house service, ...).
//!   * A direct impl of [`Embedder`] for stateful adapters that need
//!     to hold model handles, HTTP clients, or tokenizer state.
//!
//! Cross-language consistency: the cosine similarity computation
//! happens in Rust regardless of where vectors come from. As long as
//! two embedders produce comparable vectors (same dimensionality,
//! similar magnitudes), their semantic-axis output stays meaningful.
//!
//! Why no built-in ONNX backend
//! ----------------------------
//! Bundling `ort` + `tokenizers` + a real embedding model would
//! either blow past PyPI's per-wheel size limit (~100 MB) or force
//! users to download the model file on first use — both create
//! friction for the 99% of Shadow users whose semantic-axis needs
//! are already met by TF-IDF over response text. The trait keeps the
//! door open for users with paraphrase-heavy workloads to plug in
//! whatever embedding source they already run, without forcing the
//! cost on the default install.

/// A backend that produces dense embedding vectors for a slice of
/// input texts.
///
/// Implementations must be deterministic for a given input set
/// (otherwise the semantic axis becomes flappy across runs). Vector
/// dimensionality is implementation-defined; only the requirement
/// "every output vector has the same length" is enforced — the cosine
/// math handles any dimensionality.
pub trait Embedder: Send + Sync {
    /// Embed each text in `texts`. The returned vector at position `i`
    /// is the embedding for `texts[i]`. All vectors must have the same
    /// length (panics on mismatched dimensionality are a contract bug;
    /// the caller will validate before computing cosine).
    fn embed(&self, texts: &[&str]) -> Vec<Vec<f32>>;

    /// Optional: a stable identifier for the embedder, included in
    /// diagnostic output so a user can tell at a glance which embedder
    /// produced a given semantic-axis score.
    fn id(&self) -> &str {
        "anonymous"
    }
}

/// Adapter that wraps any `Fn(&[&str]) -> Vec<Vec<f32>>` closure into
/// an [`Embedder`].
///
/// Useful when the embedding source is an HTTP client, an ONNX
/// session, a Python callback (via PyO3), or any other resource the
/// caller already manages.
pub struct BoxedEmbedder<F>
where
    F: Fn(&[&str]) -> Vec<Vec<f32>> + Send + Sync,
{
    f: F,
    name: String,
}

impl<F> BoxedEmbedder<F>
where
    F: Fn(&[&str]) -> Vec<Vec<f32>> + Send + Sync,
{
    /// Wrap a closure as an [`Embedder`] with the default name `"boxed"`.
    pub fn new(f: F) -> Self {
        Self {
            f,
            name: "boxed".to_string(),
        }
    }

    /// Wrap a closure as an [`Embedder`] with a caller-supplied name
    /// (returned by [`Embedder::id`] for diagnostic output).
    pub fn named(f: F, name: impl Into<String>) -> Self {
        Self {
            f,
            name: name.into(),
        }
    }
}

impl<F> Embedder for BoxedEmbedder<F>
where
    F: Fn(&[&str]) -> Vec<Vec<f32>> + Send + Sync,
{
    fn embed(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        (self.f)(texts)
    }

    fn id(&self) -> &str {
        &self.name
    }
}

/// Cosine similarity between two equal-length dense vectors.
///
/// Returns 1.0 when both vectors are zero (consistent with the
/// TF-IDF axis: empty-vs-empty is a perfect match — neither has any
/// semantic content to differ on). Returns 0.0 when exactly one is
/// zero. Otherwise the standard `(a·b) / (‖a‖ · ‖b‖)`.
///
/// The result is clamped to `[-1.0, 1.0]` to absorb floating-point
/// drift.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot: f32 = 0.0;
    let mut na: f32 = 0.0;
    let mut nb: f32 = 0.0;
    for i in 0..a.len() {
        let x = a[i];
        let y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let na = na.sqrt();
    let nb = nb.sqrt();
    if na < 1e-12 && nb < 1e-12 {
        return 1.0;
    }
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors_is_one() {
        let v = [1.0_f32, 2.0, 3.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors_is_zero() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_both_zero_returns_one() {
        let z = [0.0_f32; 4];
        assert!((cosine(&z, &z) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_one_zero_returns_zero() {
        let a = [0.0_f32; 4];
        let b = [1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(cosine(&a, &b), 0.0);
    }

    #[test]
    fn cosine_dim_mismatch_returns_zero() {
        let a = [1.0_f32, 2.0];
        let b = [1.0_f32, 2.0, 3.0];
        assert_eq!(cosine(&a, &b), 0.0);
    }

    #[test]
    fn boxed_embedder_round_trip() {
        let emb = BoxedEmbedder::named(
            |texts: &[&str]| {
                texts
                    .iter()
                    .map(|t| vec![t.len() as f32, 1.0])
                    .collect()
            },
            "len-embed",
        );
        let v = emb.embed(&["abc", "abcdef"]);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], vec![3.0, 1.0]);
        assert_eq!(v[1], vec![6.0, 1.0]);
        assert_eq!(emb.id(), "len-embed");
    }
}
