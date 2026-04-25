# Multimodal diff

`shadow diff --multimodal-diff` adds a per-blob diff section over `blob_ref` records. Two tiers, both optional:

- **Cheap tier (dHash).** When both sides recorded a 64-bit dHash, the renderer computes Hamming distance. `≤10` bits is treated as a near-duplicate; `≤16` is "minor" drift; anything more is "moderate". Cheap to compute, robust to JPEG re-encoding and small crops.
- **Semantic tier (cosine).** When both sides recorded an embedding (e.g. CLIP), the renderer computes cosine similarity. `≥0.85` is "same content"; `0.75-0.85` is minor; `0.5-0.75` is moderate; below `0.5` is severe. The semantic tier wins when both tiers are available.

Identical `blob_id` short-circuits to `severity=none` (content-addressing means same id = same bytes).

## Recording blob_ref records

```python
from shadow.sdk import Session
from shadow.v02_records import BlobStore, record_blob_ref

store = BlobStore.at(".shadow/blobs")
with Session(output_path="trace.agentlog") as s:
    record_blob_ref(
        s,
        blob=open("screenshot.png", "rb").read(),
        mime="image/png",
        store=store,
    )
```

`compute_phash=True` (default) computes a 64-bit dHash when the blob is an image and the optional `imagehash` dependency is installed:

```bash
pip install 'shadow-diff[multimodal]'
```

For the semantic tier, attach a model embedding under `embedding` in the payload before recording. Shadow's renderer doesn't compute embeddings itself — that's intentional, since embedding choice is highly task-specific.

## Severity decision tree

When the renderer compares two `blob_ref` records at the same pair index:

1. If `baseline_blob_id == candidate_blob_id`, `severity=none`.
2. Else if both sides have an embedding, severity is driven by cosine similarity (semantic tier wins).
3. Else if both sides have a phash, severity is driven by Hamming distance (cheap tier).
4. Else (no signal but blob ids differ), severity falls back to `moderate`. A swapped blob with no similarity signal is more likely a real regression than a near-duplicate, so the renderer surfaces it for a human to eyeball. Add a phash or embedding to the recording side if you want quieter diffs.

## Limitations

- **Positional matching only.** Pair `i` of baseline is compared to pair `i` of candidate. If the candidate inserts an extra blob early, every later pair will be misaligned. A best-match alignment mode is on the wishlist; for now keep blob ordering stable across runs.
- **Unmatched blobs.** When one side has more blob_ref records than the other, the unmatched entries surface as severity=severe (one missing side).
