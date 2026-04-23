"""Real sentence-transformer embeddings for the semantic axis.

The Rust core's default semantic axis is TF-IDF cosine — honest lexical
similarity, but insensitive to paraphrase. When the user installs the
`[embeddings]` extra (`pip install shadow[embeddings]`), this module
lazy-imports `sentence-transformers/all-MiniLM-L6-v2` and re-runs axis
1 with real dense-embedding cosine similarity, splicing the result
back into the DiffReport the way `SanityJudge` splices axis 8.

Invocation:
    shadow diff baseline.agentlog candidate.agentlog --semantic embeddings

If the extra isn't installed, the CLI prints a hint and falls back to
the lexical default. Never silently pretends embeddings ran.
"""

from shadow.embeddings.semantic import SemanticEmbedder, recompute_semantic_axis

__all__ = ["SemanticEmbedder", "recompute_semantic_axis"]
