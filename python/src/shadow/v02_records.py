"""v0.2 record kinds: ``chunk``, ``harness_event``, ``blob_ref``.

These extend the v0.1 surface (``chat_request``, ``chat_response``,
``tool_call``, ``tool_result``, ``metadata``, ``error``,
``replay_summary``) with three new payload types defined in
:doc:`SPEC <../../../../../SPEC.md>` §4.8 / §4.9 / §4.10.

API:

- :func:`record_harness_event` — append a framework-level event
  (retry, rate-limit, model switch, context-trim, cache, guardrail,
  budget, stream-interrupt, tool-lifecycle) to the active session.
- :func:`record_chunk` — append a single streaming-LLM chunk record
  with ``time_unix_nano`` for replay timing fidelity.
- :class:`BlobStore` — content-addressed sha256 blob store rooted at
  a directory; resolves ``agentlog-blob://`` URIs to bytes.
- :func:`compute_phash_dhash64` — cheap perceptual hash (64-bit dHash)
  for image blobs. Optional dep on ``imagehash`` (extra ``[multimodal]``).
- :func:`record_blob_ref` — content-address a blob, write to the
  store, append a ``blob_ref`` record to the session.

The recording API delegates to ``Session._envelope`` /
``Session._records`` so v0.2 records use the same parent-chain and
content-addressing as v0.1 records. No spec-level breakage — readers
that don't recognise a kind fall back to the unknown-kind passthrough
documented in SPEC §10.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from shadow.sdk.session import Session


# RFC-style "type/subtype" check, with the optional `+suffix` and
# `; param=value` tail. We don't parse the parameters, just bound
# the shape so a caller can't pass "image/png; <html>" or "" and
# have it round-trip into a downstream renderer that interpolates
# it raw.
_MIME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}"
    r"(\s*;\s*[A-Za-z0-9!#$&^_.+-]+\s*=\s*[\w\d!#$&^_.+-]+)*$"
)


HARNESS_CATEGORIES = frozenset(
    {
        "retry",
        "rate_limit",
        "model_switch",
        "context_trim",
        "cache",
        "guardrail",
        "budget",
        "stream_interrupt",
        "tool_lifecycle",
    }
)
"""SPEC §4.9 closed taxonomy. Validated at record time."""


HARNESS_SEVERITIES = frozenset({"info", "warning", "error", "fatal"})


HarnessCategory = Literal[
    "retry",
    "rate_limit",
    "model_switch",
    "context_trim",
    "cache",
    "guardrail",
    "budget",
    "stream_interrupt",
    "tool_lifecycle",
]
HarnessSeverity = Literal["info", "warning", "error", "fatal"]


# ---- harness_event ---------------------------------------------------


def record_harness_event(
    session: Session,
    *,
    category: HarnessCategory,
    name: str,
    severity: HarnessSeverity = "info",
    attributes: dict[str, Any] | None = None,
    parent_id: str | None = None,
) -> str:
    """Append a ``harness_event`` record to ``session``. Returns the
    new record's content id.

    ``category`` MUST be one of :data:`HARNESS_CATEGORIES`. ``name``
    is a free-form identifier within the category (e.g. ``retry.attempted``,
    ``cache.hit``, ``guardrail.blocked``). ``severity`` defaults to
    ``info``. ``attributes`` is a free-form dict — typed-attribute
    validation is intentionally not enforced at the record layer so
    new event types don't require code changes.
    """
    from shadow import _core
    from shadow.errors import ShadowConfigError

    if category not in HARNESS_CATEGORIES:
        raise ShadowConfigError(
            f"unknown harness_event category {category!r}; "
            f"must be one of {sorted(HARNESS_CATEGORIES)}"
        )
    if severity not in HARNESS_SEVERITIES:
        raise ShadowConfigError(
            f"unknown harness_event severity {severity!r}; "
            f"must be one of {sorted(HARNESS_SEVERITIES)}"
        )
    if not isinstance(name, str) or not name:
        raise ShadowConfigError("harness_event `name` must be a non-empty string")
    payload: dict[str, Any] = {
        "category": category,
        "name": name,
        "severity": severity,
        "attributes": dict(attributes or {}),
    }
    payload = session._redact(payload)
    record_id = _core.content_id(payload)
    parent = parent_id if parent_id is not None else session._last_id()
    session._records.append(session._envelope("harness_event", payload, record_id, parent=parent))
    return record_id


# ---- chunk ----------------------------------------------------------


def record_chunk(
    session: Session,
    *,
    chunk_index: int,
    delta: dict[str, Any],
    is_final: bool = False,
    time_unix_nano: int | None = None,
    parent_id: str | None = None,
) -> str:
    """Append a streaming ``chunk`` record. Returns the new record's id.

    ``delta`` is a passthrough of the provider's per-chunk delta
    (Anthropic ``text_delta`` / ``input_json_delta`` / ``thinking_delta``,
    OpenAI ``{content?, tool_calls?[]}``). The format is intentionally
    unconstrained — only the assembler at recording time + the differ
    at comparison time interpret it.

    ``time_unix_nano`` defaults to ``time.time_ns()`` at the call
    site. Stored absolute (not relative) so clock-skew correction
    and partial replay survive — replay engines compute deadlines
    from the recorded absolute timestamps.
    """
    from shadow import _core
    from shadow.errors import ShadowConfigError

    chunk_index_int = int(chunk_index)
    if chunk_index_int < 0:
        raise ShadowConfigError(f"chunk_index must be >= 0 (got {chunk_index_int})")
    payload: dict[str, Any] = {
        "chunk_index": chunk_index_int,
        "time_unix_nano": int(time_unix_nano if time_unix_nano is not None else time.time_ns()),
        "delta": dict(delta),
    }
    if is_final:
        payload["is_final"] = True
    payload = session._redact(payload)
    record_id = _core.content_id(payload)
    parent = parent_id if parent_id is not None else session._last_id()
    session._records.append(session._envelope("chunk", payload, record_id, parent=parent))
    return record_id


def replay_chunk_timing(chunks: list[dict[str, Any]]) -> list[float]:
    """Return per-chunk wait times (seconds) for a replay engine to
    sleep between yielding consecutive chunks.

    Output[0] is always 0 (yield the first chunk immediately).
    Output[i] for i>0 is the seconds gap from chunks[i-1] to chunks[i],
    floored at 0 (handles non-monotonic timestamps gracefully).
    """
    out: list[float] = [0.0]
    last_ns: int | None = None
    for rec in chunks:
        payload = rec.get("payload") or {}
        ns = int(payload.get("time_unix_nano") or 0)
        if last_ns is None:
            last_ns = ns
            continue
        delta_s = max(0.0, (ns - last_ns) / 1e9)
        out.append(delta_s)
        last_ns = ns
    # First chunk doesn't get a leading 0 in our list; correct length.
    if len(out) != len(chunks):
        # Edge case: fewer than 2 chunks, or empty input.
        return [0.0] * len(chunks)
    return out


async def replay_chunks_async(
    chunks: list[dict[str, Any]],
    yielder: Any,
    *,
    speed: float = 1.0,
) -> None:
    """Replay chunk records to ``yielder`` (an async function or
    callback) preserving original inter-chunk timing.

    Uses a monotonic-deadline loop, NOT cumulative ``sleep(delta)``.
    Cumulative sleep accumulates rounding error on long streams and
    drifts measurably on multi-second streams; deadline-relative
    sleep stays accurate.

    ``speed`` is a multiplier — 1.0 = real-time, 2.0 = 2x faster,
    inf = no delay. ``yielder`` is called once per chunk with the
    chunk's payload.
    """
    import asyncio

    if not chunks:
        return
    if speed <= 0:
        raise ValueError("speed must be > 0")

    base_record_ns = int((chunks[0].get("payload") or {}).get("time_unix_nano") or 0)
    base_wall_ns = time.monotonic_ns()
    for rec in chunks:
        payload = rec.get("payload") or {}
        record_ns = int(payload.get("time_unix_nano") or 0)
        offset_ns = max(0, record_ns - base_record_ns)
        offset_s = (offset_ns / 1e9) / float(speed)
        deadline_ns = base_wall_ns + int(offset_s * 1e9)
        wait_s = max(0.0, (deadline_ns - time.monotonic_ns()) / 1e9)
        if wait_s > 0:
            await asyncio.sleep(wait_s)
        result = yielder(payload)
        if hasattr(result, "__await__"):
            await result


# ---- blob_ref + content-addressed store ------------------------------


@dataclass
class BlobStore:
    """sha256 content-addressed blob store rooted at a directory.

    Files live at ``<root>/<aa>/<rest>`` (git-objects-style sharding,
    same convention Shadow's record store already uses). Identical
    blobs collapse to one file.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def at(cls, path: str | Path) -> BlobStore:
        """Convenience: ``BlobStore.at(".shadow/blobs")`` is the same
        as ``BlobStore(root=Path(".shadow/blobs"))``. Easier to read
        in user code that doesn't already have ``Path`` imported."""
        return cls(root=Path(path))

    def put(self, data: bytes) -> str:
        """Write ``data`` and return its sha256 content id (with prefix)."""
        digest = hashlib.sha256(data).hexdigest()
        path = self._path_for(digest)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return f"sha256:{digest}"

    def get(self, blob_id: str) -> bytes:
        """Return the blob bytes for ``blob_id`` (with or without prefix).
        Raises FileNotFoundError if the blob isn't present."""
        digest = blob_id[len("sha256:") :] if blob_id.startswith("sha256:") else blob_id
        return self._path_for(digest).read_bytes()

    def has(self, blob_id: str) -> bool:
        digest = blob_id[len("sha256:") :] if blob_id.startswith("sha256:") else blob_id
        return self._path_for(digest).is_file()

    def uri_for(self, blob_id: str, *, store: str = "default") -> str:
        digest = blob_id[len("sha256:") :] if blob_id.startswith("sha256:") else blob_id
        return f"agentlog-blob://{store}/{digest}"

    def _path_for(self, digest: str) -> Path:
        return self.root / digest[:2] / digest[2:]


def compute_phash_dhash64(image_bytes: bytes) -> dict[str, str] | None:
    """Compute a 64-bit dHash for an image blob. Returns the
    SPEC-shaped phash dict, or None if the optional ``imagehash``
    dependency isn't installed.

    Cheap (≈0.3ms on a laptop) and stable under JPEG re-encoding /
    small crops. Hamming distance ≤10/64 = "near-duplicate," ≥16 =
    "different." Implementations MAY skip dHash and pay for CLIP
    later — this is the cheap tier.
    """
    try:
        import imagehash  # type: ignore[import-not-found, import-untyped, unused-ignore]
        from PIL import Image  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ImportError:
        return None
    try:
        from io import BytesIO

        img = Image.open(BytesIO(image_bytes))
        h = imagehash.dhash(img, hash_size=8)  # 8x8 = 64 bits
    except Exception:
        return None
    return {"algo": "dhash64", "hex": str(h)}


def phash_distance(a: dict[str, Any], b: dict[str, Any]) -> int | None:
    """Hamming distance between two SPEC-shaped phash dicts. Returns
    None if the algorithms don't match, the hex is missing/malformed,
    or the hex length is wrong for the algo (each algo has a fixed bit
    width, so comparing different lengths would silently produce a
    misleading distance).
    """
    algo = a.get("algo")
    if algo != b.get("algo") or not isinstance(algo, str):
        return None
    expected_hex_len = _PHASH_HEX_LEN.get(algo)
    if expected_hex_len is None:
        return None
    try:
        ah_str, bh_str = a["hex"], b["hex"]
    except KeyError:
        return None
    if not isinstance(ah_str, str) or not isinstance(bh_str, str):
        return None
    if len(ah_str) != expected_hex_len or len(bh_str) != expected_hex_len:
        return None
    try:
        ah = int(ah_str, 16)
        bh = int(bh_str, 16)
    except ValueError:
        return None
    return bin(ah ^ bh).count("1")


# Hex-character widths for known phash algorithms. dhash64 is 64 bits
# = 16 hex chars. New algorithms register their bit width here.
_PHASH_HEX_LEN: dict[str, int] = {
    "dhash64": 16,
}


def record_blob_ref(
    session: Session,
    *,
    blob: bytes,
    mime: str,
    store: BlobStore,
    store_name: str = "default",
    parent_id: str | None = None,
    compute_phash: bool = True,
) -> tuple[str, str]:
    """Write ``blob`` to ``store``, append a ``blob_ref`` record to
    the session, return ``(record_id, blob_id)``.

    ``compute_phash``: when True (default) AND the blob is an image
    AND the optional ``imagehash`` dep is installed, a 64-bit dHash
    is computed and embedded in the record. Otherwise the record
    omits ``phash``.
    """
    from shadow import _core
    from shadow.errors import ShadowConfigError

    if not isinstance(blob, bytes | bytearray):
        raise TypeError("blob must be bytes")
    mime_str = str(mime)
    if not _MIME_RE.match(mime_str):
        raise ShadowConfigError(
            f"blob_ref mime must match RFC 6838 type/subtype shape; got {mime_str!r}"
        )
    blob = bytes(blob)
    blob_id = store.put(blob)
    payload: dict[str, Any] = {
        "mime": mime_str,
        "size_bytes": len(blob),
        "blob_id": blob_id,
        "uri": store.uri_for(blob_id, store=store_name),
    }
    if compute_phash and mime.startswith("image/"):
        phash = compute_phash_dhash64(blob)
        if phash is not None:
            payload["phash"] = phash
    payload = session._redact(payload)
    record_id = _core.content_id(payload)
    parent = parent_id if parent_id is not None else session._last_id()
    session._records.append(session._envelope("blob_ref", payload, record_id, parent=parent))
    return record_id, blob_id


# ---- harness-event diff dimension ------------------------------------


@dataclass
class HarnessEventDelta:
    """One entry in a harness-event diff."""

    category: str
    name: str
    severity: str
    baseline_count: int
    candidate_count: int
    count_delta: int
    first_occurrence_baseline: int | None  # pair_index of first occurrence
    first_occurrence_candidate: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "name": self.name,
            "severity": self.severity,
            "baseline_count": self.baseline_count,
            "candidate_count": self.candidate_count,
            "count_delta": self.count_delta,
            "first_occurrence_baseline": self.first_occurrence_baseline,
            "first_occurrence_candidate": self.first_occurrence_candidate,
        }


def harness_event_diff(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> list[HarnessEventDelta]:
    """Compute per-(category, name) count deltas between two traces.

    Returns one entry per distinct (category, name) seen in either
    trace, sorted by absolute count_delta descending. Only
    ``harness_event`` records contribute; other kinds are ignored.
    """
    base_index: dict[tuple[str, str], dict[str, Any]] = {}
    cand_index: dict[tuple[str, str], dict[str, Any]] = {}

    def _scan(records: list[dict[str, Any]], target: dict[tuple[str, str], dict[str, Any]]) -> None:
        pair_idx = -1
        for rec in records:
            if rec.get("kind") == "chat_response":
                pair_idx += 1
            if rec.get("kind") != "harness_event":
                continue
            payload = rec.get("payload") or {}
            key = (str(payload.get("category") or ""), str(payload.get("name") or ""))
            slot = target.setdefault(
                key,
                {
                    "count": 0,
                    "severity": str(payload.get("severity") or "info"),
                    "first_pair": None,
                },
            )
            slot["count"] = int(slot["count"]) + 1
            if slot["first_pair"] is None:
                slot["first_pair"] = pair_idx if pair_idx >= 0 else 0

    _scan(baseline, base_index)
    _scan(candidate, cand_index)
    keys = set(base_index) | set(cand_index)
    out: list[HarnessEventDelta] = []
    for key in keys:
        b = base_index.get(key) or {}
        c = cand_index.get(key) or {}
        b_count = int(b.get("count") or 0)
        c_count = int(c.get("count") or 0)
        # Pick severity from whichever side has it (candidate first).
        severity = str(c.get("severity") or b.get("severity") or "info")
        out.append(
            HarnessEventDelta(
                category=key[0],
                name=key[1],
                severity=severity,
                baseline_count=b_count,
                candidate_count=c_count,
                count_delta=c_count - b_count,
                first_occurrence_baseline=b.get("first_pair"),
                first_occurrence_candidate=c.get("first_pair"),
            )
        )
    out.sort(key=lambda d: -abs(d.count_delta))
    return out


__all__ = [
    "HARNESS_CATEGORIES",
    "HARNESS_SEVERITIES",
    "BlobStore",
    "HarnessCategory",
    "HarnessEventDelta",
    "HarnessSeverity",
    "compute_phash_dhash64",
    "harness_event_diff",
    "phash_distance",
    "record_blob_ref",
    "record_chunk",
    "record_harness_event",
    "replay_chunk_timing",
    "replay_chunks_async",
]
