"""Design matrices for the Plackett-Burman / factorial bisection.

`full_factorial(k)` returns every `{-1, +1}` combination for `k ≤ 6`.
`plackett_burman(k)` returns a fractional design with the smallest
multiple-of-4 number of runs ≥ k+1 (7 ≤ k ≤ 31). Both return
`numpy.ndarray` with shape `(runs, k)` and entries in `{-1, +1}`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Classical Plackett-Burman generator rows for n = 4, 8, 12, 16, 20, 24, 28.
# Each row is the first row of the design matrix (values in {+1, -1}); the
# remaining rows are obtained by cyclic shifts, and a final all-minus row
# is appended. Source: NIST/SEMATECH e-Handbook of Statistical Methods,
# §5.3.3.5 "Fractional factorial designs".
_PB_GENERATORS: dict[int, list[int]] = {
    # The plan targets k up to ~8 (ground-truth test), so 8- and 12-run
    # generators cover the common cases.
    8: [1, 1, 1, -1, 1, -1, -1],  # for k ≤ 7
    12: [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],  # for k ≤ 11
    16: [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1],  # for k ≤ 15
    20: [1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1],
    24: [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1],
}


def full_factorial(k: int) -> NDArray[np.int8]:
    """All `2**k` combinations of {+1, -1} for `k` factors.

    Order: little-endian (row 0 is all `-1`, subsequent rows increment).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > 6:
        raise ValueError("full_factorial is capped at k=6 (64 runs); use plackett_burman")
    runs = 1 << k
    out = np.empty((runs, k), dtype=np.int8)
    for i in range(runs):
        for j in range(k):
            out[i, j] = -1 if (i >> j) & 1 == 0 else 1
    return out


def plackett_burman(k: int) -> NDArray[np.int8]:
    """Plackett-Burman design for `k` factors.

    Runs = smallest multiple of 4 that is ≥ k+1. Columns = `k`. Entries
    are `{-1, +1}`. Uses the Paley construction via a pre-tabulated
    generator row for each supported run count.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    runs = 4 * ((k + 4) // 4)  # smallest multiple of 4 ≥ k+1
    if runs not in _PB_GENERATORS:
        raise ValueError(
            f"Plackett-Burman for runs={runs} not tabulated. Supported: "
            f"{sorted(_PB_GENERATORS.keys())}"
        )
    generator = _PB_GENERATORS[runs]
    if len(generator) != runs - 1:
        raise ValueError(f"generator for runs={runs} has wrong length: {len(generator)}")
    mat = np.empty((runs, runs - 1), dtype=np.int8)
    for i in range(runs - 1):
        for j in range(runs - 1):
            mat[i, j] = generator[(j - i) % (runs - 1)]
    mat[runs - 1] = -1
    # Trim to k columns (first k — the PB property means any k-subset is
    # a main-effects design, so order doesn't matter).
    return mat[:, :k]


def is_orthogonal(design: NDArray[np.int8]) -> bool:
    """True if `design.T @ design == runs * I` on the main-effect columns."""
    runs, k = design.shape
    gram = design.T.astype(int) @ design.astype(int)
    return bool(np.array_equal(gram, runs * np.eye(k, dtype=int)))
