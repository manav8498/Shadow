"""Behavioral fingerprinting, Hotelling T², and sequential testing.

Four complementary primitives that supplement the nine per-axis
bootstrap tests with multivariate and sequential inference:

- :mod:`fingerprint` — extract a D-dimensional behavioral feature
  vector from a trace (tool-call patterns, stop reason, verbosity,
  latency, refusal rate).

- :mod:`hotelling` — two-sample Hotelling T² test on fingerprint
  matrices; detects joint distribution shift that individual
  univariate tests can miss.  Falls back to OAS shrinkage (Chen et al.
  2010) on the pooled covariance when n1 + n2 − 2 ≤ D, so that the
  test does not blow up in the high-dimension / small-sample regime.

- :mod:`sprt` — Wald SPRT (point-vs-point alternative) and mixture
  SPRT (always-valid inference, mSPRT).  Stops as soon as the streamed
  evidence is decisive.  mSPRT controls Type-I error simultaneously
  across all sample sizes (Robbins 1970, Johari et al. 2017) and is
  the recommended detector for production monitoring.

References
----------
Hotelling (1931); Chen, Wiesel, Eldar & Hero (2010); Wald (1945);
Robbins (1970); Johari, Pekelis & Walsh (2017).
"""

from shadow.statistical.fingerprint import (
    BehavioralVector,
    fingerprint_trace,
    mean_fingerprint,
)
from shadow.statistical.hotelling import HotellingResult, hotelling_t2
from shadow.statistical.sprt import (
    MSPRTDetector,
    MultiSPRT,
    SPRTDetector,
    SPRTState,
)

__all__ = [
    "BehavioralVector",
    "HotellingResult",
    "MSPRTDetector",
    "MultiSPRT",
    "SPRTDetector",
    "SPRTState",
    "fingerprint_trace",
    "hotelling_t2",
    "mean_fingerprint",
]
