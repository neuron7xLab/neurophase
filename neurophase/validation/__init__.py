"""Scientific validation layer — null-model confrontation and surrogates.

Program C of the Evolution Board lives here. The load-bearing rule is:
**no coupling claim without null-model confrontation.** Every
statistic the library reports about phase coupling, synchronization,
or prediction error must be able to answer: *what would this
statistic look like under a null hypothesis that destroys the thing
we want to measure while preserving the things we do not?*

This subpackage provides:

* :class:`NullModelHarness` — a seeded, deterministic harness that
  runs a statistic against ``n`` surrogate resamples and returns a
  p-value-like rejection result.
* :mod:`neurophase.validation.surrogates` — three surrogate
  generators (phase shuffle, cyclic shift, block bootstrap) with
  explicit null-hypothesis contracts.

Design principles:

1. **Seeded determinism.** Every surrogate generator accepts a
   ``np.random.Generator``; the harness threads a single seed through
   every call.
2. **Explicit null hypothesis.** Each generator's docstring states
   *which property is preserved* and *which is destroyed*. A
   surrogate that does not name its null is not a surrogate.
3. **Type-safe statistics.** ``NullModelHarness.test`` accepts a
   callable ``statistic: (x, y) -> float``; the harness never
   inspects the statistic's internal state.
4. **Replayable.** Two runs with the same seed and the same inputs
   produce identical null distributions to the bit.
"""

from __future__ import annotations

from neurophase.validation.null_model import (
    DEFAULT_N_SURROGATES,
    NullModelHarness,
    NullModelResult,
)
from neurophase.validation.surrogates import (
    block_bootstrap,
    cyclic_shift,
    phase_shuffle,
    time_reversal,
)

__all__ = [
    "DEFAULT_N_SURROGATES",
    "NullModelHarness",
    "NullModelResult",
    "block_bootstrap",
    "cyclic_shift",
    "phase_shuffle",
    "time_reversal",
]
