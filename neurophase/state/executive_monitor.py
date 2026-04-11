"""Executive monitor — online estimator of cognitive overload.

Implements the **executive function monitor** referenced in
``docs/science_basis.md`` §2.B. It fuses three channels:

1. **EEG beta power** — stress-induced overload marker.
2. **HRV (e.g., RMSSD / HF)** — autonomic regulation. High HRV protects
   against overload; low HRV amplifies it.
3. **Error-burst context** — recent behavioral errors shift the prior
   toward overload.

The monitor produces an ``OverloadIndex ∈ [0, 1]`` and a ``PacingDirective``:

    NORMAL      — keep current tempo / autonomy
    SLOW_DOWN   — reduce interface width, add guided prompts
    HARD_BLOCK  — request explicit verification step before any high-impact
                  action; pair with ``ExecutionGate`` to enforce I₁.

No synthetic fallback. If any channel is missing, the monitor returns
``HARD_BLOCK`` with ``reason="sensor_absent"``, mirroring the ``I₃`` invariant.

This module is intentionally free of SciPy dependencies — it is meant to run
in the hot path of the trading loop.

Falsifiability (Prediction 2, docs/science_basis.md §3):
    AUC/PR for predicting an ``error_burst`` in a 30–120 s window must beat
    a pure-latency baseline. If it does not, this monitor is wrong and must
    be removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Tunable defaults
# ---------------------------------------------------------------------------

#: Minimum number of samples before the monitor can emit a non-``SENSOR_ABSENT``
#: directive. Keeps the warm-up window short but non-zero.
DEFAULT_WARMUP: Final[int] = 4

#: Weight vector for the three channels (beta, HRV deficit, error burst).
#: Chosen so that in the neutral case (z=0 across channels) overload = 0.5.
DEFAULT_WEIGHTS: Final[tuple[float, float, float]] = (0.5, 0.3, 0.2)

#: Logistic slope. Picked so that ``|z| ≈ 2`` → overload ≈ 0.88.
DEFAULT_SLOPE: Final[float] = 1.0

#: Band thresholds on the overload index.
DEFAULT_SLOW_DOWN_THRESHOLD: Final[float] = 0.55
DEFAULT_HARD_BLOCK_THRESHOLD: Final[float] = 0.80

#: Maximum absolute z-score fed into the weighted sum. Larger deviations are
#: clipped — in neurophysiology anything beyond ±4σ is "saturated" and the
#: monitor should not be moved further by it.
Z_CLIP: Final[float] = 4.0


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class PacingDirective(Enum):
    """Discrete pacing decision."""

    NORMAL = auto()
    SLOW_DOWN = auto()
    HARD_BLOCK = auto()
    SENSOR_ABSENT = auto()


@dataclass(frozen=True)
class ExecutiveSample:
    """One monitor input sample.

    All channels are optional to let callers report partial data, but the
    monitor will return ``SENSOR_ABSENT`` if any required channel is missing.

    Parameters
    ----------
    beta_power
        EEG beta-band power (arbitrary units). Higher = more overload.
    hrv
        Heart-rate variability (e.g., RMSSD). Higher = more regulation capacity.
    error_burst
        Rolling count (or rate) of behavioral errors in a short window.
    timestamp
        Unix timestamp in seconds. Only used for ordering; the monitor does
        not extrapolate across gaps.
    """

    beta_power: float | None
    hrv: float | None
    error_burst: float | None
    timestamp: float


@dataclass(frozen=True)
class OverloadIndex:
    """Scalar overload estimate with provenance.

    Attributes
    ----------
    value
        Overload ∈ [0, 1]. ``0`` = fully regulated, ``1`` = saturated.
    beta_z
        z-score of the current beta-power sample against the calibration window.
    hrv_z
        z-score of the HRV *deficit* (lower HRV → higher overload).
    error_z
        z-score of the error-burst channel.
    directive
        Pacing directive derived from ``value``.
    reason
        Human-readable explanation (used by the gate and in session archives).
    """

    value: float
    beta_z: float
    hrv_z: float
    error_z: float
    directive: PacingDirective
    reason: str


@dataclass(frozen=True)
class VerificationStep:
    """Structured friction injected before a high-impact action.

    The step is deliberately small: its purpose is to break passive acceptance
    (§2.A of the science basis) without destroying flow.

    Attributes
    ----------
    prompt
        The question presented to the operator.
    expected_kind
        One of ``"counter_example"``, ``"second_source"``, ``"sanity_check"``.
    deadline_seconds
        Soft deadline before the step is treated as failed (and the gate stays
        blocked).
    """

    prompt: str
    expected_kind: str
    deadline_seconds: float


@dataclass(frozen=True)
class ExecutiveMonitorConfig:
    """Monitor configuration.

    Parameters
    ----------
    window
        Rolling window length for the calibration statistics (samples).
    warmup
        Samples required before non-``SENSOR_ABSENT`` directives are produced.
    weights
        Channel weights (beta, hrv_deficit, error_burst).
    slope
        Logistic slope on the weighted z-score.
    slow_down_threshold
        ``value >=`` this → at least ``SLOW_DOWN``.
    hard_block_threshold
        ``value >=`` this → ``HARD_BLOCK``.
    """

    window: int = 64
    warmup: int = DEFAULT_WARMUP
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS
    slope: float = DEFAULT_SLOPE
    slow_down_threshold: float = DEFAULT_SLOW_DOWN_THRESHOLD
    hard_block_threshold: float = DEFAULT_HARD_BLOCK_THRESHOLD

    def __post_init__(self) -> None:
        if self.window < 8:
            raise ValueError(f"window must be ≥ 8, got {self.window}")
        if self.warmup < 1:
            raise ValueError(f"warmup must be ≥ 1, got {self.warmup}")
        if self.warmup > self.window:
            raise ValueError("warmup cannot exceed window")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError(f"weights must sum to 1, got {self.weights}")
        if any(w < 0 for w in self.weights):
            raise ValueError(f"weights must be non-negative, got {self.weights}")
        if self.slope <= 0:
            raise ValueError(f"slope must be > 0, got {self.slope}")
        if not 0.0 < self.slow_down_threshold < self.hard_block_threshold < 1.0:
            raise ValueError(
                "thresholds must satisfy 0 < slow_down < hard_block < 1, "
                f"got slow_down={self.slow_down_threshold}, "
                f"hard_block={self.hard_block_threshold}"
            )


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


@dataclass
class _RollingStats:
    """Rolling mean / std estimator bounded by a fixed window."""

    window: int
    buffer: list[float] = field(default_factory=list)

    def update(self, value: float) -> None:
        self.buffer.append(value)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)

    def ready(self, warmup: int) -> bool:
        return len(self.buffer) >= warmup

    def zscore(self, value: float) -> float:
        if not self.buffer:
            return 0.0
        arr: NDArray[np.float64] = np.asarray(self.buffer, dtype=np.float64)
        mean = float(arr.mean())
        # ddof=0: we treat the window as the full population of recent history.
        std = float(arr.std(ddof=0))
        # Scale-aware std floor so a zero-variance window does not collapse
        # z-scores to zero. Use 10% of the larger of |mean| and |value|
        # (plus a tiny absolute epsilon) as the minimum detectable deviation.
        scale_ref = max(abs(mean), abs(value))
        scale_floor = 0.1 * scale_ref + 1e-3
        std = max(std, scale_floor)
        return (value - mean) / std


class ExecutiveMonitor:
    """Online executive-function monitor.

    Usage
    -----
    >>> monitor = ExecutiveMonitor()
    >>> for sample in samples:
    ...     result = monitor.update(sample)
    ...     if result.directive is PacingDirective.HARD_BLOCK:
    ...         raise_verification_step()
    """

    def __init__(self, config: ExecutiveMonitorConfig | None = None) -> None:
        self.config: ExecutiveMonitorConfig = config or ExecutiveMonitorConfig()
        self._beta = _RollingStats(window=self.config.window)
        self._hrv = _RollingStats(window=self.config.window)
        self._errors = _RollingStats(window=self.config.window)
        self._last_ts: float | None = None

    # -- public API ---------------------------------------------------------

    def update(self, sample: ExecutiveSample) -> OverloadIndex:
        """Ingest one sample and return the updated overload estimate.

        Monotonic timestamps are required: samples that arrive out of order
        raise ``ValueError`` rather than being silently reordered.
        """
        if self._last_ts is not None and sample.timestamp < self._last_ts:
            raise ValueError(
                f"ExecutiveMonitor requires monotonic timestamps "
                f"(got {sample.timestamp}, last was {self._last_ts})"
            )
        self._last_ts = sample.timestamp

        if (
            sample.beta_power is None
            or sample.hrv is None
            or sample.error_burst is None
            or not np.isfinite(sample.beta_power)
            or not np.isfinite(sample.hrv)
            or not np.isfinite(sample.error_burst)
        ):
            return OverloadIndex(
                value=1.0,
                beta_z=0.0,
                hrv_z=0.0,
                error_z=0.0,
                directive=PacingDirective.SENSOR_ABSENT,
                reason="sensor_absent",
            )

        # Compute z-scores against the history BEFORE folding this sample in.
        # This keeps z-scores unbiased on the first ``warmup`` samples.
        # Clip to ±Z_CLIP — beyond that the signal is saturated and should
        # not drag the weighted sum any further.
        beta_z = _clip(self._beta.zscore(sample.beta_power))
        # HRV deficit: lower HRV -> more overload, hence negated z-score.
        hrv_z = _clip(-self._hrv.zscore(sample.hrv))
        error_z = _clip(self._errors.zscore(sample.error_burst))

        # Fold the sample into the rolling buffers.
        self._beta.update(sample.beta_power)
        self._hrv.update(sample.hrv)
        self._errors.update(sample.error_burst)

        if not (
            self._beta.ready(self.config.warmup)
            and self._hrv.ready(self.config.warmup)
            and self._errors.ready(self.config.warmup)
        ):
            return OverloadIndex(
                value=0.5,
                beta_z=beta_z,
                hrv_z=hrv_z,
                error_z=error_z,
                directive=PacingDirective.NORMAL,
                reason="warmup",
            )

        w_beta, w_hrv, w_err = self.config.weights
        z = w_beta * beta_z + w_hrv * hrv_z + w_err * error_z
        value = _logistic(z, slope=self.config.slope)

        directive, reason = self._classify(value, beta_z, hrv_z, error_z)
        return OverloadIndex(
            value=value,
            beta_z=beta_z,
            hrv_z=hrv_z,
            error_z=error_z,
            directive=directive,
            reason=reason,
        )

    def verification_for(self, overload: OverloadIndex) -> VerificationStep | None:
        """Return a ``VerificationStep`` when the directive demands friction.

        ``SLOW_DOWN`` requests a lightweight sanity-check prompt; ``HARD_BLOCK``
        upgrades to a counter-example. ``NORMAL`` / ``SENSOR_ABSENT`` return
        ``None`` (the latter is already a hard stop via the gate).
        """
        match overload.directive:
            case PacingDirective.SLOW_DOWN:
                return VerificationStep(
                    prompt="State the most likely way this decision is wrong in one sentence.",
                    expected_kind="sanity_check",
                    deadline_seconds=20.0,
                )
            case PacingDirective.HARD_BLOCK:
                return VerificationStep(
                    prompt="Provide a counter-example or a second independent source for this action.",
                    expected_kind="counter_example",
                    deadline_seconds=45.0,
                )
            case _:
                return None

    def reset(self) -> None:
        """Discard the calibration window. Use at session boundaries."""
        self._beta = _RollingStats(window=self.config.window)
        self._hrv = _RollingStats(window=self.config.window)
        self._errors = _RollingStats(window=self.config.window)
        self._last_ts = None

    # -- internals ----------------------------------------------------------

    def _classify(
        self, value: float, beta_z: float, hrv_z: float, error_z: float
    ) -> tuple[PacingDirective, str]:
        if value >= self.config.hard_block_threshold:
            directive = PacingDirective.HARD_BLOCK
        elif value >= self.config.slow_down_threshold:
            directive = PacingDirective.SLOW_DOWN
        else:
            directive = PacingDirective.NORMAL
        reason = (
            f"overload={value:.3f} (beta_z={beta_z:+.2f}, hrv_z={hrv_z:+.2f}, err_z={error_z:+.2f})"
        )
        return directive, reason


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def _logistic(z: float, *, slope: float) -> float:
    """Numerically-stable logistic that stays inside [0, 1]."""
    # np.tanh is monotonically related to the logistic and avoids overflow
    # for large |z|. Map to [0, 1] via (1 + tanh) / 2.
    return float(0.5 * (1.0 + np.tanh(0.5 * slope * z)))


def _clip(z: float) -> float:
    """Clip a z-score to ±Z_CLIP."""
    if z > Z_CLIP:
        return Z_CLIP
    if z < -Z_CLIP:
        return -Z_CLIP
    return z
