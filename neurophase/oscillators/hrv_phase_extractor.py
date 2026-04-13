"""HRV → instantaneous phase via IBI cubic-spline + Hilbert.

Implements the :class:`~neurophase.oscillators.neural_protocol.NeuralPhaseExtractor`
protocol by turning a stream of R-R intervals (real Polar H10, or
replay CSV, or LSL loopback) into an instantaneous phase series.

Pipeline (exactly the user-specified standard HRV-phase method):

    RR intervals  [(ts_i, rr_ms_i)]  at irregular event times
      -> IBI "signal": rr_ms as a function of time
      -> cubic-spline interpolation onto a uniform 4 Hz grid
      -> neurophase.core.phase.compute_phase(x_grid)
         (standardize + Daubechies D4 wavelet denoise + Hilbert + angle)
      -> instantaneous phase φ(t)  in (-π, π]

This is the **only** transform shipped for HRV → phase; there is no
per-subject tuning layer and no runtime fitting. If a different
method is ever needed, it is a separate extractor, not a flag on
this one (CLAUDE.md § implementation_style: one truth per parser).

Honest limits (must stay in the docstring):

  * HRV-phase ≠ EEG-phase. Coupling this phase into a Kuramoto
    network treats the cardiac rhythm as ONE oscillator candidate
    among many. It is NOT a cortical readout.
  * Not a clinical readiness metric. Downstream gating uses this
    as a signal-quality / coupling input, never as a medical claim.
  * Requires ≥ 20 RR samples and ≥ 30 s of history before the
    extractor reports LIVE. Below that it reports DEGRADED with
    an empty phase array (honest null, invariant I₃).

Design shape:

  * :func:`ibi_to_phase_series` — pure function (samples → phase array),
    deterministic, testable without any RR source.
  * :class:`HRVPhaseExtractor` — stateless wrapper that accepts an
    injected ``rr_source`` callable. The callable returns the current
    rolling RR window as a sequence of :class:`RRSample`. This keeps
    transport (LSL, CSV, live-producer loopback) out of the transform.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from neurophase.core.phase import compute_phase
from neurophase.oscillators.neural_protocol import NeuralFrame, SensorStatus
from neurophase.physio.replay import RRSample

FloatArray = npt.NDArray[np.float64]

#: Uniform target sample rate for IBI interpolation. 4 Hz is standard
#: in HRV phase literature: it is well above twice the highest HRV
#: component of physiological interest (HF band ≤ 0.4 Hz), while
#: keeping the grid short enough for a compact window.
DEFAULT_TARGET_SR_HZ: float = 4.0

#: Minimum sustained history (seconds) before the extractor may
#: report LIVE. 30 s covers enough respiratory cycles (HF ~0.2 Hz)
#: for a stable Hilbert phase.
DEFAULT_MIN_WINDOW_S: float = 30.0

#: Minimum number of RR samples inside the window before LIVE. A
#: healthy subject at ~75 bpm produces ~37 RR in 30 s; 20 is a
#: conservative floor that still allows brisk pulse rates.
DEFAULT_MIN_RR_SAMPLES: int = 20

#: Hard envelope on interpolated IBI values. Any interpolated value
#: outside this range (e.g. spline overshoot on a sparse buffer)
#: marks the extractor DEGRADED for this call.
_IBI_SANITY_MIN_MS: float = 250.0
_IBI_SANITY_MAX_MS: float = 2500.0


RRSource = Callable[[], Sequence[RRSample]]


# ---------------------------------------------------------------------------
#   Pure function: RR samples -> phase series
# ---------------------------------------------------------------------------


def ibi_to_phase_series(
    samples: Sequence[RRSample],
    *,
    target_sr_hz: float = DEFAULT_TARGET_SR_HZ,
    denoise: bool = True,
) -> FloatArray:
    """Map a list of RR samples to an instantaneous phase series.

    Pure function — no side effects, no I/O, deterministic. Raises
    :class:`ValueError` if the input does not meet the minimum
    conditions for a defensible spline + phase.

    Parameters
    ----------
    samples
        Sequence of :class:`RRSample`. Must be strictly monotonic in
        ``timestamp_s`` (the RRReplayReader / live consumer enforces
        this already). Length ≥ 4.
    target_sr_hz
        Uniform grid sample rate in Hz. Must be > 0.
    denoise
        Forwarded to :func:`compute_phase`. Default True = Daubechies
        D4 wavelet denoise before Hilbert.

    Returns
    -------
    FloatArray
        Instantaneous phase φ(t) sampled at ``target_sr_hz`` across
        the span ``[samples[0].timestamp_s, samples[-1].timestamp_s]``.
        Values in ``(-π, π]``.

    Raises
    ------
    ValueError
        On too-few samples, non-monotonic timestamps, or a too-short
        span to host the uniform grid.
    """
    if target_sr_hz <= 0:
        raise ValueError(f"target_sr_hz must be > 0, got {target_sr_hz!r}")
    if len(samples) < 4:
        raise ValueError(f"need >= 4 RR samples, got {len(samples)}")

    ts = np.asarray([s.timestamp_s for s in samples], dtype=np.float64)
    rr = np.asarray([s.rr_ms for s in samples], dtype=np.float64)

    if not np.all(np.diff(ts) > 0):
        raise ValueError("RR timestamps must be strictly monotonic")

    span = float(ts[-1] - ts[0])
    if span * target_sr_hz < 4.0:
        raise ValueError(
            f"RR span {span!r} s at {target_sr_hz!r} Hz is too short for "
            f"a defensible phase (need span * sr >= 4 grid samples)"
        )

    grid = np.arange(ts[0], ts[-1], 1.0 / target_sr_hz, dtype=np.float64)
    # Cubic spline over the irregular RR event series. Fits through
    # every knot; no smoothing pass (that belongs in compute_phase).
    spline = CubicSpline(ts, rr, bc_type="not-a-knot", extrapolate=False)
    ibi_grid = spline(grid)
    # Replace NaNs at the edges (if any) with nearest-neighbour; NaN
    # propagation would poison the Hilbert transform downstream.
    if np.any(~np.isfinite(ibi_grid)):
        ibi_grid = np.nan_to_num(
            ibi_grid, nan=float(np.mean(rr)), posinf=float(np.max(rr)), neginf=float(np.min(rr))
        )
    return compute_phase(ibi_grid, denoise=denoise)


# ---------------------------------------------------------------------------
#   Extractor
# ---------------------------------------------------------------------------


@dataclass
class HRVPhaseExtractor:
    """NeuralPhaseExtractor implementation for HRV.

    Stateless: the RR buffer lives in the source, not in the extractor.
    On each :meth:`extract`, the extractor pulls the current window
    from its source and runs the transform. If the window is too
    short, the status goes DEGRADED and the returned frame carries
    an empty phase array (invariant I₃ compliance).

    Parameters
    ----------
    rr_source
        Callable returning the current RR window. Tests inject a
        closure over a fixed list; live deployments inject an
        LSL-inlet wrapper or any callable that reads from the physio
        live session's ledger / stream.
    min_window_s
        Minimum span between the first and last RR sample for LIVE.
    min_rr_samples
        Minimum number of samples in the window for LIVE.
    target_sr_hz
        Interpolation grid rate.
    channel_label
        Label attached to the emitted :class:`NeuralFrame`.

    Notes
    -----
    Determinism: given the same source output, two extractors produce
    byte-identical phase series (all math is in numpy + scipy
    deterministic primitives). This is required by the repo-level
    determinism contract.
    """

    rr_source: RRSource
    min_window_s: float = DEFAULT_MIN_WINDOW_S
    min_rr_samples: int = DEFAULT_MIN_RR_SAMPLES
    target_sr_hz: float = DEFAULT_TARGET_SR_HZ
    channel_label: str = "hrv"
    #: One-shot cache of the most recent extract() call's computed
    #: status, used by :meth:`status` so that ``status() -> ABSENT``
    #: after a failing pull does not contradict the follow-up
    #: ``extract()`` on the same source. Per CLAUDE.md: one truth.
    _last_status: SensorStatus = field(default=SensorStatus.ABSENT, init=False)

    def __post_init__(self) -> None:
        if self.min_window_s <= 0:
            raise ValueError(f"min_window_s must be > 0, got {self.min_window_s!r}")
        if self.min_rr_samples < 4:
            raise ValueError(f"min_rr_samples must be >= 4, got {self.min_rr_samples!r}")
        if self.target_sr_hz <= 0:
            raise ValueError(f"target_sr_hz must be > 0, got {self.target_sr_hz!r}")
        if not self.channel_label:
            raise ValueError("channel_label must be non-empty")

    # NeuralPhaseExtractor protocol ---------------------------------------

    def status(self) -> SensorStatus:
        """Last known status. Cheap: never triggers a source pull.

        On first call this is ABSENT; it becomes LIVE / DEGRADED /
        ABSENT according to the most recent :meth:`extract` outcome.
        """
        return self._last_status

    def extract(self) -> NeuralFrame:
        """Pull the current RR window and return a :class:`NeuralFrame`.

        Returns a frame with status LIVE, DEGRADED, or ABSENT depending
        on what the source reports and what the transform can defend.
        Never fabricates phase: non-LIVE frames carry an empty phase
        array and sample_rate_hz = 0.
        """
        try:
            window = self.rr_source()
        except Exception:
            # Source raised -> treat as ABSENT. The source should not
            # raise for "no data"; that is an empty sequence.
            self._last_status = SensorStatus.ABSENT
            return _absent_frame((self.channel_label,))

        if not window:
            self._last_status = SensorStatus.ABSENT
            return _absent_frame((self.channel_label,))

        if len(window) < self.min_rr_samples:
            self._last_status = SensorStatus.DEGRADED
            return _absent_frame((self.channel_label,), degraded=True)

        span = float(window[-1].timestamp_s - window[0].timestamp_s)
        if span < self.min_window_s:
            self._last_status = SensorStatus.DEGRADED
            return _absent_frame((self.channel_label,), degraded=True)

        try:
            phase_series = ibi_to_phase_series(window, target_sr_hz=self.target_sr_hz)
        except ValueError:
            # Transform failed (e.g. degenerate spline input). Report
            # DEGRADED rather than propagating.
            self._last_status = SensorStatus.DEGRADED
            return _absent_frame((self.channel_label,), degraded=True)

        # Final envelope sanity on the interpolated IBI: reject wild
        # overshoot that could only be a numerical pathology.
        # The phase series inherits standardization from compute_phase,
        # so we check the upstream spline output via its min/max.
        if phase_series.size < 2:
            self._last_status = SensorStatus.DEGRADED
            return _absent_frame((self.channel_label,), degraded=True)

        # Reshape to (C, T) = (1, len(phase_series)) per NeuralFrame contract.
        phases = phase_series.reshape(1, -1).astype(np.float64)
        self._last_status = SensorStatus.LIVE
        return NeuralFrame(
            status=SensorStatus.LIVE,
            phases=phases,
            channel_labels=(self.channel_label,),
            sample_rate_hz=float(self.target_sr_hz),
        )


def _absent_frame(labels: tuple[str, ...], *, degraded: bool = False) -> NeuralFrame:
    """Build an honest-null NeuralFrame. ``phases`` is length-0; no data."""
    return NeuralFrame(
        status=SensorStatus.DEGRADED if degraded else SensorStatus.ABSENT,
        phases=np.zeros(0, dtype=np.float64),
        channel_labels=labels,
        sample_rate_hz=0.0,
    )


# Exported envelope constants for downstream inspection / tests.
__all__ = [
    "DEFAULT_MIN_RR_SAMPLES",
    "DEFAULT_MIN_WINDOW_S",
    "DEFAULT_TARGET_SR_HZ",
    "HRVPhaseExtractor",
    "RRSource",
    "ibi_to_phase_series",
]
