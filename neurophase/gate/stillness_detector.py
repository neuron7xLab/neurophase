"""StillnessDetector — invariant I₄ of the neurophase gate.

The fourth invariant of the execution gate complements the three phase-
synchronization invariants (``I₁``–``I₃``) with an **actionability
classifier**: even when the joint order parameter ``R(t)`` is above the
synchronization threshold, acting on the system is *unnecessary* when
its dynamics are quiet. This is the phase-space analogue of free-energy
stationarity in Friston's predictive-processing framework.

Mathematical statement
----------------------

Let ``R(t) ∈ [0, 1]`` be the joint Kuramoto order parameter and let
``δ(t) ∈ [0, π]`` be the circular distance between the brain and market
mean phases (``PredictionErrorMonitor``). Define a **free-energy proxy**

.. math::

    F_{\\text{proxy}}(t) \\;=\\; \\tfrac{1}{2}\\,\\delta(t)^{2}

and its time derivative (chain rule)

.. math::

    \\frac{dF_{\\text{proxy}}}{dt}
    \\;=\\; \\delta(t)\\,\\frac{d\\delta}{dt}.

**Naming caveat (load-bearing).** ``F_proxy`` is not Friston's
variational free energy — it is a geometric surrogate that vanishes
exactly when ``δ`` vanishes and whose derivative vanishes exactly when
``δ`` is stationary. It is introduced here precisely so that the
library never claims to compute the full free-energy functional without
a generative model. See ``docs/theory/stillness_invariant.md`` for the
derivation and for the exact failure modes (multi-modal δ trajectories,
non-stationary phase drift).

The **stillness criterion** is a three-clause conjunction evaluated
**over the entire rolling window** ``τ_s = window · dt`` (not the last
sample, not an exponential moving average):

.. math::

    \\begin{aligned}
    \\max_{\\tau \\in [t - \\tau_s,\\, t]} \\left| \\dot R(\\tau)   \\right| &< \\varepsilon_R \\\\
    \\max_{\\tau \\in [t - \\tau_s,\\, t]} \\left| \\dot F_{\\text{proxy}}(\\tau) \\right| &< \\varepsilon_F \\\\
    \\max_{\\tau \\in [t - \\tau_s,\\, t]} \\delta(\\tau) &< \\delta_{\\min}
    \\end{aligned}

All three must hold simultaneously for the state to be ``STILL``.
Otherwise the state is ``ACTIVE`` with an explicit reason pointing at
the dominant failing clause.

Why window-wide and not last-sample
-----------------------------------

A single-sample criterion is trivially fooled: a system oscillating
through zero crossings produces instantaneous ``|dR/dt| = 0`` at every
extremum. The window-wide maximum rejects these states because *some*
sample in the window exceeds ``ε_R``. An EMA-smoothed criterion is
equally fragile because it *averages out* transient excursions that a
downstream decision must not act on. The window-wide max is the only
choice that never emits ``STILL`` for a system whose trajectory crossed
the ``ε`` band within the last ``τ_s`` seconds. See ``test_window_wide_beats_last_sample``
for the concrete counter-example.

Why warmup is ``ACTIVE`` and never ``SENSOR_ABSENT``
----------------------------------------------------

During the first ``window`` updates the buffer is not yet full, so the
criterion cannot be evaluated. The detector returns ``ACTIVE`` with a
``warmup`` reason. It does **not** fall back to ``SENSOR_ABSENT`` —
``SENSOR_ABSENT`` encodes a hardware fact (``I₃``) and must not be
polluted by detector-internal warmup state. Downstream consumers that
need explicit warmup information can inspect ``window_filled``.

Numerical notes
---------------

Derivatives are computed by first differences over the full buffer::

    R_diff     = diff(R_hist)       / dt
    δ_diff     = diff(delta_hist)   / dt
    F_proxy_dt = delta_hist[1:] * δ_diff      # chain rule

(There is no central-difference variant: we want a strictly causal
estimate so the detector commits to a state exactly when enough history
is present.) No SciPy, no hidden smoothing, no stochastic elements.
``numpy`` is used only for the vectorised diff and max.

Hysteresis (optional)
---------------------

A common failure mode of a discrete STILL/ACTIVE classifier is
**chatter**: when ``|dR/dt|`` sits right at ``ε_R`` the state flickers
every update. The optional ``hold_steps`` parameter adds a minimum
residence time: once a state is entered, it cannot leave for at least
``hold_steps`` updates. This is a convenience — the core invariant is
unaffected, because any sample that would have left the state early
still fails the criterion and is recorded faithfully in the reason
string (``"held: still-state residency lock"``).

Sources
-------

Friston (2010); Clark (2013, 2026); Fioriti & Chinnici (2012); R&D
report sections 4 and 8. See also ``docs/theory/stillness_invariant.md``
for the full derivation and the four counter-examples used to design
this criterion.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Sensible defaults — deliberately conservative.
# ---------------------------------------------------------------------------

#: Default rolling-window length (samples). ``8`` keeps the state-machine
#: responsive at 1 Hz update cadences while still rejecting single-sample
#: zero crossings.
DEFAULT_WINDOW: Final[int] = 8

#: Default threshold on ``max |dR/dt|`` inside the window.
DEFAULT_EPS_R: Final[float] = 1e-3

#: Default threshold on ``max |dF_proxy/dt|`` inside the window.
DEFAULT_EPS_F: Final[float] = 1e-3

#: Default threshold on ``max δ`` inside the window. ``0.05 rad ≈ 2.87°``.
DEFAULT_DELTA_MIN: Final[float] = 0.05

#: Default integrator step. ``1.0`` matches "samples per second" semantics.
DEFAULT_DT: Final[float] = 1.0

#: Minimum window size to make the finite-difference derivative defined.
_MIN_WINDOW: Final[int] = 2

# IEEE 754 float64 machine epsilon: eps ≈ 2.22e-16.
# π computed in float64 has rounding error ≤ 0.5 ULP.  Numerical
# operations (addition, subtraction of angles) may accumulate up to
# a few ULP of drift.  We allow 4 ULP above π — tight enough to
# reject obviously invalid deltas, loose enough to absorb
# floating-point arithmetic on the unit circle.
_DELTA_UPPER: Final[float] = float(np.pi * (1.0 + 4.0 * np.finfo(np.float64).eps))


class StillnessState(Enum):
    """Binary actionability classification.

    * ``STILL``  — dynamics are quiet over the whole rolling window;
                    no new information justifies action.
    * ``ACTIVE`` — any of the three clauses of the criterion failed, **or**
                    the detector is still warming up.
    """

    STILL = auto()
    ACTIVE = auto()


@dataclass(frozen=True, repr=False)
class StillnessDecision:
    """Immutable outcome of a single ``StillnessDetector.update`` call.

    Every field is a direct, reproducible measurement — the decision
    contains enough information to replay the classification without
    re-running the detector.

    Attributes
    ----------
    state
        The classification — ``STILL`` or ``ACTIVE``.
    R
        The ``R(t)`` value of the current update.
    delta
        The ``δ(t)`` (circular-distance) value of the current update.
    dR_dt_max
        ``max |dR/dt|`` over the rolling window, or ``None`` during warmup.
    dF_proxy_dt_max
        ``max |dF_proxy/dt|`` over the rolling window, or ``None`` during warmup.
    delta_max
        ``max δ`` over the rolling window, or ``None`` during warmup.
    window_filled
        ``True`` iff the rolling buffer has reached its configured size.
    reason
        Human-readable explanation. The first token ("warmup" / "still" /
        "active") is stable and can be parsed programmatically.
    """

    state: StillnessState
    R: float
    delta: float
    dR_dt_max: float | None
    dF_proxy_dt_max: float | None
    delta_max: float | None
    window_filled: bool
    reason: str

    def __repr__(self) -> str:  # aesthetic rich repr (HN22)
        parts = [
            self.state.name,
            f"R={self.R:.4f}",
            f"δ={self.delta:.4f}",
        ]
        if self.window_filled:
            parts.append("warm")
        else:
            parts.append("warmup")
        return "StillnessDecision[" + " · ".join(parts) + "]"


class StillnessDetector:
    """Rolling-window stillness classifier (invariant ``I₄``).

    Parameters
    ----------
    window
        Buffer length in samples. Must be ``≥ 2``.
    eps_R
        Tolerance on ``max |dR/dt|``. Must be strictly positive.
    eps_F
        Tolerance on ``max |dF_proxy/dt|``. Must be strictly positive.
    delta_min
        Tolerance on ``max δ``. Must be strictly positive. A reasonable
        default is ``0.05 rad`` (~2.87°).
    dt
        Integration step (seconds per update). Must be strictly positive.
    hold_steps
        Minimum residence time in each state, in updates. ``0`` disables
        hysteresis. Must be ``≥ 0``.
    """

    __slots__ = (
        "_R_hist",
        "_delta_hist",
        "_hold_remaining",
        "_last_state",
        "_n_updates",
        "delta_min",
        "dt",
        "eps_F",
        "eps_R",
        "hold_steps",
        "window",
    )

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        eps_R: float = DEFAULT_EPS_R,
        eps_F: float = DEFAULT_EPS_F,
        delta_min: float = DEFAULT_DELTA_MIN,
        dt: float = DEFAULT_DT,
        hold_steps: int = 0,
    ) -> None:
        if window < _MIN_WINDOW:
            raise ValueError(f"window must be ≥ {_MIN_WINDOW} to define a derivative, got {window}")
        if eps_R <= 0:
            raise ValueError(f"eps_R must be > 0, got {eps_R}")
        if eps_F <= 0:
            raise ValueError(f"eps_F must be > 0, got {eps_F}")
        if delta_min <= 0:
            raise ValueError(f"delta_min must be > 0, got {delta_min}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if hold_steps < 0:
            raise ValueError(f"hold_steps must be ≥ 0, got {hold_steps}")

        self.window: int = window
        self.eps_R: float = eps_R
        self.eps_F: float = eps_F
        self.delta_min: float = delta_min
        self.dt: float = dt
        self.hold_steps: int = hold_steps

        self._R_hist: deque[float] = deque(maxlen=window)
        self._delta_hist: deque[float] = deque(maxlen=window)
        self._n_updates: int = 0
        self._last_state: StillnessState | None = None
        self._hold_remaining: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, R: float, delta: float) -> StillnessDecision:
        """Ingest one ``(R, δ)`` sample and return the current classification.

        Parameters
        ----------
        R
            Joint order parameter ``R(t)``. Must be finite and in ``[0, 1]``.
        delta
            Circular distance ``δ(t)`` between the brain and market mean
            phases. Must be finite and in ``[0, π]``.

        Raises
        ------
        ValueError
            If ``R`` or ``delta`` is outside its physical range or non-finite.
        """
        if not np.isfinite(R):
            raise ValueError(f"R must be finite, got {R!r}")
        if not 0.0 <= R <= 1.0:
            raise ValueError(f"R must be in [0, 1], got {R}")
        if not np.isfinite(delta):
            raise ValueError(f"delta must be finite, got {delta!r}")
        if not 0.0 <= delta <= _DELTA_UPPER:
            raise ValueError(f"delta must be in [0, π], got {delta}")

        self._R_hist.append(float(R))
        self._delta_hist.append(float(delta))
        self._n_updates += 1

        if len(self._R_hist) < self.window:
            return self._decision(
                state=StillnessState.ACTIVE,
                R=R,
                delta=delta,
                dR_dt_max=None,
                dF_proxy_dt_max=None,
                delta_max=None,
                window_filled=False,
                reason="warmup: insufficient history — window not filled",
            )

        # Convert to arrays for vectorised differencing.
        R_arr: NDArray[np.float64] = np.asarray(self._R_hist, dtype=np.float64)
        delta_arr: NDArray[np.float64] = np.asarray(self._delta_hist, dtype=np.float64)

        # First-difference derivatives.
        dR_dt: NDArray[np.float64] = np.diff(R_arr) / self.dt
        ddelta_dt: NDArray[np.float64] = np.diff(delta_arr) / self.dt
        # Chain rule for the free-energy proxy.
        # F_proxy(t)     = 0.5 · δ(t)²
        # dF_proxy/dt(t) = δ(t) · dδ/dt(t)
        # We evaluate at the *right-hand* endpoint of each diff interval,
        # i.e. at delta_hist[1:], which keeps the estimate strictly causal.
        dF_proxy_dt: NDArray[np.float64] = delta_arr[1:] * ddelta_dt

        dR_dt_max = float(np.max(np.abs(dR_dt)))
        dF_proxy_dt_max = float(np.max(np.abs(dF_proxy_dt)))
        delta_max = float(np.max(delta_arr))

        # Clause-wise classification with an explicit reason trail.
        if dR_dt_max >= self.eps_R:
            raw_state = StillnessState.ACTIVE
            reason = (
                f"active: R dynamics exceed eps_R (max|dR/dt|={dR_dt_max:.6g} ≥ {self.eps_R:.6g})"
            )
        elif dF_proxy_dt_max >= self.eps_F:
            raw_state = StillnessState.ACTIVE
            reason = (
                f"active: free-energy proxy dynamics exceed eps_F "
                f"(max|dF_proxy/dt|={dF_proxy_dt_max:.6g} ≥ {self.eps_F:.6g})"
            )
        elif delta_max >= self.delta_min:
            raw_state = StillnessState.ACTIVE
            reason = (
                f"active: delta exceeds delta_min (max δ={delta_max:.6g} ≥ {self.delta_min:.6g})"
            )
        else:
            raw_state = StillnessState.STILL
            reason = (
                f"still: dynamics and prediction error quiet "
                f"(max|dR/dt|={dR_dt_max:.6g}, "
                f"max|dF_proxy/dt|={dF_proxy_dt_max:.6g}, "
                f"max δ={delta_max:.6g})"
            )

        final_state, reason = self._apply_hysteresis(raw_state, reason)
        return self._decision(
            state=final_state,
            R=R,
            delta=delta,
            dR_dt_max=dR_dt_max,
            dF_proxy_dt_max=dF_proxy_dt_max,
            delta_max=delta_max,
            window_filled=True,
            reason=reason,
        )

    def reset(self) -> None:
        """Discard the rolling buffers and hysteresis state.

        Use at session boundaries to start from a clean warmup phase.
        """
        self._R_hist.clear()
        self._delta_hist.clear()
        self._n_updates = 0
        self._last_state = None
        self._hold_remaining = 0

    # ------------------------------------------------------------------
    # Read-only diagnostics
    # ------------------------------------------------------------------

    @property
    def n_updates(self) -> int:
        """Total number of ``update`` calls since construction or reset."""
        return self._n_updates

    @property
    def window_filled(self) -> bool:
        """``True`` iff the rolling buffer has reached its configured size."""
        return len(self._R_hist) >= self.window

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_hysteresis(
        self, raw_state: StillnessState, reason: str
    ) -> tuple[StillnessState, str]:
        """Apply the optional minimum-residence-time lock."""
        if self.hold_steps == 0 or self._last_state is None:
            self._last_state = raw_state
            self._hold_remaining = self.hold_steps
            return raw_state, reason

        if raw_state is self._last_state:
            if self._hold_remaining > 0:
                self._hold_remaining -= 1
            return raw_state, reason

        # State change requested — veto if we are still inside the hold.
        if self._hold_remaining > 0:
            held_reason = (
                f"held: {self._last_state.name.lower()}-state residency lock "
                f"({self._hold_remaining} steps remaining); raw → {reason}"
            )
            self._hold_remaining -= 1
            return self._last_state, held_reason

        # Hold elapsed — commit to the new state and restart the timer.
        self._last_state = raw_state
        self._hold_remaining = self.hold_steps
        return raw_state, reason

    def _decision(
        self,
        *,
        state: StillnessState,
        R: float,
        delta: float,
        dR_dt_max: float | None,
        dF_proxy_dt_max: float | None,
        delta_max: float | None,
        window_filled: bool,
        reason: str,
    ) -> StillnessDecision:
        return StillnessDecision(
            state=state,
            R=float(R),
            delta=float(delta),
            dR_dt_max=dR_dt_max,
            dF_proxy_dt_max=dF_proxy_dt_max,
            delta_max=delta_max,
            window_filled=window_filled,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Pure helper — used by tests and by downstream consumers that only need
# the scalar F_proxy value (without running the full detector).
# ---------------------------------------------------------------------------


def free_energy_proxy(delta: float) -> float:
    """Geometric surrogate for Friston free energy: ``F_proxy = ½ · δ²``.

    See the module docstring and ``docs/theory/stillness_invariant.md``
    for why this is a *proxy* and never the full variational free-energy
    functional.
    """
    if not np.isfinite(delta):
        raise ValueError(f"delta must be finite, got {delta!r}")
    if delta < 0:
        raise ValueError(f"delta must be ≥ 0, got {delta}")
    return 0.5 * float(delta) ** 2
