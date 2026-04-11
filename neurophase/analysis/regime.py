"""G1 — deterministic 4-state market-regime taxonomy.

``neurophase`` computes ``R(t)`` as a scalar synchrony between the
brain and market oscillators, but the *market regime itself* has,
until now, been a structurally opaque component of the system. A
``READY`` gate on a TRENDING market is a very different object from a
``READY`` gate on a CHAOTIC market, and the downstream Program G
work (G2 regime-conditioned strategy, G5 transition model, I1
policy) cannot begin without a canonical regime label.

This module introduces a **deterministic threshold tree** that
labels every :class:`~neurophase.runtime.pipeline.DecisionFrame`
with exactly one of four regime states:

    ``TRENDING``      — high R, stable δ (locked, low-motion regime)
    ``COMPRESSING``   — rising R, narrowing δ (build-up of coupling)
    ``REVERTING``     — falling R (coupling relaxing back to baseline)
    ``CHAOTIC``       — low R OR unstable δ (no coherent regime)

Contract
--------

* **Input:** a single :class:`DecisionFrame` (the 4-dimensional
  vector ``(R, ΔR, δ, Δδ)`` is reconstructed internally from two
  successive frames — classifier is stateful and owns the delta
  computation).
* **Output:** a frozen :class:`RegimeState` carrying the label, the
  raw 4-D vector, a ``confidence_score ∈ [0, 1]``, and a one-line
  reason string.
* **No ML.** The classifier is a pure threshold tree with typed,
  reviewable parameters (:class:`RegimeThresholds`). Same frame
  sequence + same thresholds → bit-identical
  :class:`RegimeState` sequence. No randomness, no learned weights.
* **Total.** Every valid frame lands in exactly one state — the
  tree is exhaustive. ``R is None`` or ``δ is None`` raise
  :class:`ValueError`; regime classification on an invalid or
  missing ``R`` is meaningless and must not be silently coerced.

What this module does NOT do
----------------------------

* It does **not** decide trade direction, size, or any policy. G1
  is a pure labelling layer; policy is I1+.
* It does **not** predict transitions — that is G5's job. G1 only
  answers "what regime is the current frame in?".
* It does **not** interact with the gate. Regime is a property of
  the market synchrony signal; the gate's 5-state machine is a
  property of execution permission. Program G layers above Program
  B/E but is not wired into the strict priority order.

Priority tree
-------------

The label is selected by the first matching rule, in order:

1. **COMPRESSING** — ``ΔR > rising_R_min`` AND ``Δδ < -narrowing_delta_min``
   (R is actively rising AND δ is actively narrowing; this is the
   most specific shape).
2. **REVERTING** — ``ΔR < -falling_R_min``
   (R is actively falling; dominates CHAOTIC so that a crash from
   a higher level is labelled as a reversion, not as "stuck in
   low coherence").
3. **CHAOTIC** — ``R < low_R_threshold`` OR ``|Δδ| > chaotic_delta_motion``
   (either low coherence or unstable δ motion).
4. **TRENDING** — default (moderate-to-high R with low-to-moderate
   motion).

This priority ordering makes every one of the 12 non-self
transitions reachable by a crafted DecisionFrame sequence — the
reachability sweep is verified in
``tests/test_regime_taxonomy.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from neurophase.runtime.pipeline import DecisionFrame

__all__ = [
    "DEFAULT_REGIME_THRESHOLDS",
    "RegimeClassifier",
    "RegimeLabel",
    "RegimeState",
    "RegimeThresholds",
]


class RegimeLabel(Enum):
    """The four regime states.

    The order is the order the public documentation lists them
    (trending → compressing → reverting → chaotic) and has no
    semantic meaning beyond readability — the classifier's
    priority tree is independent of the enum's declaration order.
    """

    TRENDING = "trending"
    COMPRESSING = "compressing"
    REVERTING = "reverting"
    CHAOTIC = "chaotic"


@dataclass(frozen=True)
class RegimeThresholds:
    """Typed, reviewable threshold tree parameters.

    Defaults are chosen to give a clean four-way separation on the
    H1 synthetic phase-coupling battery at c ∈ {0.0, 0.5, 1.0}.
    All are dimensionless except where noted.

    Attributes
    ----------
    low_R_threshold
        R below this value triggers the CHAOTIC rule (absent a
        stronger COMPRESSING or REVERTING shape).
    high_R_threshold
        R at or above this value is "high coherence" and anchors
        the TRENDING confidence curve.
    rising_R_min
        Minimum positive ΔR to qualify as "R rising" for the
        COMPRESSING rule.
    falling_R_min
        Minimum positive ``|ΔR|`` (on a negative ΔR) to qualify as
        "R falling" for the REVERTING rule.
    narrowing_delta_min
        Minimum positive ``|Δδ|`` (on a negative Δδ) to qualify as
        "δ narrowing" for the COMPRESSING rule.
    chaotic_delta_motion
        ``|Δδ|`` above this value triggers CHAOTIC independent of
        the R level (unstable δ).
    confidence_dR_scale
        Saturation scale for the ΔR contribution to COMPRESSING /
        REVERTING confidence.
    confidence_d_delta_scale
        Saturation scale for the Δδ contribution to COMPRESSING
        confidence.
    confidence_motion_scale
        Saturation scale for ``|ΔR|+|Δδ|`` in the TRENDING
        stability factor.
    """

    low_R_threshold: float = 0.40
    high_R_threshold: float = 0.70
    rising_R_min: float = 0.010
    falling_R_min: float = 0.010
    narrowing_delta_min: float = 0.005
    chaotic_delta_motion: float = 0.300
    confidence_dR_scale: float = 0.050
    confidence_d_delta_scale: float = 0.030
    confidence_motion_scale: float = 0.050

    def __post_init__(self) -> None:
        if not 0.0 < self.low_R_threshold < self.high_R_threshold < 1.0:
            raise ValueError(
                "require 0 < low_R_threshold < high_R_threshold < 1, got "
                f"low={self.low_R_threshold}, high={self.high_R_threshold}"
            )
        for name, value in (
            ("rising_R_min", self.rising_R_min),
            ("falling_R_min", self.falling_R_min),
            ("narrowing_delta_min", self.narrowing_delta_min),
            ("chaotic_delta_motion", self.chaotic_delta_motion),
            ("confidence_dR_scale", self.confidence_dR_scale),
            ("confidence_d_delta_scale", self.confidence_d_delta_scale),
            ("confidence_motion_scale", self.confidence_motion_scale),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")


#: Default, reviewed threshold set. Do not mutate — instances are
#: frozen.
DEFAULT_REGIME_THRESHOLDS: Final[RegimeThresholds] = RegimeThresholds()


@dataclass(frozen=True, repr=False)
class RegimeState:
    """Immutable outcome of one :meth:`RegimeClassifier.classify` call.

    The four raw fields ``R``, ``dR``, ``delta``, ``d_delta``
    reconstruct the 4-D vector the tree was evaluated on. ``reason``
    is a short one-line narrative; downstream tools should key on
    the typed ``label`` field, not on the string.

    Attributes
    ----------
    label
        The assigned :class:`RegimeLabel`.
    R
        ``frame.R`` at the current tick.
    dR
        ``R(t) - R(t-1)`` (zero on the first tick).
    delta
        ``frame.delta`` at the current tick.
    d_delta
        ``δ(t) - δ(t-1)`` (zero on the first tick).
    confidence_score
        In ``[0, 1]``. Zero means the frame is on the very boundary
        of its bucket; one means it is deep inside. Derived from
        typed saturation scales in :class:`RegimeThresholds` — see
        the class docstring for the exact formula per label.
    tick_index
        Copied from the frame for log correlation.
    timestamp
        Copied from the frame.
    warm
        ``False`` on the very first tick (no ΔR / Δδ available);
        ``True`` afterwards. Downstream consumers may choose to
        discount the first tick or treat it as a valid anchor.
    reason
        One-line human-readable explainer. Not parsed by any
        downstream consumer — typed fields above are the structured
        interface.
    """

    label: RegimeLabel
    R: float
    dR: float
    delta: float
    d_delta: float
    confidence_score: float
    tick_index: int
    timestamp: float
    warm: bool
    reason: str

    def __post_init__(self) -> None:
        # confidence is a contract field — fail loud on violations.
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"confidence_score must be in [0, 1], got {self.confidence_score}")

    def __repr__(self) -> str:  # aesthetic rich repr (HN23)
        warm_flag = "warm" if self.warm else "cold"
        return (
            f"RegimeState[{self.label.name} · "
            f"R={self.R:.4f} · dR={self.dR:+.4f} · "
            f"δ={self.delta:.4f} · "
            f"conf={self.confidence_score:.2f} · {warm_flag}]"
        )


# ---------------------------------------------------------------------------
# RegimeClassifier — stateful, deterministic, threshold-only.
# ---------------------------------------------------------------------------


class RegimeClassifier:
    """Stateful four-state market-regime labeller.

    Parameters
    ----------
    thresholds
        Optional typed threshold set. Defaults to
        :data:`DEFAULT_REGIME_THRESHOLDS`.

    Notes
    -----
    The classifier owns the ``R``/``δ`` lag buffer required to
    compute ``ΔR`` and ``Δδ``. On the first frame, both deltas are
    ``0.0`` and the result is marked ``warm=False``.

    The classifier is stateful but deterministic: for any two
    classifiers constructed with the same thresholds and fed the
    same frame sequence, the emitted :class:`RegimeState` sequence
    is bit-identical.
    """

    __slots__ = (
        "_last_R",
        "_last_delta",
        "_n_ticks",
        "thresholds",
    )

    def __init__(self, thresholds: RegimeThresholds | None = None) -> None:
        self.thresholds: RegimeThresholds = (
            thresholds if thresholds is not None else DEFAULT_REGIME_THRESHOLDS
        )
        self._last_R: float | None = None
        self._last_delta: float | None = None
        self._n_ticks: int = 0

    def reset(self) -> None:
        """Clear the internal lag buffer. Use at session boundaries."""
        self._last_R = None
        self._last_delta = None
        self._n_ticks = 0

    @property
    def n_ticks(self) -> int:
        """Number of frames classified since construction / last reset."""
        return self._n_ticks

    def classify(self, frame: DecisionFrame) -> RegimeState:
        """Label one :class:`DecisionFrame` with a :class:`RegimeState`.

        Parameters
        ----------
        frame
            A pipeline-emitted decision frame. Must carry a finite
            ``R`` and ``delta``; otherwise :class:`ValueError` is
            raised — regime classification on missing inputs is
            meaningless and must not be silently coerced.

        Returns
        -------
        RegimeState
            Frozen label + 4-D vector + confidence.
        """
        R = frame.R
        delta = frame.delta
        if R is None:
            raise ValueError(
                "RegimeClassifier requires frame.R; got None. "
                "Filter DEGRADED frames upstream before classifying."
            )
        if delta is None:
            raise ValueError(
                "RegimeClassifier requires frame.delta; got None. "
                "Ensure the pipeline was driven with a delta argument."
            )

        if self._last_R is None or self._last_delta is None:
            dR = 0.0
            d_delta = 0.0
            warm = False
        else:
            dR = R - self._last_R
            d_delta = delta - self._last_delta
            warm = True

        label, reason = _classify_tree(R=R, dR=dR, d_delta=d_delta, cfg=self.thresholds)
        confidence = _confidence(label=label, R=R, dR=dR, d_delta=d_delta, cfg=self.thresholds)

        state = RegimeState(
            label=label,
            R=R,
            dR=dR,
            delta=delta,
            d_delta=d_delta,
            confidence_score=confidence,
            tick_index=frame.tick_index,
            timestamp=frame.timestamp,
            warm=warm,
            reason=reason,
        )

        self._last_R = R
        self._last_delta = delta
        self._n_ticks += 1
        return state


# ---------------------------------------------------------------------------
# Internals — pure functions over the typed threshold tree.
# ---------------------------------------------------------------------------


def _classify_tree(
    *,
    R: float,
    dR: float,
    d_delta: float,
    cfg: RegimeThresholds,
) -> tuple[RegimeLabel, str]:
    """Run the four-rule priority tree and return ``(label, reason)``.

    Rules, in priority order:

    1. COMPRESSING — rising R AND narrowing δ.
    2. REVERTING   — falling R.
    3. CHAOTIC     — low R OR unstable δ motion.
    4. TRENDING    — default.
    """
    if dR > cfg.rising_R_min and d_delta < -cfg.narrowing_delta_min:
        return (
            RegimeLabel.COMPRESSING,
            f"compressing: ΔR=+{dR:.4f} > {cfg.rising_R_min:.4f} and "
            f"Δδ={d_delta:+.4f} < -{cfg.narrowing_delta_min:.4f}",
        )
    if dR < -cfg.falling_R_min:
        return (
            RegimeLabel.REVERTING,
            f"reverting: ΔR={dR:+.4f} < -{cfg.falling_R_min:.4f}",
        )
    if cfg.low_R_threshold > R:
        return (
            RegimeLabel.CHAOTIC,
            f"chaotic: R={R:.4f} < {cfg.low_R_threshold:.4f} (low coherence)",
        )
    if abs(d_delta) > cfg.chaotic_delta_motion:
        return (
            RegimeLabel.CHAOTIC,
            f"chaotic: |Δδ|={abs(d_delta):.4f} > {cfg.chaotic_delta_motion:.4f} (unstable δ)",
        )
    return (
        RegimeLabel.TRENDING,
        f"trending: R={R:.4f} (stable), ΔR={dR:+.4f}, Δδ={d_delta:+.4f}",
    )


def _confidence(
    *,
    label: RegimeLabel,
    R: float,
    dR: float,
    d_delta: float,
    cfg: RegimeThresholds,
) -> float:
    """Return ``confidence_score ∈ [0, 1]`` for ``label`` on the frame.

    Each label has its own margin-from-boundary formula; every
    formula clips to ``[0, 1]`` and uses only the typed saturation
    scales on :class:`RegimeThresholds`. The formulas are
    deliberately simple and reviewable — no ML, no hidden
    hyperparameters.
    """
    if label is RegimeLabel.COMPRESSING:
        rising = _saturate((dR - cfg.rising_R_min) / cfg.confidence_dR_scale)
        narrowing = _saturate((-d_delta - cfg.narrowing_delta_min) / cfg.confidence_d_delta_scale)
        return rising * narrowing

    if label is RegimeLabel.REVERTING:
        falling = _saturate((-dR - cfg.falling_R_min) / cfg.confidence_dR_scale)
        return falling

    if label is RegimeLabel.CHAOTIC:
        r_chaos = _saturate((cfg.low_R_threshold - R) / cfg.low_R_threshold)
        delta_chaos = _saturate(
            (abs(d_delta) - cfg.chaotic_delta_motion) / cfg.chaotic_delta_motion
        )
        return max(r_chaos, delta_chaos)

    # TRENDING — high R stable motion.
    r_part = _saturate((R - cfg.low_R_threshold) / (cfg.high_R_threshold - cfg.low_R_threshold))
    stability = _saturate(1.0 - (abs(dR) + abs(d_delta)) / cfg.confidence_motion_scale)
    return r_part * stability


def _saturate(value: float) -> float:
    """Clip ``value`` into ``[0, 1]``."""
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value
