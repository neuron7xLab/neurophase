"""Physio gate — 4-state signal-quality admission wrapper.

Maps an :class:`~neurophase.physio.features.HRVFeatures` snapshot to a
fail-closed :class:`PhysioDecision`.  The underlying admission kernel
is the existing :class:`~neurophase.gate.execution_gate.ExecutionGate`:
the HRV ``confidence`` score is fed as ``R(t)`` to that gate, so all of
``ExecutionGate``'s invariants (``I_1 .. I_4``) are enforced for free.

Public state vocabulary:

* ``EXECUTE_ALLOWED``   — confidence >= ``threshold_allow``, and the
  underlying :class:`ExecutionGate` returned ``READY``. This is the
  only state in which downstream action is permitted.
* ``EXECUTE_REDUCED``   — confidence >= ``threshold_abstain`` but
  < ``threshold_allow``. The signal is plausible but not at full
  confidence; callers should reduce position size / disable irreversible
  actions.
* ``ABSTAIN``           — confidence < ``threshold_abstain`` but the
  feature snapshot is otherwise well-formed (buffer full, RMSSD
  plausible). Treat as "not now".
* ``SENSOR_DEGRADED``   — insufficient buffer, RMSSD outside
  plausibility envelope, or the underlying kernel gate returned
  ``DEGRADED`` / ``SENSOR_ABSENT``. The physio path is unusable; the
  caller must fall back to another input or hold.

Invariant (enforced in :class:`PhysioDecision.__post_init__`):
``execution_allowed=True`` is legal only when ``state==EXECUTE_ALLOWED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from neurophase.gate.execution_gate import ExecutionGate, GateDecision, GateState
from neurophase.physio.features import HRVFeatures

# Default admission thresholds on the confidence score. Illustrative.
DEFAULT_THRESHOLD_ALLOW: float = 0.80
DEFAULT_THRESHOLD_ABSTAIN: float = 0.50


class PhysioGateState(Enum):
    """Closed enumeration of physio-gate output states."""

    EXECUTE_ALLOWED = auto()
    EXECUTE_REDUCED = auto()
    ABSTAIN = auto()
    SENSOR_DEGRADED = auto()


@dataclass(frozen=True, repr=False)
class PhysioDecision:
    """Immutable physio-gate decision with fail-closed invariant."""

    state: PhysioGateState
    execution_allowed: bool
    confidence: float
    threshold_allow: float
    threshold_abstain: float
    reason: str
    kernel_state: GateState  # underlying ExecutionGate state, for audit

    def __post_init__(self) -> None:
        if self.execution_allowed and self.state is not PhysioGateState.EXECUTE_ALLOWED:
            raise ValueError(
                "Invariant violated: execution_allowed=True requires state=EXECUTE_ALLOWED"
            )
        if self.state is PhysioGateState.EXECUTE_ALLOWED and not self.execution_allowed:
            raise ValueError(
                "Invariant violated: state=EXECUTE_ALLOWED requires execution_allowed=True"
            )

    def __repr__(self) -> str:
        flag = "[OK]" if self.execution_allowed else "[NO]"
        return (
            f"PhysioDecision[{self.state.name} "
            f"conf={self.confidence:.3f} "
            f"kernel={self.kernel_state.name} {flag}]"
        )


class PhysioGate:
    """Fail-closed admission gate over HRV-style signal quality.

    Parameters
    ----------
    threshold_allow
        Minimum confidence for ``EXECUTE_ALLOWED``. Must be in ``(0, 1)``
        and strictly greater than ``threshold_abstain``.
    threshold_abstain
        Minimum confidence for ``EXECUTE_REDUCED``. Below this we abstain.
    kernel_gate
        The underlying :class:`ExecutionGate` instance that enforces the
        kernel invariants. Injected so tests can observe or replace it.
    mode
        ``"default"`` (illustrative repo thresholds) or ``"calibrated"``
        (per-user profile thresholds). Honor-system flag: the constructor
        does not verify that ``mode="calibrated"`` came from a real
        :class:`PhysioProfile`. Use :meth:`from_profile` for the only
        construction path that GUARANTEES profile provenance; bypassing
        that classmethod is permitted but operationally meaningless --
        the gate will trust the caller.
    profile_user_id
        Required when ``mode="calibrated"``; rejected when
        ``mode="default"``. Recorded in :class:`PhysioDecision` for audit.
    """

    __slots__ = ("kernel_gate", "mode", "profile_user_id", "threshold_abstain", "threshold_allow")

    def __init__(
        self,
        *,
        threshold_allow: float = DEFAULT_THRESHOLD_ALLOW,
        threshold_abstain: float = DEFAULT_THRESHOLD_ABSTAIN,
        kernel_gate: ExecutionGate | None = None,
        mode: str = "default",
        profile_user_id: str | None = None,
    ) -> None:
        if not 0.0 < threshold_abstain < threshold_allow < 1.0:
            raise ValueError(
                f"need 0 < threshold_abstain ({threshold_abstain}) < "
                f"threshold_allow ({threshold_allow}) < 1"
            )
        if mode not in ("default", "calibrated"):
            raise ValueError(f"mode must be 'default' or 'calibrated'; got {mode!r}")
        if mode == "calibrated" and not profile_user_id:
            raise ValueError(
                "mode='calibrated' requires a non-empty profile_user_id; "
                "construct via PhysioGate.from_profile(...) to enforce this"
            )
        if mode == "default" and profile_user_id is not None:
            raise ValueError(
                f"mode='default' must not carry a profile_user_id; "
                f"got profile_user_id={profile_user_id!r}. Either set "
                f"mode='calibrated' or omit profile_user_id."
            )
        self.threshold_allow: float = threshold_allow
        self.threshold_abstain: float = threshold_abstain
        self.mode: str = mode
        self.profile_user_id: str | None = profile_user_id
        # The kernel gate runs with its own threshold equal to our
        # abstain threshold: anything below it is BLOCKED regardless.
        self.kernel_gate: ExecutionGate = kernel_gate or ExecutionGate(threshold=threshold_abstain)

    @classmethod
    def from_profile(cls, profile: Any) -> PhysioGate:
        """Construct a calibrated gate from a :class:`PhysioProfile`.

        Importing :mod:`neurophase.physio.profile` from inside the
        classmethod keeps :mod:`neurophase.physio.gate` free of a hard
        dependency on the calibration stack — callers that do not
        calibrate never pay the import cost.
        """
        from neurophase.physio.profile import PhysioProfile

        if not isinstance(profile, PhysioProfile):
            raise TypeError(
                f"PhysioGate.from_profile expected PhysioProfile, got {type(profile).__name__}"
            )
        return cls(
            threshold_allow=profile.threshold_allow,
            threshold_abstain=profile.threshold_abstain,
            mode="calibrated",
            profile_user_id=profile.user_id,
        )

    def evaluate(self, features: HRVFeatures) -> PhysioDecision:
        """Return a fail-closed decision for one feature snapshot."""
        # 1. Hard fail-closed gates on feature validity.
        if features.window_size < 1:
            return self._degraded(features, reason="empty feature window")
        if features.confidence == 0.0 and features.window_size < features.window_size:
            # defensive, unreachable; kept for clarity.
            return self._degraded(features, reason="zero confidence, empty buffer")
        if not features.rmssd_plausible and features.window_size > 0 and features.confidence > 0:
            # RMSSD plausibility failure inside an otherwise populated
            # window is itself a hard fail-closed signal.
            return self._degraded(
                features,
                reason=("RMSSD outside plausibility envelope — flatline or artifact-dominated"),
            )

        # 2. Insufficient buffer fill -> degraded (confidence is 0 by
        #    construction when window_size < MIN_WINDOW_SIZE in features.py,
        #    so kernel_gate would BLOCK; we short-circuit for a cleaner reason).
        if features.confidence <= 0.0:
            return self._degraded(
                features,
                reason=(f"insufficient buffer ({features.window_size} samples) — confidence == 0"),
            )

        # 3. Delegate admission to the kernel gate. confidence acts as R(t).
        kernel: GateDecision = self.kernel_gate.evaluate(
            R=features.confidence, sensor_present=True, delta=None
        )

        # 4. Map the kernel state to the physio vocabulary.
        if kernel.state is GateState.DEGRADED:
            return self._degraded(features, reason=f"kernel DEGRADED: {kernel.reason}")
        if kernel.state is GateState.SENSOR_ABSENT:
            return self._degraded(features, reason="kernel SENSOR_ABSENT")
        if kernel.state is GateState.BLOCKED:
            return PhysioDecision(
                state=PhysioGateState.ABSTAIN,
                execution_allowed=False,
                confidence=features.confidence,
                threshold_allow=self.threshold_allow,
                threshold_abstain=self.threshold_abstain,
                reason=f"confidence {features.confidence:.3f} < abstain "
                f"threshold {self.threshold_abstain:.2f}",
                kernel_state=kernel.state,
            )
        # kernel.state is READY or UNNECESSARY.  Stratify on threshold_allow.
        if kernel.state is GateState.UNNECESSARY:
            # Stillness on a physio signal — treat as ABSTAIN (no change
            # in state => no new information => do not act).
            return PhysioDecision(
                state=PhysioGateState.ABSTAIN,
                execution_allowed=False,
                confidence=features.confidence,
                threshold_allow=self.threshold_allow,
                threshold_abstain=self.threshold_abstain,
                reason="kernel UNNECESSARY: signal stillness, no new information",
                kernel_state=kernel.state,
            )

        if features.confidence >= self.threshold_allow:
            return PhysioDecision(
                state=PhysioGateState.EXECUTE_ALLOWED,
                execution_allowed=True,
                confidence=features.confidence,
                threshold_allow=self.threshold_allow,
                threshold_abstain=self.threshold_abstain,
                reason=(
                    f"confidence {features.confidence:.3f} "
                    f">= allow threshold {self.threshold_allow:.2f}"
                ),
                kernel_state=kernel.state,
            )

        return PhysioDecision(
            state=PhysioGateState.EXECUTE_REDUCED,
            execution_allowed=False,
            confidence=features.confidence,
            threshold_allow=self.threshold_allow,
            threshold_abstain=self.threshold_abstain,
            reason=(
                f"confidence {features.confidence:.3f} in "
                f"[{self.threshold_abstain:.2f}, {self.threshold_allow:.2f})"
            ),
            kernel_state=kernel.state,
        )

    def _degraded(self, features: HRVFeatures, *, reason: str) -> PhysioDecision:
        """Shortcut for producing a SENSOR_DEGRADED decision (fail-closed)."""
        return PhysioDecision(
            state=PhysioGateState.SENSOR_DEGRADED,
            execution_allowed=False,
            confidence=features.confidence,
            threshold_allow=self.threshold_allow,
            threshold_abstain=self.threshold_abstain,
            reason=reason,
            kernel_state=GateState.DEGRADED,
        )
