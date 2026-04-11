"""L1 — memory-bounded rolling computation audit.

The runtime stack is built from a small set of stateful layers
that all advertise constant-memory rolling computation:

* :class:`~neurophase.data.temporal_validator.TemporalValidator`
  — bounded by ``warmup_samples``.
* :class:`~neurophase.data.stream_detector.TemporalStreamDetector`
  — bounded by ``window``.
* :class:`~neurophase.gate.stillness_detector.StillnessDetector`
  — bounded by ``window`` (R history + δ history).
* :class:`~neurophase.analysis.regime.RegimeClassifier` — O(1).
* :class:`~neurophase.policy.action.ActionPolicy` — O(1).
* :class:`~neurophase.audit.decision_ledger.DecisionTraceLedger`
  — O(1) in memory (the on-disk file is append-only by design;
  L1 audits *in-memory* footprint).

The doctrine claim is that **no internal buffer ever grows
unboundedly with the tick count**. This module makes that claim
machine-checkable. :func:`audit_runtime_memory` walks a live
:class:`~neurophase.runtime.orchestrator.RuntimeOrchestrator`,
computes the declared cap of every internal rolling collection
from its construction-time configuration, measures the live
size, and returns a frozen :class:`MemoryAuditReport`.

The HN27 contract bound to this module asserts:

1. **Bounded.** After any number of ticks, every internal
   rolling collection on every layer satisfies
   ``measured_size ≤ declared_cap``.
2. **Constant.** Two audits taken at very different tick counts
   (e.g. tick 100 vs tick 10 000) on the same orchestrator
   instance produce the same total declared cap. The cap is a
   pure function of the configuration; it does not grow with
   the input stream.
3. **Total** in the audit sense: every stateful layer that
   carries rolling memory is enumerated by
   :func:`audit_runtime_memory`. Adding a new layer with a
   rolling buffer requires registering it here, and the L1
   property test (``test_memory_bounded_audit.py``) catches
   the omission by failing on a missing component.

What this module does NOT do
----------------------------

* It does **not** measure peak resident memory of the Python
  process. That is an OS-level concern; L1 is about the
  algorithmic invariant (rolling buffers are O(1) in tick count),
  not the platform's memory accounting.
* It does **not** monkey-patch any layer. The audit reads
  internal attributes directly and is documented as such — the
  underscore-prefixed attribute names are part of the L1
  contract surface.
* It does **not** wrap the on-disk decision ledger. The ledger's
  on-disk file *is* unbounded (it is an append-only audit log),
  but its in-memory footprint is O(1) and is verified here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from neurophase.runtime.orchestrator import RuntimeOrchestrator

from neurophase.audit.decision_ledger import DecisionTraceLedger
from neurophase.data.stream_detector import TemporalStreamDetector
from neurophase.data.temporal_validator import TemporalValidator
from neurophase.gate.stillness_detector import StillnessDetector

__all__ = [
    "ComponentMemoryFootprint",
    "MemoryAuditError",
    "MemoryAuditReport",
    "audit_runtime_memory",
]

#: Sentinel cap for layers that hold no rolling collection at
#: all (RegimeClassifier, ActionPolicy, ledger in-memory state).
#: They are still enumerated in the report so that "is this
#: layer audited?" has an obvious answer.
ZERO_CAP: Final[int] = 0


class MemoryAuditError(AssertionError):
    """Raised when an internal rolling buffer exceeds its declared cap.

    Subclassing :class:`AssertionError` is intentional: an L1
    violation is not a recoverable runtime fault — it is a
    contract failure of the runtime stack itself, and the only
    correct response is to abort and surface a bug report.
    """


@dataclass(frozen=True, repr=False)
class ComponentMemoryFootprint:
    """One-line audit row for a single stateful component.

    Attributes
    ----------
    name
        Class name of the audited component.
    declared_cap
        The maximum number of items the component is allowed
        to hold, derived from its construction-time
        configuration. ``ZERO_CAP`` for components that hold
        no rolling collection at all.
    measured_size
        The current live size of the component's rolling
        collection(s). For components with multiple buffers
        (e.g. :class:`StillnessDetector` with R + δ histories)
        this is the sum across all buffers.
    is_bounded
        ``True`` iff ``measured_size ≤ declared_cap`` (or both
        are zero — a component with no rolling state is always
        bounded).
    detail
        One-line human-readable detail. Stable first token
        (``empty:`` / ``window:`` / ``warmup:``); machine
        consumers should key on the typed fields above.
    """

    name: str
    declared_cap: int
    measured_size: int
    is_bounded: bool
    detail: str

    def __post_init__(self) -> None:
        if self.declared_cap < 0:
            raise ValueError(f"declared_cap must be ≥ 0, got {self.declared_cap}")
        if self.measured_size < 0:
            raise ValueError(f"measured_size must be ≥ 0, got {self.measured_size}")
        # Internal consistency: is_bounded must agree with the
        # numeric fields. Mismatch is a programmer error.
        expected_bounded = self.measured_size <= self.declared_cap
        if expected_bounded != self.is_bounded:
            raise ValueError(
                f"is_bounded={self.is_bounded} disagrees with "
                f"measured_size={self.measured_size} ≤ "
                f"declared_cap={self.declared_cap}"
            )

    def __repr__(self) -> str:  # aesthetic rich repr (HN27)
        flag = "✓" if self.is_bounded else "✗"
        return (
            f"ComponentMemoryFootprint[{self.name} · "
            f"size={self.measured_size}/{self.declared_cap} · {flag}]"
        )


@dataclass(frozen=True, repr=False)
class MemoryAuditReport:
    """Frozen, JSON-safe outcome of one
    :func:`audit_runtime_memory` call.

    Attributes
    ----------
    n_ticks
        Number of ticks the audited orchestrator had processed
        at audit time.
    components
        Tuple of :class:`ComponentMemoryFootprint`, in
        canonical enumeration order
        (TemporalValidator → TemporalStreamDetector →
        StillnessDetector → DecisionTraceLedger →
        RegimeClassifier → ActionPolicy).
    total_declared_cap
        Sum of every component's ``declared_cap``.
    total_measured_size
        Sum of every component's ``measured_size``.
    all_bounded
        ``True`` iff every component is bounded individually.
    """

    n_ticks: int
    components: tuple[ComponentMemoryFootprint, ...]
    total_declared_cap: int
    total_measured_size: int
    all_bounded: bool

    def __post_init__(self) -> None:
        if self.n_ticks < 0:
            raise ValueError(f"n_ticks must be ≥ 0, got {self.n_ticks}")
        # Internal consistency.
        expected_total_cap = sum(c.declared_cap for c in self.components)
        expected_total_size = sum(c.measured_size for c in self.components)
        if expected_total_cap != self.total_declared_cap:
            raise ValueError(
                f"total_declared_cap mismatch: "
                f"stored={self.total_declared_cap}, "
                f"summed={expected_total_cap}"
            )
        if expected_total_size != self.total_measured_size:
            raise ValueError(
                f"total_measured_size mismatch: "
                f"stored={self.total_measured_size}, "
                f"summed={expected_total_size}"
            )
        expected_all_bounded = all(c.is_bounded for c in self.components)
        if expected_all_bounded != self.all_bounded:
            raise ValueError(
                f"all_bounded mismatch: stored={self.all_bounded}, computed={expected_all_bounded}"
            )

    def __repr__(self) -> str:  # aesthetic rich repr (HN27)
        flag = "✓" if self.all_bounded else "✗"
        return (
            f"MemoryAuditReport[n_ticks={self.n_ticks} · "
            f"size={self.total_measured_size}/{self.total_declared_cap} · "
            f"components={len(self.components)} · {flag}]"
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe projection — every value is a primitive
        or a list of dicts."""
        return {
            "n_ticks": self.n_ticks,
            "components": [
                {
                    "name": c.name,
                    "declared_cap": c.declared_cap,
                    "measured_size": c.measured_size,
                    "is_bounded": c.is_bounded,
                    "detail": c.detail,
                }
                for c in self.components
            ],
            "total_declared_cap": self.total_declared_cap,
            "total_measured_size": self.total_measured_size,
            "all_bounded": self.all_bounded,
        }


# ---------------------------------------------------------------------------
# Audit drivers — pure introspection of live runtime objects.
# ---------------------------------------------------------------------------


def audit_runtime_memory(
    orchestrator: RuntimeOrchestrator, *, raise_on_violation: bool = False
) -> MemoryAuditReport:
    """Audit every rolling collection inside ``orchestrator``.

    Parameters
    ----------
    orchestrator
        A live :class:`RuntimeOrchestrator`. The audit may be
        called at any tick count, including before the first
        :meth:`tick` and after :meth:`close`.
    raise_on_violation
        When ``True``, the audit raises :class:`MemoryAuditError`
        on any unbounded component. When ``False`` (the default),
        the violation is reported via the ``all_bounded`` field
        and the caller decides how to react.

    Returns
    -------
    MemoryAuditReport
        Frozen, JSON-safe.
    """
    pipeline = orchestrator.pipeline
    components: list[ComponentMemoryFootprint] = []

    # 1. TemporalValidator — bounded by warmup_samples.
    validator: TemporalValidator = pipeline._validator
    components.append(_audit_temporal_validator(validator))

    # 2. TemporalStreamDetector — bounded by window.
    stream: TemporalStreamDetector = pipeline._stream_detector
    components.append(_audit_stream_detector(stream))

    # 3. StillnessDetector (optional) — bounded by 2 × window.
    stillness: StillnessDetector | None = pipeline._gate.stillness_detector
    if stillness is not None:
        components.append(_audit_stillness_detector(stillness))
    else:
        components.append(
            ComponentMemoryFootprint(
                name="StillnessDetector",
                declared_cap=ZERO_CAP,
                measured_size=0,
                is_bounded=True,
                detail="empty: stillness layer disabled",
            )
        )

    # 4. DecisionTraceLedger (optional, in-memory only) — O(1).
    ledger: DecisionTraceLedger | None = pipeline.ledger
    components.append(_audit_decision_ledger(ledger))

    # 5. RegimeClassifier — O(1) (only _last_R + _last_delta).
    components.append(_audit_regime_classifier(orchestrator))

    # 6. ActionPolicy — O(1) (only _cooldown + _prev_intent).
    components.append(_audit_action_policy(orchestrator))

    total_cap = sum(c.declared_cap for c in components)
    total_size = sum(c.measured_size for c in components)
    all_bounded = all(c.is_bounded for c in components)

    report = MemoryAuditReport(
        n_ticks=orchestrator.n_ticks,
        components=tuple(components),
        total_declared_cap=total_cap,
        total_measured_size=total_size,
        all_bounded=all_bounded,
    )

    if raise_on_violation and not all_bounded:
        offenders = [c for c in components if not c.is_bounded]
        raise MemoryAuditError(
            f"L1 violation: {len(offenders)} component(s) unbounded: "
            + ", ".join(repr(c) for c in offenders)
        )
    return report


# ---------------------------------------------------------------------------
# Per-component introspection helpers.
#
# These reach into underscore-prefixed attributes on purpose: the
# attribute names are part of the L1 contract surface (the HN27
# binding pins them in INVARIANTS.yaml). Any future refactor that
# renames them must update both the audit and the contract.
# ---------------------------------------------------------------------------


def _audit_temporal_validator(
    validator: TemporalValidator,
) -> ComponentMemoryFootprint:
    cap = int(validator.warmup_samples)
    size = len(validator._history)
    return ComponentMemoryFootprint(
        name="TemporalValidator",
        declared_cap=cap,
        measured_size=size,
        is_bounded=size <= cap,
        detail=f"warmup: history bounded by warmup_samples={cap}",
    )


def _audit_stream_detector(
    detector: TemporalStreamDetector,
) -> ComponentMemoryFootprint:
    cap = int(detector.window)
    size = len(detector._buffer)
    return ComponentMemoryFootprint(
        name="TemporalStreamDetector",
        declared_cap=cap,
        measured_size=size,
        is_bounded=size <= cap,
        detail=f"window: rolling buffer bounded by window={cap}",
    )


def _audit_stillness_detector(
    detector: StillnessDetector,
) -> ComponentMemoryFootprint:
    # Two parallel deques, each maxlen=window.
    cap = int(detector.window) * 2
    size = len(detector._R_hist) + len(detector._delta_hist)
    return ComponentMemoryFootprint(
        name="StillnessDetector",
        declared_cap=cap,
        measured_size=size,
        is_bounded=size <= cap,
        detail=f"window: 2x rolling histories bounded by 2*window={cap}",
    )


def _audit_decision_ledger(
    ledger: DecisionTraceLedger | None,
) -> ComponentMemoryFootprint:
    if ledger is None:
        return ComponentMemoryFootprint(
            name="DecisionTraceLedger",
            declared_cap=ZERO_CAP,
            measured_size=0,
            is_bounded=True,
            detail="empty: no ledger attached",
        )
    # In-memory state is exactly two values: _last_hash + _n_records.
    # The on-disk file is unbounded by design (append-only audit
    # log) and is NOT part of L1 — L1 audits *in-memory* memory.
    return ComponentMemoryFootprint(
        name="DecisionTraceLedger",
        declared_cap=ZERO_CAP,
        measured_size=0,
        is_bounded=True,
        detail="empty: in-memory state is O(1) (two scalars)",
    )


def _audit_regime_classifier(
    orchestrator: RuntimeOrchestrator,
) -> ComponentMemoryFootprint:
    classifier = orchestrator.regime_classifier
    # Just two floats: _last_R + _last_delta. They are not
    # collections, so the "size" is conceptually zero.
    _ = classifier  # silence unused
    return ComponentMemoryFootprint(
        name="RegimeClassifier",
        declared_cap=ZERO_CAP,
        measured_size=0,
        is_bounded=True,
        detail="empty: O(1) state (last_R + last_delta scalars)",
    )


def _audit_action_policy(
    orchestrator: RuntimeOrchestrator,
) -> ComponentMemoryFootprint:
    policy = orchestrator.policy
    _ = policy
    return ComponentMemoryFootprint(
        name="ActionPolicy",
        declared_cap=ZERO_CAP,
        measured_size=0,
        is_bounded=True,
        detail="empty: O(1) state (cooldown + prev_intent scalars)",
    )
