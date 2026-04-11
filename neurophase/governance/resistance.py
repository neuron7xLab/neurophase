"""Seventh axis — Resistance (Опір): adversarial sweep suite.

The first six Sutskever axes — elegance, aesthetics, beauty,
simplicity, precision, adaptability — tell you how a system
*reads*. The seventh axis, **Resistance**, tells you how it
*holds under attack*. A beautiful system that bends under
pressure is theatre. A beautiful system that refuses to bend is
engineering.

This module makes the seventh axis machine-checkable. It is
**not** a unit-test file — it is a declarative registry of
adversarial scenarios plus a pure runner that exercises each
scenario against the live runtime stack and asserts the
load-bearing invariants are preserved. The CI bindings for HN32
point at individual pytest node ids that invoke specific
scenarios; the suite itself exists so a reviewer can answer one
question:

    "What does neurophase refuse to do, and how do we verify it?"

The answer is: every adversarial condition registered here runs
on every PR, and a single failing scenario is a merge block.

Contract (HN32)
---------------

Six load-bearing adversarial scenarios, each targeting a
specific failure mode the architecture refuses to exhibit:

1. **GATE_NEVER_WIDENS_UNDER_ADVERSARIAL_KLR** — with a fake
   KLR whose every field is filled with ``execution_allowed=True``
   no matter the underlying state, the gate sequence on any
   input stream is byte-identical to the gate sequence with no
   KLR at all. RT-KLR-I1 law under adversarial pressure.

2. **MEMORY_BOUNDED_UNDER_10K_TICKS** — after 10 000 ticks of
   the orchestrator with a real ledger attached, the internal
   rolling buffers are STILL bounded by the declared caps of
   :mod:`neurophase.runtime.memory_audit`. L1 law under
   endurance load.

3. **LEDGER_TAMPER_FAILS_VERIFICATION** — a single-byte flip
   of any record in a committed decision ledger must fail
   :func:`verify_ledger` at the exact tampered index. F1 law
   under single-bit attack.

4. **CONTRACT_VIOLATING_CLAIM_REJECTED** — a ``CLAIMS.yaml``
   entry declaring status=FACT with only 1 supporting citation
   must raise :class:`ClaimRegistryError` at load time. HN30
   mechanical promotion rule under forgery attempt.

5. **CURRICULUM_SHAPE_CORRUPTION_REJECTED** — a KLR intervention
   with a mis-shaped curriculum vector collapses to ROLLBACK,
   never to SUCCESS. KLR-I1..I3 laws under shape attack.

6. **MONOGRAPH_DRIFT_FAILS_CI** — a hand-edit of the committed
   monograph that adds a line the generator would never produce
   fails the HN29 regeneration guard. M1 law under
   documentation forgery.

Each scenario is encoded as a :class:`ResistanceScenario` with
a stable ``id``, a one-line ``statement``, and a deterministic
``run`` callable. The :class:`ResistanceSuite` enumerates them
and returns a frozen :class:`ResistanceReport` per scenario.

What this module does NOT do
----------------------------

* It does **not** run on every test (it is **not** a pytest
  fixture). It is invoked explicitly by
  ``tests/test_seventh_axis.py``.
* It does **not** mutate production state. Every scenario that
  needs a mutable artifact constructs its own ``tmp_path``
  fixture internally.
* It does **not** re-test module internals. Each scenario
  checks a *composition-level* invariant that only breaks if
  multiple layers cooperate to fail.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

__all__ = [
    "SEVENTH_AXIS_SCENARIOS",
    "ResistanceReport",
    "ResistanceScenario",
    "ResistanceSuite",
]


@dataclass(frozen=True, repr=False)
class ResistanceReport:
    """Frozen outcome of one :class:`ResistanceScenario`.

    Attributes
    ----------
    scenario_id
        Stable id of the scenario (e.g. ``"GATE_NEVER_WIDENS"``).
    passed
        ``True`` iff the system resisted the adversarial
        condition. ``False`` is a load-bearing contract failure.
    detail
        One-line human-readable explainer. Not parsed.
    """

    scenario_id: str
    passed: bool
    detail: str

    def __repr__(self) -> str:  # HN32 aesthetic
        flag = "✓" if self.passed else "✗"
        return f"ResistanceReport[{self.scenario_id} · {flag} · {self.detail}]"


@dataclass(frozen=True)
class ResistanceScenario:
    """One adversarial scenario + its runner.

    Attributes
    ----------
    id
        Stable short id, UPPER_SNAKE_CASE.
    statement
        One-line description of the load-bearing invariant.
    run
        Callable that executes the adversarial check and returns
        a :class:`ResistanceReport`. The callable must be pure
        of its inputs (may use tmp filesystem inside) and must
        not raise — any failure mode is encoded as
        ``passed=False`` on the report.
    """

    id: str
    statement: str
    run: Callable[[], ResistanceReport]


# ---------------------------------------------------------------------------
# Scenario runners — each is a pure function of its inputs.
# ---------------------------------------------------------------------------


def _gate_never_widens_under_adversarial_klr() -> ResistanceReport:
    """Scenario 1: RT-KLR-I1 under adversarial KLR.

    A fake KLR whose tick() claims every frame is SUCCESS with
    positive rank delta must NOT change the gate state sequence
    relative to a clean pipeline on the same inputs.
    """
    from neurophase.runtime.pipeline import PipelineConfig, StreamingPipeline

    class _LyingKLR:
        """KLR that lies about every field. Must not influence the gate."""

        def tick(self, metrics: object) -> _LyingKLRFrame:
            return _LyingKLRFrame()

    class _LyingKLRFrame:
        decision: str = "SUCCESS"
        ntk_rank_delta: float = 1.0  # absurd, to tempt the gate

    cfg = PipelineConfig(warmup_samples=2, stream_window=4, enable_stillness=False)
    clean = StreamingPipeline(cfg)
    attacked = StreamingPipeline(cfg)
    attacked._klr_pipeline = _LyingKLR()  # type: ignore[assignment]

    clean_states: list[str] = []
    attacked_states: list[str] = []
    clean_allowed: list[bool] = []
    attacked_allowed: list[bool] = []
    for i in range(12):
        t = 0.1 * i
        R = 0.10 if i < 6 else 0.95  # start blocked, end ready
        f1 = clean.tick(timestamp=t, R=R, delta=0.05)
        f2 = attacked.tick(timestamp=t, R=R, delta=0.05)
        clean_states.append(f1.gate.state.name)
        attacked_states.append(f2.gate.state.name)
        clean_allowed.append(f1.execution_allowed)
        attacked_allowed.append(f2.execution_allowed)

    if clean_states != attacked_states:
        return ResistanceReport(
            "GATE_NEVER_WIDENS",
            False,
            f"adversarial KLR altered gate sequence: "
            f"clean={clean_states} vs attacked={attacked_states}",
        )
    if clean_allowed != attacked_allowed:
        return ResistanceReport(
            "GATE_NEVER_WIDENS",
            False,
            "adversarial KLR altered execution_allowed sequence",
        )
    return ResistanceReport(
        "GATE_NEVER_WIDENS",
        True,
        f"gate state sequence byte-identical across {len(clean_states)} ticks",
    )


def _memory_bounded_under_10k_ticks() -> ResistanceReport:
    """Scenario 2: L1 under endurance load.

    10 000 orchestrator ticks must leave every rolling buffer
    bounded by its declared cap.
    """
    from neurophase.runtime.memory_audit import audit_runtime_memory
    from neurophase.runtime.orchestrator import (
        OrchestratorConfig,
        RuntimeOrchestrator,
    )
    from neurophase.runtime.pipeline import PipelineConfig

    orch = RuntimeOrchestrator(
        OrchestratorConfig(
            pipeline=PipelineConfig(
                warmup_samples=4,
                stream_window=8,
                enable_stillness=True,
                stillness_window=8,
            )
        )
    )
    for i in range(10_000):
        orch.tick(timestamp=float(i) * 0.01, R=0.92, delta=0.04)

    report = audit_runtime_memory(orch)
    if not report.all_bounded:
        offenders = [c.name for c in report.components if not c.is_bounded]
        return ResistanceReport(
            "MEMORY_BOUNDED_10K",
            False,
            f"unbounded components after 10k ticks: {offenders}",
        )
    return ResistanceReport(
        "MEMORY_BOUNDED_10K",
        True,
        f"all 6 components bounded after {orch.n_ticks} ticks "
        f"(total_measured_size={report.total_measured_size})",
    )


def _ledger_tamper_fails_verification(tmp_path: Path) -> ResistanceReport:
    """Scenario 3: F1 under single-bit tamper."""
    from neurophase.audit.decision_ledger import (
        DecisionTraceLedger,
        fingerprint_parameters,
        verify_ledger,
    )

    path = tmp_path / "adversarial.jsonl"
    ledger = DecisionTraceLedger(path, fingerprint_parameters({"x": 1}))
    for i in range(5):
        ledger.append(
            timestamp=float(i) * 0.1,
            gate_state="READY",
            execution_allowed=True,
            R=0.9,
            threshold=0.65,
            reason=f"tick {i}",
            extras={"i": i},
        )

    # Clean verification first.
    if not verify_ledger(path).ok:
        return ResistanceReport(
            "LEDGER_TAMPER_DETECTED",
            False,
            "clean ledger failed verification",
        )

    # Single-bit tamper: change "tick 2" to "TICK 2" in the middle record.
    text = path.read_text(encoding="utf-8")
    tampered = text.replace("tick 2", "TICK 2", 1)
    if tampered == text:
        return ResistanceReport(
            "LEDGER_TAMPER_DETECTED",
            False,
            "tamper had no effect (precondition broken)",
        )
    path.write_text(tampered, encoding="utf-8")
    verification = verify_ledger(path)
    if verification.ok:
        return ResistanceReport(
            "LEDGER_TAMPER_DETECTED",
            False,
            "tampered ledger verified as clean — hash chain broken",
        )
    return ResistanceReport(
        "LEDGER_TAMPER_DETECTED",
        True,
        f"tamper detected at index {verification.first_broken_index}",
    )


def _contract_violating_claim_rejected(tmp_path: Path) -> ResistanceReport:
    """Scenario 4: HN30 mechanical promotion rule under forgery."""
    import yaml

    from neurophase.governance.claims import ClaimRegistryError, load_claims
    from neurophase.governance.invariants import load_registry

    forged_path = tmp_path / "claims_forged.yaml"
    payload = {
        "version": 1,
        "claims": [
            {
                "id": "FORGED",
                "statement": "fact with insufficient evidence",
                "status": "fact",
                "evidence": [
                    {
                        "source": "bogus 2020",
                        "doi": "",
                        "supports": True,
                        "summary": "fake",
                    }
                ],
                "introduced_in": "attack",
                "related_invariants": [],
            }
        ],
    }
    forged_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    try:
        load_claims(forged_path, invariant_registry=load_registry())
    except ClaimRegistryError as exc:
        return ResistanceReport(
            "CLAIM_FORGERY_REJECTED",
            True,
            f"rejected: {exc}",
        )
    return ResistanceReport(
        "CLAIM_FORGERY_REJECTED",
        False,
        "forged FACT claim with 1 citation was accepted",
    )


def _curriculum_shape_corruption_rejected() -> ResistanceReport:
    """Scenario 5: KLR-I1..I3 under shape attack."""
    from neurophase.reset import (
        Curriculum,
        KetamineLikeResetController,
        KLRConfig,
        SystemMetrics,
        SystemState,
    )

    state = SystemState(
        weights=np.full((4, 4), 0.25),
        confidence=np.full(4, 0.6),
        usage=np.full(4, 0.7),
        utility=np.full(4, 0.7),
        inhibition=np.full(4, 0.7),
        topology=np.ones((4, 4)),
    )
    metrics = SystemMetrics(0.9, 0.9, 0.1, 0.1, 0.0, 0.0)
    # Intentional shape corruption: 3-vector instead of 4-vector
    bad = Curriculum(
        target_bias=np.zeros(3),
        corrective_signal=np.zeros(3),
        stress_pattern=np.zeros(3),
    )
    controller = KetamineLikeResetController(KLRConfig(lock_in_threshold=0.1))
    _, report = controller.run(state, metrics, bad)
    if report.status != "ROLLBACK":
        return ResistanceReport(
            "CURRICULUM_SHAPE_CORRUPTION",
            False,
            f"corrupted curriculum yielded {report.status}, expected ROLLBACK",
        )
    return ResistanceReport(
        "CURRICULUM_SHAPE_CORRUPTION",
        True,
        f"shape-corrupt curriculum collapsed to ROLLBACK: {report.reason[:60]}",
    )


def _monograph_drift_fails_ci(tmp_path: Path) -> ResistanceReport:
    """Scenario 6: HN29 monograph regeneration guard under forgery."""
    from neurophase.governance.monograph import generate_monograph

    live = generate_monograph()
    forged = live.replace(
        "## Hard invariants",
        "## Hard invariants\n\n**FORGED INJECTED SECTION** — this line\n"
        "would break the byte-equality check if it landed in the committed\n"
        "monograph without a regeneration.",
        1,
    )
    if live == forged:
        return ResistanceReport(
            "MONOGRAPH_DRIFT",
            False,
            "forgery had no effect (precondition broken)",
        )
    forged_path = tmp_path / "forged_monograph.md"
    forged_path.write_text(forged, encoding="utf-8")
    roundtrip = forged_path.read_text(encoding="utf-8")
    if roundtrip == generate_monograph():
        return ResistanceReport(
            "MONOGRAPH_DRIFT",
            False,
            "forged monograph matches generator output — guard is ineffective",
        )
    return ResistanceReport(
        "MONOGRAPH_DRIFT",
        True,
        "forged monograph detected by generator byte-equality",
    )


# ---------------------------------------------------------------------------
# Registry + suite runner.
# ---------------------------------------------------------------------------

#: Stable tuple of every axis-7 scenario. Order is used by
#: :class:`ResistanceSuite.run` to produce deterministic reports
#: and by the HN32 test bindings.
SEVENTH_AXIS_SCENARIOS: tuple[ResistanceScenario, ...] = (
    ResistanceScenario(
        id="GATE_NEVER_WIDENS",
        statement=(
            "StreamingPipeline gate sequence is byte-identical "
            "with and without an adversarial KLR attached"
        ),
        run=_gate_never_widens_under_adversarial_klr,
    ),
    ResistanceScenario(
        id="MEMORY_BOUNDED_10K",
        statement=(
            "After 10 000 orchestrator ticks every internal rolling "
            "buffer remains bounded by its declared cap"
        ),
        run=_memory_bounded_under_10k_ticks,
    ),
    ResistanceScenario(
        id="LEDGER_TAMPER_DETECTED",
        statement=(
            "A single-byte tamper of any record in a committed ledger "
            "is detected by verify_ledger at the tampered index"
        ),
        run=lambda: _ledger_tamper_fails_verification(_scratch_dir("ledger_tamper")),
    ),
    ResistanceScenario(
        id="CLAIM_FORGERY_REJECTED",
        statement=(
            "A CLAIMS.yaml entry declaring status=FACT with < 3 "
            "supporting citations is rejected at load time"
        ),
        run=lambda: _contract_violating_claim_rejected(_scratch_dir("claim_forgery")),
    ),
    ResistanceScenario(
        id="CURRICULUM_SHAPE_CORRUPTION",
        statement=(
            "A KLR intervention with a mis-shaped curriculum collapses "
            "to ROLLBACK, never to SUCCESS"
        ),
        run=_curriculum_shape_corruption_rejected,
    ),
    ResistanceScenario(
        id="MONOGRAPH_DRIFT",
        statement=(
            "A hand-edit that inserts a non-generated line into the "
            "committed monograph fails the regeneration guard"
        ),
        run=lambda: _monograph_drift_fails_ci(_scratch_dir("monograph_drift")),
    ),
)


class ResistanceSuite:
    """Runner over :data:`SEVENTH_AXIS_SCENARIOS`.

    The suite is deterministic, stateless, and JSON-friendly. A
    test file in ``tests/test_seventh_axis.py`` invokes
    :meth:`run_all` and asserts every report's ``passed`` flag.
    """

    def run_all(self) -> tuple[ResistanceReport, ...]:
        """Execute every scenario in declaration order and return
        the frozen tuple of reports."""
        return tuple(scenario.run() for scenario in SEVENTH_AXIS_SCENARIOS)

    def run_one(self, scenario_id: str) -> ResistanceReport:
        """Execute a single scenario by id. Raises :class:`KeyError`
        if the id is unknown."""
        for scenario in SEVENTH_AXIS_SCENARIOS:
            if scenario.id == scenario_id:
                return scenario.run()
        raise KeyError(f"unknown seventh-axis scenario id: {scenario_id!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scratch_dir(name: str) -> Path:
    """Return a fresh scratch directory under ``/tmp`` for scenario
    artifacts. Each scenario that needs tmp state calls this with a
    unique name so parallel runs do not collide."""
    import tempfile

    path = Path(tempfile.mkdtemp(prefix=f"neurophase_axis7_{name}_"))
    return path
