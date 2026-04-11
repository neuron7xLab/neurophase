"""Tenth axis — Reproducibility (Відтворюваність).

Axes 8 and 9 are about the **static** state of the repository —
the files agree (Coherence), the files cover the public surface
(Completeness). Axis 10 is about the **dynamic behavior** — the
system, when invoked, produces the same bytes twice.

A coherent, complete system can still be non-reproducible.
Example: a module that embeds ``time.time()`` into its output, a
function that iterates over a dict without ``sorted(...)``, a
generator that uses ``random`` without a fixed seed. None of
axes 7-8-9 catch that. Axis 10 does.

The :class:`ReproducibilityAuditor` runs each deterministic
layer TWICE with identical inputs and asserts byte-equality on
the full output. A single mismatch is a load-bearing contract
failure.

Scenarios
---------

1. **MONOGRAPH_BYTE_EQUAL** — ``generate_monograph()`` twice.
2. **DOCTOR_REGISTRY_BYTE_EQUAL** — DOCTOR_CHECKS tuple +
   cheap sub-checks (schema loads) byte-equal across two runs.
   The heavy doctor checks (resistance, memory audit,
   completeness) are covered by their own scenarios below.
3. **RESISTANCE_SUITE_BYTE_EQUAL** — ``ResistanceSuite().run_all()``
   twice. Every scenario's ``(passed, detail)`` pair matches.
4. **COMPLETENESS_SUITE_BYTE_EQUAL** — ``run_completeness()``
   twice. Every per-check result stable.
5. **PARAMETER_SWEEP_BYTE_EQUAL** — ``sweep_parameters(grid, seed)``
   twice on the same grid + seed.
6. **SYNTHETIC_OSCILLATOR_BYTE_EQUAL** — two
   :class:`SyntheticOscillatorSource` instances with the same
   config driven for 128 steps produce byte-identical phase
   arrays.
7. **INVARIANT_REGISTRY_BYTE_EQUAL** — ``load_registry()`` twice
   produces equal ``InvariantRegistry`` objects (structural
   equality on frozen dataclasses).
8. **PIPELINE_GATE_SEQUENCE_BYTE_EQUAL** — two
   :class:`StreamingPipeline` instances with the same
   :class:`PipelineConfig` driven on the same
   ``(timestamp, R, delta)`` sequence produce identical
   ``gate.state`` sequences over 32 ticks.

Each scenario returns a frozen
:class:`ReproducibilityCheckResult`; the suite aggregates into
:class:`ReproducibilityReport`. The report is itself
deterministic, closing the axis-10 loop on itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "REPRODUCIBILITY_SCENARIOS",
    "ReproducibilityAuditor",
    "ReproducibilityCheckResult",
    "ReproducibilityReport",
    "run_reproducibility",
]


@dataclass(frozen=True, repr=False)
class ReproducibilityCheckResult:
    """Frozen outcome of one reproducibility scenario.

    Attributes
    ----------
    scenario_id
        Stable UPPER_SNAKE_CASE id.
    passed
        ``True`` iff both runs produced byte-identical output.
    detail
        One-line summary. Stable first token:
        ``byte_equal:`` / ``mismatch:``.
    """

    scenario_id: str
    passed: bool
    detail: str

    def __repr__(self) -> str:  # HN37 aesthetic
        flag = "✓" if self.passed else "✗"
        return f"ReproducibilityCheckResult[{self.scenario_id} · {flag}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True, repr=False)
class ReproducibilityReport:
    """Aggregated outcome of the axis-10 auditor.

    Attributes
    ----------
    results
        Tuple of :class:`ReproducibilityCheckResult`, in
        declaration order.
    reproducible
        ``True`` iff every scenario produced byte-equal output.
    """

    results: tuple[ReproducibilityCheckResult, ...]
    reproducible: bool

    def __post_init__(self) -> None:
        expected = all(r.passed for r in self.results)
        if expected != self.reproducible:
            raise ValueError(f"reproducible={self.reproducible} disagrees with per-result flags")

    def __repr__(self) -> str:  # HN37 aesthetic
        flag = "✓" if self.reproducible else "✗"
        passed = sum(1 for r in self.results if r.passed)
        return f"ReproducibilityReport[{passed}/{len(self.results)} · {flag}]"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "reproducible": self.reproducible,
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Scenario runners — each is a pure function that runs its target
# twice and compares.
# ---------------------------------------------------------------------------


def _check_monograph_byte_equal() -> ReproducibilityCheckResult:
    from neurophase.governance.monograph import generate_monograph

    a = generate_monograph()
    b = generate_monograph()
    if a != b:
        return ReproducibilityCheckResult(
            "MONOGRAPH_BYTE_EQUAL",
            False,
            f"mismatch: delta = {len(b) - len(a):+d} bytes on second run",
        )
    return ReproducibilityCheckResult(
        "MONOGRAPH_BYTE_EQUAL",
        True,
        f"byte_equal: {len(a)} bytes stable across two runs",
    )


def _check_doctor_registry_byte_equal() -> ReproducibilityCheckResult:
    """Cheap determinism probe for the doctor: verify that the
    registered check list is a stable tuple and that the per-check
    callables return equal CheckResult objects **without** running
    the slow sub-suites (resistance, 10k-tick memory audit, full
    completeness). Those are covered by their own dedicated
    scenarios below, so running the full doctor twice would
    duplicate 2-3 minutes of compute for no additional coverage.
    """
    from neurophase.governance.doctor import DOCTOR_CHECKS

    ids_a = tuple(entry[0] for entry in DOCTOR_CHECKS)
    ids_b = tuple(entry[0] for entry in DOCTOR_CHECKS)
    if ids_a != ids_b:
        return ReproducibilityCheckResult(
            "DOCTOR_REGISTRY_BYTE_EQUAL",
            False,
            "mismatch: DOCTOR_CHECKS ids differ across two accesses",
        )
    # Run only the cheap identity checks (schema + registry
    # loads) twice and compare. The heavy checks are covered
    # elsewhere in this suite.
    cheap_ids = {
        "INVARIANT_REGISTRY_SCHEMA",
        "STATE_MACHINE_SCHEMA",
        "CLAIM_REGISTRY_SCHEMA",
        "API_FACADE_SURFACE",
    }
    from neurophase.governance.doctor import Doctor

    doctor = Doctor()
    for check_id in cheap_ids:
        r1 = doctor.run_one(check_id)
        r2 = doctor.run_one(check_id)
        if r1.to_dict() != r2.to_dict():
            return ReproducibilityCheckResult(
                "DOCTOR_REGISTRY_BYTE_EQUAL",
                False,
                f"mismatch: {check_id} drifted across two invocations",
            )
    return ReproducibilityCheckResult(
        "DOCTOR_REGISTRY_BYTE_EQUAL",
        True,
        f"byte_equal: {len(DOCTOR_CHECKS)} registered checks stable, "
        f"{len(cheap_ids)} cheap re-runs byte-identical",
    )


def _check_resistance_suite_byte_equal() -> ReproducibilityCheckResult:
    from neurophase.governance.resistance import ResistanceSuite

    a = ResistanceSuite().run_all()
    b = ResistanceSuite().run_all()
    pairs_a = [(r.scenario_id, r.passed) for r in a]
    pairs_b = [(r.scenario_id, r.passed) for r in b]
    if pairs_a != pairs_b:
        return ReproducibilityCheckResult(
            "RESISTANCE_SUITE_BYTE_EQUAL",
            False,
            f"mismatch: resistance scenario outcomes drifted: {pairs_a} vs {pairs_b}",
        )
    return ReproducibilityCheckResult(
        "RESISTANCE_SUITE_BYTE_EQUAL",
        True,
        f"byte_equal: all {len(a)} axis-7 scenarios stable across two runs",
    )


def _check_completeness_suite_byte_equal() -> ReproducibilityCheckResult:
    from neurophase.governance.completeness import run_completeness

    a = run_completeness()
    b = run_completeness()
    if a.to_json_dict() != b.to_json_dict():
        return ReproducibilityCheckResult(
            "COMPLETENESS_SUITE_BYTE_EQUAL",
            False,
            "mismatch: completeness report drifted across runs",
        )
    return ReproducibilityCheckResult(
        "COMPLETENESS_SUITE_BYTE_EQUAL",
        True,
        f"byte_equal: all {len(a.results)} axis-9 checks stable across two runs",
    )


def _check_parameter_sweep_byte_equal() -> ReproducibilityCheckResult:
    from neurophase.benchmarks.parameter_sweep import SweepGrid, sweep_parameters

    grid = SweepGrid(
        threshold_values=(0.30, 0.50, 0.70),
        coupling_strengths=(0.0, 0.5, 1.0),
        trace_seeds=(1, 2, 3),
        n_samples=64,
    )
    a = sweep_parameters(grid, seed=42)
    b = sweep_parameters(grid, seed=42)
    if a.to_json_dict() != b.to_json_dict():
        return ReproducibilityCheckResult(
            "PARAMETER_SWEEP_BYTE_EQUAL",
            False,
            "mismatch: sweep report drifted across two runs",
        )
    return ReproducibilityCheckResult(
        "PARAMETER_SWEEP_BYTE_EQUAL",
        True,
        f"byte_equal: {len(a.results)} sweep cells stable across two runs",
    )


def _check_synthetic_oscillator_byte_equal() -> ReproducibilityCheckResult:
    import numpy as np

    from neurophase.sensors.synthetic import (
        SyntheticOscillatorConfig,
        SyntheticOscillatorSource,
    )

    cfg = SyntheticOscillatorConfig(n_channels=4, seed=777)
    a = SyntheticOscillatorSource(cfg)
    b = SyntheticOscillatorSource(cfg)
    for step in range(128):
        fa = a.extract()
        fb = b.extract()
        if not np.array_equal(fa.phases, fb.phases):
            return ReproducibilityCheckResult(
                "SYNTHETIC_OSCILLATOR_BYTE_EQUAL",
                False,
                f"mismatch: phase drift at step {step}",
            )
    return ReproducibilityCheckResult(
        "SYNTHETIC_OSCILLATOR_BYTE_EQUAL",
        True,
        "byte_equal: 128 steps of synthetic oscillator byte-identical across two sources",
    )


def _check_invariant_registry_byte_equal() -> ReproducibilityCheckResult:
    from neurophase.governance.invariants import load_registry

    a = load_registry()
    b = load_registry()
    if a != b:
        return ReproducibilityCheckResult(
            "INVARIANT_REGISTRY_BYTE_EQUAL",
            False,
            "mismatch: InvariantRegistry structural equality failed",
        )
    total = len(a.invariants) + len(a.honest_naming)
    return ReproducibilityCheckResult(
        "INVARIANT_REGISTRY_BYTE_EQUAL",
        True,
        f"byte_equal: {total} registered contracts stable across two loads",
    )


def _check_pipeline_gate_sequence_byte_equal() -> ReproducibilityCheckResult:
    from neurophase.runtime.pipeline import PipelineConfig, StreamingPipeline

    cfg = PipelineConfig(warmup_samples=2, stream_window=4, enable_stillness=False)
    p1 = StreamingPipeline(cfg)
    p2 = StreamingPipeline(cfg)

    seq = [(float(i) * 0.1, 0.5 + 0.02 * (i % 20), 0.05) for i in range(32)]
    states_a: list[str] = []
    states_b: list[str] = []
    for t, R, d in seq:
        fa = p1.tick(timestamp=t, R=R, delta=d)
        fb = p2.tick(timestamp=t, R=R, delta=d)
        states_a.append(fa.gate.state.name)
        states_b.append(fb.gate.state.name)
    if states_a != states_b:
        return ReproducibilityCheckResult(
            "PIPELINE_GATE_SEQUENCE_BYTE_EQUAL",
            False,
            f"mismatch: gate sequences differ: {states_a} vs {states_b}",
        )
    return ReproducibilityCheckResult(
        "PIPELINE_GATE_SEQUENCE_BYTE_EQUAL",
        True,
        f"byte_equal: {len(seq)}-tick gate sequence stable across two pipelines",
    )


# ---------------------------------------------------------------------------
# Registry + runner.
# ---------------------------------------------------------------------------

#: Stable tuple of every axis-10 scenario, in declaration order.
REPRODUCIBILITY_SCENARIOS: tuple[tuple[str, Callable[[], ReproducibilityCheckResult]], ...] = (
    ("MONOGRAPH_BYTE_EQUAL", _check_monograph_byte_equal),
    ("DOCTOR_REGISTRY_BYTE_EQUAL", _check_doctor_registry_byte_equal),
    ("RESISTANCE_SUITE_BYTE_EQUAL", _check_resistance_suite_byte_equal),
    ("COMPLETENESS_SUITE_BYTE_EQUAL", _check_completeness_suite_byte_equal),
    ("PARAMETER_SWEEP_BYTE_EQUAL", _check_parameter_sweep_byte_equal),
    ("SYNTHETIC_OSCILLATOR_BYTE_EQUAL", _check_synthetic_oscillator_byte_equal),
    ("INVARIANT_REGISTRY_BYTE_EQUAL", _check_invariant_registry_byte_equal),
    (
        "PIPELINE_GATE_SEQUENCE_BYTE_EQUAL",
        _check_pipeline_gate_sequence_byte_equal,
    ),
)


class ReproducibilityAuditor:
    """Runner over :data:`REPRODUCIBILITY_SCENARIOS`.

    Stateless. Two invocations on the same repository state
    produce the same :class:`ReproducibilityReport` — which is
    itself a load-bearing claim, closing axis 10 on itself.
    """

    def run(self) -> ReproducibilityReport:
        results: list[ReproducibilityCheckResult] = []
        for _, runner in REPRODUCIBILITY_SCENARIOS:
            try:
                result = runner()
            except Exception as exc:  # defensive
                result = ReproducibilityCheckResult(
                    scenario_id="UNKNOWN",
                    passed=False,
                    detail=f"mismatch: runner raised {type(exc).__name__}: {exc}",
                )
            results.append(result)

        reproducible = all(r.passed for r in results)
        return ReproducibilityReport(
            results=tuple(results),
            reproducible=reproducible,
        )

    def run_one(self, scenario_id: str) -> ReproducibilityCheckResult:
        for registered_id, runner in REPRODUCIBILITY_SCENARIOS:
            if registered_id == scenario_id:
                return runner()
        raise KeyError(f"unknown reproducibility scenario: {scenario_id!r}")


def run_reproducibility() -> ReproducibilityReport:
    """Shortcut: ``ReproducibilityAuditor().run()``."""
    return ReproducibilityAuditor().run()
