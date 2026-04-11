"""Tests for ``neurophase.benchmarks.phase_coupling`` (H1).

Validates the ground-truth contract: the endpoint cases (``c = 0``,
``c = 1``) produce PLV that matches ``ground_truth_plv`` exactly or
within tight finite-sample tolerances, and intermediate values
produce PLV that converges to ``c`` as ``n → ∞``.

Also certifies determinism, phase wrapping, config validation, and
integration with ``plv_significance`` from PR #15 (the harness
rejects independent pairs and accepts tightly locked pairs).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from neurophase.benchmarks.phase_coupling import (
    DEFAULT_N_SAMPLES,
    PhaseCouplingConfig,
    PhaseCouplingTrace,
    generate_anti_coupled,
    generate_phase_coupling,
)
from neurophase.metrics.plv import plv, plv_significance

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_default_config(self) -> None:
        c = PhaseCouplingConfig()
        assert c.n == DEFAULT_N_SAMPLES
        assert c.coupling_strength == 0.5
        assert c.phi_offset == 0.0
        assert c.seed == 42

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n": 0},
            {"n": 1},
            {"n": -10},
            {"coupling_strength": -0.1},
            {"coupling_strength": 1.1},
            {"phi_offset": float("nan")},
            {"phi_offset": float("inf")},
        ],
    )
    def test_invalid_config_rejected(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            PhaseCouplingConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Ground-truth endpoints (the load-bearing contract)
# ---------------------------------------------------------------------------


class TestGroundTruthEndpoints:
    def test_fully_locked_gives_plv_exactly_one(self) -> None:
        trace = generate_phase_coupling(PhaseCouplingConfig(n=1024, coupling_strength=1.0, seed=1))
        assert trace.ground_truth_plv == 1.0
        assert math.isclose(plv(trace.phi_x, trace.phi_y), 1.0, abs_tol=1e-12)

    def test_fully_locked_with_offset_still_plv_one(self) -> None:
        """PLV measures locking, not direction — any constant offset gives PLV=1."""
        trace = generate_phase_coupling(
            PhaseCouplingConfig(
                n=512,
                coupling_strength=1.0,
                phi_offset=math.pi / 3,
                seed=3,
            )
        )
        assert math.isclose(plv(trace.phi_x, trace.phi_y), 1.0, abs_tol=1e-12)

    def test_independent_pair_has_plv_near_zero(self) -> None:
        trace = generate_phase_coupling(PhaseCouplingConfig(n=4096, coupling_strength=0.0, seed=5))
        assert trace.ground_truth_plv == 0.0
        # Finite-sample PLV for independent uniforms is O(1/√n).
        measured = plv(trace.phi_x, trace.phi_y)
        assert measured < 0.05

    def test_anti_coupled_helper_matches_direct_zero(self) -> None:
        a = generate_anti_coupled(n=1024, seed=7)
        b = generate_phase_coupling(PhaseCouplingConfig(n=1024, coupling_strength=0.0, seed=7))
        np.testing.assert_array_equal(a.phi_x, b.phi_x)
        np.testing.assert_array_equal(a.phi_y, b.phi_y)


# ---------------------------------------------------------------------------
# Monotonicity: stronger coupling → higher finite-sample PLV
# ---------------------------------------------------------------------------


class TestCouplingMonotonicity:
    def test_plv_is_monotone_in_coupling_strength(self) -> None:
        """Average PLV over many seeds is monotone increasing in c."""
        strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
        avg_plvs: list[float] = []
        for c in strengths:
            samples: list[float] = []
            for seed in range(11, 21):
                trace = generate_phase_coupling(
                    PhaseCouplingConfig(n=1024, coupling_strength=c, seed=seed)
                )
                samples.append(plv(trace.phi_x, trace.phi_y))
            avg_plvs.append(float(np.mean(samples)))

        # Each consecutive pair must be non-decreasing (strict for most).
        for i in range(len(avg_plvs) - 1):
            assert avg_plvs[i + 1] >= avg_plvs[i] - 1e-3, (
                f"monotonicity violated at c={strengths[i]}→{strengths[i + 1]}: "
                f"{avg_plvs[i]} → {avg_plvs[i + 1]}"
            )


# ---------------------------------------------------------------------------
# Phase wrapping
# ---------------------------------------------------------------------------


class TestPhaseWrapping:
    @pytest.mark.parametrize("coupling", [0.0, 0.5, 1.0])
    def test_output_is_wrapped(self, coupling: float) -> None:
        trace = generate_phase_coupling(
            PhaseCouplingConfig(n=256, coupling_strength=coupling, phi_offset=2 * math.pi, seed=9)
        )
        assert np.all(trace.phi_x > -math.pi - 1e-12)
        assert np.all(trace.phi_x <= math.pi + 1e-12)
        assert np.all(trace.phi_y > -math.pi - 1e-12)
        assert np.all(trace.phi_y <= math.pi + 1e-12)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_bit_identical(self) -> None:
        cfg = PhaseCouplingConfig(n=512, coupling_strength=0.5, seed=99)
        a = generate_phase_coupling(cfg)
        b = generate_phase_coupling(cfg)
        np.testing.assert_array_equal(a.phi_x, b.phi_x)
        np.testing.assert_array_equal(a.phi_y, b.phi_y)
        assert a.ground_truth_plv == b.ground_truth_plv

    def test_different_seed_different_trace(self) -> None:
        a = generate_phase_coupling(PhaseCouplingConfig(n=256, coupling_strength=0.5, seed=1))
        b = generate_phase_coupling(PhaseCouplingConfig(n=256, coupling_strength=0.5, seed=2))
        assert not np.array_equal(a.phi_x, b.phi_x)


# ---------------------------------------------------------------------------
# Integration with the PLV significance harness (closes the C loop)
# ---------------------------------------------------------------------------


class TestHarnessIntegration:
    def test_null_is_not_rejected(self) -> None:
        """An independent pair must not be flagged as significant by the
        shared NullModelHarness at α = 0.05."""
        trace = generate_anti_coupled(n=1024, seed=101)
        result = plv_significance(trace.phi_x, trace.phi_y, n_surrogates=200, seed=103)
        assert not result.significant

    def test_tight_lock_is_rejected(self) -> None:
        trace = generate_phase_coupling(
            PhaseCouplingConfig(n=1024, coupling_strength=1.0, seed=111)
        )
        result = plv_significance(trace.phi_x, trace.phi_y, n_surrogates=200, seed=113)
        assert result.significant
        assert math.isclose(result.plv, 1.0, abs_tol=1e-12)

    def test_moderate_coupling_is_usually_rejected(self) -> None:
        """Expected PLV ≈ 0.75 should be rejected by the harness with
        near-certainty at n=1024, n_surrogates=200."""
        trace = generate_phase_coupling(
            PhaseCouplingConfig(n=1024, coupling_strength=0.75, seed=121)
        )
        result = plv_significance(trace.phi_x, trace.phi_y, n_surrogates=200, seed=123)
        assert result.significant


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_config_frozen(self) -> None:
        import dataclasses

        c = PhaseCouplingConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.n = 99  # type: ignore[misc]

    def test_trace_frozen(self) -> None:
        import dataclasses

        trace = generate_anti_coupled(n=64, seed=1)
        assert isinstance(trace, PhaseCouplingTrace)
        with pytest.raises(dataclasses.FrozenInstanceError):
            trace.ground_truth_plv = 0.5  # type: ignore[misc]
