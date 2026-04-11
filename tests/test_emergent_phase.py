"""Tests for neurophase.gate.emergent_phase."""

from __future__ import annotations

import pytest

from neurophase.gate.emergent_phase import (
    DEFAULT_CRITERIA,
    EmergentPhaseCriteria,
    detect_emergent_phase,
)


def test_all_four_pass_is_emergent() -> None:
    decision = detect_emergent_phase(R=0.82, dH=-0.08, kappa=-0.15, ism=1.0)
    assert decision.is_emergent
    assert decision.R_ok
    assert decision.dH_ok
    assert decision.kappa_ok
    assert decision.ism_ok
    assert decision.reasons() == []


def test_R_below_threshold_fails() -> None:
    decision = detect_emergent_phase(R=0.60, dH=-0.08, kappa=-0.15, ism=1.0)
    assert not decision.is_emergent
    assert not decision.R_ok
    assert any("R=" in r for r in decision.reasons())


def test_dH_positive_fails() -> None:
    decision = detect_emergent_phase(R=0.82, dH=0.01, kappa=-0.15, ism=1.0)
    assert not decision.is_emergent
    assert not decision.dH_ok


def test_kappa_positive_fails() -> None:
    decision = detect_emergent_phase(R=0.82, dH=-0.08, kappa=0.05, ism=1.0)
    assert not decision.is_emergent
    assert not decision.kappa_ok


def test_ism_above_band_fails() -> None:
    decision = detect_emergent_phase(R=0.82, dH=-0.08, kappa=-0.15, ism=1.5)
    assert not decision.is_emergent
    assert not decision.ism_ok


def test_ism_below_band_fails() -> None:
    decision = detect_emergent_phase(R=0.82, dH=-0.08, kappa=-0.15, ism=0.3)
    assert not decision.is_emergent
    assert not decision.ism_ok


def test_boundary_values_do_not_cross() -> None:
    # All strict comparisons — equality must NOT satisfy the criterion.
    decision = detect_emergent_phase(
        R=DEFAULT_CRITERIA.R_min,
        dH=DEFAULT_CRITERIA.dH_max,
        kappa=DEFAULT_CRITERIA.kappa_max,
        ism=DEFAULT_CRITERIA.ism_low,
    )
    assert not decision.R_ok  # R must be strictly greater than R_min
    assert not decision.dH_ok  # dH must be strictly less than dH_max


def test_custom_criteria() -> None:
    criteria = EmergentPhaseCriteria(
        R_min=0.5, dH_max=0.0, kappa_max=0.0, ism_low=0.5, ism_high=2.0
    )
    decision = detect_emergent_phase(R=0.6, dH=-0.01, kappa=-0.01, ism=1.0, criteria=criteria)
    assert decision.is_emergent


def test_rejects_inverted_ism_band() -> None:
    with pytest.raises(ValueError, match="ism_low"):
        EmergentPhaseCriteria(ism_low=1.5, ism_high=0.8)


def test_rejects_R_min_out_of_range() -> None:
    with pytest.raises(ValueError, match="R_min must be"):
        EmergentPhaseCriteria(R_min=1.5)
