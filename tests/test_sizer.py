"""Tests for neurophase.risk.sizer."""

from __future__ import annotations

import pytest

from neurophase.risk.sizer import RiskProfile, size_position


def test_gate_blocked_returns_zero() -> None:
    out = size_position(R=0.5, threshold=0.65, cvar=0.05)
    assert out.fraction == 0.0
    assert "gate blocked" in out.reason


def test_zero_cvar_refuses() -> None:
    out = size_position(R=0.9, threshold=0.65, cvar=0.0)
    assert out.fraction == 0.0
    assert "CVaR" in out.reason


def test_full_synchronization_uses_full_scale() -> None:
    """R = 1 maximises scale_R; result approaches cvar_cap · scale_m."""
    profile = RiskProfile(
        risk_per_trade=0.02, confidence=0.99, max_leverage=5.0, multifractal_penalty=0.0
    )
    out = size_position(R=1.0, threshold=0.65, cvar=0.05, profile=profile)
    assert out.scale_R == 1.0
    assert out.scale_m == 1.0
    # cvar_cap = 0.02 / 0.05 = 0.4 → below the 5.0 leverage cap.
    assert out.fraction == pytest.approx(0.4)


def test_partial_sync_scales_linearly() -> None:
    profile = RiskProfile(multifractal_penalty=0.0)
    low = size_position(R=0.70, threshold=0.65, cvar=0.05, profile=profile)
    high = size_position(R=0.90, threshold=0.65, cvar=0.05, profile=profile)
    assert low.fraction < high.fraction
    assert high.scale_R > low.scale_R


def test_multifractal_penalty_shrinks_size() -> None:
    profile = RiskProfile(multifractal_penalty=1.5)
    calm = size_position(R=0.9, threshold=0.65, cvar=0.05, profile=profile)
    rough = size_position(
        R=0.9,
        threshold=0.65,
        cvar=0.05,
        multifractal_instability_value=0.4,
        profile=profile,
    )
    assert rough.fraction < calm.fraction


def test_hard_cap_at_max_leverage() -> None:
    """A tiny CVaR with strong sync may trigger the hard leverage cap."""
    profile = RiskProfile(
        risk_per_trade=0.1, confidence=0.99, max_leverage=2.0, multifractal_penalty=0.0
    )
    out = size_position(R=1.0, threshold=0.5, cvar=0.001, profile=profile)
    assert out.fraction == pytest.approx(2.0)
    assert "max_leverage" in out.reason


def test_profile_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="risk_per_trade"):
        RiskProfile(risk_per_trade=0.0)
    with pytest.raises(ValueError, match="confidence"):
        RiskProfile(confidence=1.0)
    with pytest.raises(ValueError, match="max_leverage"):
        RiskProfile(max_leverage=0.0)
    with pytest.raises(ValueError, match="multifractal_penalty"):
        RiskProfile(multifractal_penalty=-0.5)


def test_threshold_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="threshold must be"):
        size_position(R=0.9, threshold=1.5, cvar=0.05)


def test_multifractal_penalty_can_collapse_size() -> None:
    """Very wide spectrum forces scale_m = 0 and fraction = 0."""
    profile = RiskProfile(multifractal_penalty=10.0)
    out = size_position(
        R=1.0,
        threshold=0.5,
        cvar=0.05,
        multifractal_instability_value=0.2,
        profile=profile,
    )
    assert out.scale_m == 0.0
    assert out.fraction == 0.0
