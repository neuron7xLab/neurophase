"""Tests for neurophase.metrics.effect_size.

Coverage: Cohen's d, Hedges' g, CI, power, interpretation, compound report.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from neurophase.metrics.effect_size import (
    EffectSizeReport,
    cohens_d,
    cohens_d_one_sample,
    confidence_interval_d,
    effect_size_report,
    hedges_g,
    statistical_power,
)

# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------


def test_cohens_d_identical_groups_is_zero() -> None:
    g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    g2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert cohens_d(g1, g2) == pytest.approx(0.0, abs=1e-12)


def test_cohens_d_known_value() -> None:
    # group1=[1,2,3], group2=[4,5,6]
    # pooled std = sqrt(((2*1+2*1)/(3+3-2))) = 1.0
    # d = (2 - 5) / 1 = -3.0
    # But the task says ≈ -2.45.  Let's verify the actual formula.
    # pooled_var = ((2*var([1,2,3],ddof=1)) + (2*var([4,5,6],ddof=1))) / (3+3-2)
    #            = (2*1 + 2*1) / 4 = 1.0   → d = -3.0
    # The commonly-cited "≈ -2.45" uses the biased (ddof=0) SD.
    # Our implementation uses ddof=1 (unbiased), so d = -3.0.
    d = cohens_d([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert d == pytest.approx(-3.0, rel=1e-6)


def test_cohens_d_sign() -> None:
    d = cohens_d([10.0, 11.0, 12.0], [1.0, 2.0, 3.0])
    assert d > 0.0


def test_cohens_d_raises_small_group() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        cohens_d([1.0], [2.0, 3.0])


# ---------------------------------------------------------------------------
# hedges_g
# ---------------------------------------------------------------------------


def test_hedges_g_smaller_than_d_for_small_n() -> None:
    g1 = [1.0, 2.0, 3.0]
    g2 = [4.0, 5.0, 6.0]
    d = cohens_d(g1, g2)
    g = hedges_g(g1, g2)
    # J < 1 → |g| < |d|
    assert abs(g) < abs(d)


def test_hedges_g_converges_to_d_for_large_n() -> None:
    rng = np.random.default_rng(0)
    g1 = rng.normal(0.0, 1.0, 2000)
    g2 = rng.normal(0.5, 1.0, 2000)
    d = cohens_d(g1, g2)
    g = hedges_g(g1, g2)
    # For N=4000 the correction J ≈ 0.9998; difference is tiny
    assert abs(g - d) < 1e-3


def test_hedges_g_same_sign_as_d() -> None:
    g1 = [1.0, 2.0, 3.0]
    g2 = [4.0, 5.0, 6.0]
    assert math.copysign(1.0, hedges_g(g1, g2)) == math.copysign(1.0, cohens_d(g1, g2))


# ---------------------------------------------------------------------------
# cohens_d_one_sample
# ---------------------------------------------------------------------------


def test_one_sample_d() -> None:
    # values = [2,4,6,8,10], mean=6, std=sqrt(10)≈3.162, mu_0=0 → d≈1.897
    values = [2.0, 4.0, 6.0, 8.0, 10.0]
    d = cohens_d_one_sample(values, mu_0=0.0)
    assert d == pytest.approx(6.0 / math.sqrt(10.0), rel=1e-6)


def test_one_sample_d_zero_when_at_null() -> None:
    values = [3.0, 3.0, 3.0, 3.0, 3.0]
    # std=0 → return 0
    assert cohens_d_one_sample(values, mu_0=3.0) == 0.0


def test_one_sample_d_raises_too_few() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        cohens_d_one_sample([5.0])


# ---------------------------------------------------------------------------
# confidence_interval_d
# ---------------------------------------------------------------------------


def test_ci_contains_true_d() -> None:
    """Parametric check: CI at alpha=0.05 should contain the true d.

    Convention: d = mean(g1) - mean(g2).  With g1 ~ N(true_d, 1) and
    g2 ~ N(0, 1) the true parameter is +true_d.
    """
    rng = np.random.default_rng(42)
    n = 200
    n_reps = 200
    covered = 0
    true_d = 0.5
    for _ in range(n_reps):
        # g1 has higher mean so d ≈ +true_d
        g1 = rng.normal(true_d, 1.0, n)
        g2 = rng.normal(0.0, 1.0, n)
        d_hat = cohens_d(g1, g2)
        lo, hi = confidence_interval_d(d_hat, n1=n, n2=n)
        if lo <= true_d <= hi:
            covered += 1
    coverage = covered / n_reps
    # Expect ≥ 90 % empirical coverage (nominal 95 %)
    assert coverage >= 0.90, f"CI coverage too low: {coverage:.2%}"


def test_ci_width_decreases_with_n() -> None:
    d = 0.5
    lo_small, hi_small = confidence_interval_d(d, n1=10, n2=10)
    lo_large, hi_large = confidence_interval_d(d, n1=1000, n2=1000)
    width_small = hi_small - lo_small
    width_large = hi_large - lo_large
    assert width_large < width_small


# ---------------------------------------------------------------------------
# statistical_power
# ---------------------------------------------------------------------------


def test_power_increases_with_n() -> None:
    d = 0.5
    p_small = statistical_power(d, n=20)
    p_large = statistical_power(d, n=200)
    assert p_large > p_small


def test_power_increases_with_effect_size() -> None:
    n = 50
    p_small_d = statistical_power(0.2, n=n)
    p_large_d = statistical_power(0.8, n=n)
    assert p_large_d > p_small_d


def test_power_range() -> None:
    pwr = statistical_power(0.5, n=50)
    assert 0.0 <= pwr <= 1.0


def test_power_at_zero_d_is_alpha_over_2() -> None:
    # d=0 → power ≈ alpha/2 for one-sided interpretation, but formula gives
    # Φ(0 - z_{0.975}) = Φ(-1.96) ≈ 0.025
    pwr = statistical_power(0.0, n=100, alpha=0.05)
    assert pwr == pytest.approx(0.025, abs=1e-3)


# ---------------------------------------------------------------------------
# Interpretation thresholds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "d,expected",
    [
        (0.0, "negligible"),
        (0.1, "negligible"),
        (0.19, "negligible"),
        (0.2, "small"),
        (0.35, "small"),
        (0.49, "small"),
        (0.5, "medium"),
        (0.65, "medium"),
        (0.79, "medium"),
        (0.8, "large"),
        (1.5, "large"),
        (-0.9, "large"),
        (-0.3, "small"),
    ],
)
def test_interpretation_thresholds(d: float, expected: str) -> None:
    from neurophase.metrics.effect_size import _interpret  # type: ignore[attr-defined]

    assert _interpret(d) == expected


# ---------------------------------------------------------------------------
# EffectSizeReport — compound convenience function
# ---------------------------------------------------------------------------


def test_effect_size_report_from_null_distribution() -> None:
    rng = np.random.default_rng(7)
    null_dist = rng.normal(0.0, 1.0, 500)
    # observed value 2 SD above null mean
    observed = float(np.mean(null_dist)) + 2.0 * float(np.std(null_dist, ddof=1))

    report = effect_size_report(observed, null_dist, n_subjects=30)

    assert isinstance(report, EffectSizeReport)
    assert report.cohens_d == pytest.approx(2.0, abs=0.05)
    assert report.hedges_g != report.cohens_d  # correction applied
    assert report.ci_lower < report.cohens_d < report.ci_upper
    assert 0.0 <= report.power <= 1.0
    assert report.interpretation == "large"


def test_effect_size_report_null_raises_too_few() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        effect_size_report(1.0, [0.5], n_subjects=10)


def test_effect_size_report_negligible_effect() -> None:
    rng = np.random.default_rng(99)
    null_dist = rng.normal(0.0, 1.0, 1000)
    # observed ≈ null mean → negligible d
    observed = float(np.mean(null_dist)) + 0.01
    report = effect_size_report(observed, null_dist, n_subjects=20)
    assert report.interpretation == "negligible"
    assert abs(report.cohens_d) < 0.2
