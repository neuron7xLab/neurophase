"""Effect-size computation for null-model confrontation results.

Standard effect-size metrics for reporting alongside p-values,
following APA 7th edition reporting guidelines. Every null result
in results/ must carry an effect size so that a reader can distinguish
"null because N is small" from "null because the effect is absent."

References:
    Cohen (1988) Statistical Power Analysis for the Behavioral Sciences.
    Hedges & Olkin (1985) Statistical Methods for Meta-Analysis.
    Faul et al. (2007) G*Power 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

# ---------------------------------------------------------------------------
# Interpretation thresholds (Cohen 1988)
# ---------------------------------------------------------------------------


def _interpret(d: float) -> str:
    """Map |d| to Cohen's verbal label."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# Core effect-size estimators
# ---------------------------------------------------------------------------


def cohens_d(group1: ArrayLike, group2: ArrayLike) -> float:
    """Standardized mean difference between two independent groups.

    Uses pooled standard deviation (equal-variance assumption).

    Parameters
    ----------
    group1, group2:
        1-D array-like of observations.

    Returns
    -------
    float
        Cohen's d.  Positive when mean(group1) > mean(group2).
    """
    a = np.asarray(group1, dtype=float)
    b = np.asarray(group2, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("Each group must have at least 2 observations.")
    pooled_var = ((n1 - 1) * float(np.var(a, ddof=1)) + (n2 - 1) * float(np.var(b, ddof=1))) / (
        n1 + n2 - 2
    )
    if pooled_var <= 0.0:
        return 0.0
    return (float(np.mean(a)) - float(np.mean(b))) / math.sqrt(pooled_var)


def hedges_g(group1: ArrayLike, group2: ArrayLike) -> float:
    """Bias-corrected effect size for small samples (Hedges & Olkin 1985).

    Applies correction factor J = 1 - 3 / (4*(n1+n2) - 9).

    Parameters
    ----------
    group1, group2:
        1-D array-like of observations.

    Returns
    -------
    float
        Hedges' g.
    """
    a = np.asarray(group1, dtype=float)
    b = np.asarray(group2, dtype=float)
    d = cohens_d(a, b)
    n_total = len(a) + len(b)
    j = 1.0 - 3.0 / (4.0 * n_total - 9.0)
    return d * j


def cohens_d_one_sample(values: ArrayLike, mu_0: float = 0.0) -> float:
    """Cohen's d for a one-sample test against a null mean.

    Parameters
    ----------
    values:
        1-D array-like of observations.
    mu_0:
        Null hypothesis mean (default 0.0).

    Returns
    -------
    float
        Cohen's d = (mean - mu_0) / std.
    """
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        raise ValueError("Need at least 2 observations.")
    sd = float(np.std(v, ddof=1))
    if sd <= 0.0:
        return 0.0
    return (float(np.mean(v)) - mu_0) / sd


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------


def confidence_interval_d(
    d: float,
    n1: int,
    n2: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Approximate 95 % CI for Cohen's d via non-central t distribution.

    Uses the Hedges & Olkin (1985) variance approximation:
        Var(d) ≈ (n1+n2)/(n1*n2)  +  d²/(2*(n1+n2))

    Parameters
    ----------
    d:
        Observed Cohen's d.
    n1, n2:
        Group sizes.
    alpha:
        Significance level (default 0.05 → 95 % CI).

    Returns
    -------
    tuple[float, float]
        (lower, upper) confidence bounds.
    """
    from scipy.stats import norm

    se_d = math.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2.0 * (n1 + n2)))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return (d - z * se_d, d + z * se_d)


# ---------------------------------------------------------------------------
# Post-hoc statistical power
# ---------------------------------------------------------------------------


def statistical_power(d: float, n: int, alpha: float = 0.05) -> float:
    """Post-hoc power for a two-sample t-test via normal approximation.

    power = Φ(|d| · √(n/2) − z_{1−α/2})

    Parameters
    ----------
    d:
        Cohen's d (effect size).
    n:
        Total sample size (n1 = n2 = n/2 assumed).
    alpha:
        Type-I error rate (default 0.05).

    Returns
    -------
    float
        Power in [0, 1].
    """
    from scipy.stats import norm

    z_crit = float(norm.ppf(1.0 - alpha / 2.0))
    ncp = abs(d) * math.sqrt(n / 2.0)
    return float(norm.cdf(ncp - z_crit))


# ---------------------------------------------------------------------------
# Compound report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectSizeReport:
    """Full effect-size summary for one comparison.

    Attributes
    ----------
    cohens_d:
        Standardized mean difference.
    hedges_g:
        Bias-corrected d (Hedges & Olkin 1985).
    ci_lower:
        Lower bound of 95 % CI for Cohen's d.
    ci_upper:
        Upper bound of 95 % CI for Cohen's d.
    power:
        Post-hoc statistical power.
    interpretation:
        Verbal label per Cohen (1988) benchmarks:
        "negligible" / "small" / "medium" / "large".
    """

    cohens_d: float
    hedges_g: float
    ci_lower: float
    ci_upper: float
    power: float
    interpretation: str


def effect_size_report(
    observed: float,
    null_distribution: ArrayLike,
    n_subjects: int,
) -> EffectSizeReport:
    """Convenience factory: compute full EffectSizeReport from surrogate output.

    Treats the observed statistic as the "treatment group" scalar and the
    null distribution as the "control group" sample, then computes d, g,
    CI, and power.

    Parameters
    ----------
    observed:
        The observed test statistic (scalar).
    null_distribution:
        1-D array of surrogate / null statistics.
    n_subjects:
        Number of subjects / observations in the real data (used for power).

    Returns
    -------
    EffectSizeReport
    """
    null = np.asarray(null_distribution, dtype=float)
    if len(null) < 2:
        raise ValueError("null_distribution must have at least 2 elements.")

    # One-sample d: how many SDs is the observed value from the null mean?
    null_mean = float(np.mean(null))
    null_std = float(np.std(null, ddof=1))

    d = 0.0 if null_std <= 0.0 else (observed - null_mean) / null_std

    n_null = len(null)
    g = d * (1.0 - 3.0 / (4.0 * (1 + n_null) - 9.0))

    # CI: treat as one vs. n_null (n1=1 causes degenerate SE; use n_subjects instead)
    ci = confidence_interval_d(d, n1=n_subjects, n2=n_null)

    pwr = statistical_power(d, n=n_subjects)

    return EffectSizeReport(
        cohens_d=d,
        hedges_g=g,
        ci_lower=ci[0],
        ci_upper=ci[1],
        power=pwr,
        interpretation=_interpret(d),
    )
