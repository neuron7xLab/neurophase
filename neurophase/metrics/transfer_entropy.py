"""Transfer entropy TE(X → Y) — directed information flow between two series.

    TE(X → Y) = Σ p(y_{t+1}, y_t^{(k)}, x_t^{(k)}) · log
                    [ p(y_{t+1} | y_t^{(k)}, x_t^{(k)})
                      / p(y_{t+1} | y_t^{(k)}) ]

Transfer entropy answers the question symmetric synchrony metrics (PLV,
iPLV, ISM, correlation) cannot: *given both series' pasts, does the
past of X reduce uncertainty about the next value of Y beyond what the
past of Y already knows?* It is model-free, non-parametric, and
strictly non-negative.

In ``neurophase`` the canonical use is the trader–market coupling:

    TE(market → trader) − TE(trader → market)

is a scalar **directional** signal with the same sign convention as the
classical "lead–lag". Unlike cross-correlation it survives non-linear
coupling and does not require either series to be Gaussian.

Estimator
---------
Plug-in TE on histogram-quantised symbol sequences. Equal-frequency
(quantile) binning keeps the joint state space populated. ``k`` is the
history depth per variable; ``n_levels`` is the per-variable alphabet
size. Joint counts are accumulated via ``numpy.bincount`` over a single
base-``n_levels`` integer encoding of the triple
``(y_{t+1}, y_t^{(k)}, x_t^{(k)})``.

No additive smoothing is required for numerical stability: the
log-ratio is evaluated only over observed joint cells, and
``c(y, h_y, h_x) > 0`` implies both ``c(y, h_y) > 0`` and
``c(h_y, h_x) > 0``.

Finite-sample bias
------------------
Plug-in TE is positively biased for finite samples — even independent
series yield a small positive estimate. The surrogate-aware entry
:func:`transfer_entropy_with_significance` estimates and subtracts that
bias via the mean TE of **circular-shift** surrogates, which preserve
each series' marginal distribution while destroying cross-temporal
coupling. The same surrogates yield a one-sided empirical p-value.

Units
-----
Nats (natural logarithm), matching :mod:`neurophase.metrics.entropy`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Cap the joint state-space code range: n_levels^(2k+1) must fit safely
# into int64 bincount indices. 2**40 leaves ample headroom while
# rejecting pathological k / n_levels combinations early.
_MAX_STATE_SPACE = 1 << 40


@dataclass(frozen=True)
class TEResult:
    """Bidirectional transfer entropy with bias correction and p-values.

    Attributes
    ----------
    te_xy, te_yx : float
        Bias-corrected transfer entropies ``TE(X → Y)`` and ``TE(Y → X)``
        in nats. Non-negative by construction.
    te_net : float
        ``te_xy − te_yx`` — the directional coupling asymmetry. Positive
        means X leads Y; negative means Y leads X.
    p_xy, p_yx : float
        One-sided empirical p-values against the circular-shift null.
    n_surrogates : int
        Number of surrogate shuffles used for the null distribution.
    k : int
        History depth per variable.
    n_levels : int
        Per-variable alphabet size used during quantisation.
    """

    te_xy: float
    te_yx: float
    te_net: float
    p_xy: float
    p_yx: float
    n_surrogates: int
    k: int
    n_levels: int


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------


def _quantise(series: FloatArray, n_levels: int) -> IntArray:
    """Equal-frequency (quantile) quantisation into ``n_levels`` symbols.

    Binary case is the median split; higher ``n_levels`` use interior
    quantile cuts. Duplicate cut points — from ties or near-constant
    input — collapse, so the effective alphabet may be smaller than
    ``n_levels``. This is intentional: a quantiser that invents
    distinctions that do not exist in the data produces spurious TE.
    """
    if n_levels == 2:
        median = float(np.median(series))
        return (series > median).astype(np.int64)
    cuts = np.quantile(series, np.linspace(0.0, 1.0, n_levels + 1)[1:-1])
    cuts = np.unique(cuts)
    return np.digitize(series, cuts).astype(np.int64)


# ---------------------------------------------------------------------------
# Core plug-in estimator
# ---------------------------------------------------------------------------


def _plug_in_te(source: IntArray, target: IntArray, k: int, n_levels: int) -> float:
    """Plug-in TE on already-quantised integer sequences."""
    T = target.size
    n_samples = T - k
    if n_samples <= 1:
        return 0.0

    hist_base = n_levels**k

    # Length-k history codes via Horner-style base-n_levels accumulation.
    tgt_hist = np.zeros(n_samples, dtype=np.int64)
    src_hist = np.zeros(n_samples, dtype=np.int64)
    for lag in range(k):
        tgt_hist = tgt_hist * n_levels + target[k - 1 - lag : T - 1 - lag]
        src_hist = src_hist * n_levels + source[k - 1 - lag : T - 1 - lag]
    y_future = target[k:]

    joint_yhx = (y_future * hist_base + tgt_hist) * hist_base + src_hist
    joint_yh = y_future * hist_base + tgt_hist
    joint_hx = tgt_hist * hist_base + src_hist

    c_yhx = np.bincount(joint_yhx)
    c_yh = np.bincount(joint_yh)
    c_hx = np.bincount(joint_hx)
    c_h = np.bincount(tgt_hist)

    nz = np.flatnonzero(c_yhx)
    if nz.size == 0:
        return 0.0

    # Decode composite indices into their parts — cheaper than materialising
    # three parallel arrays during the forward pass.
    src_code = nz % hist_base
    rem = nz // hist_base
    tgt_code = rem % hist_base
    y_code = rem // hist_base

    yh_codes = y_code * hist_base + tgt_code
    hx_codes = tgt_code * hist_base + src_code

    p_yhx = c_yhx[nz].astype(np.float64) / float(n_samples)
    # log( p(y | h_y, h_x) / p(y | h_y) ) = log( c_yhx · c_h / (c_yh · c_hx) ).
    # All four counts are strictly positive on the observed support ``nz``.
    # The combined-ratio form collapses to log(1) ≡ 0 exactly when the
    # numerator equals the denominator (self-TE); the arithmetically
    # separated log-sum form leaks ~1e-16 per cell from summation noise.
    ratio = (
        c_yhx[nz].astype(np.float64)
        * c_h[tgt_code].astype(np.float64)
        / (c_yh[yh_codes].astype(np.float64) * c_hx[hx_codes].astype(np.float64))
    )
    te = float(np.sum(p_yhx * np.log(ratio)))
    # Plug-in TE is non-negative in expectation; clamp numerical noise.
    return max(te, 0.0)


def _prepare(
    source: npt.ArrayLike, target: npt.ArrayLike, k: int, n_levels: int
) -> tuple[IntArray, IntArray] | None:
    """Validate inputs and return quantised (source, target) or None.

    Returns None on honest-null inputs (length < k + 2) so the public
    entry points can surface 0.0 cleanly without repeating the guard.
    """
    if k < 1:
        raise ValueError(f"k must be ≥ 1; got {k!r}")
    if n_levels < 2:
        raise ValueError(f"n_levels must be ≥ 2; got {n_levels!r}")
    if n_levels ** (2 * k + 1) > _MAX_STATE_SPACE:
        raise ValueError(
            f"joint state space n_levels**(2k+1) = {n_levels ** (2 * k + 1)} "
            f"exceeds safety cap {_MAX_STATE_SPACE}; reduce k or n_levels"
        )

    src = np.asarray(source, dtype=np.float64).ravel()
    tgt = np.asarray(target, dtype=np.float64).ravel()
    if src.shape != tgt.shape:
        raise ValueError(f"source and target must share shape; got {src.shape} vs {tgt.shape}")
    if src.size < k + 2:
        return None
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(tgt))):
        raise ValueError("source and target must be finite")

    return _quantise(src, n_levels), _quantise(tgt, n_levels)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transfer_entropy(
    source: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    k: int = 1,
    n_levels: int = 2,
) -> float:
    """Plug-in transfer entropy ``TE(source → target)`` in nats.

    Parameters
    ----------
    source, target : array_like
        Scalar series of identical length. Must be finite and have at
        least ``k + 2`` samples for a non-trivial estimate.
    k : int
        History depth per variable (default 1).
    n_levels : int
        Per-variable alphabet size used during quantile quantisation
        (default 2 — binary median split, the most robust choice).

    Returns
    -------
    float
        ``TE(source → target)`` in nats. Returns ``0.0`` on honest-null
        inputs (insufficient samples). Does **not** correct for
        finite-sample bias; use
        :func:`transfer_entropy_with_significance` for a de-biased
        estimate and a surrogate p-value.
    """
    prepared = _prepare(source, target, k, n_levels)
    if prepared is None:
        return 0.0
    src_q, tgt_q = prepared
    return _plug_in_te(src_q, tgt_q, k, n_levels)


def transfer_entropy_with_significance(
    source: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    k: int = 1,
    n_levels: int = 2,
    n_surrogates: int = 200,
    bias_correct: bool = True,
    seed: int | None = None,
) -> TEResult:
    """Bidirectional TE with circular-shift surrogate bias correction.

    For each direction the surrogate null is built by circularly shifting
    the *source* series by a random offset ``δ ∈ [k + 1, T − k − 1]``.
    This preserves each series' marginal distribution while destroying
    temporal coupling. The bias-corrected estimate is

        TE_corrected = max(0, TE_raw − ⟨TE_surrogate⟩)

    and the one-sided p-value is the fraction of surrogate TEs that
    meet or exceed the raw estimate.

    Parameters
    ----------
    source, target : array_like
        Scalar series of identical length. Interpreted here as X and Y
        in ``TE(X → Y)`` and ``TE(Y → X)``.
    k : int
        History depth.
    n_levels : int
        Quantisation alphabet size.
    n_surrogates : int
        Number of circular-shift surrogates per direction. ≥ 1.
    bias_correct : bool
        Subtract ``⟨TE_surrogate⟩`` from the raw estimate when True.
    seed : int | None
        Seed for the surrogate RNG. ``None`` uses fresh non-determinism.

    Returns
    -------
    TEResult
        Bias-corrected TEs, net flow, p-values, and the configuration
        used. Degenerate inputs yield an all-zero result with
        ``p_xy = p_yx = 1.0``.
    """
    if n_surrogates < 1:
        raise ValueError(f"n_surrogates must be ≥ 1; got {n_surrogates!r}")

    prepared = _prepare(source, target, k, n_levels)
    if prepared is None:
        return TEResult(
            te_xy=0.0,
            te_yx=0.0,
            te_net=0.0,
            p_xy=1.0,
            p_yx=1.0,
            n_surrogates=n_surrogates,
            k=k,
            n_levels=n_levels,
        )
    src_q, tgt_q = prepared
    T = src_q.size

    te_xy_raw = _plug_in_te(src_q, tgt_q, k, n_levels)
    te_yx_raw = _plug_in_te(tgt_q, src_q, k, n_levels)

    rng = np.random.default_rng(seed)
    low, high = k + 1, max(k + 2, T - k)  # inclusive low, exclusive high
    sur_xy = np.empty(n_surrogates, dtype=np.float64)
    sur_yx = np.empty(n_surrogates, dtype=np.float64)
    for s in range(n_surrogates):
        shift = int(rng.integers(low, high))
        src_shifted = np.roll(src_q, shift)
        tgt_shifted = np.roll(tgt_q, shift)
        sur_xy[s] = _plug_in_te(src_shifted, tgt_q, k, n_levels)
        sur_yx[s] = _plug_in_te(tgt_shifted, src_q, k, n_levels)

    te_xy = max(te_xy_raw - float(np.mean(sur_xy)), 0.0) if bias_correct else te_xy_raw
    te_yx = max(te_yx_raw - float(np.mean(sur_yx)), 0.0) if bias_correct else te_yx_raw

    # One-sided empirical p-value with the classical (count + 1)/(N + 1)
    # correction so the null never produces an impossible p = 0.
    p_xy = float((np.sum(sur_xy >= te_xy_raw) + 1) / (n_surrogates + 1))
    p_yx = float((np.sum(sur_yx >= te_yx_raw) + 1) / (n_surrogates + 1))

    return TEResult(
        te_xy=te_xy,
        te_yx=te_yx,
        te_net=te_xy - te_yx,
        p_xy=p_xy,
        p_yx=p_yx,
        n_surrogates=n_surrogates,
        k=k,
        n_levels=n_levels,
    )


__all__ = [
    "TEResult",
    "transfer_entropy",
    "transfer_entropy_with_significance",
]
