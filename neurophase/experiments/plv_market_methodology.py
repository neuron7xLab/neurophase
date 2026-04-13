"""PLV methodology validation on REAL BTC OHLCV + synthetic neural phase.

Goal: prove that the repo's PLV + cyclic-shift-surrogate pipeline
(:func:`neurophase.metrics.plv.plv_on_held_out`) correctly detects
known phase coupling when the market side is a real, non-stationary
price series. This is a **methodology validation**, NOT a utility
claim about BTC, neural signals, or trading.

Pipeline:

    close prices (real Binance BTC/USDT 1m, one full day)
      -> neurophase.core.phase.compute_phase (standardize + D4 denoise + Hilbert)
      -> phi_market
    synthetic phase ladder indexed by coupling c in [0, 1]:
        d_phi_neural(t) = c * d_phi_market(t) + (1 - c) * xi(t)
        phi_neural      = cumsum(d_phi_neural)
      where xi is i.i.d. Gaussian.
    -> PLV on a held-out second half of the day
    -> cyclic-shift surrogate p-value (Phipson-Smyth smoothed)

Expected result matrix (ground truth is KNOWN, by construction):

    c = 1.0  ->  PLV close to 1.0, p < alpha (clearly coupled)
    c = 0.5  ->  PLV intermediate (mixed)
    c = 0.0  ->  PLV close to 1/sqrt(N_test), p >= alpha
                 (null — Gaussian-walk neural is not locked to the market)

The bundled CSV is ``data/market/btc_1m_sample.csv`` (public Binance
data, 2024-06-01 UTC, 1440 one-minute candles; provenance marker on
first line).

Run::

    python -m neurophase.experiments.plv_market_methodology

Output: ``results/plv_market_methodology_<YYYYMMDD>.json``.

Honest scope (must stay in-module):

* The market-side phase is extracted from real historical prices. It
  is not a live feed. It is a static fixture.
* The neural-side phase is SYNTHETIC. No claim is made that any real
  neural system tracks this particular coupling construction.
* A passing methodology sweep here proves nothing about whether any
  biological signal couples to any market. It proves only that the
  PLV pipeline itself detects coupling where it exists and rejects
  where it does not.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from neurophase.core.phase import compute_phase
from neurophase.metrics.plv import HeldOutSplit, plv_on_held_out

FloatArray = npt.NDArray[np.float64]

#: Path to the bundled real-market fixture. Relative to the repo root.
BTC_SAMPLE_PATH: Path = Path("data/market/btc_1m_sample.csv")

#: Default coupling ladder for the methodology sweep.
DEFAULT_COUPLINGS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Default surrogate count. Matches the repo's PLV default.
DEFAULT_N_SURROGATES: int = 1000

#: Alpha for significance labels in the sweep output.
DEFAULT_ALPHA: float = 0.05


@dataclass(frozen=True)
class MethodologyRow:
    coupling: float
    plv: float
    p_value: float
    significant: bool

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "coupling": self.coupling,
            "plv": self.plv,
            "p_value": self.p_value,
            "significant": self.significant,
        }


# ---------------------------------------------------------------------------
#   Data loading
# ---------------------------------------------------------------------------


def load_btc_close_prices(path: str | Path = BTC_SAMPLE_PATH) -> FloatArray:
    """Load BTC close prices from the bundled CSV.

    The file carries leading ``#``-comment provenance lines and a CSV
    header row; this loader skips the comments and reads only the
    ``close`` column.

    Raises :class:`FileNotFoundError` if the bundled sample is missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"BTC sample not found at {p}")
    closes: list[float] = []
    header_seen = False
    close_col: int | None = None
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if not header_seen:
            try:
                close_col = fields.index("close")
            except ValueError as exc:
                raise ValueError(f"{p}: header missing 'close' column") from exc
            header_seen = True
            continue
        if close_col is None:
            continue
        closes.append(float(fields[close_col]))
    if not closes:
        raise ValueError(f"{p}: no data rows after header")
    return np.asarray(closes, dtype=np.float64)


# ---------------------------------------------------------------------------
#   Synthetic coupled neural phase
# ---------------------------------------------------------------------------


def coupled_neural_phase(
    phi_market: FloatArray,
    *,
    coupling: float,
    rng: np.random.Generator,
    noise_sigma: float = 1.0,
) -> FloatArray:
    """Build a synthetic neural phase with a prescribed coupling strength.

    Construction is in the phase-difference domain:

        d_phi_neural(t) = coupling * d_phi_market(t) + (1 - coupling) * xi(t)
        phi_neural(t)   = cumsum(d_phi_neural(t))  + phi_market(0)

    where ``xi`` is i.i.d. ``N(0, noise_sigma)``. The scalar offset
    ``phi_market(0)`` anchors the two series at t=0; the PLV itself is
    invariant under that anchor.

    Parameters
    ----------
    phi_market
        Market phase series, shape (T,), in radians. Does NOT need to
        be unwrapped; internally we difference and diff-unwrap.
    coupling
        Scalar ``c`` in ``[0, 1]``. ``c=1`` => phi_neural equals
        phi_market (up to numerical roundoff); ``c=0`` => phi_neural
        is a pure random walk with variance ``noise_sigma ** 2`` per
        step.
    rng
        Deterministic ``numpy.random.Generator``.
    noise_sigma
        Scale of the i.i.d. Gaussian noise driving the neural walk.
        Defaults to 1 rad / step, which is the canonical matched
        scale for a random-walk phase.

    Returns
    -------
    FloatArray
        ``phi_neural``, shape (T,), in radians. Not wrapped; PLV does
        not require wrapping.
    """
    if not 0.0 <= coupling <= 1.0:
        raise ValueError(f"coupling must be in [0, 1], got {coupling!r}")
    if noise_sigma <= 0:
        raise ValueError(f"noise_sigma must be > 0, got {noise_sigma!r}")
    phi_market = np.asarray(phi_market, dtype=np.float64)
    if phi_market.ndim != 1 or phi_market.size < 2:
        raise ValueError(f"phi_market must be 1-D with >= 2 samples, got {phi_market.shape}")

    # Diff-unwrap to get the "true" per-step phase increment, then mix.
    d_market = np.diff(np.unwrap(phi_market))
    xi = rng.normal(loc=0.0, scale=noise_sigma, size=d_market.shape)
    d_neural = coupling * d_market + (1.0 - coupling) * xi
    # Cumulative sum puts us back in phase land.
    phi_neural = np.concatenate([[phi_market[0]], phi_market[0] + np.cumsum(d_neural)])
    return phi_neural.astype(np.float64)


# ---------------------------------------------------------------------------
#   Sweep runner
# ---------------------------------------------------------------------------


def _build_held_out_split(n: int) -> HeldOutSplit:
    """First-half train, second-half test. Honest temporal split."""
    half = n // 2
    return HeldOutSplit(
        train_indices=np.arange(half, dtype=np.int64),
        test_indices=np.arange(half, n, dtype=np.int64),
        total_length=n,
    )


def run_methodology_sweep(
    *,
    csv_path: str | Path = BTC_SAMPLE_PATH,
    couplings: tuple[float, ...] = DEFAULT_COUPLINGS,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full methodology sweep and return a structured report."""
    close = load_btc_close_prices(csv_path)
    phi_market = compute_phase(close)
    n = phi_market.size
    split = _build_held_out_split(n)

    rng = np.random.default_rng(seed)
    rows: list[MethodologyRow] = []
    for c in couplings:
        # Independent RNG per coupling to keep results reproducible
        # regardless of sweep order.
        per_rng = np.random.default_rng(seed + round(c * 1_000_000))
        phi_neural = coupled_neural_phase(phi_market, coupling=c, rng=per_rng)
        result = plv_on_held_out(
            phi_market,
            phi_neural,
            split=split,
            n_surrogates=n_surrogates,
            alpha=alpha,
            seed=seed,
        )
        rows.append(
            MethodologyRow(
                coupling=c,
                plv=result.plv,
                p_value=result.p_value,
                significant=result.significant,
            )
        )
        # Drain any spare entropy (keeps rng determinism across versions).
        _ = rng.standard_normal(1)

    return {
        "experiment": "plv_market_methodology",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset": "Binance public BTC/USDT 1m, 2024-06-01 UTC",
        "dataset_path": str(csv_path),
        "n_samples": int(n),
        "held_out_train_size": int(split.train_indices.size),
        "held_out_test_size": int(split.test_indices.size),
        "n_surrogates": n_surrogates,
        "alpha": alpha,
        "seed": seed,
        "noise_sigma": 1.0,
        "rows": [r.to_json_dict() for r in rows],
        "scope": (
            "METHODOLOGY VALIDATION ONLY. Market-side phase is real "
            "historical BTC price; neural-side phase is synthetic with "
            "known coupling c in [0, 1]. A pass here proves the PLV + "
            "surrogate pipeline detects coupling where it exists and "
            "rejects where it does not. It proves NOTHING about whether "
            "any biological signal couples to any market."
        ),
    }


def save_report(report: dict[str, Any], output_dir: str | Path = "results") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d")
    path = out / f"plv_market_methodology_{date_str}.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
#   CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neurophase.experiments.plv_market_methodology",
        description=(
            "PLV methodology validation on REAL BTC OHLCV + synthetic "
            "neural phase. NOT a utility claim."
        ),
    )
    p.add_argument("--csv", type=Path, default=BTC_SAMPLE_PATH)
    p.add_argument(
        "--couplings",
        type=float,
        nargs="+",
        default=list(DEFAULT_COUPLINGS),
        help="Coupling ladder (values in [0, 1]).",
    )
    p.add_argument("--n-surrogates", type=int, default=DEFAULT_N_SURROGATES)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    report = run_methodology_sweep(
        csv_path=args.csv,
        couplings=tuple(args.couplings),
        n_surrogates=args.n_surrogates,
        alpha=args.alpha,
        seed=args.seed,
    )
    path = save_report(report, output_dir=args.output_dir)
    print(f"Results written to: {path}")
    print()
    print("Coupling sweep:")
    print(f"  {'c':>6}  {'plv':>8}  {'p':>8}  {'sig':>5}")
    for row in report["rows"]:
        marker = "*" if row["significant"] else ""
        print(f"  {row['coupling']:>6.3f}  {row['plv']:>8.4f}  {row['p_value']:>8.4f}  {marker:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
