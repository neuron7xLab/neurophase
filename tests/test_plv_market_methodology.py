"""Tests for neurophase.experiments.plv_market_methodology.

The experiment is a methodology validation: it must (a) run on the
bundled REAL BTC sample, (b) report the perfect-coupling case
(c=1.0) as significant, (c) report the null case (c=0.0) as NOT
significant, and (d) produce a structurally well-formed output dict.

These tests are mandatory for any future change that touches either
the real-market fixture or the coupled-neural construction — silent
drift of the ground-truth matrix is exactly what this file defends
against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurophase.experiments.plv_market_methodology import (
    BTC_SAMPLE_PATH,
    MethodologyRow,
    coupled_neural_phase,
    load_btc_close_prices,
    run_methodology_sweep,
    save_report,
)

# ---------------------------------------------------------------------------
#   Data fixture + loader
# ---------------------------------------------------------------------------


class TestBundledFixture:
    def test_bundled_csv_present(self) -> None:
        assert BTC_SAMPLE_PATH.exists(), (
            "bundled BTC sample missing; the methodology experiment has no data to validate against"
        )

    def test_bundled_csv_has_provenance_marker(self) -> None:
        first_line = BTC_SAMPLE_PATH.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("#")
        upper = first_line.upper()
        assert "BINANCE" in upper
        assert "PUBLIC" in upper

    def test_load_btc_close_prices_shape(self) -> None:
        close = load_btc_close_prices()
        # 1440 minutes in a UTC trading day.
        assert close.shape == (1440,)
        assert close.dtype == np.float64
        # Sanity: BTC in 2024-06 range was roughly [60_000, 80_000] USD.
        assert 50_000.0 < close.min() < close.max() < 90_000.0

    def test_load_rejects_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_btc_close_prices(tmp_path / "does-not-exist.csv")


# ---------------------------------------------------------------------------
#   Coupled neural phase constructor
# ---------------------------------------------------------------------------


class TestCoupledNeuralPhase:
    def test_perfect_coupling_reproduces_market_phase(self) -> None:
        """c=1.0 must reproduce the market phase series up to numerical
        noise. This is the ground-truth anchor of the whole matrix."""
        rng = np.random.default_rng(0)
        phi_market = np.linspace(0.0, 20.0, 256)  # monotone phase
        phi_neural = coupled_neural_phase(phi_market, coupling=1.0, rng=rng)
        # c=1 -> d_neural == d_market -> cumsum matches phi_market.
        np.testing.assert_allclose(phi_neural, phi_market, atol=1e-12)

    def test_zero_coupling_is_pure_random_walk(self) -> None:
        """c=0.0 must not depend on the market phase at all (the
        contribution of d_market is scaled to zero)."""
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        phi_market_a = np.linspace(0.0, 10.0, 128)
        phi_market_b = np.linspace(0.0, 50.0, 128)  # very different
        neural_a = coupled_neural_phase(phi_market_a, coupling=0.0, rng=rng_a)
        neural_b = coupled_neural_phase(phi_market_b, coupling=0.0, rng=rng_b)
        # Offsets differ (anchored at phi_market[0]) but the *increments*
        # must be identical under identical rngs.
        np.testing.assert_allclose(np.diff(neural_a), np.diff(neural_b), atol=1e-12)

    def test_rejects_coupling_out_of_bounds(self) -> None:
        rng = np.random.default_rng(0)
        phi = np.linspace(0.0, 1.0, 64)
        with pytest.raises(ValueError, match=r"coupling"):
            coupled_neural_phase(phi, coupling=-0.01, rng=rng)
        with pytest.raises(ValueError, match=r"coupling"):
            coupled_neural_phase(phi, coupling=1.01, rng=rng)

    def test_rejects_non_positive_noise_sigma(self) -> None:
        rng = np.random.default_rng(0)
        phi = np.linspace(0.0, 1.0, 64)
        with pytest.raises(ValueError, match=r"noise_sigma"):
            coupled_neural_phase(phi, coupling=0.5, rng=rng, noise_sigma=0.0)

    def test_rejects_short_or_nd_input(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match=r"phi_market"):
            coupled_neural_phase(np.array([1.0]), coupling=0.5, rng=rng)
        with pytest.raises(ValueError, match=r"phi_market"):
            coupled_neural_phase(np.zeros((4, 4)), coupling=0.5, rng=rng)


# ---------------------------------------------------------------------------
#   Full sweep on the bundled fixture
# ---------------------------------------------------------------------------


class TestFullSweep:
    @pytest.fixture(scope="class")
    def sweep_report(self) -> dict[str, object]:
        """Run the sweep ONCE for the class (1000 surrogates is the slow
        step; parametrising individual assertions over the same report
        keeps this test file under a couple of seconds)."""
        return run_methodology_sweep(n_surrogates=1000, seed=42)

    def test_report_shape(self, sweep_report: dict[str, object]) -> None:
        for key in (
            "experiment",
            "timestamp_utc",
            "dataset",
            "dataset_path",
            "n_samples",
            "n_surrogates",
            "alpha",
            "rows",
            "scope",
        ):
            assert key in sweep_report, f"missing key {key!r} in report"
        rows = sweep_report["rows"]
        assert isinstance(rows, list)
        # Default coupling ladder has 5 rungs (0.0, 0.25, 0.5, 0.75, 1.0).
        assert len(rows) == 5

    def test_perfect_coupling_is_significant(self, sweep_report: dict[str, object]) -> None:
        rows = sweep_report["rows"]
        assert isinstance(rows, list)
        c1 = next(r for r in rows if r["coupling"] == 1.0)
        assert c1["significant"] is True, c1
        assert c1["plv"] > 0.95, c1  # trivially near 1.0 by construction
        assert c1["p_value"] < 0.05, c1

    def test_zero_coupling_is_not_significant(self, sweep_report: dict[str, object]) -> None:
        rows = sweep_report["rows"]
        assert isinstance(rows, list)
        c0 = next(r for r in rows if r["coupling"] == 0.0)
        assert c0["significant"] is False, c0
        assert c0["p_value"] >= 0.05, c0

    def test_scope_disclaimer_present(self, sweep_report: dict[str, object]) -> None:
        """The experiment's honest-scope statement MUST travel with every
        generated report. Removing it would let a downstream reader
        mistake this for a utility claim."""
        scope = sweep_report["scope"]
        assert isinstance(scope, str)
        assert "METHODOLOGY VALIDATION ONLY" in scope
        assert "NOTHING" in scope  # the absolute disclaimer


# ---------------------------------------------------------------------------
#   save_report round-trip
# ---------------------------------------------------------------------------


def test_save_report_round_trip(tmp_path: Path) -> None:
    report = {"experiment": "plv_market_methodology", "rows": [], "scope": "x"}
    path = save_report(report, output_dir=tmp_path)
    assert path.exists()
    import json

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["experiment"] == "plv_market_methodology"


# ---------------------------------------------------------------------------
#   MethodologyRow dataclass contract
# ---------------------------------------------------------------------------


def test_methodology_row_to_json_dict() -> None:
    row = MethodologyRow(coupling=0.5, plv=0.3, p_value=0.1, significant=False)
    d = row.to_json_dict()
    assert d == {"coupling": 0.5, "plv": 0.3, "p_value": 0.1, "significant": False}
