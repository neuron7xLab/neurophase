"""Tests for ds003458 real-data analysis pipeline.

These tests validate the loader, preprocessor, and analysis pipeline
structure. Tests that require actual EEG data are skipped if the
dataset is not downloaded.

Invariants:
    DS-I1: Pre-registration commit exists before analysis
    DS-I2: Per-subject verdict JSON saved before group test
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest


def _dataset_available() -> bool:
    """Check if ds003458 has at least one subject downloaded."""
    p = Path("data/ds003458")
    if not p.exists():
        return False
    subs = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    return len(subs) > 0


SKIP_NO_DATA = pytest.mark.skipif(
    not _dataset_available(),
    reason="ds003458 not downloaded",
)


class TestPreregistration:
    def test_preregistration_frozen(self) -> None:
        """DS-I1: Pre-registration file exists and is committed."""
        prereg = Path("results/ds003458_preregistration.md")
        assert prereg.exists(), "Pre-registration file not found"

        # Check it's tracked by git
        result = subprocess.run(
            ["git", "log", "--oneline", "--", str(prereg)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0, "Pre-registration not committed to git"

    def test_preregistration_contains_required_fields(self) -> None:
        """Pre-registration has all required analysis parameters."""
        prereg = Path("results/ds003458_preregistration.md")
        text = prereg.read_text()
        required = [
            "8 Hz",  # FMθ band upper bound
            "Fz",  # channel
            "PPC",  # primary metric
            "0.05",  # significance
            "Held-out",  # held-out discipline
            "three-gate",  # verdict
        ]
        for field in required:
            assert field.lower() in text.lower(), f"Pre-registration missing: {field}"


class TestRewardProbabilities:
    def test_oscillation_formula(self) -> None:
        """Reward probabilities follow the documented sinusoidal formula."""
        from neurophase.data.ds003458_loader import _compute_reward_probabilities

        probs = _compute_reward_probabilities(480)
        assert probs.shape == (3, 480)
        # All probabilities in [0.2, 1.0] (0.6 ± 0.4)
        assert float(np.min(probs)) >= 0.19
        assert float(np.max(probs)) <= 1.01
        # Check oscillation: should have roughly 2 cycles in 480 trials
        lo = probs[0]
        # Count zero crossings of demeaned signal
        demeaned = lo - float(np.mean(lo))
        crossings = np.sum(np.diff(np.sign(demeaned)) != 0)
        # ~4 crossings = ~2 cycles
        assert 2 <= crossings <= 8, f"Expected ~4 crossings, got {crossings}"


@SKIP_NO_DATA
class TestLoaderIntegration:
    def test_load_subject(self) -> None:
        from neurophase.data.ds003458_loader import DS003458Loader

        loader = DS003458Loader("data/ds003458")
        subjects = loader.list_subjects()
        assert len(subjects) >= 1

        sub = loader.load_subject(subjects[0])
        assert sub.fs == 500.0
        assert sub.reward_prob_chosen.shape[0] > 400
        assert sub.trial_onsets_sec.shape[0] > 400
        assert np.all(np.isfinite(sub.reward_prob_chosen))


@SKIP_NO_DATA
class TestPreprocessorIntegration:
    def test_extract_phases(self) -> None:
        from neurophase.data.ds003458_loader import DS003458Loader
        from neurophase.data.eeg_preprocessor import extract_phases

        loader = DS003458Loader("data/ds003458")
        sub = loader.load_subject(loader.list_subjects()[0])
        phases = extract_phases(sub)

        assert phases.phi_neural.shape == phases.phi_market.shape
        assert phases.n_samples > 10000
        assert np.all(np.isfinite(phases.phi_neural))
        assert np.all(np.isfinite(phases.phi_market))


@SKIP_NO_DATA
class TestVerdictIntegration:
    def test_verdict_runs(self) -> None:
        from neurophase.data.ds003458_loader import DS003458Loader
        from neurophase.data.eeg_preprocessor import extract_phases
        from neurophase.metrics.plv_verdict import compute_verdict

        loader = DS003458Loader("data/ds003458")
        sub = loader.load_subject(loader.list_subjects()[0])
        phases = extract_phases(sub)

        verdict = compute_verdict(
            phases.phi_neural,
            phases.phi_market,
            coupling_k=None,
            n_surrogates=50,
            seed=42,
        )
        assert verdict.verdict in {"CONFIRMED", "MARGINAL", "REJECTED"}
        assert 0.0 <= verdict.ppc <= 1.0
        assert 0.0 <= verdict.rayleigh_r <= 1.0


class TestAnalysisStructure:
    def test_results_saved(self, tmp_path: Path) -> None:
        """DS-I2: save_results produces valid JSON."""
        from neurophase.experiments.ds003458_analysis import save_results

        mock_results = {
            "experiment": "ds003458_analysis",
            "n_confirmed": 5,
            "n_marginal": 10,
            "n_rejected": 8,
            "evidence_status": "Tentative (unchanged)",
            "per_subject": [],
        }
        path = save_results(mock_results, output_dir=tmp_path)
        assert path.exists()
        import json

        with open(path) as f:
            data = json.load(f)
        assert data["experiment"] == "ds003458_analysis"
