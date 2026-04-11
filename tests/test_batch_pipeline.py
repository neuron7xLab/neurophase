"""Tests for the ``StreamingPipeline.tick_batch`` vectorised API."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neurophase.audit.decision_ledger import verify_ledger
from neurophase.runtime.pipeline import (
    PipelineConfig,
    StreamingPipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_inputs() -> pd.DataFrame:
    """The canonical healthy sequence used across tests."""
    return pd.DataFrame(
        {
            "timestamp": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "R": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.99, 0.30],
            "delta": [0.01] * 8,
        }
    )


def _config(**overrides: object) -> PipelineConfig:
    defaults: dict[str, object] = {
        "warmup_samples": 2,
        "stream_window": 4,
        "max_fault_rate": 0.50,
        "enable_stillness": False,
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Required-column validation.
# ---------------------------------------------------------------------------


class TestColumnContract:
    def test_rejects_missing_timestamp(self) -> None:
        p = StreamingPipeline(_config())
        bad = pd.DataFrame({"R": [0.9, 0.9]})
        with pytest.raises(ValueError, match="required columns"):
            p.tick_batch(bad)

    def test_rejects_missing_R(self) -> None:
        p = StreamingPipeline(_config())
        bad = pd.DataFrame({"timestamp": [0.0, 0.1]})
        with pytest.raises(ValueError, match="required columns"):
            p.tick_batch(bad)

    def test_accepts_missing_optional_columns(self) -> None:
        p = StreamingPipeline(_config())
        only_required = pd.DataFrame({"timestamp": [0.0, 0.1], "R": [0.9, 0.9]})
        out = p.tick_batch(only_required)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Semantic parity with the serial path — the load-bearing contract.
# ---------------------------------------------------------------------------


class TestSerialParity:
    def test_batch_equals_serial_gate_state_sequence(self) -> None:
        inputs = _canonical_inputs()

        # Serial path.
        p_serial = StreamingPipeline(_config())
        serial_states: list[str] = []
        for _, row in inputs.iterrows():
            frame = p_serial.tick(
                timestamp=float(row["timestamp"]),
                R=float(row["R"]),
                delta=float(row["delta"]),
            )
            serial_states.append(frame.gate_state.name)

        # Batch path (fresh pipeline).
        p_batch = StreamingPipeline(_config())
        batch_df = p_batch.tick_batch(inputs)
        batch_states = list(batch_df["gate_state"])

        assert serial_states == batch_states

    def test_batch_produces_same_tick_indices(self) -> None:
        inputs = _canonical_inputs()
        p = StreamingPipeline(_config())
        out = p.tick_batch(inputs)
        assert list(out["tick_index"]) == list(range(len(inputs)))

    def test_batch_preserves_pipeline_state(self) -> None:
        """Calling ``tick_batch`` twice in a row must continue from the
        rolling state — not reset it. The second call sees a warmed-up
        pipeline."""
        p = StreamingPipeline(_config())

        first_half = _canonical_inputs().iloc[:4].reset_index(drop=True)
        second_half = _canonical_inputs().iloc[4:].reset_index(drop=True)

        out_a = p.tick_batch(first_half)
        out_b = p.tick_batch(second_half)

        # Compare to a single pass through the whole thing on a fresh pipeline.
        p_full = StreamingPipeline(_config())
        out_full = p_full.tick_batch(_canonical_inputs())

        combined_states = list(out_a["gate_state"]) + list(out_b["gate_state"])
        full_states = list(out_full["gate_state"])
        assert combined_states == full_states


# ---------------------------------------------------------------------------
# Output schema.
# ---------------------------------------------------------------------------


class TestOutputSchema:
    def test_output_columns(self) -> None:
        p = StreamingPipeline(_config())
        out = p.tick_batch(_canonical_inputs())
        expected = {
            "tick_index",
            "timestamp",
            "R",
            "delta",
            "gate_state",
            "execution_allowed",
            "time_quality",
            "stream_regime",
            "stream_fault_rate",
            "gate_reason",
            "ledger_record_hash",
        }
        assert expected <= set(out.columns)

    def test_output_row_count_matches_input(self) -> None:
        p = StreamingPipeline(_config())
        out = p.tick_batch(_canonical_inputs())
        assert len(out) == len(_canonical_inputs())

    def test_ledger_hash_column_is_none_without_ledger(self) -> None:
        p = StreamingPipeline(_config())
        out = p.tick_batch(_canonical_inputs())
        assert out["ledger_record_hash"].isna().all()


# ---------------------------------------------------------------------------
# NaN / missing handling in the R column.
# ---------------------------------------------------------------------------


class TestMissingValues:
    def test_nan_R_is_treated_as_none(self) -> None:
        p = StreamingPipeline(_config())
        inputs = pd.DataFrame(
            {
                "timestamp": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "R": [0.9, 0.9, 0.9, 0.9, 0.9, float("nan")],
                "delta": [0.01] * 6,
            }
        )
        out = p.tick_batch(inputs)
        assert out["gate_state"].iloc[-1] == "DEGRADED"

    def test_missing_delta_is_handled(self) -> None:
        p = StreamingPipeline(_config(enable_stillness=True))
        inputs = pd.DataFrame(
            {
                "timestamp": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "R": [0.9, 0.9, 0.9, 0.9, 0.9, 0.99],
                "delta": [0.01, 0.01, 0.01, 0.01, 0.01, float("nan")],
            }
        )
        out = p.tick_batch(inputs)
        # NaN delta should route through the "stillness skipped" path
        # and still produce a READY frame for the last tick.
        assert out["gate_state"].iloc[-1] in {"READY", "UNNECESSARY"}


# ---------------------------------------------------------------------------
# Ledger byte-identical replay with batch API.
# ---------------------------------------------------------------------------


class TestBatchLedgerCompat:
    def test_batch_ledger_is_byte_identical_to_serial_ledger(self, tmp_path: Path) -> None:
        """Serial ticks vs batch ticks against attached ledgers — the
        two resulting files must be byte-identical. This is the
        strongest possible parity contract."""
        inputs = _canonical_inputs()

        serial_path = tmp_path / "serial.jsonl"
        p_serial = StreamingPipeline(_config(ledger_path=serial_path))
        for _, row in inputs.iterrows():
            p_serial.tick(
                timestamp=float(row["timestamp"]),
                R=float(row["R"]),
                delta=float(row["delta"]),
            )

        batch_path = tmp_path / "batch.jsonl"
        p_batch = StreamingPipeline(_config(ledger_path=batch_path))
        p_batch.tick_batch(inputs)

        assert serial_path.read_bytes() == batch_path.read_bytes()

        v_serial = verify_ledger(serial_path)
        v_batch = verify_ledger(batch_path)
        assert v_serial.ok
        assert v_batch.ok
        assert v_serial.n_records == v_batch.n_records == len(inputs)


# ---------------------------------------------------------------------------
# Extra-column tolerance.
# ---------------------------------------------------------------------------


class TestExtraColumns:
    def test_ignores_unknown_columns(self) -> None:
        """Extra columns in the input DataFrame are silently ignored."""
        p = StreamingPipeline(_config())
        inputs = _canonical_inputs().assign(psi_brain=[0.1] * 8, scratch=[42] * 8)
        out = p.tick_batch(inputs)
        assert len(out) == len(inputs)
