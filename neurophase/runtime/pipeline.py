"""E1 + E2 — stateful streaming pipeline + DecisionFrame contract.

The runtime pipeline is the **single correct way** for callers to
drive the gate online. It composes:

1. :class:`~neurophase.data.temporal_validator.TemporalValidator`
   (B1) — per-sample temporal contract check.
2. :class:`~neurophase.data.stream_detector.TemporalStreamDetector`
   (B2 + B6) — rolling stream-level regime classification.
3. :class:`~neurophase.gate.execution_gate.ExecutionGate` with an
   optional :class:`~neurophase.gate.stillness_detector.StillnessDetector`
   (I₁–I₄) — the 5-state gate.
4. Optional :class:`~neurophase.audit.decision_ledger.DecisionTraceLedger`
   (F1) — append-only SHA256-chained audit log.

Everything the caller needs — invariant enforcement, temporal
integrity, stillness classification, replay-ready provenance — is
bolted together inside one ``tick`` call. The pipeline never
exposes intermediate module state; the caller reads the output as a
single immutable :class:`DecisionFrame`.

Design notes
------------

* **Stateful.** Each tick advances the validator, stream detector,
  and stillness buffers in lockstep. The pipeline owns the state
  and is the only correct path for its mutation.
* **Deterministic.** Two pipelines configured identically and fed
  the same ``(ts, R, delta)`` stream emit identical
  ``DecisionFrame`` sequences **including** identical ledger
  record hashes. This is covered by the F3 certification suite.
* **No policy layer.** The pipeline stops at the gate decision —
  it does not size positions, throttle actions, or act on external
  systems. Those are Program I tasks and must be layered above, not
  inside.
* **No SciPy.** All underlying modules are numpy-only; the pipeline
  itself adds no numerics.
* **Optional ledger.** If ``PipelineConfig.ledger_path`` is set,
  each tick emits a record into a ``DecisionTraceLedger`` using the
  pipeline's parameter fingerprint. If absent, the ledger layer is
  skipped and the ``DecisionFrame.ledger_record`` field is ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    import pandas as pd

from neurophase.audit.decision_ledger import (
    DecisionTraceLedger,
    DecisionTraceRecord,
    fingerprint_parameters,
)
from neurophase.data.stream_detector import (
    DEFAULT_MAX_FAULT_RATE,
    DEFAULT_STREAM_WINDOW,
    StreamQualityDecision,
    StreamRegime,
    TemporalStreamDetector,
)
from neurophase.data.temporal_validator import (
    DEFAULT_MAX_GAP_SECONDS,
    TemporalQualityDecision,
    TemporalValidator,
    TimeQuality,
)
from neurophase.gate.execution_gate import (
    DEFAULT_THRESHOLD,
    ExecutionGate,
    GateDecision,
    GateState,
)
from neurophase.gate.stillness_detector import StillnessDetector

#: Default runtime warmup samples for the temporal validator layer.
DEFAULT_PIPELINE_WARMUP: Final[int] = 4


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for a :class:`StreamingPipeline`.

    Attributes
    ----------
    threshold
        Gate threshold passed to :class:`ExecutionGate`.
    max_gap_seconds
        Gap tolerance for :class:`TemporalValidator`.
    warmup_samples
        Warmup window for :class:`TemporalValidator`.
    stream_window
        Window size for :class:`TemporalStreamDetector`.
    max_fault_rate
        Fault-rate threshold for :class:`TemporalStreamDetector`.
    stream_hold_steps
        Hysteresis hold length for the stream detector.
    enable_stillness
        When ``True`` the pipeline attaches a :class:`StillnessDetector`
        to the gate; when ``False`` the gate runs in 4-state mode
        and never emits ``UNNECESSARY``.
    stillness_window
        Rolling window for the stillness detector.
    stillness_eps_R
        ``max |dR/dt|`` tolerance for the stillness detector.
    stillness_eps_F
        ``max |dF_proxy/dt|`` tolerance.
    stillness_delta_min
        ``max δ`` tolerance.
    stillness_dt
        Step size passed to :class:`StillnessDetector`. Does not
        affect the pipeline's own clock.
    ledger_path
        Optional path to an append-only decision ledger. When set,
        each tick emits a :class:`DecisionTraceRecord`.
    """

    threshold: float = DEFAULT_THRESHOLD
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS
    warmup_samples: int = DEFAULT_PIPELINE_WARMUP
    stream_window: int = DEFAULT_STREAM_WINDOW
    max_fault_rate: float = DEFAULT_MAX_FAULT_RATE
    stream_hold_steps: int = 0
    enable_stillness: bool = True
    stillness_window: int = 8
    stillness_eps_R: float = 1e-3
    stillness_eps_F: float = 1e-3
    stillness_delta_min: float = 0.10
    stillness_dt: float = 0.01
    ledger_path: Path | None = None


@dataclass(frozen=True)
class DecisionFrame:
    """Runtime envelope — single immutable record per tick.

    Carries every provenance field downstream consumers need: the
    raw input, the temporal-layer outcome, the stream regime, the
    gate decision, and the optional ledger record. The frame is
    deliberately flat (no nested optionality traps) and JSON-friendly
    via :meth:`to_json_dict`.

    Attributes
    ----------
    tick_index
        Zero-based monotonic tick counter since pipeline construction.
    timestamp
        Caller-supplied timestamp for this tick (seconds).
    R
        Raw order parameter input at this tick.
    delta
        Raw circular-distance input at this tick. Optional because
        callers without a stillness detector may not supply it.
    temporal
        :class:`TemporalQualityDecision` emitted by B1.
    stream
        :class:`StreamQualityDecision` emitted by B2 + B6.
    gate
        :class:`GateDecision` emitted by the 5-state gate.
    ledger_record
        The append-only ledger record, if a ledger is attached.
    """

    tick_index: int
    timestamp: float
    R: float | None
    delta: float | None
    temporal: TemporalQualityDecision
    stream: StreamQualityDecision
    gate: GateDecision
    ledger_record: DecisionTraceRecord | None = field(default=None)

    @property
    def execution_allowed(self) -> bool:
        """Shortcut for ``frame.gate.execution_allowed``."""
        return self.gate.execution_allowed

    @property
    def gate_state(self) -> GateState:
        return self.gate.state

    @property
    def stream_regime(self) -> StreamRegime:
        return self.stream.regime

    @property
    def time_quality(self) -> TimeQuality:
        return self.temporal.quality

    def to_json_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for ``json.dumps``.

        Intentionally flat: every field is a primitive, enum name,
        or ``None``. No nested dataclass objects.
        """
        return {
            "tick_index": self.tick_index,
            "timestamp": self.timestamp,
            "R": self.R,
            "delta": self.delta,
            "time_quality": self.temporal.quality.name,
            "temporal_reason": self.temporal.reason,
            "stream_regime": self.stream.regime.name,
            "stream_reason": self.stream.reason,
            "stream_fault_rate": self.stream.stats.fault_rate,
            "gate_state": self.gate.state.name,
            "gate_reason": self.gate.reason,
            "execution_allowed": self.gate.execution_allowed,
            "ledger_record_hash": (
                self.ledger_record.record_hash if self.ledger_record is not None else None
            ),
        }


# ---------------------------------------------------------------------------
# StreamingPipeline
# ---------------------------------------------------------------------------


class StreamingPipeline:
    """Stateful composition of B1 → B2+B6 → ExecutionGate (→ ledger).

    Parameters
    ----------
    config
        Immutable :class:`PipelineConfig`.

    Raises
    ------
    ValueError
        If the config produces an invalid downstream component
        (e.g. threshold outside ``(0, 1)`` — surfaced at gate
        construction time).
    """

    __slots__ = (
        "_gate",
        "_ledger",
        "_n_ticks",
        "_stream_detector",
        "_validator",
        "config",
        "parameter_fingerprint",
    )

    def __init__(self, config: PipelineConfig) -> None:
        self.config: PipelineConfig = config
        self._validator = TemporalValidator(
            max_gap_seconds=config.max_gap_seconds,
            warmup_samples=max(2, config.warmup_samples),
        )
        self._stream_detector = TemporalStreamDetector(
            window=config.stream_window,
            max_fault_rate=config.max_fault_rate,
            hold_steps=config.stream_hold_steps,
        )

        stillness: StillnessDetector | None = None
        if config.enable_stillness:
            stillness = StillnessDetector(
                window=config.stillness_window,
                eps_R=config.stillness_eps_R,
                eps_F=config.stillness_eps_F,
                delta_min=config.stillness_delta_min,
                dt=config.stillness_dt,
            )
        self._gate = ExecutionGate(
            threshold=config.threshold,
            stillness_detector=stillness,
        )

        self.parameter_fingerprint: str = fingerprint_parameters(
            {
                "threshold": config.threshold,
                "max_gap_seconds": config.max_gap_seconds,
                "warmup_samples": config.warmup_samples,
                "stream_window": config.stream_window,
                "max_fault_rate": config.max_fault_rate,
                "stream_hold_steps": config.stream_hold_steps,
                "enable_stillness": config.enable_stillness,
                "stillness_window": config.stillness_window,
                "stillness_eps_R": config.stillness_eps_R,
                "stillness_eps_F": config.stillness_eps_F,
                "stillness_delta_min": config.stillness_delta_min,
                "stillness_dt": config.stillness_dt,
            }
        )

        self._ledger: DecisionTraceLedger | None = None
        if config.ledger_path is not None:
            self._ledger = DecisionTraceLedger(config.ledger_path, self.parameter_fingerprint)

        self._n_ticks: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(
        self,
        *,
        timestamp: float,
        R: float | None,
        delta: float | None = None,
        reference_now: float | None = None,
    ) -> DecisionFrame:
        """Advance the pipeline by one tick and return the frame.

        Parameters
        ----------
        timestamp
            Sample timestamp (seconds since an arbitrary epoch).
        R
            Joint order parameter at this tick. ``None`` signals a
            failed upstream computation and is routed through the
            gate's ``DEGRADED`` state.
        delta
            Optional circular distance. Required only when the
            pipeline has stillness enabled and the caller wants the
            ``UNNECESSARY`` state to be reachable.
        reference_now
            Optional wall-clock reference for the staleness check.
        """
        tick_idx = self._n_ticks
        self._n_ticks += 1

        temporal = self._validator.validate(timestamp, reference_now=reference_now)
        stream = self._stream_detector.update(temporal)

        # Inside the pipeline we gate on the *stream-level* regime,
        # not the per-sample quality. A sample that is individually
        # VALID but lands in a DEGRADED stream regime must still be
        # rejected (invariant B2+B6 at the runtime level).
        if stream.regime is not StreamRegime.HEALTHY:
            # Synthesize a non-VALID TemporalQualityDecision equivalent
            # so the gate's B1 layer treats this as DEGRADED. We re-use
            # the per-sample decision but replace its quality with a
            # stream-regime-labelled placeholder.
            gated_decision = self._gate.evaluate(
                R=R,
                delta=delta,
                time_quality=_stream_quality_placeholder(stream, temporal),
            )
        else:
            gated_decision = self._gate.evaluate(
                R=R,
                delta=delta,
                time_quality=temporal,
            )

        ledger_record: DecisionTraceRecord | None = None
        if self._ledger is not None:
            ledger_record = self._ledger.append(
                timestamp=timestamp,
                gate_state=gated_decision.state.name,
                execution_allowed=gated_decision.execution_allowed,
                R=gated_decision.R,
                threshold=gated_decision.threshold,
                reason=gated_decision.reason,
                extras={
                    "tick_index": tick_idx,
                    "time_quality": temporal.quality.name,
                    "stream_regime": stream.regime.name,
                    "stream_fault_rate": stream.stats.fault_rate,
                    "delta": delta,
                },
            )

        return DecisionFrame(
            tick_index=tick_idx,
            timestamp=timestamp,
            R=R,
            delta=delta,
            temporal=temporal,
            stream=stream,
            gate=gated_decision,
            ledger_record=ledger_record,
        )

    def tick_batch(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Advance the pipeline over every row of a DataFrame in one call.

        This is the **batch complement** to :meth:`tick`: for offline
        replay and calibration-grade scoring, paying the Python
        function-call cost per sample is wasteful. ``tick_batch``
        iterates the underlying stateful layers internally and
        materialises one row in the output DataFrame per input row,
        preserving the exact same state transitions as a sequence of
        :meth:`tick` calls would produce.

        Required columns
        ----------------
        * ``timestamp`` — float, monotonically non-decreasing.
        * ``R``         — float or ``None`` / ``NaN``.

        Optional columns
        ----------------
        * ``delta``         — float; treated as ``None`` when absent.
        * ``reference_now`` — float; treated as ``None`` when absent.

        Output columns
        --------------
        ``tick_index``, ``timestamp``, ``R``, ``delta``,
        ``gate_state``, ``execution_allowed``, ``time_quality``,
        ``stream_regime``, ``stream_fault_rate``, ``gate_reason``,
        ``ledger_record_hash``.

        Contract
        --------
        * **Semantic parity.** For the same inputs,
          ``tick_batch`` produces exactly the same ``gate_state``
          sequence as iterating :meth:`tick` row by row. This is
          certified by ``tests/test_batch_pipeline.py``.
        * **Statefulness.** ``tick_batch`` continues from the
          current pipeline state — calling it twice in a row
          appends to the same rolling windows. Use :meth:`reset`
          at session boundaries.
        * **Ledger compatible.** When a ledger is attached, every
          batch row is appended to the ledger exactly as if the
          caller had invoked :meth:`tick` for each row. The final
          ledger file is byte-identical to the serial path.

        Parameters
        ----------
        frame
            ``pandas.DataFrame`` with the required columns above.
            The column order is irrelevant; missing optional
            columns are treated as ``None``.

        Returns
        -------
        pandas.DataFrame
            One row per input row. Row order matches the input.
        """
        import pandas as pd  # lazy import — top-level pipeline avoids it

        required = {"timestamp", "R"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"tick_batch input missing required columns: {sorted(missing)}")

        has_delta = "delta" in frame.columns
        has_ref = "reference_now" in frame.columns

        rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            ts = float(row["timestamp"])
            raw_R = row["R"]
            r_val: float | None
            if raw_R is None or (isinstance(raw_R, float) and pd.isna(raw_R)):
                r_val = None
            else:
                r_val = float(raw_R)
            delta_val: float | None = None
            if has_delta:
                raw_delta = row["delta"]
                if raw_delta is not None and not (
                    isinstance(raw_delta, float) and pd.isna(raw_delta)
                ):
                    delta_val = float(raw_delta)
            ref_val: float | None = None
            if has_ref:
                raw_ref = row["reference_now"]
                if raw_ref is not None and not (isinstance(raw_ref, float) and pd.isna(raw_ref)):
                    ref_val = float(raw_ref)

            tick_frame = self.tick(
                timestamp=ts,
                R=r_val,
                delta=delta_val,
                reference_now=ref_val,
            )
            rows.append(
                {
                    "tick_index": tick_frame.tick_index,
                    "timestamp": tick_frame.timestamp,
                    "R": tick_frame.R,
                    "delta": tick_frame.delta,
                    "gate_state": tick_frame.gate_state.name,
                    "execution_allowed": tick_frame.execution_allowed,
                    "time_quality": tick_frame.time_quality.name,
                    "stream_regime": tick_frame.stream_regime.name,
                    "stream_fault_rate": tick_frame.stream.stats.fault_rate,
                    "gate_reason": tick_frame.gate.reason,
                    "ledger_record_hash": (
                        tick_frame.ledger_record.record_hash
                        if tick_frame.ledger_record is not None
                        else None
                    ),
                }
            )

        return pd.DataFrame(rows)

    def reset(self) -> None:
        """Reset all stateful layers. Use at session boundaries."""
        self._validator.reset()
        self._stream_detector.reset()
        if self._gate.stillness_detector is not None:
            self._gate.stillness_detector.reset()
        self._n_ticks = 0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def n_ticks(self) -> int:
        return self._n_ticks

    @property
    def ledger(self) -> DecisionTraceLedger | None:
        return self._ledger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stream_quality_placeholder(
    stream: StreamQualityDecision, last_temporal: TemporalQualityDecision
) -> TemporalQualityDecision:
    """Synthesize a non-VALID ``TemporalQualityDecision`` from a stream regime.

    When the stream-level classifier (B2+B6) rejects the stream, the
    gate must see a non-VALID ``TemporalQualityDecision`` regardless
    of the per-sample B1 outcome. We preserve the provenance of the
    original per-sample decision but mark the quality as ``GAPPED``
    (the coarsest "something's wrong with the stream" signal) and
    carry the stream reason into the decision's reason string.
    """
    return TemporalQualityDecision(
        quality=TimeQuality.GAPPED,
        ts=last_temporal.ts,
        last_ts=last_temporal.last_ts,
        gap_seconds=last_temporal.gap_seconds,
        staleness_seconds=last_temporal.staleness_seconds,
        warmup_remaining=last_temporal.warmup_remaining,
        reason=f"stream {stream.regime.name.lower()}: {stream.reason}",
    )
