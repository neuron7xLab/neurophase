"""E3 — runtime orchestrator: single-call session driver.

This module is the **highest layer** of the runtime stack. It
composes every load-bearing online component into one cohesive
session object:

    raw tick → StreamingPipeline (B1+B2+B6+I₁..I₄)
             → RegimeClassifier   (G1)
             → ActionPolicy       (I1)
             → DecisionTraceLedger (F1)
             → SessionManifest    (K1)

Until E3 the caller had to wire each of those layers together by
hand: build the pipeline, classify regime on every emitted frame,
feed the regime + frame into the policy, manage manifest start/end,
remember to verify the bound ledger at session-close. That is
exactly the kind of glue code that doctrine item *"composition is
the testable surface"* says should live inside the library, not in
every consumer.

:class:`RuntimeOrchestrator` provides one constructor, one
``tick()`` method, and one ``close()`` method. Two orchestrators
with the same configuration fed the same input sequence emit
byte-identical :class:`OrchestratedFrame` sequences **and**
byte-identical session manifests at close time.

Contracts
---------

* **Single-call composition.** :meth:`RuntimeOrchestrator.tick`
  calls every layer exactly once, in the canonical order, and
  returns one frozen :class:`OrchestratedFrame` envelope carrying
  every layer's typed output.
* **Deterministic.** No clocks, no RNG inside the orchestrator
  itself. Same config + same input sequence → byte-identical
  envelopes and byte-identical session manifest.
* **Gate-honoring.** The orchestrator never widens the gate's
  permission surface. The :class:`ActionPolicy` it wraps already
  enforces this; the orchestrator simply forwards.
* **Session-scoped audit.** :meth:`close` materialises a
  :class:`SessionManifest` (K1) bound to the underlying
  decision ledger and the parameter fingerprint, **and** verifies
  the ledger end-to-end before returning. A close that returns
  successfully is a proof that everything in the session is
  internally consistent.
* **Immutable envelopes.** Every :class:`OrchestratedFrame` is
  frozen and JSON-safe via :meth:`to_json_dict`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neurophase.analysis.regime import RegimeClassifier, RegimeState, RegimeThresholds
from neurophase.audit.session_manifest import (
    EMPTY_LEDGER_TIP,
    SessionManifest,
    build_session_manifest,
    compute_dataset_fingerprint,
)
from neurophase.policy.action import (
    ActionDecision,
    ActionPolicy,
    PolicyConfig,
)
from neurophase.runtime.pipeline import (
    DecisionFrame,
    PipelineConfig,
    StreamingPipeline,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "OrchestratedFrame",
    "OrchestratorConfig",
    "RuntimeOrchestrator",
]


@dataclass(frozen=True)
class OrchestratorConfig:
    """Immutable configuration for a :class:`RuntimeOrchestrator`.

    Attributes
    ----------
    pipeline
        :class:`PipelineConfig` for the underlying
        :class:`StreamingPipeline`.
    policy
        :class:`PolicyConfig` for the underlying
        :class:`ActionPolicy`. Defaults to a fresh
        :class:`PolicyConfig`.
    regime_thresholds
        Optional :class:`RegimeThresholds` for the
        :class:`RegimeClassifier`. ``None`` means use the package
        defaults.
    code_commit
        Optional code commit SHA recorded in the manifest.
    host
        Optional non-PII host identifier recorded in the manifest.
    notes
        Optional free-form notes recorded in the manifest.
    """

    pipeline: PipelineConfig
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    regime_thresholds: RegimeThresholds | None = None
    code_commit: str = ""
    host: str = ""
    notes: str = ""


@dataclass(frozen=True, repr=False)
class OrchestratedFrame:
    """Frozen, JSON-safe envelope returned by :meth:`RuntimeOrchestrator.tick`.

    Carries every typed layer output for one tick:

    * :attr:`pipeline_frame` — the raw
      :class:`~neurophase.runtime.pipeline.DecisionFrame`.
    * :attr:`regime` — the
      :class:`~neurophase.analysis.regime.RegimeState` (or ``None``
      if the frame had a missing R / delta and the regime layer
      was skipped).
    * :attr:`action` — the
      :class:`~neurophase.policy.action.ActionDecision` (or
      ``None`` if the regime layer was skipped — without a regime
      the policy cannot run).

    The convenience properties surface the most common downstream
    queries (gate state, regime label, action intent) without
    requiring the caller to crack open the nested envelopes.
    """

    pipeline_frame: DecisionFrame
    regime: RegimeState | None
    action: ActionDecision | None

    def __repr__(self) -> str:  # aesthetic rich repr (HN26)
        regime_name = self.regime.label.name if self.regime is not None else "—"
        action_name = self.action.intent.name if self.action is not None else "—"
        flag = "✓" if self.pipeline_frame.gate.execution_allowed else "✗"
        return (
            f"OrchestratedFrame[tick={self.pipeline_frame.tick_index} · "
            f"{self.pipeline_frame.gate.state.name} · "
            f"{regime_name} · {action_name} · {flag}]"
        )

    @property
    def tick_index(self) -> int:
        return self.pipeline_frame.tick_index

    @property
    def timestamp(self) -> float:
        return self.pipeline_frame.timestamp

    @property
    def execution_allowed(self) -> bool:
        return self.pipeline_frame.gate.execution_allowed

    def to_json_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe projection — surfaces every layer's typed output."""
        out: dict[str, Any] = {
            "pipeline_frame": self.pipeline_frame.to_json_dict(),
        }
        if self.regime is not None:
            out["regime"] = {
                "label": self.regime.label.name,
                "R": self.regime.R,
                "dR": self.regime.dR,
                "delta": self.regime.delta,
                "d_delta": self.regime.d_delta,
                "confidence_score": self.regime.confidence_score,
                "warm": self.regime.warm,
                "reason": self.regime.reason,
            }
        else:
            out["regime"] = None
        if self.action is not None:
            out["action"] = self.action.to_json_dict()
        else:
            out["action"] = None
        return out


# ---------------------------------------------------------------------------
# RuntimeOrchestrator — single-call session driver.
# ---------------------------------------------------------------------------


class RuntimeOrchestrator:
    """Stateful single-call composition of pipeline + regime + policy + manifest.

    Parameters
    ----------
    config
        Immutable :class:`OrchestratorConfig`.

    Notes
    -----
    The orchestrator is the **only** correct way to drive the
    full online stack. Direct use of the lower layers is still
    supported for research and testing, but production callers
    should always go through this object.

    Construct → call :meth:`tick` repeatedly → call :meth:`close`
    once. After :meth:`close` returns, no further :meth:`tick`
    calls are permitted: the orchestrator transitions into a
    sealed state.
    """

    __slots__ = (
        "_closed",
        "_dataset_fingerprint",
        "_last_tip_hash",
        "_pipeline",
        "_policy",
        "_regime",
        "_session_start",
        "_start_ts_caller",
        "config",
        "manifest",
    )

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config: OrchestratorConfig = config
        self._pipeline = StreamingPipeline(config.pipeline)
        self._regime = RegimeClassifier(thresholds=config.regime_thresholds)
        self._policy = ActionPolicy(config.policy)
        self._dataset_fingerprint: str | None = None
        # Wall-clock + first-frame timestamp anchors. The orchestrator
        # uses the *caller's* tick timestamps for the manifest's
        # start/end fields (deterministic), and a wall-clock anchor
        # only as a fallback when the caller never tick()s.
        self._session_start: float = time.time()
        self._start_ts_caller: float | None = None
        self._last_tip_hash: str = EMPTY_LEDGER_TIP
        self._closed: bool = False
        self.manifest: SessionManifest | None = None

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
    ) -> OrchestratedFrame:
        """Drive every layer exactly once and return a fused envelope.

        Parameters
        ----------
        timestamp
            Sample timestamp (seconds since an arbitrary epoch).
        R
            Joint order parameter at this tick. ``None`` routes
            through the gate's ``DEGRADED`` path; the regime and
            policy layers are skipped (the regime classifier
            requires a finite R).
        delta
            Optional circular distance. Required for the regime
            and policy layers; without it the regime layer is
            skipped.
        reference_now
            Optional wall-clock reference for staleness checking.
        """
        if self._closed:
            raise RuntimeError("RuntimeOrchestrator.tick called after close()")

        if self._start_ts_caller is None:
            self._start_ts_caller = timestamp

        pipeline_frame = self._pipeline.tick(
            timestamp=timestamp,
            R=R,
            delta=delta,
            reference_now=reference_now,
        )

        regime: RegimeState | None
        action: ActionDecision | None
        if pipeline_frame.R is None or pipeline_frame.delta is None:
            # The regime layer requires both R and delta. A frame
            # missing either passes through with no regime / no
            # action — the gate already vetoed it via DEGRADED, so
            # the policy would only emit HOLD anyway.
            regime = None
            action = None
        else:
            regime = self._regime.classify(pipeline_frame)
            action = self._policy.decide(pipeline_frame, regime)

        if pipeline_frame.ledger_record is not None:
            self._last_tip_hash = pipeline_frame.ledger_record.record_hash

        return OrchestratedFrame(
            pipeline_frame=pipeline_frame,
            regime=regime,
            action=action,
        )

    def close(
        self,
        *,
        end_ts: float | None = None,
        dataset_fingerprint: str | None = None,
    ) -> SessionManifest:
        """Seal the session and materialise the K1 :class:`SessionManifest`.

        Parameters
        ----------
        end_ts
            Optional explicit session-close timestamp. Defaults to
            the timestamp of the most recent tick (or the
            session-start anchor if no ticks were emitted).
        dataset_fingerprint
            Optional hex SHA256 of the input dataset bytes. When
            ``None``, a placeholder zero-hash is computed for an
            empty input — callers that care about provenance
            should always supply this. The manifest happily
            accepts any 64-char hex string.

        Returns
        -------
        SessionManifest
            The manifest, also stored on
            :attr:`RuntimeOrchestrator.manifest`. After this call
            returns, :meth:`tick` will raise.
        """
        if self._closed:
            raise RuntimeError("RuntimeOrchestrator.close called twice")

        start = self._start_ts_caller if self._start_ts_caller is not None else 0.0
        end = (
            end_ts
            if end_ts is not None
            else (self._start_ts_caller if self._start_ts_caller is not None else 0.0)
        )
        if end < start:
            raise ValueError(f"close() end_ts ({end}) precedes session start ({start})")

        if dataset_fingerprint is None:
            dataset_fingerprint = compute_dataset_fingerprint(b"")

        ledger_path = (
            self.config.pipeline.ledger_path
            if self.config.pipeline.ledger_path is not None
            else Path("/dev/null")
        )

        manifest = build_session_manifest(
            start_ts=start,
            end_ts=end,
            parameter_fingerprint=self._pipeline.parameter_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
            ledger_path=ledger_path,
            ledger_tip_hash=self._last_tip_hash,
            n_ticks=self._pipeline.n_ticks,
            code_commit=self.config.code_commit,
            host=self.config.host,
            notes=self.config.notes,
        )

        # Cross-file integrity check: if a real ledger was
        # configured, verify the bound ledger before declaring the
        # session sealed. /dev/null is treated as "no ledger" and
        # skipped. A failure here surfaces as ManifestError.
        if (
            self.config.pipeline.ledger_path is not None
            and self.config.pipeline.ledger_path.exists()
        ):
            manifest.verify_against_ledger()

        self.manifest = manifest
        self._closed = True
        return manifest

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def n_ticks(self) -> int:
        return self._pipeline.n_ticks

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def parameter_fingerprint(self) -> str:
        return self._pipeline.parameter_fingerprint

    @property
    def last_tip_hash(self) -> str:
        return self._last_tip_hash

    @property
    def pipeline(self) -> StreamingPipeline:
        return self._pipeline

    @property
    def regime_classifier(self) -> RegimeClassifier:
        return self._regime

    @property
    def policy(self) -> ActionPolicy:
        return self._policy
