"""Physio — replay-first RR/HRV signal-quality gating path.

This subpackage is a minimal, truthful vertical slice that turns
timestamped RR-interval (heart-rate) samples from a replay file into
fail-closed gate decisions using the existing
:class:`~neurophase.gate.execution_gate.ExecutionGate` kernel.

Scope and honest limits
-----------------------

* **Replay only.** The only supported input is a CSV of timestamped RR
  intervals (see :class:`RRReplayReader`). There is no live device
  driver in this repository. A live adapter can be built against the
  same :class:`RRSample` contract later; nothing here pretends that a
  live path exists today.
* **Signal-quality gating, not medical assessment.** HRV-style features
  (RMSSD, RR stability, continuity, confidence) are used as
  physiological-plausibility / signal-integrity indicators. The physio
  gate is NOT a clinical instrument, a readiness biomarker, or a
  trading-alpha primitive.
* **Fail-closed by construction.** Missing, malformed, non-monotonic,
  or physiologically impossible input raises
  :class:`ReplayIngestError`; an insufficient or artifact-heavy window
  returns :attr:`PhysioGateState.SENSOR_DEGRADED`; only a clean,
  high-confidence window can emit :attr:`PhysioGateState.EXECUTE_ALLOWED`.
* **Kernel reuse.** The underlying admission decision is delegated to
  the existing :class:`ExecutionGate` (5-state invariant enforcement,
  ``I_1 .. I_4``). The physio layer only owns input validation, feature
  extraction, and the 4-state physio-facing vocabulary.

Public surface::

    from neurophase.physio import (
        HRVFeatures,
        PhysioDecision,
        PhysioFrame,
        PhysioGate,
        PhysioGateState,
        PhysioReplayPipeline,
        ReplayIngestError,
        RRReplayReader,
        RRSample,
    )

Demo::

    python -m neurophase.physio.demo
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neurophase.physio.features import HRVFeatures, HRVWindow
from neurophase.physio.gate import PhysioDecision, PhysioGate, PhysioGateState
from neurophase.physio.pipeline import PhysioFrame, PhysioReplayPipeline
from neurophase.physio.replay import ReplayIngestError, RRReplayReader, RRSample

# Reachability declaration for ``governance.completeness`` — the demo
# module is a CLI entry point, intentionally not re-exported at import
# time (to avoid pulling argparse into ``from neurophase.physio import *``).
if TYPE_CHECKING:
    from neurophase.physio.demo import main as _demo_main  # noqa: F401

__all__ = [
    "HRVFeatures",
    "HRVWindow",
    "PhysioDecision",
    "PhysioFrame",
    "PhysioGate",
    "PhysioGateState",
    "PhysioReplayPipeline",
    "RRReplayReader",
    "RRSample",
    "ReplayIngestError",
]
