"""Auditability layer — decision trace ledger and replay primitives.

Program F of the Evolution Board lives here. Doctrine #5: *an
unreplayable result is not yet a system result.* Every decision
emitted by the gate should be persistable into an append-only,
tamper-evident ledger so that postmortem analysis, replay, and
determinism tests can reconstruct the exact decision from the exact
input.

Public API:

* :class:`DecisionTraceRecord` — frozen dataclass with the full
  provenance of a single gate decision.
* :class:`DecisionTraceLedger` — append-only ledger that maintains a
  SHA256 hash chain across records.
* :func:`verify_ledger` — stateless verification of a ledger file.
* :class:`LedgerError` — raised on tamper detection or schema mismatch.
* :func:`replay_ledger` — **F2** replay engine: re-run a pipeline
  against the original input stream and verify the resulting
  ledger matches the stored one byte-for-byte. Postmortem
  precondition.
"""

from __future__ import annotations

from neurophase.audit.decision_ledger import (
    GENESIS_HASH,
    DecisionTraceLedger,
    DecisionTraceRecord,
    LedgerError,
    LedgerVerification,
    fingerprint_parameters,
    verify_ledger,
)
from neurophase.audit.replay import ReplayInput, ReplayResult, replay_ledger

__all__ = [
    "GENESIS_HASH",
    "DecisionTraceLedger",
    "DecisionTraceRecord",
    "LedgerError",
    "LedgerVerification",
    "ReplayInput",
    "ReplayResult",
    "fingerprint_parameters",
    "replay_ledger",
    "verify_ledger",
]
