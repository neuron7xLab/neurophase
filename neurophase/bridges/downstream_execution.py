"""Downstream execution adapter — gate-first egress contract.

A :class:`DownstreamAdapter` is the **only** permitted egress surface
for a kernel-approved frame. It enforces:

* The frame was produced by the canonical runtime path (it is an
  :class:`OrchestratedFrame`, serialized via
  :func:`~neurophase.contracts.as_canonical_dict` and validated).
* ``execution_allowed`` is ``True``. A frame with a failed gate is
  **never** dispatched; the adapter raises.
* Transport failure is surfaced as
  :class:`DownstreamAdapterError`, **never** swallowed.

The adapter owns one callable — the transport — and wraps it. It does
not retry, buffer, or implement trading logic. Those concerns are for
layers *above* the adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from neurophase.bridges.errors import BridgeError
from neurophase.contracts import (
    CANONICAL_FRAME_SCHEMA_VERSION,
    as_canonical_dict,
    validate_canonical_dict,
)
from neurophase.runtime.orchestrator import OrchestratedFrame


class DownstreamAdapterError(BridgeError):
    """Raised when the downstream transport fails or the frame is not dispatchable."""


@dataclass(frozen=True)
class DownstreamDispatchResult:
    """Outcome of a single dispatch attempt.

    Attributes
    ----------
    dispatched
        Always ``True`` on a successful return (``False`` appears only
        in a rejected-frame report via :attr:`reason`).
    schema_version
        The schema version embedded in the outbound payload.
    transport_reply
        Opaque object returned by the transport callable. Stored
        verbatim so callers can cross-reference their own
        order-id / message-id / etc.
    frame_tick_index
        The ``tick_index`` of the dispatched frame, for audit.
    ledger_record_hash
        The ledger tip at dispatch time, or ``None`` if the
        pipeline has no ledger.
    reason
        Free-form reason. On success, the adapter sets this to the
        empty string; on refusal it explains why.
    """

    dispatched: bool
    schema_version: str
    transport_reply: Any
    frame_tick_index: int
    ledger_record_hash: str | None
    reason: str


class DownstreamAdapter:
    """Wraps a transport callable and enforces the gate-first egress law."""

    __slots__ = ("_transport",)

    def __init__(self, transport: Callable[[Mapping[str, Any]], Any]) -> None:
        if not callable(transport):
            raise BridgeError("DownstreamAdapter.transport must be callable")
        self._transport = transport

    def dispatch(self, frame: OrchestratedFrame) -> DownstreamDispatchResult:
        """Send ``frame`` to the downstream transport.

        The frame is serialised via
        :func:`~neurophase.contracts.as_canonical_dict`, validated, and
        then forwarded. A frame with ``execution_allowed=False`` is
        rejected immediately — the transport is never called.

        Raises
        ------
        DownstreamAdapterError
            When the gate rejected the frame, or when the transport
            callable raised.
        """
        if not isinstance(frame, OrchestratedFrame):
            raise DownstreamAdapterError(
                f"dispatch() expected OrchestratedFrame, got {type(frame).__name__}"
            )

        if not frame.execution_allowed:
            gate_state = frame.pipeline_frame.gate.state.name
            gate_reason = frame.pipeline_frame.gate.reason
            raise DownstreamAdapterError(
                f"gate refused dispatch: state={gate_state} reason={gate_reason!r}"
            )

        payload = as_canonical_dict(frame)
        # Validate the frame ourselves before the transport sees it —
        # a bug in as_canonical_dict must never ship to production.
        validate_canonical_dict(payload)

        try:
            reply = self._transport(payload)
        except Exception as exc:
            raise DownstreamAdapterError(f"downstream transport raised: {exc!r}") from exc

        return DownstreamDispatchResult(
            dispatched=True,
            schema_version=CANONICAL_FRAME_SCHEMA_VERSION,
            transport_reply=reply,
            frame_tick_index=frame.pipeline_frame.tick_index,
            ledger_record_hash=(
                frame.pipeline_frame.ledger_record.record_hash
                if frame.pipeline_frame.ledger_record is not None
                else None
            ),
            reason="",
        )
