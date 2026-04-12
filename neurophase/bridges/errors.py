"""Bridge-layer base exception.

Every bridge (``ingress``, ``clock_sync``, ``frame_mux``,
``downstream_execution``) raises a subclass of :class:`BridgeError`
when it refuses a frame. Distinct subclasses exist so callers can
react differently to, e.g., a clock desync vs a downstream 503.
"""

from __future__ import annotations


class BridgeError(RuntimeError):
    """Base class for every bridge-layer rejection.

    Catching :class:`BridgeError` at the session boundary is the
    idiomatic way to reject an entire tick: every bridge contract
    is designed to raise rather than return a degenerate frame,
    so the kernel is never fed invalid data.
    """
