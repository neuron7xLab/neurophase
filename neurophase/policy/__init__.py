"""Policy layer — typed action intents above the execution gate.

This package contains the **policy layer** that sits *above* the
gate. It maps a fully-resolved ``(DecisionFrame, RegimeState)``
pair into a typed :class:`ActionIntent` token. It does **not**
execute trades, send orders, or talk to any broker — those are
strictly downstream concerns. The policy's job is to tell the
caller *what* should happen, not *how*.

The policy is the first place in the stack where Program G
(regime intelligence) and Program B/E/I (gate + permission
machinery) come together. It is also the last place before the
external world: anything the policy emits is auditable, frozen,
and JSON-serialisable.
"""

from __future__ import annotations

from neurophase.policy.action import (
    ActionDecision,
    ActionIntent,
    ActionPolicy,
    PolicyConfig,
)

__all__ = [
    "ActionDecision",
    "ActionIntent",
    "ActionPolicy",
    "PolicyConfig",
]
