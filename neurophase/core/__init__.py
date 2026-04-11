"""Core physics — phase extraction, Kuramoto dynamics, order parameter."""

from __future__ import annotations

from neurophase.core.kuramoto import KuramotoNetwork, KuramotoParams
from neurophase.core.order_parameter import OrderParameterResult, order_parameter
from neurophase.core.phase import adaptive_threshold, compute_phase, preprocess_signal

__all__ = [
    "KuramotoNetwork",
    "KuramotoParams",
    "OrderParameterResult",
    "adaptive_threshold",
    "compute_phase",
    "order_parameter",
    "preprocess_signal",
]
