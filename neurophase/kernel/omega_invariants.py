"""Ω-Invariants enforcement kernel.

Provides runtime decorators for:
- invariant(expression)
- falsifiable(criterion)
- evidence_bound(status)

These decorators are composable and attach auditable metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

_ALLOWED_EVIDENCE = {"Established", "Strongly Plausible", "Tentative", "Unsupported"}


@dataclass(frozen=True)
class InvariantSpec:
    name: str
    threshold: float
    comparator: str = ">="

    def evaluate(self, value: float) -> bool:
        if self.comparator != ">=":
            raise ValueError(f"Unsupported comparator: {self.comparator}")
        return value >= self.threshold


def _coerce_float(v: Any) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    raise TypeError(f"Invariant value must be numeric, got {type(v)!r}")


def invariant(*, name: str, threshold: float, extractor: Callable[[Any], float]) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Enforce post-condition invariant on a callable return value.

    Example: @invariant(name="R_t", threshold=0.92, extractor=lambda x: x.r_t)
    """

    spec = InvariantSpec(name=name, threshold=threshold)

    def deco(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            out = fn(*args, **kwargs)
            value = _coerce_float(extractor(out))
            if not spec.evaluate(value):
                raise RuntimeError(
                    f"Ω invariant violated: {spec.name} {spec.comparator} {spec.threshold}, got {value:.6f}"
                )
            return out

        wrapped.__omega_invariant__ = spec  # type: ignore[attr-defined]
        return wrapped

    return deco


def falsifiable(*, criterion: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Attach mandatory falsification criterion metadata to a callable."""

    criterion_norm = criterion.strip()
    if not criterion_norm:
        raise ValueError("criterion must be non-empty")

    def deco(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            return fn(*args, **kwargs)

        wrapped.__falsifiable_criterion__ = criterion_norm  # type: ignore[attr-defined]
        return wrapped

    return deco


def evidence_bound(*, status: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Attach evidence status to callable and enforce valid taxonomy value."""

    normalized = status.strip()
    if normalized not in _ALLOWED_EVIDENCE:
        raise ValueError(f"Unsupported evidence status: {status}")

    def deco(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            return fn(*args, **kwargs)

        wrapped.__evidence_status__ = normalized  # type: ignore[attr-defined]
        return wrapped

    return deco


def verify_public_callable(fn: Callable[..., Any]) -> list[str]:
    """Return list of missing Ω-governance metadata fields for callable."""
    missing: list[str] = []
    if not hasattr(fn, "__evidence_status__"):
        missing.append("evidence_status")
    if not hasattr(fn, "__falsifiable_criterion__"):
        missing.append("falsifiable_criterion")
    return missing
