"""Read-only ``DomainAdapter`` projection from :class:`SystemState` into the
external ``neosynaptex`` integrating mirror.

The adapter is the bridge between the KLR substrate of ``neurophase`` and
the cross-substrate γ-scaling diagnostic in ``neosynaptex``. It is a
**passive observer**: it reads from :class:`SystemState` snapshots and
exposes the ``DomainAdapter`` protocol (``domain``, ``state_keys``,
``state``, ``topo``, ``thermo_cost``). It MUST never mutate the state it
observes.

The projection is deliberately small — three scalar features:

* ``ntk_rank``      — normalized NTK-rank proxy (plasticity headroom)
* ``frozen_ratio``  — fraction of frozen nodes
* ``usage_entropy`` — normalized Shannon entropy of the usage distribution

Topology signal: non-frozen node count (``topo``). Thermodynamic cost:
``1 − ntk_rank`` (inverse of plasticity headroom — high cost when the
system approaches rank saturation). Both values are clamped away from
zero to avoid spurious log-transforms downstream.

Contracts
---------
* :meth:`NeosynaptexResetAdapter.update` is the **only** write path.
* All values returned by :meth:`state`, :meth:`topo` and
  :meth:`thermo_cost` are finite by construction.
* Until :meth:`update` has been called, :meth:`state` yields ``NaN``
  placeholders and :meth:`topo` / :meth:`thermo_cost` return their
  floor values — this is the witness ``WARMUP`` signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurophase.reset.ntk_monitor import NTKMonitor
from neurophase.reset.state import SystemState

__all__ = ["NeosynaptexResetAdapter"]

_STATE_KEYS: tuple[str, ...] = ("ntk_rank", "frozen_ratio", "usage_entropy")
_TOPO_FLOOR: float = 0.01
_COST_FLOOR: float = 0.01


@dataclass(frozen=True)
class _Snapshot:
    ntk_rank: float
    frozen_ratio: float
    usage_entropy: float
    non_frozen_count: float


def _normalized_entropy(values: NDArray[np.float64]) -> float:
    """Shannon entropy of a non-negative vector normalized into ``[0, 1]``.

    Guaranteed finite. Returns ``0.0`` for degenerate inputs (all-zero,
    single-bin or NaN-bearing vectors).
    """

    if values.size == 0:
        return 0.0
    clean = np.where(np.isfinite(values), values, 0.0)
    clean = np.clip(clean, 0.0, None)
    total = float(np.sum(clean))
    if total <= 0.0:
        return 0.0
    p = clean / total
    mask = p > 0.0
    if not np.any(mask):
        return 0.0
    h = float(-np.sum(p[mask] * np.log(p[mask])))
    h_max = float(np.log(values.size)) if values.size > 1 else 1.0
    if h_max <= 0.0:
        return 0.0
    value = h / h_max
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


class NeosynaptexResetAdapter:
    """``DomainAdapter`` that turns :class:`SystemState` snapshots into
    neosynaptex-compatible state vectors.

    The adapter holds *only* cached scalars derived from the last observed
    :class:`SystemState`. It never stores a reference to the state itself
    and never mutates the state passed to :meth:`update`.
    """

    _DOMAIN: str = "klr_reset"

    def __init__(self) -> None:
        self._ntk = NTKMonitor()
        self._snapshot: _Snapshot | None = None

    # ------------------------------------------------------------------
    # DomainAdapter protocol
    # ------------------------------------------------------------------
    @property
    def domain(self) -> str:
        return self._DOMAIN

    @property
    def state_keys(self) -> list[str]:
        return list(_STATE_KEYS)

    def state(self) -> dict[str, float]:
        snap = self._snapshot
        if snap is None:
            return {k: float("nan") for k in _STATE_KEYS}
        return {
            "ntk_rank": snap.ntk_rank,
            "frozen_ratio": snap.frozen_ratio,
            "usage_entropy": snap.usage_entropy,
        }

    def topo(self) -> float:
        snap = self._snapshot
        if snap is None:
            return _TOPO_FLOOR
        return max(snap.non_frozen_count, _TOPO_FLOOR)

    def thermo_cost(self) -> float:
        snap = self._snapshot
        if snap is None:
            return _COST_FLOOR
        cost = 1.0 - snap.ntk_rank
        return max(cost, _COST_FLOOR)

    # ------------------------------------------------------------------
    # Observation entry point (the sole write path)
    # ------------------------------------------------------------------
    def update(self, state: SystemState) -> None:
        """Record a derived snapshot from ``state``.

        The function performs only read accesses; ``state`` and every
        array it exposes are left untouched. This is enforced by
        invariant ``NEO-I1`` and validated by
        ``tests/test_gamma_witness.py::test_witness_readonly``.
        """

        ntk_rank = float(self._ntk.rank_proxy(state.weights))
        if not np.isfinite(ntk_rank):
            ntk_rank = 0.0
        ntk_rank = float(np.clip(ntk_rank, 0.0, 1.0))

        n = int(state.weights.shape[0])
        frozen_count = 0 if state.frozen is None else int(np.count_nonzero(state.frozen))
        non_frozen = float(max(n - frozen_count, 0))
        frozen_ratio = float(frozen_count / n) if n > 0 else 0.0

        usage_entropy = _normalized_entropy(state.usage)

        self._snapshot = _Snapshot(
            ntk_rank=ntk_rank,
            frozen_ratio=frozen_ratio,
            usage_entropy=usage_entropy,
            non_frozen_count=non_frozen,
        )

    def has_snapshot(self) -> bool:
        """Return ``True`` once :meth:`update` has been called at least once."""

        return self._snapshot is not None
