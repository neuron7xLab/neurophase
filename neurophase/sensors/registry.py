"""Adapter registry — named factories for sensor sources.

Real hardware adapters (Tobii, OpenBCI, Polar, Muse, Emotiv)
live in separate packages that depend on vendor SDKs. The
registry is the **architectural seam** where those packages
plug in: they register a named factory, the registry returns
it, and downstream code uses the factory to construct an
adapter without ever importing the vendor library.

The in-repo registry ships with two built-in factories —
``"synthetic"`` and ``"null"`` — so that the seam is exercised
end-to-end even without any external dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from neurophase.oscillators.neural_protocol import NeuralPhaseExtractor

__all__ = [
    "DEFAULT_ADAPTER_REGISTRY",
    "AdapterFactory",
    "AdapterRegistry",
    "SensorAdapterError",
]

#: Type alias for a zero-argument factory returning an extractor.
AdapterFactory = Callable[[], "NeuralPhaseExtractor"]


class SensorAdapterError(ValueError):
    """Raised on registration or build failures in the sensor registry."""


class AdapterRegistry:
    """Named-factory registry for sensor adapters.

    Every registered factory must return an object implementing
    :class:`~neurophase.oscillators.neural_protocol.NeuralPhaseExtractor`.
    The registry does a runtime protocol check via
    ``isinstance(obj, NeuralPhaseExtractor)`` on :meth:`build` and
    raises :class:`SensorAdapterError` on mismatch.

    The registry is **mutable but deterministic**: two registries
    with the same registrations, queried in the same order,
    produce equal :class:`NeuralPhaseExtractor` instances (modulo
    factory non-determinism, which the factory owner is
    responsible for).
    """

    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, name: str, factory: AdapterFactory) -> None:
        """Register a named factory.

        Parameters
        ----------
        name
            Stable string key. Must be non-empty.
        factory
            Zero-argument callable returning a
            :class:`NeuralPhaseExtractor`.

        Raises
        ------
        SensorAdapterError
            If ``name`` is empty or already registered.
        """
        if not name:
            raise SensorAdapterError("adapter name must be non-empty")
        if name in self._factories:
            raise SensorAdapterError(f"adapter already registered: {name!r}")
        self._factories[name] = factory

    def unregister(self, name: str) -> None:
        """Remove a named factory. Raises :class:`KeyError` if unknown."""
        if name not in self._factories:
            raise KeyError(f"unknown adapter: {name!r}")
        del self._factories[name]

    def build(self, name: str) -> NeuralPhaseExtractor:
        """Instantiate an adapter by name.

        Raises
        ------
        SensorAdapterError
            If ``name`` is not registered, or if the factory
            returns an object that does not implement
            :class:`NeuralPhaseExtractor`.
        """
        from neurophase.oscillators.neural_protocol import NeuralPhaseExtractor

        if name not in self._factories:
            raise SensorAdapterError(f"unknown adapter: {name!r}; registered: {self.names()}")
        obj = self._factories[name]()
        if not isinstance(obj, NeuralPhaseExtractor):
            raise SensorAdapterError(
                f"factory for {name!r} returned {type(obj).__name__}, "
                f"which does not implement NeuralPhaseExtractor"
            )
        return obj

    def names(self) -> tuple[str, ...]:
        """Return the tuple of registered adapter names in insertion order."""
        return tuple(self._factories.keys())

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def __len__(self) -> int:
        return len(self._factories)

    def __repr__(self) -> str:
        return f"AdapterRegistry[n={len(self._factories)} · {list(self._factories.keys())}]"


def _default_synthetic_factory() -> NeuralPhaseExtractor:
    from neurophase.sensors.synthetic import SyntheticOscillatorSource

    return SyntheticOscillatorSource()


def _default_null_factory() -> NeuralPhaseExtractor:
    from neurophase.oscillators.neural_protocol import NullNeuralExtractor

    return NullNeuralExtractor()


def _build_default_registry() -> AdapterRegistry:
    reg = AdapterRegistry()
    reg.register("synthetic", _default_synthetic_factory)
    reg.register("null", _default_null_factory)
    return reg


#: Module-level singleton registry pre-populated with the two
#: in-repo factories. Downstream code can mutate it by calling
#: :meth:`AdapterRegistry.register`, or construct its own via
#: :class:`AdapterRegistry`.
DEFAULT_ADAPTER_REGISTRY: Final[AdapterRegistry] = _build_default_registry()
