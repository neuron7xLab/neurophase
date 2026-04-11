"""BTC Field Order v3.2 — structured LLM intelligence payload builder.

Reference prompt protocol from the BTC_Field_Order_v3_2 document
(neuron7xLab, 2026). This module does **not** call any LLM. It only:

    1. Validates a strictly-typed request payload against the v3.2 schema.
    2. Serialises it into the JSON shape the LLM wrapper consumes.
    3. Provides constructors for the IF–THEN rule section.

Once this payload is built, the caller can dispatch it to a chosen LLM
(via their own SDK, Anthropic / OpenAI / local). Keeping the payload
builder here — inside neurophase — lets the physics layer drive LLM
calls without entangling the two substrates.

All dataclasses are frozen and pure-data — no state, no network, no
secrets. The honest-null contract applies: missing required fields
produce a ``ValueError`` at construction, not a silent default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal

Timeframe = Literal["scalp", "intraday", "swing", "cycle"]
Mode = Literal["quick_brief", "pro_report", "signal_scan", "execution_plan", "stress_test"]

PROMPT_VERSION: Final[str] = "v3.2"
JURISDICTION: Final[str] = "UA"
TIMEZONE: Final[str] = "Europe/Kyiv"


@dataclass(frozen=True)
class SpotBlock:
    """Spot-market snapshot fields required by the v3.2 schema."""

    price: float
    vol: float
    dvwap: float
    dominance: float

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"spot.price must be > 0, got {self.price}")
        if self.vol < 0:
            raise ValueError(f"spot.vol must be >= 0, got {self.vol}")
        if not 0.0 <= self.dominance <= 100.0:
            raise ValueError(f"spot.dominance must be in [0, 100], got {self.dominance}")


@dataclass(frozen=True)
class DerivativesBlock:
    """Derivatives snapshot: OI, funding, basis, gamma, liquidation clusters."""

    oi: float
    fr: float
    basis: float
    gamma: float
    liq_clusters: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.oi < 0:
            raise ValueError(f"derivatives.oi must be >= 0, got {self.oi}")
        if not -1.0 <= self.fr <= 1.0:
            raise ValueError(f"derivatives.fr must be in [-1, 1], got {self.fr}")


@dataclass(frozen=True)
class OrderBookBlock:
    """Level-2 order book snapshot.

    ``bids`` and ``asks`` are tuples of (price, size) pairs; walls are
    marked separately by side.
    """

    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]
    walls: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.bids or not self.asks:
            raise ValueError("order book must have at least one bid and one ask")


@dataclass(frozen=True)
class WhaleEvent:
    """A single whale movement event."""

    direction: Literal["in", "out"]
    size_btc: float

    def __post_init__(self) -> None:
        if self.size_btc <= 0:
            raise ValueError(f"whale size_btc must be > 0, got {self.size_btc}")


@dataclass(frozen=True)
class OnchainBlock:
    """On-chain flow snapshot."""

    exch_in: float
    exch_out: float
    whale_moves: tuple[WhaleEvent, ...] = ()
    miner_reserves: float = 0.0

    def __post_init__(self) -> None:
        if self.exch_in < 0 or self.exch_out < 0:
            raise ValueError("exch_in / exch_out must be non-negative")
        if self.miner_reserves < 0:
            raise ValueError(f"miner_reserves must be non-negative, got {self.miner_reserves}")


@dataclass(frozen=True)
class Scenario:
    """One of the three Bull / Base / Bear scenarios."""

    name: Literal["Bull", "Base", "Bear"]
    prob: float
    triggers: tuple[str, ...]
    invalidation: tuple[str, ...]
    targets: tuple[float, ...]
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.prob <= 100.0:
            raise ValueError(f"scenario.prob must be in [0, 100], got {self.prob}")
        if not 1 <= self.confidence <= 10:
            raise ValueError(f"scenario.confidence must be in [1, 10], got {self.confidence}")


@dataclass(frozen=True)
class BTCFieldOrderRequest:
    """Strictly-typed request payload for the v3.2 protocol.

    Downstream LLM wrappers should serialise this via
    ``build_signal_scan_payload`` (or a future ``build_pro_report_payload``)
    before dispatching to the model.
    """

    timeframe: Timeframe
    mode: Mode
    window_from: str  # ISO-8601
    window_to: str  # ISO-8601
    spot: SpotBlock
    derivatives: DerivativesBlock
    orderbook: OrderBookBlock
    onchain: OnchainBlock
    scenarios: tuple[Scenario, ...] = field(default_factory=tuple)
    max_leverage: float = 3.0
    risk_per_trade: float = 0.01

    def __post_init__(self) -> None:
        if self.max_leverage <= 0:
            raise ValueError(f"max_leverage must be > 0, got {self.max_leverage}")
        if not 0.0 < self.risk_per_trade < 1.0:
            raise ValueError(f"risk_per_trade must be in (0, 1), got {self.risk_per_trade}")


def validate_request(req: BTCFieldOrderRequest) -> list[str]:
    """Return a list of non-fatal warnings about the request.

    Fatal errors are raised at dataclass construction. This function
    adds softer hygiene checks that map to guardrails in the v3.2
    protocol (``assurance.missing_inputs``, ``quality_flags.coverage``).
    """
    warnings: list[str] = []
    if len(req.scenarios) < 3:
        warnings.append("v3.2 protocol expects Bull / Base / Bear — at least 3 scenarios")
    if req.derivatives.fr >= 0.05:
        warnings.append(
            f"funding rate {req.derivatives.fr:.4f} extreme — expect counter-trend bias"
        )
    elif req.derivatives.fr <= -0.05:
        warnings.append(f"funding rate {req.derivatives.fr:.4f} extreme negative — squeeze risk")
    if not req.onchain.whale_moves:
        warnings.append("no whale_moves provided — on-chain coverage partial")
    return warnings


def build_signal_scan_payload(req: BTCFieldOrderRequest) -> dict[str, Any]:
    """Serialise a request into the v3.2 JSON shape for ``signal_scan`` mode.

    Output follows the ``meta``/``dashboard``/``scenarios``/... layout
    from section 8 of the Field Order v3.2 document. The produced dict
    is ready to be passed as a user message to an LLM SDK.

    Parameters
    ----------
    req : BTCFieldOrderRequest
        Validated request payload.

    Returns
    -------
    dict[str, Any]
    """
    payload: dict[str, Any] = {
        "strict_mode": True,
        "meta": {
            "prompt_version": PROMPT_VERSION,
            "jurisdiction_tag": JURISDICTION,
            "tz": TIMEZONE,
            "timeframe": req.timeframe,
            "mode": req.mode,
            "window": {"from": req.window_from, "to": req.window_to},
        },
        "spot": {
            "price": req.spot.price,
            "vol": req.spot.vol,
            "dvwap": req.spot.dvwap,
            "dominance": req.spot.dominance,
        },
        "derivatives": {
            "oi": req.derivatives.oi,
            "fr": req.derivatives.fr,
            "basis": req.derivatives.basis,
            "gamma": req.derivatives.gamma,
            "liq_clusters": list(req.derivatives.liq_clusters),
        },
        "orderbook": {
            "bids": [list(b) for b in req.orderbook.bids],
            "asks": [list(a) for a in req.orderbook.asks],
            "walls": list(req.orderbook.walls),
        },
        "onchain": {
            "exch_in": req.onchain.exch_in,
            "exch_out": req.onchain.exch_out,
            "whale_moves": [
                {"dir": w.direction, "size_btc": w.size_btc} for w in req.onchain.whale_moves
            ],
            "miner_reserves": req.onchain.miner_reserves,
        },
        "scenarios": [
            {
                "name": s.name,
                "prob": s.prob,
                "triggers": list(s.triggers),
                "invalidation": list(s.invalidation),
                "targets": list(s.targets),
                "confidence": s.confidence,
            }
            for s in req.scenarios
        ],
        "constraints": {
            "max_leverage": req.max_leverage,
            "risk_per_trade": req.risk_per_trade,
        },
        "qa": {"warnings": validate_request(req)},
    }
    return payload
