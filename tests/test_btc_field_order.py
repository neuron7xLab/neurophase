"""Tests for neurophase.intel.btc_field_order."""

from __future__ import annotations

import pytest

from neurophase.intel.btc_field_order import (
    BTCFieldOrderRequest,
    DerivativesBlock,
    OnchainBlock,
    OrderBookBlock,
    Scenario,
    SpotBlock,
    WhaleEvent,
    build_signal_scan_payload,
    validate_request,
)


def _spot() -> SpotBlock:
    return SpotBlock(price=70000.0, vol=1200.0, dvwap=0.002, dominance=53.5)


def _derivatives() -> DerivativesBlock:
    return DerivativesBlock(
        oi=1.2e9, fr=0.01, basis=0.001, gamma=0.0, liq_clusters=(68000.0, 72000.0)
    )


def _orderbook() -> OrderBookBlock:
    return OrderBookBlock(
        bids=((69900.0, 2.1), (69850.0, 3.4)),
        asks=((70100.0, 1.5), (70200.0, 2.8)),
    )


def _onchain() -> OnchainBlock:
    return OnchainBlock(
        exch_in=1200.0,
        exch_out=800.0,
        whale_moves=(WhaleEvent(direction="in", size_btc=600.0),),
    )


def _request(scenarios: tuple[Scenario, ...] = ()) -> BTCFieldOrderRequest:
    return BTCFieldOrderRequest(
        timeframe="intraday",
        mode="signal_scan",
        window_from="2026-04-11T00:00:00+02:00",
        window_to="2026-04-11T04:00:00+02:00",
        spot=_spot(),
        derivatives=_derivatives(),
        orderbook=_orderbook(),
        onchain=_onchain(),
        scenarios=scenarios,
    )


def test_build_signal_scan_payload_shape() -> None:
    bull = Scenario(
        name="Bull",
        prob=45.0,
        triggers=("exch_out > 500",),
        invalidation=("fr > 0.05",),
        targets=(72000.0, 74000.0),
        confidence=7.0,
    )
    base = Scenario(
        name="Base",
        prob=35.0,
        triggers=("chop",),
        invalidation=("close < 68000",),
        targets=(71000.0,),
        confidence=6.0,
    )
    bear = Scenario(
        name="Bear",
        prob=20.0,
        triggers=("whale_out > 500",),
        invalidation=("close > 72000",),
        targets=(66000.0,),
        confidence=6.0,
    )
    req = _request(scenarios=(bull, base, bear))
    payload = build_signal_scan_payload(req)
    assert payload["strict_mode"] is True
    assert payload["meta"]["prompt_version"] == "v3.2"
    assert payload["meta"]["tz"] == "Europe/Kyiv"
    assert payload["meta"]["timeframe"] == "intraday"
    assert payload["spot"]["price"] == 70000.0
    assert len(payload["scenarios"]) == 3
    assert payload["onchain"]["whale_moves"][0]["size_btc"] == 600.0


def test_validate_request_flags_missing_scenarios() -> None:
    req = _request(scenarios=())
    warnings = validate_request(req)
    assert any("Bull / Base / Bear" in w for w in warnings)


def test_validate_request_flags_extreme_funding() -> None:
    req = BTCFieldOrderRequest(
        timeframe="intraday",
        mode="signal_scan",
        window_from="2026-04-11T00:00:00+02:00",
        window_to="2026-04-11T04:00:00+02:00",
        spot=_spot(),
        derivatives=DerivativesBlock(oi=1.0e9, fr=0.08, basis=0.0, gamma=0.0),
        orderbook=_orderbook(),
        onchain=_onchain(),
    )
    warnings = validate_request(req)
    assert any("funding rate" in w for w in warnings)


def test_rejects_bad_spot_price() -> None:
    with pytest.raises(ValueError, match=r"spot\.price"):
        SpotBlock(price=0.0, vol=1.0, dvwap=0.0, dominance=50.0)


def test_rejects_bad_dominance() -> None:
    with pytest.raises(ValueError, match="dominance"):
        SpotBlock(price=70000.0, vol=1.0, dvwap=0.0, dominance=150.0)


def test_rejects_bad_funding_rate() -> None:
    with pytest.raises(ValueError, match="fr"):
        DerivativesBlock(oi=1.0, fr=2.0, basis=0.0, gamma=0.0)


def test_rejects_empty_orderbook_side() -> None:
    with pytest.raises(ValueError, match="at least one"):
        OrderBookBlock(bids=(), asks=((70000.0, 1.0),))


def test_rejects_bad_whale_size() -> None:
    with pytest.raises(ValueError, match="size_btc"):
        WhaleEvent(direction="in", size_btc=-5.0)


def test_scenario_rejects_bad_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Scenario(
            name="Bull",
            prob=50.0,
            triggers=("x",),
            invalidation=("y",),
            targets=(1.0,),
            confidence=15.0,
        )


def test_rejects_bad_leverage() -> None:
    with pytest.raises(ValueError, match="max_leverage"):
        BTCFieldOrderRequest(
            timeframe="intraday",
            mode="signal_scan",
            window_from="2026-04-11T00:00:00+02:00",
            window_to="2026-04-11T04:00:00+02:00",
            spot=_spot(),
            derivatives=_derivatives(),
            orderbook=_orderbook(),
            onchain=_onchain(),
            max_leverage=0.0,
        )
