# `data/market/` — bundled market fixtures

This directory is the only sub-tree under `data/` that is tracked in
git. Everything else under `/data/` is `.gitignore`d (large-dataset
policy — see `.gitignore`).

## `btc_1m_sample.csv`

**Source.** Binance public market-data archive.
  https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2024-06-01.zip
(archive last-modified 2024-06-02; retrieved 2026-04-13).

**Content.** One full UTC trading day of BTC/USDT 1-minute klines:
1440 candles, normalised to a single CSV with leading `#`-comment
provenance lines followed by one header row and the 12-column Binance
klines schema:

```
open_time_ms, open, high, low, close, volume, close_time_ms,
quote_volume, trades, taker_buy_base, taker_buy_quote, ignore
```

**Purpose.** Used by `neurophase/experiments/plv_market_methodology.py`
as a **real market-side fixture** for PLV methodology validation. Not
a trading signal, not financial advice, not claimed as forward-looking
data. It is a fixed historical fixture that lets the PLV + surrogate
pipeline be validated against real non-stationary prices rather than
purely synthetic ones.

**License.** Binance public data archive terms apply. See the Binance
Data Vision portal for the current terms.

**Do not** rotate this file silently. If the experiment needs a
different window or a different symbol, bundle a new file under a
different name (`btc_1m_<YYYYMM>.csv`, `ethusdt_1m_sample.csv`, etc.)
and update the corresponding experiment pointer — preserve
reproducibility of older results.
