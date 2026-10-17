"""Market data: download OHLCV from an exchange (with CSV cache) or generate synthetic data."""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


# Binance's official market-data mirror; reachable from regions where api.binance.com is not.
BINANCE_PUBLIC_DATA = "https://data-api.binance.vision/api/v3"


def make_exchange(name: str, api_key: str | None = None, secret: str | None = None):
    import ccxt

    params = {"enableRateLimit": True}
    if api_key and secret:
        params.update(apiKey=api_key, secret=secret)
    elif name == "binance":
        params["options"] = {"fetchMarkets": ["spot"]}
    ex = getattr(ccxt, name)(params)
    if name == "binance" and not api_key:
        ex.urls["api"]["public"] = BINANCE_PUBLIC_DATA
    # Honour an HTTPS proxy and custom CA bundle from the environment (ccxt ignores them by default).
    if os.environ.get("HTTPS_PROXY"):
        ex.https_proxy = os.environ["HTTPS_PROXY"]
    ca = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if ca:
        ex.session.verify = ca
    return ex


def to_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", *COLUMNS])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").set_index("ts").sort_index().astype(float)


def fetch_ohlcv(exchange, symbol: str, timeframe: str, since: pd.Timestamp, limit: int = 1000) -> pd.DataFrame:
    """Page through the exchange's OHLCV endpoint from `since` until now."""
    step_ms = exchange.parse_timeframe(timeframe) * 1000
    cursor = int(since.timestamp() * 1000)
    now_ms = exchange.milliseconds()
    rows: list[list] = []
    while cursor < now_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + step_ms
        time.sleep(exchange.rateLimit / 1000)
    if not rows:
        raise RuntimeError(f"No data returned for {symbol} {timeframe}")
    return to_frame(rows)


def load_history(exchange_name: str, symbol: str, timeframe: str, days: int, cache_dir: str = "data") -> pd.DataFrame:
    """Load OHLCV from the local CSV cache, downloading if missing."""
    path = Path(cache_dir) / f"{exchange_name}_{symbol.replace('/', '-')}_{timeframe}_{days}d.csv"
    if path.exists():
        return load_csv(path)
    exchange = make_exchange(exchange_name)
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    df = fetch_ohlcv(exchange, symbol, timeframe, since)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df[COLUMNS].astype(float)


def synthetic_ohlcv(bars: int = 5000, start_price: float = 30_000, freq: str = "1h", seed: int = 7) -> pd.DataFrame:
    """Regime-switching random walk, useful for testing the pipeline offline."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.0008, bars // 250 + 1), 250)[:bars]
    vol = np.repeat(rng.uniform(0.004, 0.012, bars // 250 + 1), 250)[:bars]
    rets = rng.normal(drift, vol)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, vol, bars)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    idx = pd.date_range("2024-01-01", periods=bars, freq=freq, tz="UTC")
    volume = rng.lognormal(3, 0.5, bars)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
