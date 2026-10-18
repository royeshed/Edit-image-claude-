"""Which coins the portfolio bot trades: re-chosen every quarter from Binance data, without hindsight.

On each refresh date R (00:00 UTC on the first day of the calendar's months, by default 1 Jan/Apr/Jul/Oct)
the universe is the `size` coins with the highest average daily USDT volume over the 90 days before R,
among USDT pairs that, judged only from daily candles that closed by R:
  * traded in the last 5 days and have >= `min_age` days of uninterrupted daily history
    (a gap of more than 10 days starts the history again: a relisted coin is a new coin),
  * moved at least 10% (max / min close) over those 90 days (catches stablecoins missing from the list),
  * are not stablecoins, fiat, wrapped or staked tokens, leveraged tokens, tokenised gold or tokenised stocks.
This is the "mature_volume" rule of research/universe_selection.py, chosen on 2018-2022 and checked on
2023-2026; tests/test_universe.py checks that both give the same coins.

A universe chosen at R governs decisions on 4h candles that open at or after R; decisions on earlier
candles keep the previous universe.
"""
import itertools
import logging
import re
import time

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CALENDARS = {"Q-jan": (1, 4, 7, 10), "Q-feb": (2, 5, 8, 11), "Q-mar": (3, 6, 9, 12), "monthly": tuple(range(1, 13)),
             "Y-jan": (1,)}
GAP_DAYS = 10
# Not coins: stablecoins and fiat, wrapped/staked tokens, tokenised gold and stocks (CoinMarketCap tags, Oct 2026).
EXCLUDE = {
    "AEUR", "AUD", "BETH", "BFUSD", "BIDR", "BKRW", "BNSOL", "BRL", "BUSD", "DAI", "EUR", "EURI", "FDUSD", "FRAX", "GBP",
    "IDRT", "PAX", "PAXG", "PYUSD", "RLUSD", "SUSD", "TRY", "TUSD", "U", "USD1", "USDC", "USDE", "USDP", "USDS", "USDSB",
    "USDSOLD", "USDT", "UST", "USTC", "WBETH", "WBTC", "WNXM", "XAUT", "XUSD",
    "AVGOB", "BNCB", "COINB", "CRCLB", "DRAMB", "EWYB", "GOOGLB", "INTCB", "MSTRB", "MUB", "MUUB", "NVDAB", "QQQB",
    "SKHYB", "SNDKB", "SOXLB", "SPCXB", "SPYB", "TSLAB",
}


def refresh_date(t: pd.Timestamp, calendar: str = "Q-jan") -> pd.Timestamp:
    """The latest refresh date at or before t (00:00 UTC on the 1st of a calendar month)."""
    months = CALENDARS[calendar]
    y, m = t.year, max((mm for mm in months if mm <= t.month), default=None)
    if m is None:
        y, m = y - 1, max(months)
    return pd.Timestamp(year=y, month=m, day=1, tz="UTC")


def is_coin(base: str, bases: set[str]) -> bool:
    lev = re.fullmatch(r"(.+?)(UP|DOWN|BULL|BEAR)", base)
    return base not in EXCLUDE and not (lev and lev.group(1) in bases)


def score(daily: pd.DataFrame, R: pd.Timestamp, min_age: int = 365, lookback: int = 90) -> float:
    """Average daily USDT volume over the `lookback` days before R, or NaN if the coin is not eligible.

    `daily` has columns close and quote_volume, indexed by candle open time (UTC midnight)."""
    d = daily[daily.index < R].dropna(subset=["close"])
    if d.empty:
        return np.nan
    D = R - pd.Timedelta("1D")                      # the last candle that closed by R
    gaps = np.flatnonzero(np.diff(d.index.values).astype("timedelta64[D]").astype(int) > GAP_DAYS)
    seg = d.iloc[gaps[-1] + 1:] if len(gaps) else d   # uninterrupted history up to R
    if seg.index[-1] < D - pd.Timedelta("4D") or len(seg) < min_age:
        return np.nan
    win = seg[seg.index > D - pd.Timedelta(days=lookback)]
    if win.close.max() < 1.10 * win.close.min():
        return np.nan
    return float(win.quote_volume.mean())


def select(daily: dict[str, pd.DataFrame], R: pd.Timestamp, size: int = 15, min_age: int = 365) -> list[str]:
    """The `size` symbols with the highest score at R, best first."""
    s = pd.Series({sym: score(d, R, min_age) for sym, d in daily.items()}).dropna()
    return list(s.sort_values(ascending=False, kind="stable").index[:size])


# ---------- Binance ----------

def candidates(exchange, quote: str = "USDT", tradable_only: bool = True) -> list[str]:
    """Spot pairs against `quote` that are coins. With tradable_only=False also halted/delisted pairs
    (their history still counts when replaying the past)."""
    info = exchange.publicGetExchangeInfo()["symbols"]
    pairs = [s for s in info if s["quoteAsset"] == quote and (s["status"] == "TRADING" or not tradable_only)]
    bases = {s["baseAsset"] for s in pairs}
    return sorted(f"{s['baseAsset']}/{quote}" for s in pairs if is_coin(s["baseAsset"], bases))


def fetch_daily(exchange, symbol: str, R: pd.Timestamp, days: int = 1000) -> pd.DataFrame:
    """The last `days` daily candles that closed by R (close and USDT volume)."""
    rows = exchange.publicGetKlines({"symbol": symbol.replace("/", ""), "interval": "1d",
                                     "endTime": int(R.timestamp() * 1000) - 1, "limit": days})
    d = pd.DataFrame([[int(r[0]), float(r[4]), float(r[7])] for r in rows], columns=["ts", "close", "quote_volume"])
    d.index = pd.to_datetime(d.pop("ts"), unit="ms", utc=True)
    return d[d.index + pd.Timedelta("1D") <= R]


def choose(exchange, R: pd.Timestamp, size: int = 15, min_age: int = 365, tradable_only: bool = True,
           retries: int = 3, pause: float = 2.0, tradable_at=None) -> list[str]:
    """Fetch every candidate's daily candles and select. Raises instead of returning a list built from
    partial data: a coin skipped because of a network error would change the universe for a whole quarter.
    `tradable_at(symbol, R)`, when given, decides from data whether a pair was trading at R (a replay of the
    past cannot ask the exchange for the status it had then)."""
    import ccxt

    daily, failed = {}, []
    for sym in candidates(exchange, tradable_only=tradable_only):
        for attempt in range(retries):
            try:
                daily[sym] = fetch_daily(exchange, sym, R)
                break
            except ccxt.BadSymbol:            # the exchange does not serve this pair: it cannot be chosen
                log.info("Universe: %s unknown to the exchange, skipped", sym)
                break
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:   # timeouts, 5xx, rate limits
                if attempt == retries - 1:
                    failed.append(sym)
                    log.warning("Universe: %s failed %d times: %s", sym, retries, e)
                else:
                    time.sleep(pause * 2 ** attempt)
    if failed:
        raise RuntimeError(f"universe for {R.date()} not chosen: no daily candles for {len(failed)} pairs ({', '.join(failed[:5])}...)")
    ranked = select(daily, R, len(daily), min_age)
    chosen = list(itertools.islice((s for s in ranked if tradable_at(s, R)), size)) if tradable_at else ranked[:size]
    if not chosen:
        raise RuntimeError(f"universe for {R.date()} not chosen: no eligible coin among {len(daily)} pairs")
    return chosen
