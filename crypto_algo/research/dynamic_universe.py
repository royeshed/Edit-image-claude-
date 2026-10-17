"""Point-in-time coin universes chosen by trailing trading volume (no hindsight, delisted coins included).

At each refresh date D the universe is the `n` assets with the highest average daily USDT volume
over the 90 days before D, among assets with at least `min_days` of history that still traded in
the last few days. A symbol that stopped trading for more than GAP_DAYS and came back (LUNA in
2022) is split into two assets, SYM and SYM#2, because the relisted coin is a different asset.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

UNI = Path(__file__).resolve().parents[1] / "data" / "universe"
GAP_DAYS = 10
FREQ = {"yearly": "YS", "quarterly": "QS", "monthly": "MS"}


def segments(idx: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous trading periods separated by gaps longer than GAP_DAYS."""
    if len(idx) == 0:
        return []
    gaps = np.flatnonzero(np.diff(idx.values).astype("timedelta64[D]").astype(int) > GAP_DAYS)
    bounds = [0, *(gaps + 1), len(idx)]
    return [(idx[a], idx[b - 1]) for a, b in zip(bounds[:-1], bounds[1:])]


def load_daily() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Daily close and USDT volume per asset, plus each asset's (symbol, start, end) for splitting other timeframes."""
    closes, vols, spans = {}, {}, {}
    for f in sorted(UNI.glob("*_1d.csv")):
        sym = f.name[:-7]
        d = pd.read_csv(f, index_col=0)
        if d.empty:
            continue
        d.index = pd.to_datetime(d.index, utc=True)
        for i, (a, b) in enumerate(segments(d.index)):
            name = sym if i == 0 else f"{sym}#{i + 1}"
            seg = d.loc[a:b]
            closes[name], vols[name], spans[name] = seg["close"], seg["quote_volume"], (sym, a, b)
    return pd.DataFrame(closes), pd.DataFrame(vols), spans


def universes(vol: pd.DataFrame, freq: str, n: int, min_days: int = 60, lookback: int = 90,
              start: str = "2018-01-01") -> dict[pd.Timestamp, list[str]]:
    """{refresh date: [assets]} using only data before each refresh date (the start date is always one)."""
    out = {}
    hist = vol.notna().cumsum()
    first = pd.Timestamp(start, tz="UTC")
    dates = pd.date_range(first, vol.index[-1], freq=FREQ.get(freq, freq))  # also a pandas alias like "QS-FEB"
    for d in dates if len(dates) and dates[0] == first else dates.insert(0, first):
        past = vol[vol.index < d]
        if past.empty:
            continue
        recent_ok = past.iloc[-5:].notna().any()
        enough = hist[hist.index < d].iloc[-1] >= min_days
        score = past.iloc[-lookback:].mean().where(recent_ok & enough).dropna()
        out[d] = list(score.sort_values(ascending=False).index[:n])
    return out


def membership(univ: dict, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """1 while an asset is in the universe chosen at the latest refresh date."""
    m = pd.DataFrame(0.0, index=index, columns=columns)
    dates = sorted(univ)
    for i, d in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else index[-1] + pd.Timedelta("1D")
        cols = [c for c in univ[d] if c in m.columns]
        m.loc[(m.index >= d) & (m.index < end), cols] = 1.0
    return m


def load_bars(assets, spans: dict, tf: str = "4h") -> dict:
    """OHLC panels for the given assets on a common grid, split the same way as the daily data."""
    frames = {}
    for name in assets:
        sym, a, b = spans[name]
        f = UNI / f"{sym}_{tf}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, index_col=0)
        d.index = pd.to_datetime(d.index, utc=True)
        seg = d.loc[a: b + pd.Timedelta("1D")]
        if b >= d.index[-1].floor("D") - pd.Timedelta("1D"):
            seg = seg.iloc[:-1]  # still trading: the last bar is still forming
        frames[name] = seg
    grid = pd.date_range(pd.Timestamp("2018-01-01", tz="UTC"), max(f.index[-1] for f in frames.values()), freq=tf.replace("d", "D"))
    panel = {k: pd.DataFrame({c: f[k].reindex(grid) for c, f in frames.items()}) for k in ("open", "high", "low", "close", "volume")}
    c = panel["close"]
    panel["ret"] = c.ffill().pct_change(fill_method=None).where(c.notna())
    return panel
