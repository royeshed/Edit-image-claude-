"""Long-only, short-horizon strategy families for spot (positions are 0 or 1 per coin).

All parameters are in bars of the panel's timeframe, except `gate_days` (daily trend filter).
Signals use closes up to bar t; the lab charges bar t+1's return to the position decided at t.
"""
import numpy as np
import pandas as pd

from .features import rsi


def _state(entry: pd.DataFrame, exit_: pd.DataFrame) -> pd.DataFrame:
    """1 from an entry until the next exit (exit wins on the same bar)."""
    s = pd.DataFrame(np.where(exit_, 0.0, np.where(entry, 1.0, np.nan)), index=entry.index, columns=entry.columns)
    return s.ffill().fillna(0.0)


def bars_per_day(index: pd.DatetimeIndex) -> int:
    return int(pd.Timedelta("1D") / (index[1] - index[0]))


def daily_gate(p: dict, gate_days: int = 50) -> pd.DataFrame:
    """Price above its ~gate_days-day EMA (computed on the panel's own bars)."""
    c = p["close"]
    return c > c.ewm(span=gate_days * bars_per_day(c.index), adjust=False).mean()


def breakout(p, n: int, m: int):
    c = p["close"]
    return _state(c > p["high"].rolling(n).max().shift(), c < p["low"].rolling(m).min().shift())


def ema_cross(p, fast: int, slow: int):
    c = p["close"]
    return (c.ewm(span=fast, adjust=False).mean() > c.ewm(span=slow, adjust=False).mean()).astype(float)


def short_momentum(p, base: int):
    c = p["close"]
    s = sum(np.sign(c / c.shift(int(base * m)) - 1) for m in (0.25, 0.5, 1, 2, 4)) / 5
    return (s > 0).astype(float)


def rsi2_reversion(p, level: float, trend: int = 200):
    """Connors-style: buy RSI(2) < level above the `trend`-bar SMA, sell on a close above the 5-bar SMA."""
    c = p["close"]
    entry = (rsi(c, 2) < level) & (c > c.rolling(trend).mean())
    return _state(entry, c > c.rolling(5).mean())


def bollinger_reversion(p, n: int = 20, k: float = 2.0, trend: int = 200):
    c = p["close"]
    mid, sd = c.rolling(n).mean(), c.rolling(n).std()
    entry = (c < mid - k * sd) & (c > c.rolling(trend).mean())
    return _state(entry, c > mid)


def session_breakout(p, range_end: int = 8, trade_until: int = 22):
    """Buy a break of the Asia-session (00:00-range_end UTC) high, exit by `trade_until` UTC or below the Asia low."""
    c, h, l = p["close"], p["high"], p["low"]
    hour = np.asarray(c.index.hour)
    day = c.index.floor("D")
    in_range = np.broadcast_to((hour < range_end)[:, None], c.shape)
    asia_hi = h.where(in_range).groupby(day).cummax().groupby(day).ffill()
    asia_lo = l.where(in_range).groupby(day).cummin().groupby(day).ffill()
    window = np.broadcast_to(((hour >= range_end) & (hour < trade_until))[:, None], c.shape)
    entry = window & (c > asia_hi)
    exit_ = ~window | (c < asia_lo)
    return _state(entry, exit_)


def squeeze_breakout(p, n: int = 20, pct_window: int = 120, pct: float = 0.2):
    """Bollinger-width in its lowest `pct` of the last `pct_window` bars, then a close above the upper band."""
    c = p["close"]
    mid, sd = c.rolling(n).mean(), c.rolling(n).std()
    width = (4 * sd / mid)
    squeezed = width.rolling(pct_window).rank(pct=True).shift() <= pct
    entry = squeezed.rolling(n).max().astype(bool) & (c > mid + 2 * sd)
    return _state(entry, c < mid)


def families(tf: str) -> dict:
    """Parameter grid per family; values are in bars of `tf`."""
    g = {
        **{f"breakout {n}/{m}": (breakout, dict(n=n, m=m)) for n, m in [(20, 10), (20, 20), (50, 20), (50, 50), (100, 20), (100, 50)]},
        **{f"ema {f}/{s}": (ema_cross, dict(fast=f, slow=s)) for f, s in [(10, 30), (20, 50), (50, 200)]},
        **{f"momentum {b}": (short_momentum, dict(base=b)) for b in (24, 48, 96)},
        **{f"rsi2<{lv}": (rsi2_reversion, dict(level=lv)) for lv in (5, 10, 20)},
        "bollinger reversion": (bollinger_reversion, {}),
        **{f"squeeze {pc}": (squeeze_breakout, dict(pct=pc)) for pc in (0.1, 0.2)},
    }
    if tf in ("15m", "1h"):
        g["session breakout 8-22"] = (session_breakout, dict(range_end=8, trade_until=22))
        g["session breakout 8-16"] = (session_breakout, dict(range_end=8, trade_until=16))
    return g
