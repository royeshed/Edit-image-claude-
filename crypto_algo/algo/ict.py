"""ICT (Inner Circle Trader) building blocks, computed without look-ahead.

Every series is aligned to the bar at whose close the information is known.
A swing point at bar i, for example, is only confirmed `n` bars later, so it
first appears at i + n.
"""
import numpy as np
import pandas as pd


def swings(df: pd.DataFrame, n: int = 3) -> tuple[pd.Series, pd.Series]:
    """Latest confirmed swing-high and swing-low *levels* (fractals of 2n+1 bars)."""
    w = 2 * n + 1
    is_high = df["high"] == df["high"].rolling(w, center=True).max()
    is_low = df["low"] == df["low"].rolling(w, center=True).min()
    swing_high = df["high"].where(is_high).shift(n).ffill()
    swing_low = df["low"].where(is_low).shift(n).ffill()
    return swing_high, swing_low


def structure(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Market structure: +1 after a close above the last swing high (bullish BOS), -1 after a close below the last swing low.

    `choch_up` / `choch_down` mark the bar where structure flips (change of character).
    """
    sh, sl = swings(df, n)
    up = df["close"] > sh.shift()
    down = df["close"] < sl.shift()
    state = pd.Series(np.where(up, 1.0, np.where(down, -1.0, np.nan)), index=df.index).ffill().fillna(0)
    prev = state.shift(fill_value=0)
    return pd.DataFrame({
        "swing_high": sh, "swing_low": sl, "structure": state,
        "choch_up": (state == 1) & (prev == -1), "choch_down": (state == -1) & (prev == 1),
        # Premium / discount within the current dealing range.
        "range_mid": (sh + sl) / 2,
    })


def liquidity_sweeps(df: pd.DataFrame, sh: pd.Series, sl: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Wick through a prior swing that closes back inside: sell-side sweep (bullish), buy-side sweep (bearish)."""
    prev_sl, prev_sh = sl.shift(), sh.shift()
    bull = (df["low"] < prev_sl) & (df["close"] > prev_sl)
    bear = (df["high"] > prev_sh) & (df["close"] < prev_sh)
    return bull, bear


def fair_value_gaps(df: pd.DataFrame, max_age: int = 24) -> pd.DataFrame:
    """Most recent bullish / bearish FVG still within `max_age` bars, and whether this bar retests it.

    Bullish FVG at bar i: low[i] > high[i-2]; the gap is [high[i-2], low[i]].
    A retest is a later bar whose low trades into the gap while closing above its bottom.
    """
    h2, l2 = df["high"].shift(2), df["low"].shift(2)
    bull_new = df["low"] > h2
    bear_new = df["high"] < l2
    idx = pd.Series(np.arange(len(df)), index=df.index, dtype=float)

    def track(new, top, bottom):
        t = top.where(new).ffill()
        b = bottom.where(new).ffill()
        age = idx - idx.where(new).ffill()
        return t.shift(), b.shift(), age.shift()  # the gap is known from the next bar on

    bt, bb, bage = track(bull_new, df["low"], h2)
    st, sb, sage = track(bear_new, l2, df["high"])
    return pd.DataFrame({
        "bull_fvg_retest": (bage <= max_age) & (df["low"] <= bt) & (df["close"] > bb),
        "bear_fvg_retest": (sage <= max_age) & (df["high"] >= sb) & (df["close"] < st),
    })


def killzone(index: pd.DatetimeIndex) -> pd.Series:
    """London (07-10 UTC) and New York (12-16 UTC) sessions, when most displacement happens."""
    h = index.hour
    return pd.Series(((h >= 7) & (h < 10)) | ((h >= 12) & (h < 16)), index=index)
