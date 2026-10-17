"""Portfolio construction for the long-only spot bot: how capital is spread across coins.

weights[t] (fractions of total equity, each >= 0, sum <= 1) are decided at bar t's close
and earn bar t+1's returns; turnover |w[t] - w[t-1]| pays `cost` per side.
"""
import numpy as np
import pandas as pd

from .lab import BARS_PER_YEAR


def run(weights: pd.DataFrame, ret: pd.DataFrame, cost: float, cash_yield: float = 0.0, tf: str = "4h") -> pd.Series:
    w = weights.reindex_like(ret).fillna(0.0)
    gross = (w.shift(1) * ret.fillna(0)).sum(axis=1)
    fees = cost * w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    idle = (1 - w.shift(1).sum(axis=1)).clip(lower=0).fillna(1)
    return gross - fees + idle * cash_yield / BARS_PER_YEAR[tf]


def equal_sleeves(size: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Current bot: every trading coin owns 1/N of the account, used only while its signal is on."""
    return size.where(live, 0).div(live.sum(axis=1), axis=0)


def concentrated(size: pd.DataFrame, live: pd.DataFrame, floor: float, cap: float) -> pd.DataFrame:
    """Spread the account over coins whose signal is on: w_i = size_i / max(sum(size), floor), capped at `cap`."""
    s = size.where(live, 0)
    w = s.div(np.maximum(s.sum(axis=1), floor), axis=0)
    return w.clip(upper=cap)


def top_k(size: pd.DataFrame, live: pd.DataFrame, close: pd.DataFrame, k: int, lookback: int, cap: float) -> pd.DataFrame:
    """Only the k strongest coins (by `lookback`-bar return) among those whose signal is on, equal weight."""
    s = size.where(live, 0)
    mom = (close / close.shift(lookback) - 1).where(s > 0)
    rank = mom.rank(axis=1, ascending=False)
    keep = (rank <= k) & (s > 0)
    w = s.where(keep, 0)
    return w.div(np.maximum(w.sum(axis=1), 1.0), axis=0).clip(upper=cap) if cap < 1 else w.div(np.maximum(keep.sum(axis=1), 1), axis=0) * 0 + w.div(np.maximum(w.sum(axis=1), 1e-9), axis=0).clip(upper=cap)


def eligible_by_track_record(size: pd.DataFrame, ret: pd.DataFrame, live: pd.DataFrame, window: int, min_age: int) -> pd.DataFrame:
    """A coin may be traded only if it has `min_age` bars of history and the strategy's own P&L on it
    over the previous `window` bars is not negative (known at the time: uses past bars only)."""
    sleeve = (size.shift(1) * ret).fillna(0)
    track = sleeve.rolling(window, min_periods=window // 2).sum().shift(1)
    age = live.cumsum()
    return (age >= min_age) & ~(track < 0)
