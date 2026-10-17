"""Vectorised strategy constructors returning (time x coin) positions in [-1, 1]."""
import numpy as np
import pandas as pd


def hold(entry: pd.DataFrame, bars: int) -> pd.DataFrame:
    """1 while any entry fired within the last `bars` bars (entry at t -> in position t..t+bars-1)."""
    return entry.astype(float).rolling(bars, min_periods=1).max().fillna(0)


def vol_target(pos: pd.DataFrame, ret: pd.DataFrame, target: float, n: int = 24 * 30, cap: float = 2.0) -> pd.DataFrame:
    """Scale each coin to `target` annualised volatility (hourly bars), capped at `cap`x."""
    vol = ret.rolling(n, min_periods=n // 2).std() * np.sqrt(8760)
    return (pos * (target / vol)).clip(-cap, cap)
