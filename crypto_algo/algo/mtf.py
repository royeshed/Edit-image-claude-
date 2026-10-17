"""Multi-timeframe helpers: resample a base timeframe and align higher-timeframe values back onto it."""
import pandas as pd

def to_freq(tf: str) -> str:
    """Exchange timeframe ("5m", "1h", "1d") -> pandas frequency ("5min", "1h", "1D")."""
    unit = {"m": "min", "h": "h", "d": "D", "w": "W"}[tf[-1]]
    return tf[:-1] + unit


AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(to_freq(rule), label="left", closed="left").agg(AGG).dropna()


def align(htf: pd.DataFrame | pd.Series, base_index: pd.DatetimeIndex, rule: str) -> pd.DataFrame | pd.Series:
    """Map higher-timeframe values onto base bars, visible only once the HTF bar has closed.

    An HTF bar labelled t closes at t + rule, which is the close of the base bar labelled
    t + rule - base_step. From that base bar onward its values are known.
    """
    base_step = base_index[1] - base_index[0]
    shifted = htf.copy()
    shifted.index = shifted.index + pd.Timedelta(to_freq(rule)) - base_step
    return shifted.reindex(base_index, method="ffill")
