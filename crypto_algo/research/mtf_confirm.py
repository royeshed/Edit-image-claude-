"""Higher-timeframe confirmation research; the implementation lives in algo/htf.py (shared with the live bot)."""
from algo.htf import FREQ, align, confirmations, confirmed_breakouts, resample, state  # noqa: F401


def breakout_signals(p: dict, n: int, m: int):
    c = p["close"]
    return c > p["high"].rolling(n).max().shift(), c < p["low"].rolling(m).min().shift()
