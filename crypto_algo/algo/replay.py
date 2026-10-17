"""Replay history through the live Trader with a paper broker.

The backtester and the live loop are separate code paths; replaying the same candles
through both is a check that the bot will actually trade the way the backtest says.
The fake exchange serves candles up to the current bar (the last one counted as still
forming) and quotes that bar's close, i.e. one poll per candle.
"""
import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .broker import PaperBroker
from .config import Config
from .live import Trader


class ReplayExchange:
    def __init__(self, df: pd.DataFrame, start: int):
        self.df, self.i = df, start

    def fetch_ohlcv(self, symbol, timeframe, limit=500):
        w = self.df.iloc[max(0, self.i - limit + 1): self.i + 1]
        return [[int(ts.timestamp() * 1000), *row] for ts, row in zip(w.index, w.to_numpy())]

    def fetch_ticker(self, symbol):
        return {"last": float(self.df["close"].iloc[self.i])}


def replay(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Return the paper equity curve from stepping the Trader through `df`."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = replace(cfg, state_file=str(Path(tmp) / "state.json"))
        broker = PaperBroker(cfg.initial_capital, cfg.costs.fee_rate, cfg.costs.slippage)
        trader = Trader(cfg, None, broker)
        start = trader.strategy.warmup + 1
        trader.ex = ex = ReplayExchange(df, start)
        equity = {}
        for i in range(start, len(df)):
            ex.i = i
            trader.step()
            cash, base = broker.balance()
            equity[df.index[i]] = cash + base * df["close"].iloc[i]
    return pd.Series(equity, name="equity")
