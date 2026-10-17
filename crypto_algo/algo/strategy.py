"""Trend-following strategy: EMA crossover, filtered by a long-term trend EMA and RSI momentum."""
import pandas as pd

from .config import StrategyConfig
from .indicators import atr, ema, rsi

LONG, FLAT = 1, 0


class EmaRsiTrendStrategy:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    @property
    def warmup(self) -> int:
        return max(self.cfg.ema_slow, self.cfg.ema_trend, self.cfg.rsi_period, self.cfg.atr_period) + 1

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators plus `entry` / `exit` booleans, evaluated on each bar's close."""
        c = self.cfg
        out = df.copy()
        out["ema_fast"] = ema(out["close"], c.ema_fast)
        out["ema_slow"] = ema(out["close"], c.ema_slow)
        out["ema_trend"] = ema(out["close"], c.ema_trend)
        out["rsi"] = rsi(out["close"], c.rsi_period)
        out["atr"] = atr(out, c.atr_period)

        above = out["ema_fast"] > out["ema_slow"]
        cross_up = above & ~above.shift(fill_value=False)
        cross_down = ~above & above.shift(fill_value=False)
        uptrend = out["close"] > out["ema_trend"]
        momentum_ok = out["rsi"].between(c.rsi_min, c.rsi_max)

        out["entry"] = cross_up & uptrend & momentum_ok & out["atr"].notna()
        out["exit"] = cross_down
        return out
