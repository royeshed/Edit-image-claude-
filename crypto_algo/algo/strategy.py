"""Trend-following strategies. Each adds `entry` / `exit` booleans evaluated on each bar's close."""
import pandas as pd

from .config import StrategyConfig
from .ict import fair_value_gaps, killzone, liquidity_sweeps, structure
from .indicators import atr, ema, rsi
from .mtf import align, resample

LONG, FLAT = 1, 0


class EmaRsiTrendStrategy:
    """EMA crossover, filtered by a long-term trend EMA and RSI momentum."""

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


class DonchianBreakoutStrategy:
    """Classic channel breakout: buy a new N-bar high in an uptrend, sell a new M-bar low."""

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    @property
    def warmup(self) -> int:
        return max(self.cfg.donchian_entry, self.cfg.ema_trend, self.cfg.atr_period) + 1

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        out = df.copy()
        out["ema_trend"] = ema(out["close"], c.ema_trend)
        out["rsi"] = rsi(out["close"], c.rsi_period)
        out["atr"] = atr(out, c.atr_period)
        # Channels use prior bars only, so the current close can break out of them.
        upper = out["high"].rolling(c.donchian_entry).max().shift()
        lower = out["low"].rolling(c.donchian_exit).min().shift()
        out["entry"] = (out["close"] > upper) & (out["close"] > out["ema_trend"]) & out["atr"].notna()
        out["exit"] = out["close"] < lower
        return out


class ICTMultiTimeframeStrategy:
    """Top-down ICT entries, long and short, with every timeframe aligned.

    1. Bias (bias_tf):   above EMA-trend with bullish market structure -> longs only (mirror for shorts).
    2. Setup (setup_tf): price in discount of the dealing range, or a recent sell-side liquidity sweep.
    3. Trigger (base):   change of character up and/or a bullish fair-value-gap retest,
                         inside a killzone, RSI not stretched.
    Stop goes beyond the latest base-timeframe swing (clamped to 0.5-4 ATR); exits are the
    stop, the R-multiple target, or the bias structure flipping against the trade.
    """

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    @property
    def warmup(self) -> int:
        return 600

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        out = df.copy()
        out["atr"] = atr(out, c.atr_period)
        out["rsi"] = rsi(out["close"], c.rsi_period)

        # 1. Bias
        hb = resample(df, c.bias_tf)
        sb = structure(hb, c.swing_n)
        trend = ema(hb["close"], c.ema_trend)
        bias = pd.DataFrame({
            "long": (hb["close"] > trend) & (sb["structure"] == 1),
            "short": (hb["close"] < trend) & (sb["structure"] == -1),
            "struct": sb["structure"],
        })
        bias = align(bias, out.index, c.bias_tf)

        # 2. Setup
        hs = resample(df, c.setup_tf)
        ss = structure(hs, c.swing_n)
        bull_sweep, bear_sweep = liquidity_sweeps(hs, ss["swing_high"], ss["swing_low"])
        setup = pd.DataFrame({
            "long": (hs["close"] < ss["range_mid"]) | bull_sweep.rolling(c.setup_lookback, min_periods=1).max().astype(bool),
            "short": (hs["close"] > ss["range_mid"]) | bear_sweep.rolling(c.setup_lookback, min_periods=1).max().astype(bool),
        })
        setup = align(setup, out.index, c.setup_tf)

        # 3. Trigger
        st = structure(df, c.swing_n)
        fvg = fair_value_gaps(df, c.fvg_max_age)
        if c.trigger == "choch":
            trig_l, trig_s = st["choch_up"], st["choch_down"]
        elif c.trigger == "fvg":
            trig_l, trig_s = fvg["bull_fvg_retest"], fvg["bear_fvg_retest"]
        else:
            trig_l = st["choch_up"] | fvg["bull_fvg_retest"]
            trig_s = st["choch_down"] | fvg["bear_fvg_retest"]
        session = killzone(out.index) if c.use_killzone else True

        bl = bias["long"].fillna(False).astype(bool)
        bs = bias["short"].fillna(False).astype(bool)
        out["long_entry"] = bl & setup["long"].fillna(False).astype(bool) & trig_l & (out["rsi"] < c.rsi_max) & session
        out["short_entry"] = (bs & setup["short"].fillna(False).astype(bool) & trig_s & (out["rsi"] > 100 - c.rsi_max)
                              & session & c.allow_shorts)
        out["long_exit"] = bias["struct"] == -1
        out["short_exit"] = bias["struct"] == 1

        if c.entry == "limit":
            self._limit_entries(out, bl, bs, setup, session)
            out["entry"], out["exit"] = out["long_entry"], out["long_exit"]
            return out

        lo, hi = 0.5 * out["atr"], 4 * out["atr"]
        dist_l = (out["close"] - (st["swing_low"] - 0.1 * out["atr"])).clip(lower=lo, upper=hi)
        dist_s = ((st["swing_high"] + 0.1 * out["atr"]) - out["close"]).clip(lower=lo, upper=hi)
        out["stop_long"] = out["close"] - dist_l
        out["stop_short"] = out["close"] + dist_s
        out["entry"], out["exit"] = out["long_entry"], out["long_exit"]  # for long-only consumers
        return out

    def _limit_entries(self, out, bias_long, bias_short, setup, session) -> None:
        """A fresh displacement FVG in the bias direction places a limit order inside the gap.

        Bullish gap at bar i: [high[i-2], low[i]]. Limit at `limit_level` depth (0.5 = CE),
        stop below the displacement candle's low, valid for `fvg_max_age` bars.
        """
        c = self.cfg
        a = out["atr"]
        h2, l2 = out["high"].shift(2), out["low"].shift(2)
        bull = (out["low"] - h2) >= c.fvg_min_atr * a
        bear = (l2 - out["high"]) >= c.fvg_min_atr * a
        lim_l = out["low"] - c.limit_level * (out["low"] - h2)
        lim_s = out["high"] + c.limit_level * (l2 - out["high"])
        stop_l = out["low"].shift(1).combine(h2, min) - 0.1 * a
        stop_s = out["high"].shift(1).combine(l2, max) + 0.1 * a
        lo, hi = 0.5 * a, 4 * a
        out["limit_long"] = lim_l
        out["limit_short"] = lim_s
        out["stop_long"] = lim_l - (lim_l - stop_l).clip(lower=lo, upper=hi)
        out["stop_short"] = lim_s + (stop_s - lim_s).clip(lower=lo, upper=hi)
        out["long_entry"] = bias_long & setup["long"].fillna(False).astype(bool) & bull & session
        out["short_entry"] = bias_short & setup["short"].fillna(False).astype(bool) & bear & session & c.allow_shorts
        out.attrs["limit_bars"] = c.fvg_max_age


STRATEGIES = {"ema_rsi": EmaRsiTrendStrategy, "donchian": DonchianBreakoutStrategy,
              "ict_mtf": ICTMultiTimeframeStrategy}


def make_strategy(cfg: StrategyConfig):
    return STRATEGIES[cfg.name](cfg)
