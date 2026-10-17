"""Ensemble time-series momentum: the strategy family that survived the research."""
import numpy as np
import pandas as pd

from .strategies import vol_target


def ensemble_signal(close: pd.DataFrame, base: int, mults=(0.25, 0.5, 1, 2, 4), continuous: bool = True) -> pd.DataFrame:
    """Average of sign(return over L) for L = base * m. In [-1, 1]."""
    sigs = [np.sign(close / close.shift(int(base * m)) - 1) for m in mults]
    s = sum(sigs) / len(sigs)
    return s if continuous else np.sign(s)


def decide_at(sig: pd.DataFrame, hours: tuple[int, ...]) -> pd.DataFrame:
    """Only change position at the close of the given UTC hours; hold in between."""
    h = np.asarray(sig.index.hour)
    mask = np.isin(h, hours)
    return sig.where(np.broadcast_to(mask[:, None], sig.shape)).ffill()


def trend_positions(p: dict, base: int = 24 * 20, hours=(13,), long_only: bool = True, target_vol: float | None = 0.5,
                    btc_filter: bool = False, continuous: bool = True, cap: float = 2.0) -> pd.DataFrame:
    c, r = p["close"], p["ret"]
    s = ensemble_signal(c, base, continuous=continuous)
    if long_only:
        s = s.clip(lower=0)
    if btc_filter:  # alts only while BTC's own trend is positive
        btc_up = (ensemble_signal(c[["BTC"]], base)["BTC"] > 0).astype(float)
        s = s.apply(lambda col: col if col.name == "BTC" else col * btc_up)
    if target_vol:
        s = vol_target(s, r, target_vol, cap=cap)
    return decide_at(s, hours)
