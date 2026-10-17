"""Quant research lab: panel data, forward returns, signal tests and a vectorised backtest.

Conventions (no look-ahead):
  * every feature at time t uses data up to and including bar t's close;
  * a position decided at t earns bar t+1's return: pnl[t+1] = pos[t] * ret[t+1];
  * costs are charged on turnover |pos[t] - pos[t-1]| at `cost` per unit of notional.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
COINS = ["BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL", "AVAX"]
START = pd.Timestamp("2018-01-01", tz="UTC")
PERIODS = {
    "train": (pd.Timestamp("2018-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC")),
    "valid": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
    "test": (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2100-01-01", tz="UTC")),
}
COST = 0.0007  # futures taker 0.05% + 0.02% slippage, per side
BARS_PER_YEAR = {"1h": 8760, "4h": 2190, "1d": 365}


def load_panel(tf: str = "1h", coins=COINS) -> dict[str, pd.DataFrame]:
    """OHLCV per field as (time x coin) frames on a common grid. The still-forming last bar is dropped."""
    frames = {}
    for c in coins:
        d = pd.read_csv(DATA / f"binance_{c}-USDT_{tf}_3400d.csv", index_col=0)
        d.index = pd.to_datetime(d.index, utc=True)
        frames[c] = d.iloc[:-1]
    grid = pd.date_range(START, max(f.index[-1] for f in frames.values()), freq=tf.replace("m", "min").replace("d", "D"))
    panel = {k: pd.DataFrame({c: f[k].reindex(grid) for c, f in frames.items()}) for k in ("open", "high", "low", "close", "volume")}
    panel["ret"] = panel["close"].pct_change(fill_method=None)
    return panel


def period_mask(index: pd.DatetimeIndex, name: str) -> np.ndarray:
    lo, hi = PERIODS[name]
    return (index >= lo) & (index < hi)


def forward_return(close: pd.DataFrame, h: int) -> pd.DataFrame:
    """Return from close[t] to close[t+h]."""
    return close.shift(-h) / close - 1


def ic(feature: pd.DataFrame, fwd: pd.DataFrame, period: str, step: int = 1) -> pd.Series:
    """Spearman rank correlation per coin between feature[t] and fwd[t], sampled every `step` bars
    (use step = horizon to avoid overlapping returns)."""
    m = period_mask(feature.index, period)
    out = {}
    for c in feature.columns:
        x = pd.concat([feature[c][m], fwd[c][m]], axis=1).iloc[::step].dropna()
        out[c] = x.iloc[:, 0].rank().corr(x.iloc[:, 1].rank()) if len(x) > 100 else np.nan
    return pd.Series(out)


def backtest(pos: pd.DataFrame, ret: pd.DataFrame, cost: float = COST) -> pd.DataFrame:
    """Per-coin net returns of holding `pos` (fraction of that coin's sleeve, -1..1)."""
    pos = pos.reindex_like(ret).fillna(0)
    gross = pos.shift(1) * ret.fillna(0)
    fees = cost * pos.diff().abs().fillna(pos.abs())
    return (gross - fees).where(ret.notna())  # NaN while a coin isn't trading yet


def portfolio(pnl: pd.DataFrame) -> pd.Series:
    """Equal-weight across coins that are trading at each bar."""
    return pnl.mean(axis=1, skipna=True).fillna(0)


def metrics(r: pd.Series, tf: str = "1h", period: str | None = None, pos: pd.DataFrame | None = None) -> dict:
    if period:
        r = r[period_mask(r.index, period)]
    r = r.fillna(0)
    ppy = BARS_PER_YEAR[tf]
    eq = (1 + r).cumprod()
    years = len(r) / ppy
    out = {
        "cagr": eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else -1.0,
        "vol": r.std() * np.sqrt(ppy),
        "sharpe": r.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else 0.0,
        "maxdd": (eq / eq.cummax() - 1).min(),
        "total": eq.iloc[-1] - 1,
    }
    if pos is not None:
        p = pos[period_mask(pos.index, period)] if period else pos
        out["turnover_y"] = p.diff().abs().sum(axis=1).mean() * ppy / max(1, p.shape[1])
        out["exposure"] = (p.abs() > 0).mean().mean()
    return out


def report(name: str, r: pd.Series, tf: str = "1h", pos=None) -> dict:
    row = {"name": name}
    for p in PERIODS:
        m = metrics(r, tf, p, pos)
        row.update({f"{p}_sharpe": m["sharpe"], f"{p}_cagr": m["cagr"], f"{p}_dd": m["maxdd"]})
    return row
