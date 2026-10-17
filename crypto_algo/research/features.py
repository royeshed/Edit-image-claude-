"""Feature library: classic indicators plus custom ones, all as (time x coin) frames known at bar close."""
import numpy as np
import pandas as pd


def rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def zscore(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return (x - x.rolling(n).mean()) / x.rolling(n).std()


def realized_vol(ret: pd.DataFrame, n: int) -> pd.DataFrame:
    return ret.rolling(n).std()


def atr(p: dict, n: int) -> pd.DataFrame:
    pc = p["close"].shift()
    tr = np.maximum(p["high"] - p["low"], np.maximum((p["high"] - pc).abs(), (p["low"] - pc).abs()))
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def build(p: dict) -> dict[str, pd.DataFrame]:
    c, r, v = p["close"], p["ret"], p["volume"]
    vol24 = realized_vol(r, 24 * 7)
    f = {}
    # --- returns, vol-scaled (custom: "shock" = move in units of its own volatility)
    for L in (1, 4, 12, 24, 72):
        f[f"shock_{L}h"] = (c / c.shift(L) - 1) / (vol24 * np.sqrt(L))
    # --- oscillators
    for n in (14, 56, 336):  # 14h, ~14 4h-bars, 14 days
        f[f"rsi_{n}"] = rsi(c, n)
    for n in (20, 100, 480):
        f[f"bb_z_{n}"] = zscore(c, n)
    # --- trend distance
    for n in (50, 200, 1200):
        f[f"dist_ema_{n}"] = c / c.ewm(span=n, adjust=False).mean() - 1
    hi, lo = p["high"].rolling(168).max(), p["low"].rolling(168).min()
    f["donch_pos_7d"] = (c - lo) / (hi - lo)
    # --- stochastic-style close location within the bar and over 24h
    f["clv_1h"] = ((c - p["low"]) - (p["high"] - c)) / (p["high"] - p["low"])
    f["clv_24h"] = ((c - p["low"].rolling(24).min()) - (p["high"].rolling(24).max() - c)) / (p["high"].rolling(24).max() - p["low"].rolling(24).min())
    # --- volume / volatility regime
    dv = c * v
    f["volume_z_24"] = zscore(np.log(dv.rolling(24).sum()), 24 * 30)
    f["volume_z_1h"] = zscore(np.log(dv + 1), 24 * 7)
    f["vol_ratio"] = realized_vol(r, 24) / realized_vol(r, 24 * 30)
    f["vol_level"] = realized_vol(r, 24 * 7) / realized_vol(r, 24 * 7).rolling(24 * 365, min_periods=24 * 90).median()
    # --- cross-sectional (custom): coin's rank of 24h / 7d return among the 8, centred at 0
    f["xs_rank_24h"] = (c / c.shift(24) - 1).rank(axis=1, pct=True) - 0.5
    f["xs_rank_7d"] = (c / c.shift(168) - 1).rank(axis=1, pct=True) - 0.5
    # --- BTC lead (custom): BTC's last-hour move vs the coin's own
    btc1 = r["BTC"]
    f["btc_lead_1h"] = pd.DataFrame({k: btc1 for k in c.columns}, index=c.index) - r
    f["btc_ret_4h"] = pd.DataFrame({k: c["BTC"] / c["BTC"].shift(4) - 1 for k in c.columns}, index=c.index)
    # --- range expansion: today's range vs average
    rng = (p["high"].rolling(24).max() - p["low"].rolling(24).min()) / c
    f["range_z"] = zscore(rng, 24 * 30)
    # --- session-relative moves (custom): return since 00:00 UTC (Asia open) and since 13:00 UTC (US open)
    at = lambda h: np.broadcast_to((c.index.hour == h)[:, None], c.shape)
    day_open = p["open"].where(at(0)).ffill()
    f["ret_since_utc0"] = c / day_open - 1
    us_open = p["open"].where(at(13)).ffill()
    f["ret_since_us_open"] = c / us_open - 1
    return f
