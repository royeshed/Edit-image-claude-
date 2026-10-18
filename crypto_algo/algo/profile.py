"""Volume profile: where the volume traded over the last `window` candles.

POC (point of control) = the price bin with the most volume; VAH = the top of the value area, the smallest set of
bins holding 70% of the volume. Volume is weighted in quote currency (volume x typical price) and binned by typical
price between the window's low and high. Values are computed on the candle that closes each UTC day, from candles up
to and including it, and held until the next day's value.
"""
import numpy as np
import pandas as pd


def volume_profile(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame,
                   window: int = 180, bins: int = 24, with_val: bool = False):
    """(POC, VAH), or (POC, VAH, VAL) with with_val (VAL = bottom of the value area)."""
    h, l, c = high.to_numpy(float), low.to_numpy(float), close.to_numpy(float)
    tp = (h + l + c) / 3
    w = volume.to_numpy(float) * tp
    poc = np.full(c.shape, np.nan)
    vah = np.full(c.shape, np.nan)
    val = np.full(c.shape, np.nan)
    step = close.index[1] - close.index[0] if len(close.index) > 1 else pd.Timedelta("4h")
    days = np.flatnonzero(((close.index + step).hour == 0) & ((close.index + step).minute == 0))
    for j in range(c.shape[1]):
        for t in days:
            if t < window - 1 or np.isnan(c[t, j]):
                continue
            s = slice(t - window + 1, t + 1)
            p, v = tp[s, j], w[s, j]
            m = ~(np.isnan(p) | np.isnan(v))
            if m.sum() < window // 2:
                continue
            lo, hi = np.nanmin(l[s, j]), np.nanmax(h[s, j])
            if not hi > lo:
                continue
            hist, edges = np.histogram(p[m], bins=bins, range=(lo, hi), weights=v[m])
            if hist.sum() <= 0:
                continue
            k = int(hist.argmax())
            poc[t, j] = (edges[k] + edges[k + 1]) / 2
            order = np.argsort(hist, kind="stable")[::-1]
            keep = order[: np.searchsorted(np.cumsum(hist[order]), 0.7 * hist.sum()) + 1]
            vah[t, j] = edges[keep.max() + 1]
            val[t, j] = edges[keep.min()]
    hold = round(pd.Timedelta("1D") / step) if step <= pd.Timedelta("1D") else 1
    mk = lambda a: pd.DataFrame(a, index=close.index, columns=close.columns).ffill(limit=hold - 1)
    return (mk(poc), mk(vah), mk(val)) if with_val else (mk(poc), mk(vah))


def atr_panel(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wilder's average true range of every column."""
    c1 = close.shift()
    tr = np.maximum(high - low, np.maximum((high - c1).abs(), (low - c1).abs()))
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def cmf_panel(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Chaikin money flow: volume-weighted position of each close inside its candle, over n candles (-1..1)."""
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    return (mfm * volume).rolling(n).sum() / volume.rolling(n).sum()
