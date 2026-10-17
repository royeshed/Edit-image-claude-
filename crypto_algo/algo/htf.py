"""Higher-timeframe (HTF) helpers shared by the live bot and the research.

An HTF value becomes visible on the base timeframe only once its HTF candle has
closed: a daily candle opening at D 00:00 closes at D+1 00:00, so it is first usable
on the base bar whose close is D+1 00:00. Days start at 00:00 UTC, weeks on Monday.
"""
import numpy as np
import pandas as pd

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
FREQ = {"1h": "1h", "2h": "2h", "4h": "4h", "1d": "1D", "1w": "W-MON"}


def resample(bars: dict, tf: str) -> dict:
    """Resample close/high/low(/open/volume) panels to `tf` (weeks start Monday 00:00 UTC)."""
    kw = dict(label="left", closed="left")
    out = {}
    for k in ("open", "high", "low", "close", "volume"):
        if k in bars:
            out[k] = getattr(bars[k].resample(FREQ[tf], **kw), AGG[k])()
    valid = bars["close"].resample(FREQ[tf], **kw).count() > 0
    out = {k: v.where(valid) for k, v in out.items()}
    out["ret"] = out["close"].pct_change(fill_method=None)
    return out


def tf_delta(tf: str) -> pd.Timedelta:
    return pd.Timedelta("7D") if tf == "1w" else pd.Timedelta(FREQ[tf])


def align(htf: pd.DataFrame, base_index: pd.DatetimeIndex, tf: str) -> pd.DataFrame:
    """Shift each HTF value to the base bar on whose close the HTF candle closes, then hold it."""
    step = base_index[1] - base_index[0]
    shifted = htf.copy()
    shifted.index = shifted.index + tf_delta(tf) - step
    return shifted.reindex(base_index, method="ffill")


def state(entry: pd.DataFrame, exit_: pd.DataFrame) -> pd.DataFrame:
    """1 from an entry until the next exit (exit wins on the same bar)."""
    s = pd.DataFrame(np.where(exit_, 0.0, np.where(entry, 1.0, np.nan)), index=entry.index, columns=entry.columns)
    return s.ffill().fillna(0.0)


def confirmations(ph: dict) -> dict[str, pd.DataFrame]:
    """Boolean 'trend is up' definitions computed on a higher-timeframe panel."""
    c, hi, lo = ph["close"], ph["high"], ph["low"]
    ema = lambda n: c.ewm(span=n, adjust=False).mean()
    donch = state(c > hi.rolling(20).max().shift(), c < lo.rolling(10).min().shift()).astype(bool)
    mom = sum(np.sign(c / c.shift(k) - 1) for k in (5, 10, 20, 40, 80)) > 0
    # market structure: the latest confirmed 5-bar swing low is above the previous one
    piv = lo == lo.rolling(5, center=True).min()
    swing = lo.where(piv).shift(2)  # confirmed 2 bars after the low
    prev = swing.apply(lambda col: col.dropna().shift().reindex(col.index)).ffill()
    return {"close>ema20": c > ema(20), "close>ema50": c > ema(50), "ema10>ema30": ema(10) > ema(30),
            "donchian20/10": donch, "momentum": mom, "higher-lows": swing.ffill() > prev}


def confirmation(bars: dict, base_index: pd.DatetimeIndex, tfs, name: str) -> pd.DataFrame:
    """AND of confirmation `name` on every timeframe in `tfs`, aligned to `base_index`."""
    out = None
    for tf in tfs:
        a = align(confirmations(resample(bars, tf))[name].astype(float), base_index, tf).fillna(0).astype(bool)
        out = a if out is None else (out & a)
    return out


def confirmed_breakouts(p: dict, systems, confirm: pd.DataFrame | None, mode: str = "entry") -> pd.DataFrame:
    """Average position of breakout systems (entry n-bar high, exit m-bar low) with optional HTF confirmation.

    mode "entry": a system may only enter while `confirm` is true; it exits on its own rule.
    mode "gate":  as "entry", and the position is also closed whenever `confirm` turns false.
    """
    c = p["close"]
    total = 0
    for n, m in systems:
        e = c > p["high"].rolling(n).max().shift()
        x = c < p["low"].rolling(m).min().shift()
        if confirm is not None:
            e = e & confirm
            if mode == "gate":
                x = x | ~confirm
        total = total + state(e, x)
    return total / len(systems)
