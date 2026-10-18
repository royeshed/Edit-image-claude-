"""regime - a market-regime multiplier in [0, 1] on every coin's signal.

Idea: the live bot already asks each coin for its own daily confirmation (daily close > daily EMA50) before a
breakout may enter or stay open. What it does not look at is the market around the coin. Crypto is one big
factor: when bitcoin is in a bear trend or most liquid coins are below their trend, alt breakouts are mostly
bear-market rallies that fail (2018, mid-2019..2020, May-July 2021, 2022). A regime multiplier scales every
coin's signal down (to 0, or to a partial value) while the market regime is bad, and leaves it alone otherwise.
It only ever REDUCES exposure, so the per-coin <= 1/6 cap, spot/long-only and the daily confirmation of every
entry are untouched (the regime adds a second, market-level higher-timeframe confirmation on top).

Kinds (every value is built from COMPLETED daily/weekly candles only, via algo.htf.resample + align or lab.gate):
  btc          BTC trend: BTC daily close > daily SMA(n) / EMA(n), or BTC weekly close > weekly EMA(n) ("wema").
               multiplier = 1 when on, `off` when not.
  btc_score    graded BTC trend: mean of [daily close > SMA200, daily close > EMA50, weekly close > weekly EMA20]
               (lengths times `scale`), so 0, 1/3, 2/3 or 1.
  coin_weekly  each coin's own weekly confirmation on top of the daily one: weekly close > weekly EMA(n).
               (n=50 is lab.gate(("1d", "1w"), "close>ema50"), since the signal already carries the daily gate.)
  breadth      market breadth: share of the 30 most liquid mature coins (the "mature volume" rule, same quarterly
               calendar as the variant) whose daily close > daily EMA50. 1 when breadth >= thr, `off` below.
  vol          volatility regime: BTC 30-day realised vol of daily returns / its trailing 365-day realised vol.
               1 when the ratio <= `ratio`, `off` above (extreme-volatility episodes are scaled down).
  altbtc       alts vs BTC: ETH/BTC daily close > its daily EMA(n). Alts get 1 when on, `off` when not; BTC is
               never scaled by this rule.
Parameter `len` is the moving-average length (in candles of the rule's timeframe).
Layer: regime (lab.build(regime=Series indexed like lab.index, or a DataFrame like lab.close)).
"""
import json
import sys

import numpy as np
import pandas as pd

from algo.htf import align, resample
from research.harness import METHODS, TRAIN, Lab, show, summary

FAMILY = "regime"
BTC, ETH = "BTCUSDT", "ETHUSDT"
BREADTH_POOL = 30     # the 30 most liquid mature coins (fixed, not tuned)

# ----- GRID, declared before any evaluation (stage 1: 11 trials) -----
GRID = [
    # A. BTC trend, one multiplier for every coin
    {"kind": "btc", "ma": "sma", "len": 200, "off": 0.0},     # 0 the classic 200-day filter
    {"kind": "btc", "ma": "sma", "len": 200, "off": 0.5},     # 1 same, half size instead of flat
    {"kind": "btc", "ma": "ema", "len": 50, "off": 0.0},      # 2 faster: BTC passes the same daily test as the coin
    {"kind": "btc", "ma": "wema", "len": 20, "off": 0.0},     # 3 weekly gate on BTC ("bull market support band")
    {"kind": "btc_score", "scale": 1.0},                    # 4 graded mean of 0, 2, 3
    # B. per-coin weekly confirmation added to the daily one
    {"kind": "coin_weekly", "len": 20},                       # 5
    {"kind": "coin_weekly", "len": 50},                       # 6 = lab.gate(("1d", "1w"), "close>ema50")
    # C. market breadth
    {"kind": "breadth", "thr": 0.5, "off": 0.0},            # 7
    {"kind": "breadth", "thr": 0.5, "off": 0.5},            # 8
    # D. volatility regime
    {"kind": "vol", "ratio": 1.5, "off": 0.5},              # 9
    # E. alts vs BTC
    {"kind": "altbtc", "len": 50, "off": 0.5},                # 10
]
# ----- Stage 2, a rule fixed in advance (<= 5 more trials, <= 16 in total) -----
#   a) passing = stage-1 trials with median Sharpe > baseline median AND beats_base >= 0.67. None -> stop, FROZEN=None.
#   b) take the passing trial with the highest median Sharpe and evaluate its two neighbours(params) below.
#      Freeze it only if BOTH neighbours also have a median Sharpe above the baseline's (a plateau, not a spike).
#      If it fails the plateau, do the same for the best passing trial of a different kind (2 more trials).
#   c) if passing trials of two different kinds exist, evaluate the product of the two best (1 trial); prefer it
#      over the single only if it also passes and its median Sharpe is > 0.05 above the single's.
MAX_TRIALS = 20
# Stage 2 as run (3 trials, 14 in total): b) best passing = trial 3 (BTC weekly close > weekly EMA20); neighbours
# len 14 and 28 both beat the baseline median -> plateau, frozen. c) passing groups: A (BTC trend) and D (vol); "kind"
# read as grid group, since btc_score is the same BTC-trend family and already contains trial 3 -> product 3 x 9 tested:
# +0.01 Sharpe over trial 3 alone (< 0.05), so the simpler single rule is kept.
STAGE2 = [{"kind": "btc", "ma": "wema", "len": 14, "off": 0.0}, {"kind": "btc", "ma": "wema", "len": 28, "off": 0.0},
          {"kind": "product", "a": GRID[3], "b": GRID[9]}]
FROZEN = {"kind": "btc", "ma": "wema", "len": 20, "off": 0.0}  # trade only while BTC's weekly close > weekly EMA20
# Outcome (train 2018-11..2022, 14 trials, baseline median Sharpe 1.567): BTC weekly EMA20 1.75 (9/9 variants, worst DD
# -25% vs -38%), neighbours EMA14 1.72 / EMA28 1.63 (9/9). btc_score 1.66, vol 1.59, BTC daily EMA50 1.57 (tie) passed
# but are weaker; BTC daily SMA200 1.46 (whipsaws: 28 switches vs 14 weekly), per-coin weekly EMA20/50 1.55/1.44,
# breadth 1.28/1.45 (0/9), ETH/BTC 1.50 were worse. The gain comes from 2022 (-17% -> 0%) and mid-2021; 2019 is worse.


def neighbours(p: dict) -> list[dict]:
    """The two pre-declared neighbours of a stage-1 trial: main length x0.7 / x1.4, thresholds +-0.1 / ratio 1.25/2."""
    q = dict(p)
    if p["kind"] in ("btc", "coin_weekly", "altbtc"):
        return [{**q, "len": int(round(p["len"] * f))} for f in (0.7, 1.4)]
    if p["kind"] == "btc_score":
        return [{**q, "scale": f} for f in (0.7, 1.4)]
    if p["kind"] == "breadth":
        return [{**q, "thr": round(p["thr"] + d, 2)} for d in (-0.1, 0.1)]
    if p["kind"] == "vol":
        return [{**q, "ratio": r} for r in (1.25, 2.0)]
    raise ValueError(p)


# ---------------- building blocks (bar t uses bars <= t only; HTF values only once their candle closed) ----------------
def _htf_close(lab: Lab, cols, tf: str) -> pd.DataFrame:
    """Completed `tf` candles' closes (indexed by candle open time) of the given 4h columns."""
    return resample({"close": lab.close[list(cols)]}, tf)["close"]


def _on_base(lab: Lab, x: pd.DataFrame | pd.Series, tf: str) -> pd.DataFrame | pd.Series:
    """Move a `tf` value (float) to the 4h bar whose close completes its candle, then hold it; NaN -> 0."""
    df = x.to_frame() if isinstance(x, pd.Series) else x
    out = align(df.astype(float), lab.index, tf).fillna(0.0)
    return out.iloc[:, 0] if isinstance(x, pd.Series) else out


def _above(c: pd.Series | pd.DataFrame, ma: str, n: int):
    m = c.rolling(n).mean() if ma == "sma" else c.ewm(span=n, adjust=False).mean()
    return (c > m).where(c.notna())


def btc_trend(lab: Lab, ma: str, n: int) -> pd.Series:
    """1.0 while BTC's last completed daily (weekly for 'wema') close is above its moving average, else 0.0."""
    def go():
        tf = "1w" if ma == "wema" else "1d"
        c = _htf_close(lab, [BTC], tf)[BTC]
        return _on_base(lab, _above(c, "sma" if ma == "sma" else "ema", n), tf)
    return lab.memo((FAMILY, "btc", ma, n), go)


def coin_weekly(lab: Lab, n: int) -> pd.DataFrame:
    """1.0 while a coin's last completed weekly close is above its weekly EMA(n), else 0.0 (per coin)."""
    def go():
        c = _htf_close(lab, lab.assets, "1w")
        return _on_base(lab, _above(c, "ema", n), "1w")
    return lab.memo((FAMILY, "coin_weekly", n), go)


def breadth(lab: Lab, calendar: str) -> pd.Series:
    """Share of the BREADTH_POOL most liquid mature coins (point-in-time pool) whose daily close > daily EMA50."""
    def go():
        pool = lab.mask(BREADTH_POOL, calendar) & lab.live
        g = lab.gate(lab.cfg.confirm_tfs, lab.cfg.confirm)
        return ((g & pool).sum(axis=1) / pool.sum(axis=1).replace(0, np.nan)).fillna(0.0)
    return lab.memo((FAMILY, "breadth", calendar), go)


def vol_ratio(lab: Lab) -> pd.Series:
    """BTC 30-day / 365-day realised vol of completed daily log returns (on the 4h grid once the day closed)."""
    def go():
        c = _htf_close(lab, [BTC], "1d")[BTC]
        r = np.log(c).diff()
        ratio = r.rolling(30, min_periods=20).std() / r.rolling(365, min_periods=180).std()
        return align(ratio.to_frame(), lab.index, "1d").iloc[:, 0]      # NaN until history exists
    return lab.memo((FAMILY, "vol_ratio"), go)


def alt_strength(lab: Lab, n: int) -> pd.Series:
    """1.0 while ETH/BTC's last completed daily close is above its daily EMA(n), else 0.0."""
    def go():
        c = _htf_close(lab, [BTC, ETH], "1d")
        return _on_base(lab, _above(c[ETH] / c[BTC], "ema", n), "1d")
    return lab.memo((FAMILY, "altbtc", n), go)


def _mix(on, off: float):
    """on in [0, 1] -> 1 where on, `off` where not."""
    return on + off * (1.0 - on)


def layer(lab: Lab, calendar: str, kind: str, **p) -> pd.Series | pd.DataFrame:
    """The regime multiplier lab.build(regime=...) expects, values in [0, 1]."""
    if kind == "btc":
        return _mix(btc_trend(lab, p["ma"], p["len"]), p.get("off", 0.0))
    if kind == "btc_score":
        s = p.get("scale", 1.0)
        n = lambda x: max(2, int(round(x * s)))
        return (btc_trend(lab, "sma", n(200)) + btc_trend(lab, "ema", n(50)) + btc_trend(lab, "wema", n(20))) / 3
    if kind == "coin_weekly":
        return coin_weekly(lab, p["len"])
    if kind == "breadth":
        return _mix((breadth(lab, calendar) >= p["thr"]).astype(float), p.get("off", 0.0))
    if kind == "vol":
        return _mix((vol_ratio(lab) <= p["ratio"]).astype(float), p.get("off", 0.5))   # NaN ratio -> scaled down
    if kind == "altbtc":
        r = _mix(alt_strength(lab, p["len"]), p.get("off", 0.5))
        out = pd.DataFrame(np.repeat(r.to_numpy()[:, None], len(lab.assets), axis=1), index=lab.index, columns=lab.assets)
        out[BTC] = 1.0
        return out
    if kind == "product":
        a, b = layer(lab, calendar, **p["a"]), layer(lab, calendar, **p["b"])
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            return a * b
        a = a if isinstance(a, pd.DataFrame) else pd.DataFrame({c: a for c in lab.assets})
        return a.mul(b, axis=0) if isinstance(b, pd.Series) else a * b
    raise ValueError(kind)


def regime(lab: Lab, calendar: str, **params) -> pd.Series | pd.DataFrame:
    per_cal = params["kind"] in ("breadth", "product")
    key = (FAMILY, "layer", json.dumps(params, sort_keys=True), calendar if per_cal else None)
    return lab.memo(key, lambda: layer(lab, calendar, **params))


def weights(lab: Lab, n: int, calendar: str, **params) -> pd.DataFrame:
    return lab.build(n, calendar, regime=regime(lab, calendar, **params))


# ---------------- driver ----------------
OUT = METHODS / f"{FAMILY}_train.csv"


def explore() -> None:
    """How often is each stage-1 regime 'on' during TRAIN (no performance is computed here)."""
    lab = Lab.load()
    tr = (lab.index >= TRAIN[0]) & (lab.index < TRAIN[1])
    for p in GRID:
        r = regime(lab, "Q-jan", **p)
        v = r[tr] if isinstance(r, pd.Series) else r[tr].where(lab.live[tr] & lab.mask(15, "Q-jan")[tr]).stack()
        print(json.dumps(p), f"mean multiplier {float(np.nanmean(v)):.2f}  share==1 {float(np.nanmean(v == 1)):.2f}", flush=True)
    b = breadth(lab, "Q-jan")[tr]
    print("breadth quantiles", b.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_dict())
    vr = vol_ratio(lab)[tr]
    print("vol ratio quantiles", vr.quantile([0.1, 0.5, 0.9, 0.95]).round(2).to_dict())


def run(grid: list[dict]) -> None:
    lab = Lab.load()
    base = lab.baseline_results
    print("baseline :", show(base), flush=True)
    for params in grid:
        d = lab.evaluate(weights, FAMILY, params)
        print(json.dumps(params), ":", show(d, base), flush=True)
        row = pd.DataFrame([{"params": json.dumps(params, sort_keys=True), **summary(d, base).to_dict(),
                             **{f"sh_{r.n}_{r.calendar}": r.train_sharpe for r in d.itertuples()},
                             **{f"dd_{r.n}_{r.calendar}": r.train_maxdd for r in d.itertuples()}}])
        row.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "explore":
        explore()
    elif cmd == "run":                      # run 0,1,2   (indices into GRID)
        run([GRID[int(i)] for i in sys.argv[2].split(",")])
    elif cmd == "run_json":                 # run_json '[{...}, {...}]'   (stage-2 trials chosen by the rule above)
        run(json.loads(sys.argv[2]))


if __name__ == "__main__":
    main()
