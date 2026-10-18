"""ranking - which coins to keep when more than 3 have a signal.

Idea: the live bot keeps, among the coins of its universe that have a confirmed Donchian signal, the 3 with the
strongest 14-day momentum. On train that cut binds on 26-35% of bars (whenever 4+ universe coins signal, mostly in
bull phases, often 8-15 of them), so the ranking decides what is held exactly when the bot earns most. This family
changes only that ordering (lab.build(rank=R), higher = preferred) and keeps k=3, slots=6, the baseline signal and
its daily close > EMA50 confirmation, so entries/exits, the 1/6 cap and the <= 50% exposure are untouched (except
`sigstr`, which prefers coins with more systems long and therefore also holds a little more).

Candidate orderings (all causal: rolling windows ending at bar t, cross-sectional ranks within bar t):
  * mom      - plain momentum over 7/28/56 days (baseline: 14);
  * riskadj  - log return over h days / (4h-return volatility over the same h days * sqrt(bars)): rewards smooth
               advances over jumpy ones;
  * mhrank   - average of the cross-sectional percentile ranks of 7/14/28/56-day momentum (horizon diversification);
  * resid    - residual ("idiosyncratic") momentum versus BTC: log return over h days minus beta * BTC's log return,
               beta from 90 days of 4h returns. NOTE plain relative strength vs BTC (r_coin - r_btc or
               (1+r_coin)/(1+r_btc)) subtracts the same number from every coin at a bar, so it orders coins exactly
               like plain momentum - it is not a separate trial; the beta-adjusted residual is;
  * breakout - breakout strength: (close - prior n-bar high) / ATR(14 days): how far above its entry channel;
  * trendq   - trend quality (Clenow): slope of a rolling log-price regression over h days * its R^2;
  * volsurge - volume surge: mean quote volume of the last 7 days / of the last 90 days (attention behind a move);
  * sigstr   - signal strength (share of the 6 systems long) as the primary key, 14-day momentum as tie-break;
  * lowvol   - lowest 30-day volatility first (calmer coins, smaller swings per 1/6 slot);
  * liquid   - highest 30-day mean quote volume first (deepest coins).
A coin whose score is undefined is not kept, as in the bot (rare; window scores need only half their window).
Layer: rank (lab.build(rank=rank_layer(lab, **FROZEN), k=3, slots=6)).
"""
import json
import sys

import numpy as np
import pandas as pd

from research.harness import METHODS, Lab, show, summary

FAMILY = "ranking"
K, SLOTS = 3, 6          # fixed: how many coins and the 1/6 slot size are the sizing family's job
BPD = 6                  # 4h bars per day
BTC = "BTCUSDT"

# ----- GRID, declared before any evaluation -----
# Descriptive fact only (no performance computed): on train 26-35% of bars have > 3 universe coins with a signal
# (n=10/15/20, Q-jan), so the ordering is decisive in bull phases.
# Stage 1 (15 trials), one idea each, against the baseline (14-day momentum):
GRID = [
    {"kind": "mom", "days": 7},
    {"kind": "mom", "days": 28},
    {"kind": "mom", "days": 56},
    {"kind": "riskadj", "days": 14},
    {"kind": "riskadj", "days": 28},
    {"kind": "mhrank"},                                  # 7/14/28/56-day ranks averaged
    {"kind": "resid", "days": 28},                       # beta-adjusted momentum vs BTC (beta: 90 days)
    {"kind": "breakout", "chan": 50},                    # distance above the 50-bar entry channel in ATRs
    {"kind": "breakout", "chan": 200},                   # ... above the 200-bar entry channel
    {"kind": "trendq", "days": 14},
    {"kind": "trendq", "days": 28},
    {"kind": "volsurge"},                                # 7-day / 90-day quote volume
    {"kind": "sigstr"},                                  # systems long first, 14-day momentum as tie-break
    {"kind": "lowvol"},                                  # lowest 30-day volatility first
    {"kind": "liquid"},                                  # highest 30-day quote volume first
]
# Stage 2 (<= 5 trials), rule fixed in advance:
#   PASS = stage-1 trials with median Sharpe > baseline median Sharpe AND beats_base >= 0.67.
#   If PASS is empty: stop, FROZEN = None.
#   Otherwise, for the best PASS trial: evaluate its untested neighbours along its own parameter (days 14<->28<->56,
#   chan 50<->100<->200; mhrank: 14/28/56/90 horizons; sigstr: tie-break by the best other PASS ranker), <= 4 calls,
#   and freeze it only if the neighbours are also >= the baseline median Sharpe (a plateau, not a spike). Within 0.05
#   Sharpe prefer the simpler ordering (plain momentum > one-score orderings > composites).
# Stage 1 outcome (train medians, baseline 1.57): PASS = {mom 7d: 1.68, beats 8/9; sigstr: 1.59, beats 8/9}; the best
# PASS is mom 7d, so stage 2 = its untested neighbours along `days` (14 = the baseline itself, 1.57, is already one):
STAGE2 = [
    {"kind": "mom", "days": 4},
    {"kind": "mom", "days": 10},
]
# Stage 2 outcome: mom 4d 1.60 (beats 9/9), mom 10d 1.49 (beats 5/9). The 10-day neighbour of 7 days is below the
# baseline, so the 4-7-10-14 day profile (1.60 / 1.68 / 1.49 / 1.57) is a noisy band around the baseline, not a
# plateau above it (paired daily-return t-stats vs the baseline ~1 for all of them). sigstr's +0.02 is within 0.05 of
# the simpler baseline. By the rule declared above nothing is frozen.
FROZEN = None


# ---------------------------------------------------------------- scores (bar t uses bars <= t only)
def _clean(x: pd.DataFrame) -> pd.DataFrame:
    return x.replace([np.inf, -np.inf], np.nan)


def _logret(lab: Lab, bars: int) -> pd.DataFrame:
    return lab.memo((FAMILY, "logret", bars), lambda: np.log(lab.close / lab.close.shift(bars)))


def _vol(lab: Lab, bars: int) -> pd.DataFrame:
    return lab.memo((FAMILY, "vol", bars), lambda: lab.ret.rolling(bars, min_periods=bars // 2).std())


def _quote_volume(lab: Lab) -> pd.DataFrame:
    return lab.memo((FAMILY, "qv"), lambda: lab.volume.astype("float64") * lab.close)


def _beta(lab: Lab, bars: int = 90 * BPD) -> pd.DataFrame:
    """Rolling OLS beta of each coin's 4h returns on BTC's, over the bars where both traded."""
    def go():
        x = lab.ret
        y = pd.DataFrame(np.broadcast_to(lab.ret[BTC].to_numpy()[:, None], x.shape), index=x.index, columns=x.columns)
        y = y.where(x.notna())
        x = x.where(y.notna())
        r = lambda d: d.rolling(bars, min_periods=bars // 2).mean()
        my, mx = r(y), r(x)
        cov = r(x * y) - mx * my
        var = r(y * y) - my * my
        return _clean(cov / var)
    return lab.memo((FAMILY, "beta", bars), go)


def _atr(lab: Lab, bars: int = 14 * BPD) -> pd.DataFrame:
    def go():
        pc = lab.close.shift()
        tr = np.maximum(lab.high, pc).sub(np.minimum(lab.low, pc))
        tr = tr.where(pc.notna(), lab.high - lab.low)
        return tr.rolling(bars, min_periods=bars // 2).mean()
    return lab.memo((FAMILY, "atr", bars), go)


def _trend_quality(lab: Lab, bars: int) -> pd.DataFrame:
    """slope * R^2 of an OLS fit of log close on time over the last `bars` bars, using the bars that traded
    (exact OLS with missing bars; at least half the window must be present)."""
    y = np.log(lab.close)
    t = np.arange(len(y), dtype="float64") - len(y) / 2.0          # centred clock: smaller sums, same slope
    T = pd.DataFrame(np.broadcast_to(t[:, None], y.shape), index=y.index, columns=y.columns).where(y.notna())
    roll = lambda d: d.rolling(bars, min_periods=bars // 2).sum()
    n = roll(y.notna().astype("float64"))
    mt, my = roll(T) / n, roll(y) / n
    cov = roll(T * y) / n - mt * my
    var_t = roll(T * T) / n - mt * mt
    var_y = roll(y * y) / n - my * my
    slope = cov / var_t
    r2 = (cov * cov / (var_t * var_y)).clip(0, 1)
    return _clean(slope * r2).where(n >= bars // 2)


def score(lab: Lab, kind: str = "mom", days: int = 14, chan: int = 50, signal: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw preference score aligned to lab.close (higher = preferred), NaN where undefined."""
    h = days * BPD
    if kind == "mom":
        return lab.momentum(days)
    if kind == "riskadj":
        return _clean(_logret(lab, h) / (_vol(lab, h) * np.sqrt(h)))
    if kind == "mhrank":
        hs = (7, 14, 28, 56) if days == 14 else (14, 28, 56, 90)
        ranks = [lab.momentum(d).rank(axis=1, pct=True) for d in hs]
        return sum(r.fillna(0.0) for r in ranks) / sum(r.notna().astype(float) for r in ranks)
    if kind == "resid":
        lr = _logret(lab, h)
        return lr - _beta(lab).mul(lr[BTC], axis=0)
    if kind == "breakout":
        return _clean((lab.close - lab.high.rolling(chan, min_periods=chan // 2).max().shift()) / _atr(lab))
    if kind == "trendq":
        return _trend_quality(lab, h)
    if kind == "volsurge":
        qv = _quote_volume(lab)
        return _clean(qv.rolling(7 * BPD, min_periods=7 * BPD // 2).mean() / qv.rolling(90 * BPD, min_periods=45 * BPD).mean())
    if kind == "sigstr":
        S = lab.baseline_signal() if signal is None else signal
        tie = lab.momentum(days).rank(axis=1, pct=True).fillna(0.0)      # in [0, 1]
        return S + 0.1 * tie                                             # 0.1 < 1/6, the smallest step of S
    if kind == "lowvol":
        return -_vol(lab, 30 * BPD)
    if kind == "liquid":
        return _quote_volume(lab).rolling(30 * BPD, min_periods=15 * BPD).mean()
    raise ValueError(kind)


def rank_layer(lab: Lab, kind: str = "mom", days: int = 14, chan: int = 50, signal: pd.DataFrame | None = None) -> pd.DataFrame:
    """The rank DataFrame for lab.build(rank=...). As in the bot, a signalled coin with an undefined score is not kept
    (rare: exchange-wide outage bars; the window scores need only half their window)."""
    return score(lab, kind, days, chan, signal)


def _key(params: dict) -> tuple:
    return (FAMILY, "rank", json.dumps(params, sort_keys=True))


def weights(lab: Lab, n: int, calendar: str, **params) -> pd.DataFrame:
    R = lab.memo(_key(params), lambda: rank_layer(lab, **params))
    return lab.build(n, calendar, rank=R, k=K, slots=SLOTS)


def _drop(lab: Lab) -> None:
    """Free this family's cached frames between trials (memory: each frame is ~85 MB); keep the bot's own."""
    for key in [k for k in lab._cache if isinstance(k, tuple) and (k[0] == FAMILY or (k[0] == "mom" and k[1] != 14))]:
        del lab._cache[key]


def main() -> None:
    lab = Lab.load()
    base = lab.baseline_results
    print("baseline :", show(base), flush=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "grid"
    pool = {"grid": GRID, "stage2": STAGE2}[which]
    idx = range(len(pool)) if len(sys.argv) < 3 else [int(i) for i in sys.argv[2].split(",")]
    out = METHODS / f"{FAMILY}_train.csv"
    for i in idx:
        params = pool[i]
        d = lab.evaluate(weights, FAMILY, params)
        print(json.dumps(params), ":", show(d, base), flush=True)
        row = pd.DataFrame([{"stage": which, "params": json.dumps(params, sort_keys=True), **summary(d, base).to_dict(),
                             **{f"sh_{r.n}_{r.calendar}": r.train_sharpe for r in d.itertuples()},
                             **{f"dd_{r.n}_{r.calendar}": r.train_maxdd for r in d.itertuples()}}])
        row.to_csv(out, mode="a", header=not out.exists(), index=False)
        _drop(lab)


if __name__ == "__main__":
    main()
