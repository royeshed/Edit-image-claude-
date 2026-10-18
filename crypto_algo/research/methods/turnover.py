"""Turnover family - trade less without losing the edge.

Idea: the bot trades ~54x its equity per year (0.12% per side -> ~6.5% a year in fees). A train-only decomposition of
the baseline's turnover (n=15 Q-jan, no returns looked at) shows where it comes from:
    rank churn (a coin swapped in/out of the top 3 by 14-day momentum while its own signal stays on)  ~36x/yr (68%)
    fresh entries / signal exits                                                                       ~12x/yr
    resizes when one of the 6 Donchian systems flips (steps of 1/36 of equity)                          ~5x/yr
and 24% of all holding spells last a single 4h bar. So the main lever is hysteresis on top-k membership; a no-trade
band can only touch the ~5x of resizes. Methods (each one only delays or skips trades; it never buys a coin whose
signal - and so its daily close > EMA50 confirmation - is off, and never holds a coin outside the universe):
  * buffer       - top-k hysteresis: a held coin keeps its slot while it ranks within the top k+buffer signalling
                   coins; a free slot goes to the best-ranked coin not held. buffer=99: rank only fills free slots,
                   a held coin leaves on its own exit signal (or the universe/gate) only;
  * min_hold     - a coin cannot be pushed out by rank during its first `min_hold` 4h bars;
  * daily_select - which coins are held is decided once a day, at the 4h bar closing 00:00 UTC (when the daily
                   candle and the daily gate close); between decisions no coin is added or swapped, but signal
                   exits (Donchian or gate) still execute on every 4h close; select_every = the same with a
                   decision every 12 / 24 / 48 hours (daily_select == select_every 24);
  * daily        - the whole weight vector changes only at the 00:00 UTC close (exits wait too);
  * band         - no-trade band for held coins: keep a held coin's weight unless the target moves by >= band
                   (entries and full exits always execute).
Layers: selection through lab.build(rank=sticky_rank(...)) and lab.build(smoothing=make_smoothing(...)), both
stackable with other families (sticky_rank takes the final signal S; make_smoothing works on any final weights).
"""
import json
import sys

import numpy as np
import pandas as pd

from research.harness import METHODS, MAX_WEIGHT, Lab, show, summary

FAMILY = "turnover"
DECIDE_HOUR = 20     # the 4h bar opening 20:00 UTC closes at 00:00 UTC, exactly when the daily candle/gate closes

# Declared before running anything. Stage 1 = 12 trials. Stage 2 (<= 4 more, rule fixed now): the best stage-1
# selection setting (A/B/C, by median Sharpe) combined with the better band of D, plus one plateau-neighbour check
# if the best stage-1 value sits on the edge of its grid. Hard cap: 20 evaluate() calls for the family.
GRID = [
    # A. top-k hysteresis (attacks the ~36x/yr rank churn)
    {"buffer": 1}, {"buffer": 2}, {"buffer": 4}, {"buffer": 99},
    # B. minimum holding period before a rank exit (1 day, 5 days)
    {"min_hold": 6}, {"min_hold": 30},
    # C. once-a-day decisions at the 00:00 UTC close
    {"daily_select": True}, {"daily": True},
    # D. no-trade band on held coins' resizes (one system flip = 1/36 = 0.028 of equity)
    {"band": 0.03}, {"band": 0.06},
    # E. a-priori combinations (buffer 2 = middle of A)
    {"buffer": 2, "daily_select": True}, {"buffer": 2, "band": 0.03},
]
# Stage 2, filled in by the rule above after stage 1 had run. Best stage-1 selection setting = daily_select (median
# Sharpe 1.58), better band = 0.06. daily_select is a decision interval of 24h whose only tested neighbour is the
# baseline's 4h, so its plateau is checked on the decision-interval axis: 48h (declared edge check) and 12h (interior).
STAGE2 = [{"daily_select": True, "band": 0.06}, {"select_every": 48}, {"select_every": 12}]
# Stage 3 - robustness only, declared after stage 2 (48h: median 1.69, 9/9; 12h: 1.51; 24h: 1.58) and before running.
# 48h sits on the edge of the decision-interval axis and a 48h calendar has two phases, so: 48h on the other days
# (phase 24) and the next interval out, 96h. Pre-committed rule: freeze {"select_every": 48} only if the other phase
# also has median Sharpe > baseline and beats it in >= 6/9 variants AND 96h is decent (median >= baseline - 0.05);
# otherwise 48h is treated as a calendar spike and no decision interval is frozen. 17 evaluate() calls in total.
STAGE3 = [{"select_every": 48, "select_phase": 24}, {"select_every": 96}]
# Verdict after 17 trials: nothing frozen. Hysteresis (buffer / min_hold) halves turnover but costs 0.05-0.21 Sharpe:
# the top-3 rank churn is paid for by the momentum edge. daily_select (+band 0.06) keeps Sharpe (+0.01, within noise)
# with ~38% less turnover, but the decision-interval axis has no plateau (12h 1.51, 48h 1.69 on one calendar phase
# and 1.39 on the other, 96h 1.50), so its +0.01 is calendar noise; within 0.05 Sharpe the simpler baseline wins.
FROZEN = None


# ---------------------------------------------------------------- selection layer (bar t uses bars <= t only)
def decision_bars(index: pd.DatetimeIndex, every: int = 24, phase: int = 0) -> np.ndarray:
    """4h bars whose close falls on a multiple of `every` hours since 1970-01-01 00:00 UTC (24 -> the 00:00 UTC close,
    i.e. the bar opening 20:00; 12 -> 00:00 and 12:00; 48 -> 00:00 every other day). A fixed calendar: no look-ahead."""
    close = index + pd.Timedelta("4h")
    hrs = (close - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1h")
    return np.asarray((hrs - phase) % every == 0)


def sticky_select(S: pd.DataFrame, R: pd.DataFrame, k: int, buffer: int = 0, min_hold: int = 0,
                  every: int | None = None, phase: int = 0) -> pd.DataFrame:
    """Path-dependent top-k choice among coins with S > 0 and a known rank R (higher R = better).
    buffer=min_hold=0 and daily=False reproduce lab.build's plain top-k cut exactly."""
    E = ((S.to_numpy() > 0) & np.isfinite(R.to_numpy(dtype=float)))
    Rv = R.to_numpy(dtype=float)
    dec = decision_bars(S.index, every, phase) if every else np.ones(len(S), bool)
    T, N = E.shape
    out = np.zeros((T, N), bool)
    sel = np.zeros(N, bool)
    age = np.zeros(N, np.int64)
    pos = np.full(N, np.iinfo(np.int64).max)
    for t in range(T):
        e = E[t]
        held = sel & e
        if not dec[t]:
            new = held                                   # between daily decisions: only exits
        else:
            idx = np.flatnonzero(e)
            new = np.zeros(N, bool)
            if idx.size:
                order = idx[np.argsort(-Rv[t, idx], kind="stable")]   # ties -> column order, like method="first"
                pos[:] = np.iinfo(np.int64).max
                pos[order] = np.arange(1, idx.size + 1)
                new = held & ((pos <= k + buffer) | (age < min_hold))
                free = k - int(new.sum())
                if free > 0:
                    cand = order[~new[order]][:free]
                    new[cand] = True
        age = np.where(new, np.where(sel, age + 1, 0), 0)
        sel = new
        out[t] = new
    return pd.DataFrame(out, index=S.index, columns=S.columns)


def sticky_rank(lab: Lab, S: pd.DataFrame, R: pd.DataFrame | None = None, k: int | None = None, buffer: int = 0,
                min_hold: int = 0, every: int | None = None, phase: int = 0) -> pd.DataFrame:
    """Rank override for lab.build(rank=...): 1 for the sticky selection, 0 for other eligible coins, NaN otherwise.
    The selection always holds min(k, #eligible) coins, so build's top-k keeps exactly the selected ones.
    S must be the signal build will rank (after regime and universe masking)."""
    k = lab.cfg.top_k if k is None else k
    R = lab.momentum(lab.cfg.rank_days) if R is None else R
    sel = sticky_select(S, R, k, buffer, min_hold, every, phase)
    elig = (S > 0) & R.notna()
    return sel.astype(float).where(elig)


# ---------------------------------------------------------------- smoothing layer (works on any final weights)
def make_smoothing(band: float | None = None, daily: bool = False):
    """smoothing(lab, w) -> w for lab.build. Never raises a weight above its target's caps, never holds a coin whose
    target is 0 longer than the next decision (band: exits always execute at once)."""
    def smoothing(lab: Lab, w: pd.DataFrame) -> pd.DataFrame:
        W = np.nan_to_num(w.to_numpy(dtype=float))
        if daily:
            dec = decision_bars(w.index)
            W = pd.DataFrame(np.where(dec[:, None], W, np.nan)).ffill().fillna(0.0).to_numpy()
        if band is not None:
            out = np.empty_like(W)
            prev = np.zeros(W.shape[1])
            for t in range(len(W)):
                tgt = W[t]
                keep = (prev > 0) & (tgt > 0) & (np.abs(tgt - prev) < band)
                prev = np.where(keep, prev, tgt)
                out[t] = prev
            W = out
        return pd.DataFrame(np.clip(W, 0.0, MAX_WEIGHT), index=w.index, columns=w.columns)
    return smoothing


# ---------------------------------------------------------------- the candidate
def weights(lab: Lab, n: int, calendar: str, buffer: int = 0, min_hold: int = 0, daily_select: bool = False,
            band: float | None = None, daily: bool = False, select_every: int | None = None,
            select_phase: int = 0) -> pd.DataFrame:
    rank = None
    every = select_every or (24 if daily_select else None)
    if buffer or min_hold or every:
        m = lab.mask(n, calendar)
        S = lab.baseline_signal().where(m & lab.live, 0.0).clip(0, 1)     # what build will rank
        rank = sticky_rank(lab, S, buffer=buffer, min_hold=min_hold, every=every, phase=select_phase)
    smooth = make_smoothing(band, daily) if (band is not None or daily) else None
    return lab.build(n, calendar, rank=rank, smoothing=smooth)


def main() -> None:
    lab = Lab.load()
    base = lab.baseline_results
    print("baseline :", show(base), flush=True)
    allg = GRID + STAGE2 + STAGE3
    grid = allg if len(sys.argv) < 2 else [allg[int(i)] for i in sys.argv[1].split(",")]
    out = METHODS / f"{FAMILY}_train.csv"
    for params in grid:
        d = lab.evaluate(weights, FAMILY, params)
        print(json.dumps(params), ":", show(d, base), flush=True)
        row = pd.DataFrame([{"params": json.dumps(params, sort_keys=True), **summary(d, base).to_dict(),
                             **{f"sh_{r.n}_{r.calendar}": r.train_sharpe for r in d.itertuples()},
                             **{f"dd_{r.n}_{r.calendar}": r.train_maxdd for r in d.itertuples()}}])
        row.to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
