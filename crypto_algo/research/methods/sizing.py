"""Sizing family - how much to hold.

Idea: the live bot holds at most 3 coins at <= 1/6 each (<= 50% invested, ~18% on average), and a coin's weight is
its signal strength (the share of its 6 Donchian systems that are long) / 6. This family changes only how much is
held, never which entries are allowed, so every entry keeps the daily close > EMA50 confirmation:
  * k        - how many signalling coins (strongest 14-day momentum first) may be held at once; slots stay 6, so
               each coin stays <= 1/6 and the total <= 100%;
  * power    - signal-strength weighting: weight = S**power / max(sum S**power, 6). power 1 is the bot today,
               0 gives a full 1/6 slot as soon as any system is long, 0.5 is in between;
  * ivol     - inverse-volatility weights: each coin's weight is multiplied by min(1, ivol / its trailing 30-day
               annualised volatility), so coins wilder than `ivol` get a smaller slot (never a bigger one);
  * vt       - portfolio volatility targeting: all weights are multiplied by min(1, vt / ex-ante portfolio vol),
               the ex-ante vol being the trailing 30-day volatility of today's holdings (their covariance);
  * dd       - drawdown de-risking: weights are halved while the un-scaled strategy is more than `dd` below its peak.
Layer: sizing (lab.build(k=..., slots=6, sizing=fn(lab, w, S) -> w)).
"""
import json
import sys

import numpy as np
import pandas as pd

from research.harness import METHODS, MAX_WEIGHT, PPY, Lab, show, summary

FAMILY = "sizing"
SLOTS = 6            # 1 / MAX_WEIGHT: each coin <= 1/6, total <= 100%
VOL_BARS = 180       # 30 days of 4h bars for the volatility estimates (fixed, not tuned)

# Declared before running anything (17 of the 20 allowed trials). The power=0 trial ran twice (the first run had a
# bug: nan ** 0 == 1 spread weight over every coin), so 18 + 1 plateau check = 19 evaluate() calls in total.
GRID = [
    # A. more coins at once (slots stay 6)
    {"k": 4}, {"k": 5}, {"k": 6},
    # B. signal-strength weighting
    {"k": 3, "power": 0.0}, {"k": 3, "power": 0.5}, {"k": 6, "power": 0.5},
    # C. inverse-volatility slots (held coins' 30d vol: quartiles 0.8 / 1.0 / 1.3 on train)
    {"k": 3, "ivol": 0.7}, {"k": 3, "ivol": 1.0}, {"k": 6, "ivol": 0.7}, {"k": 6, "ivol": 1.0},
    # D. portfolio volatility targeting, de-risk only (ex-ante vol when invested, train median: k=3 0.27, k=6 0.32)
    {"k": 3, "vt": 0.25}, {"k": 3, "vt": 0.40}, {"k": 6, "vt": 0.30}, {"k": 6, "vt": 0.45}, {"k": 6, "vt": 0.60},
    # E. drawdown de-risking
    {"k": 3, "dd": 0.15}, {"k": 6, "dd": 0.20},
    # F. plateau check only, added after the 17 above had run: ivol=0.7 (best) sat on the edge of the grid, so its
    #    other neighbour is checked. The frozen choice stays restricted to the 17 declared trials above.
    {"k": 3, "ivol": 0.5},
]
FROZEN = {"k": 3, "ivol": 0.7}  # best median Sharpe of the 17 declared trials, inside the ivol 0.5-0.7-1.0 plateau


# ---------------------------------------------------------------- building blocks (bar t uses bars <= t only)
def coin_vol(lab: Lab) -> pd.DataFrame:
    """Trailing 30-day annualised volatility of each coin's 4h returns (NaN until 60 bars exist)."""
    return lab.memo((FAMILY, "coin_vol"), lambda: lab.ret.rolling(VOL_BARS, min_periods=60).std() * np.sqrt(PPY))


def exante_vol(lab: Lab, w: pd.DataFrame) -> pd.Series:
    """Annualised RMS over the last 30 days of the return today's weights would have earned (w' Sigma w)."""
    R = lab.ret.reindex(columns=w.columns).fillna(0.0).to_numpy()
    W = w.to_numpy()
    out = np.zeros(len(W))
    for t in np.flatnonzero((W > 0).any(axis=1)):
        idx = np.flatnonzero(W[t] > 0)
        lo = max(0, t - VOL_BARS + 1)
        p = R[lo:t + 1, idx] @ W[t, idx]
        out[t] = np.sqrt(np.mean(p * p) * PPY) if len(p) > 1 else 0.0
    return pd.Series(out, index=w.index)


def strategy_drawdown(lab: Lab, w: pd.DataFrame) -> pd.Series:
    """Drawdown of the un-scaled weights' equity at each close (weights at t-1 earn bar t, after costs)."""
    eq = (1 + lab.pnl(w).fillna(0.0)).cumprod()
    return eq / eq.cummax() - 1


def make_sizing(power: float = 1.0, ivol: float | None = None, vt: float | None = None, dd: float | None = None):
    """sizing(lab, w, S) -> w for lab.build. S is the signal after the top-k cut, w = S / max(sum S, 6)."""
    def sizing(lab: Lab, w: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
        if power != 1.0:
            Sp = (S ** power).where(S > 0, 0.0)      # after the power: 0 ** 0 and nan ** 0 are both 1
            w = Sp.div(np.maximum(Sp.sum(axis=1), SLOTS), axis=0)
        if ivol is not None:
            f = (ivol / coin_vol(lab).reindex(index=w.index, columns=w.columns)).clip(upper=1.0).fillna(1.0)
            w = w * f
        if vt is not None:
            s = (vt / exante_vol(lab, w).replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
            w = w.mul(s, axis=0)
        if dd is not None:
            down = strategy_drawdown(lab, w) < -dd
            w = w.mul(np.where(down, 0.5, 1.0), axis=0)
        return w.clip(lower=0.0, upper=MAX_WEIGHT)
    return sizing


def weights(lab: Lab, n: int, calendar: str, k: int = 3, power: float = 1.0, ivol: float | None = None,
            vt: float | None = None, dd: float | None = None) -> pd.DataFrame:
    plain = power == 1.0 and ivol is None and vt is None and dd is None
    return lab.build(n, calendar, k=k, slots=SLOTS, sizing=None if plain else make_sizing(power, ivol, vt, dd))


def main() -> None:
    lab = Lab.load()
    base = lab.baseline_results
    print("baseline :", show(base), flush=True)
    grid = GRID if len(sys.argv) < 2 else [GRID[int(i)] for i in sys.argv[1].split(",")]
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
