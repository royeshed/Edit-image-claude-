"""signal_ensemble - diversify the entry/exit signal across trend-following signal TYPES.

Idea: the live bot's signal is the fraction of 6 Donchian breakout systems that are long. All six are the
same kind of rule, so they tend to be wrong together (false breakouts in a range). Other trend-following
rules (EMA crossovers, time-series momentum, Keltner/ATR channels, Supertrend, moving-average slope, MACD)
react to different features of the price path, so averaging across types should give a smoother, more
reliable "how much of the evidence says trend" score. Every system, of every type, is gated like the
baseline's "gate" mode: it may only be long while the daily close > daily EMA50 (lab.gate()), and an open
position is closed as soon as that daily confirmation turns off. So entries ALWAYS keep a daily confirmation.

Layer: signal (lab.build(signal=S), S in [0, 1] = weighted fraction of systems long).

Every type uses the same horizon ladder L in {50, 100, 200} 4h bars (8, 17, 33 days), times `scale`:
  don    Donchian: close > prior L-bar high, exit close < prior 10/20-bar low (exactly the live bot at scale 1)
  ema    EMA crossover: EMA(L/4) > EMA(L)
  tsmom  time-series momentum: close > close 2L bars ago (horizons 100/200/400 bars = 17/33/67 days)
  kelt   Keltner/ATR channel: enter close > EMA(L) + 2 ATR(L), exit close < EMA(L)
  st     Supertrend, ATR(20) with multipliers 3/4/5 (x sqrt(scale)): long while the supertrend is up
  slope  moving-average slope: EMA(L) above its value L/4 bars ago
  macd   MACD with slow = L, fast = 12/26 L, signal = 9/26 L: long while MACD > 0 and MACD > its signal line
S = equal-weight average over the chosen types of each type's fraction of systems long.
"""
import json
import sys

import numpy as np
import pandas as pd

from algo.htf import state
from research.harness import METHODS, Lab, show, summary

FAMILY = "signal_ensemble"
ALL = ["don", "ema", "tsmom", "kelt", "st", "slope", "macd"]
NEW = [t for t in ALL if t != "don"]
LADDER = (50, 100, 200)

# ----- GRID, declared before any evaluation -----
# Stage 1 (12 trials): each new type alone (what does it carry?), fixed ensembles, a majority-vote variant,
# and two alternative Donchian length sets (the baseline itself at scale 0.7 / 1.4).
GRID = [
    {"types": ["ema"]},
    {"types": ["tsmom"]},
    {"types": ["kelt"]},
    {"types": ["st"]},
    {"types": ["slope"]},
    {"types": ["macd"]},
    {"types": ["don", "ema", "tsmom"]},
    {"types": ALL},
    {"types": NEW},
    {"types": ALL, "min_agree": 0.34},
    {"types": ["don"], "scale": 0.7},
    {"types": ["don"], "scale": 1.4},
]
# Stage 2 (<= 3 trials), a rule fixed in advance:
#   a) A = new types whose single-type median Sharpe >= baseline median Sharpe - 0.10. If A is non-empty and
#      ["don"] + A differs from every stage-1 ensemble, evaluate {"types": ["don"] + A}.
#   b) Among the ensembles (stage-1 trials 7-10 and a), take the one with the highest median Sharpe; if it
#      passes the improvement test (median Sharpe > baseline and beats_base >= 0.67), evaluate it at
#      scale 0.7 and 1.4 to check for a plateau. Freeze it only if both neighbours stay above the baseline.
STAGE2_THRESHOLD = 0.10
MAX_TRIALS = 20
FROZEN = None  # set after running GRID on train; None if nothing beat the baseline
# Outcome (train 2018-11..2022, 13 trials): no candidate beat the baseline (median Sharpe 1.57). Every new type
# alone was worse (0.95-1.50), every ensemble that adds them was worse (1.13-1.53), stage 2a = don+macd (1.53,
# beats 4/9), so stage 2b never triggered. Donchian at scale 0.7 ties (1.57), at 1.4 is worse (1.44).


# ---------------- building blocks ----------------
def _n(x: float, scale: float) -> int:
    return max(2, int(round(x * scale)))


def _px(lab: Lab):
    """Forward-filled close/high/low (causal) so a missing bar does not break a rolling window."""
    return lab.memo((FAMILY, "px"), lambda: (lab.close.ffill(), lab.high.ffill(), lab.low.ffill()))


def _gate(lab: Lab) -> pd.DataFrame:
    return lab.gate(lab.cfg.confirm_tfs, lab.cfg.confirm)


def _ema(lab: Lab, n: int) -> pd.DataFrame:
    return lab.memo((FAMILY, "ema", n), lambda: _px(lab)[0].ewm(span=n, adjust=False).mean())


def _atr(lab: Lab, n: int) -> pd.DataFrame:
    def go():
        c, h, l = _px(lab)
        cp = c.shift()
        tr = np.maximum(h - l, np.maximum((h - cp).abs(), (l - cp).abs()))
        return tr.ewm(alpha=1 / n, adjust=False).mean()           # Wilder ATR
    return lab.memo((FAMILY, "atr", n), go)


def _level(cond: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """A 'long while condition holds' rule under the daily gate: state(cond & g, ~cond | ~g) == cond & g."""
    return (cond & g).astype("float32")


def _hyst(entry: pd.DataFrame, exit_: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Entry/exit rule in the baseline's gate mode: enter only while confirmed, leave when unconfirmed."""
    return state(entry & g, exit_ | ~g).astype("float32")


def _supertrend_up(c: np.ndarray, h: np.ndarray, l: np.ndarray, atr: np.ndarray, m: float) -> np.ndarray:
    """Classic Supertrend direction (True = up), vectorised across coins, causal loop over bars."""
    hl2 = (h + l) / 2
    ub, lb = hl2 + m * atr, hl2 - m * atr
    T, N = c.shape
    up = np.zeros((T, N), dtype=bool)
    fub, flb = ub[0].copy(), lb[0].copy()
    cur = np.zeros(N, dtype=bool)                     # start "down": a fresh breakout is needed to go long
    with np.errstate(invalid="ignore"):
        for t in range(1, T):
            cp = c[t - 1]
            nfub = np.where(np.isnan(fub) | (ub[t] < fub) | (cp > fub), ub[t], fub)
            nflb = np.where(np.isnan(flb) | (lb[t] > flb) | (cp < flb), lb[t], flb)
            cur = np.where(cur, ~(c[t] < nflb), c[t] > nfub)
            up[t] = cur
            fub, flb = nfub, nflb
    return up


# ---------------- signal types (each: fraction of its systems long, gated) ----------------
def t_don(lab, scale, g):
    tot = 0
    for n in lab.cfg.entries:
        for m in lab.cfg.exits:
            e, x = lab.donchian(_n(n, scale), _n(m, scale))
            tot = tot + state(e & g, x | ~g)
    return (tot / (len(lab.cfg.entries) * len(lab.cfg.exits))).astype("float32")


def t_ema(lab, scale, g):
    return sum(_level(_ema(lab, _n(L / 4, scale)) > _ema(lab, _n(L, scale)), g) for L in LADDER) / len(LADDER)


def t_tsmom(lab, scale, g):
    c = _px(lab)[0]
    return sum(_level(c > c.shift(_n(2 * L, scale)), g) for L in LADDER) / len(LADDER)


def t_kelt(lab, scale, g):
    c = _px(lab)[0]
    tot = 0
    for L in LADDER:
        mid, atr = _ema(lab, _n(L, scale)), _atr(lab, _n(L, scale))
        tot = tot + _hyst(c > mid + 2 * atr, c < mid, g)
    return tot / len(LADDER)


def t_st(lab, scale, g):
    c, h, l = _px(lab)
    atr = _atr(lab, _n(20, scale)).to_numpy()
    C, H, Lo = c.to_numpy(), h.to_numpy(), l.to_numpy()
    tot = 0
    for m in (3.0, 4.0, 5.0):
        up = pd.DataFrame(_supertrend_up(C, H, Lo, atr, m * np.sqrt(scale)), index=c.index, columns=c.columns)
        tot = tot + _level(up, g)
    return tot / 3


def t_slope(lab, scale, g):
    out = 0
    for L in LADDER:
        e = _ema(lab, _n(L, scale))
        out = out + _level(e > e.shift(_n(L / 4, scale)), g)
    return out / len(LADDER)


def t_macd(lab, scale, g):
    out = 0
    for L in LADDER:
        macd = _ema(lab, _n(L * 12 / 26, scale)) - _ema(lab, _n(L, scale))
        sig = macd.ewm(span=_n(L * 9 / 26, scale), adjust=False).mean()
        out = out + _level((macd > 0) & (macd > sig), g)
    return out / len(LADDER)


TYPES = {"don": t_don, "ema": t_ema, "tsmom": t_tsmom, "kelt": t_kelt, "st": t_st, "slope": t_slope, "macd": t_macd}


def type_signal(lab: Lab, name: str, scale: float = 1.0) -> pd.DataFrame:
    def go():
        out = TYPES[name](lab, scale, _gate(lab))
        for k in [k for k in lab._cache if isinstance(k, tuple) and k[:2] in ((FAMILY, "ema"), (FAMILY, "atr"))]:
            del lab._cache[k]                                   # indicators are cheap to redo; keep memory low
        return out
    return lab.memo((FAMILY, "type", name, float(scale)), go)


def layer(lab: Lab, types=("don",), scale: float = 1.0, min_agree: float = 0.0) -> pd.DataFrame:
    """S in [0, 1]: equal-weight average over `types` of each type's fraction of gated systems that are long.
    With min_agree > 0, a coin's signal is zeroed unless at least that fraction of the evidence agrees."""
    S = sum(type_signal(lab, t, scale) for t in types) / len(types)
    if min_agree > 0:
        S = S.where(S >= min_agree, 0.0)
    return S.where(lab.live, 0.0).astype("float32")


def weights(lab: Lab, n: int, calendar: str, **params) -> pd.DataFrame:
    return lab.build(n, calendar, signal=lab.memo((FAMILY, json.dumps(params, sort_keys=True)), lambda: layer(lab, **params)))


# ---------------- runner ----------------
RESULTS = METHODS / f"{FAMILY}_train.csv"


def _done() -> pd.DataFrame:
    return pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame(columns=["params"])


def _run(lab: Lab, base: pd.DataFrame, params: dict) -> dict:
    key = json.dumps(params, sort_keys=True)
    done = _done()
    if key in set(done["params"]):
        print("already run:", key)
        return done[done["params"] == key].iloc[0].to_dict()
    if len(done) >= MAX_TRIALS:
        raise SystemExit("trial budget used up")
    d = lab.evaluate(weights, FAMILY, params)
    print(key, ":", show(d, base), flush=True)
    row = {"params": key, **summary(d, base).to_dict(),
           **{f"sharpe_{n}_{c}": s for n, c, s in zip(d.n, d.calendar, d.train_sharpe)}}
    pd.concat([done, pd.DataFrame([row])], ignore_index=True).to_csv(RESULTS, index=False)
    # free this trial's signal (the per-type signals stay cached for reuse)
    lab._cache.pop((FAMILY, key), None)
    return row


def stage2_candidates(done: pd.DataFrame, base_sharpe: float) -> list[dict]:
    res = {r["params"]: r for _, r in done.iterrows()}
    out = []
    A = [t for t in NEW if res.get(json.dumps({"types": [t]}, sort_keys=True), {}).get("sharpe", -9) >= base_sharpe - STAGE2_THRESHOLD]
    ens = [GRID[i] for i in (6, 7, 8, 9)]
    if A:
        cand = {"types": ["don"] + A}
        if all(sorted(cand["types"]) != sorted(e["types"]) or e.get("min_agree") for e in ens):
            out.append(cand)
    return out


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    lab = Lab.load()
    base = lab.baseline_results
    bs = summary(base)
    print("baseline :", show(base), flush=True)
    if cmd == "check":   # the 'don' type at scale 1 must be the live bot's signal exactly
        S = layer(lab, ["don"])
        print("max |don - baseline| =", float((S - lab.baseline_signal()).abs().max().max()))
    elif cmd == "diag":  # signal properties only (no returns), train dates, one universe variant
        m = lab.mask(15, "Q-jan") & lab.live
        tr = lab.index < pd.Timestamp("2023-01-01", tz="UTC")
        mm = m[tr].to_numpy()
        don = type_signal(lab, "don")[tr].to_numpy()[mm]
        g = _gate(lab)[tr].to_numpy()[mm]
        print(f"gate on={g.mean():.2f}")
        for t in ALL:
            S = type_signal(lab, t)[tr]
            v = S.to_numpy()[mm]
            print(f"{t:6s} on={(v > 0).mean():.2f} meanS={v.mean():.3f} "
                  f"dS/yr={float(S.where(m[tr], 0).diff().abs().sum(axis=1).mean() * 2190):.0f} "
                  f"corr_don={np.corrcoef(v, don)[0, 1]:.2f} corr_gate={np.corrcoef(v, g)[0, 1]:.2f}", flush=True)
    elif cmd == "stage1":
        for params in GRID:
            _run(lab, base, params)
    elif cmd == "stage2a":
        for params in stage2_candidates(_done(), bs.sharpe):
            _run(lab, base, params)
    elif cmd == "stage2b":
        params = json.loads(sys.argv[2])
        for s in (0.7, 1.4):
            _run(lab, base, {**params, "scale": s})
    elif cmd == "leak":
        params = json.loads(sys.argv[2])
        a = lab.leak_check(weights, params)
        b = lab.leak_check(weights, params, dates=("2019-09-10 12:00", "2021-05-19 04:00"))
        print("leak_check", a, b)


if __name__ == "__main__":
    main()
