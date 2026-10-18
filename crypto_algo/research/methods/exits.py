"""exits - how positions are closed.

Idea: the live bot enters on Donchian 50/100/200-bar breakouts confirmed by the daily close > EMA50 and
closes each of its 6 systems on a 10- or 20-bar low or on the daily-gate flip. Those exits are tight (1.7-3.3
days), so trends are cut early and the same coin is re-bought later (turnover 54x/yr). This family keeps the
baseline ENTRIES exactly (Donchian breakout AND daily confirmation - the user's rule) and changes only how a
position is closed:
  * Donchian exit lengths (5/10, 20/40 instead of 10/20),
  * which higher-timeframe flip closes a position: daily (baseline), weekly, or none,
  * an ATR chandelier trailing stop (exit when close < highest close since entry - k*ATR) instead of the
    Donchian low,
  * a time stop (exit after N bars without a new high since entry),
  * partial profit-taking (halve a system's position once the close reached entry + x*ATR),
  * a break-even stop (after +x*ATR, exit if the close falls back below the entry price).
Layer: signal (lab.build(signal=S), S in [0, 1] = fraction of entry x exit-leg systems that are long).

Systems = entries (50, 100, 200) x exit legs. An exit leg is a Donchian low ("don", m), a chandelier stop
("ch", k) or, when neither is given, only the higher-timeframe flip. The HTF flip and the global path rules
(time stop, partial, break-even) apply to every leg. Path-dependent exits are run by a per-coin trade loop over
numpy arrays (entry -> scan forward in growing chunks -> first exit bar), which is exact and causal: the exit
decision at bar t uses only the close/ATR up to t and the entry bar's close/ATR.
"""
import json
import sys

import numpy as np
import pandas as pd

from algo.htf import state
from research.harness import METHODS, Lab, show, summary

FAMILY = "exits"
DEFAULTS = {"exits": [10, 20], "chand": [], "gate_exit": "1d", "atr_n": 42, "tstop": 0, "partial": 0.0, "be": 0.0}

# ----- GRID, declared before any evaluation -----
# Descriptive facts about the baseline's trades on train (no performance of any new rule was computed first):
# median trade 21-42 bars; 84-98% of exits come from the Donchian low and only 2-16% from the daily-gate flip;
# win rate 33-38%; max favourable excursion median 2-4 ATR(42), p75 6-10, p90 13-23; median giveback from the
# peak to the exit bar 3.6-5.8 ATR. So chandelier k ~3-6 is about as tight as the 10/20-bar lows, the gate-exit
# choice should matter little, and profit-taking at +6/+12 ATR touches the best quarter / tenth of trades.
# Stage 1 (13 trials), each one idea against the baseline:
GRID = [
    {"exits": [5, 10]},                      # tighter Donchian exits
    {"exits": [20, 40]},                     # looser Donchian exits: let winners run
    {"exits": [5, 10, 20, 40]},              # diversify exit speeds (12 systems)
    {"exits": []},                           # hold until the daily confirmation flips
    {"gate_exit": "1w"},                     # close on the weekly flip (weekly close < EMA20) instead of daily
    {"exits": [], "chand": [3, 6]},          # chandelier stop instead of the Donchian low (~ as tight as 10/20)
    {"exits": [], "chand": [4, 8]},
    {"exits": [], "chand": [6, 12]},         # looser chandelier
    {"tstop": 30},                           # + time stop: 5 days without a new high
    {"tstop": 60},                           # + time stop: 10 days without a new high
    {"partial": 6.0},                        # + halve each system at entry + 6 ATR
    {"partial": 12.0},                       # + halve each system at entry + 12 ATR
    {"be": 4.0},                             # + break-even stop once +4 ATR was reached
]
# Stage 2 (<= 6 trials), a rule fixed in advance:
#   PASS = stage-1 trials with median Sharpe > baseline median Sharpe AND beats_base >= 0.67.
#   If PASS is empty: stop, FROZEN = None.
#   Otherwise, for the best PASS trial by median Sharpe (and the second best if it is a different idea and
#   within 0.05), evaluate its NEIGHBOURS not yet run. It is a plateau only if every neighbour has median
#   Sharpe >= baseline median - 0.05. If two PASS trials of different ideas are both plateaus, evaluate their
#   combination once; keep it only if it beats the simpler one by > 0.05 median Sharpe and passes the test.
#   FROZEN = the simplest plateau candidate within 0.05 of the best; None if no candidate is a plateau.
NEIGHBOURS = {
    0: [{"exits": [3, 6]}],                                   # [10, 20] = baseline is the other neighbour
    1: [{"exits": [15, 30]}, {"exits": [30, 60]}],
    2: [{"exits": [5, 10, 20]}, {"exits": [10, 20, 40]}],
    3: [{"exits": [40, 80]}],                                 # [20, 40] is in stage 1
    4: [{"gate_exit": "none"}],                               # daily = baseline is the other neighbour
    5: [{"exits": [], "chand": [2, 4]}],                      # [4, 8] is in stage 1
    6: [],                                                    # [3, 6] and [6, 12] are in stage 1
    7: [{"exits": [], "chand": [8, 16]}],                     # [4, 8] is in stage 1
    8: [{"tstop": 18}],                                       # 60 is in stage 1
    9: [{"tstop": 90}],                                       # 30 is in stage 1
    10: [{"partial": 4.0}],                                   # 12 is in stage 1
    11: [{"partial": 18.0}],                                  # 6 is in stage 1
    12: [{"be": 2.0}, {"be": 6.0}],
}
STAGE1_NEIGHBOURS = {0: [], 1: [], 2: [], 3: [1], 4: [], 5: [6], 6: [5, 7], 7: [6], 8: [9], 9: [8], 10: [11], 11: [10], 12: []}
MAX_TRIALS = 20
FROZEN = {"exits": [5, 10]}  # Donchian exits on the 5- and 10-bar low (instead of 10/20), daily-gate exit kept
# Outcome (train 2018-11..2022, 14 trials; baseline median Sharpe 1.567):
#   Stage 1 PASS: exits [5, 10] (1.65, beats 9/9) and break-even after +4 ATR (1.58, 9/9; not within 0.05 of
#   the best, so not followed up). Stage 2: neighbour exits [3, 6] = 1.69 (9/9); the other neighbour is the
#   baseline [10, 20] = 1.57, so [5, 10] sits on a plateau. [3, 6] is within 0.05 and its tighter side was not
#   tested, so the rule keeps [5, 10] (equally simple, closer to the live bot, plateau confirmed on both sides).
#   Looser exits all hurt: [20, 40] 1.30, gate-only 1.22, chandelier 3/6, 4/8, 6/12 ATR = 1.45, 1.38, 1.29.
#   Weekly-flip exit 1.58 (5/9), time stop 30/60 bars 1.56/1.51, partial profit at +6/+12 ATR 1.40/1.51,
#   [5, 10, 20, 40] 1.52 - none passes.


# ---------------- building blocks ----------------
def _px(lab: Lab):
    return lab.memo((FAMILY, "px"), lambda: lab.close.ffill())


def atr(lab: Lab, n: int) -> pd.DataFrame:
    """Wilder ATR on 4h bars (true range from the previous traded close), causal."""
    def go():
        cp = _px(lab).shift()
        tr = np.maximum(lab.high, cp) - np.minimum(lab.low, cp)
        tr = tr.where(lab.high.notna())
        return tr.ewm(alpha=1 / n, adjust=False, ignore_na=True).mean()
    return lab.memo((FAMILY, "atr", n), go)


def gate_exit(lab: Lab, mode: str) -> pd.DataFrame:
    """Bool: the higher-timeframe confirmation is OFF (closes positions). 'none' never closes."""
    if mode == "1d":
        return ~lab.gate(lab.cfg.confirm_tfs, lab.cfg.confirm)
    if mode == "1w":
        return ~lab.gate(("1w",), "close>ema20")
    if mode == "none":
        return pd.DataFrame(False, index=lab.index, columns=lab.assets)
    raise ValueError(mode)


def _trade_loop(c, ent_idx, xs, a, k, tstop, part_x, be_x):
    """Positions (0 / 0.5 / 1) of one coin for one system.

    c: close (NaN while not trading), ent_idx: sorted bars where the entry is on, xs: static exit (bool),
    a: ATR. A position opened at the close of bar i is closed at the first bar j > i where an exit fires
    (pos[j] = 0, exit wins over a same-bar entry); the next trade opens at the first entry bar > j.
    """
    T = len(c)
    pos = np.zeros(T, np.float32)
    p = 0
    ne = len(ent_idx)
    while p < ne:
        i = int(ent_idx[p])
        e_px, a_e = c[i], a[i]
        target = e_px + part_x * a_e if part_x else np.inf
        be_lvl = e_px + be_x * a_e if be_x else np.inf
        cm_prev, last_high, reached, half_at = e_px, i, False, T
        start, chunk, j = i + 1, 64, T
        while start < T:
            end = min(T, start + chunk)
            cc = c[start:end]
            cm = np.fmax.accumulate(np.concatenate(([cm_prev], cc)))[1:]
            ex = xs[start:end].copy()
            if k:
                ex |= cc < cm - k * a[start:end]
            if tstop:
                ar = np.arange(start, end)
                last = np.maximum.accumulate(np.where(cc >= cm, ar, last_high))
                ex |= (ar - last) >= tstop
            if part_x and half_at == T:
                hit = np.flatnonzero(cm >= target)
                if hit.size:
                    half_at = start + int(hit[0])
            if be_x:
                r = np.logical_or.accumulate(cm >= be_lvl) | reached
                ex |= r & (cc < e_px)
            hit = np.flatnonzero(ex)
            if hit.size:
                j = start + int(hit[0])
                break
            cm_prev = cm[-1]
            if tstop:
                last_high = int(last[-1])
            if be_x:
                reached = bool(r[-1])
            start, chunk = end, chunk * 2
        pos[i:j] = 1.0
        if half_at < j:
            pos[half_at:j] = 0.5
        p = int(np.searchsorted(ent_idx, j, side="right"))
    return pos


def _system(lab: Lab, entry: pd.DataFrame, xs: pd.DataFrame, k: float, tstop: int, part_x: float, be_x: float,
            atr_n: int) -> np.ndarray:
    ent = (entry & ~xs).to_numpy()
    if not (k or tstop or part_x or be_x):
        return state(entry, xs).to_numpy()
    c = lab.close.to_numpy()
    a = atr(lab, atr_n).to_numpy()
    X = xs.to_numpy()
    out = np.zeros(c.shape, np.float32)
    for col in range(c.shape[1]):
        idx = np.flatnonzero(ent[:, col])
        if idx.size:
            out[:, col] = _trade_loop(c[:, col], idx, X[:, col], a[:, col], k, tstop, part_x, be_x)
    return out


def layer(lab: Lab, exits=(10, 20), chand=(), gate_exit_tf: str = "1d", atr_n: int = 42, tstop: int = 0,
          partial: float = 0.0, be: float = 0.0) -> pd.DataFrame:
    """Signal: average over entries (50/100/200 Donchian, daily-gated) x exit legs of each system's position."""
    g = lab.gate(lab.cfg.confirm_tfs, lab.cfg.confirm)          # entries ALWAYS need the daily confirmation
    gx = gate_exit(lab, gate_exit_tf)
    legs = [("don", m) for m in exits] + [("ch", k) for k in chand] or [("htf", 0)]
    total, nsys = 0.0, 0
    for n in lab.cfg.entries:
        e, _ = lab.donchian(n, 10)
        e = e & g
        for kind, v in legs:
            xs = gx | lab.donchian(n, v)[1] if kind == "don" else gx
            total = total + _system(lab, e, xs, v if kind == "ch" else 0.0, tstop, partial, be, atr_n)
            nsys += 1
    S = pd.DataFrame(total / nsys, index=lab.index, columns=lab.assets)
    return S.where(lab.live, 0.0)


def _layer(lab: Lab, params: dict) -> pd.DataFrame:
    p = {**DEFAULTS, **params}
    return layer(lab, exits=tuple(p["exits"]), chand=tuple(p["chand"]), gate_exit_tf=p["gate_exit"],
                 atr_n=p["atr_n"], tstop=p["tstop"], partial=p["partial"], be=p["be"])


def signal(lab: Lab, **params) -> pd.DataFrame:
    """Memoised signal for lab.build(signal=...), shared by the 9 universe variants."""
    return lab.memo((FAMILY, json.dumps(params, sort_keys=True)), lambda: _layer(lab, params))


def weights(lab: Lab, n: int, calendar: str, **params) -> pd.DataFrame:
    return lab.build(n, calendar, signal=signal(lab, **params))


# ---------------- running the grid ----------------
def n_trials() -> int:
    f = METHODS / "trials.jsonl"
    return sum(json.loads(l)["name"] == FAMILY for l in open(f)) if f.exists() else 0


def _run(lab: Lab, base: pd.DataFrame, params: dict, tag: str) -> dict:
    if n_trials() >= MAX_TRIALS:
        raise SystemExit(f"trial budget of {MAX_TRIALS} used up")
    d = lab.evaluate(weights, FAMILY, params)
    lab._cache.pop((FAMILY, json.dumps(params, sort_keys=True)), None)      # keep memory flat
    print(f"[{tag}] {json.dumps(params)}: {show(d, base)}", flush=True)
    return {"tag": tag, "params": json.dumps(params, sort_keys=True), **summary(d, base).to_dict()}


def _save(rows: list[dict]) -> None:
    f = METHODS / f"{FAMILY}_train.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(f, index=False)


def stage1(lab: Lab, base: pd.DataFrame) -> None:
    print("baseline :", show(base), flush=True)
    rows = []
    for i, params in enumerate(GRID):
        rows.append(_run(lab, base, params, f"s1-{i}"))
        _save([rows[-1]])


def _passes(r: pd.Series, b: float) -> bool:
    return r.sharpe > b and r.beats_base >= 0.67


def stage2(lab: Lab, base: pd.DataFrame) -> None:
    """Applies the stage-2 rule declared next to GRID to the stage-1 results."""
    b = summary(base).sharpe
    res = pd.read_csv(METHODS / f"{FAMILY}_train.csv")
    s1 = res[res.tag.str.startswith("s1-")].reset_index(drop=True)
    ok = s1[[_passes(r, b) for _, r in s1.iterrows()]].sort_values("sharpe", ascending=False)
    print("baseline median Sharpe", round(b, 3), "| stage-1 PASS:", ok[["tag", "params", "sharpe", "beats_base"]].to_dict("records"))
    if ok.empty:
        print("no stage-1 trial passes the improvement test -> FROZEN = None, no stage 2")
        return
    picks = [int(ok.iloc[0].tag[3:])]
    if len(ok) > 1 and ok.iloc[1].sharpe >= ok.iloc[0].sharpe - 0.05:
        picks.append(int(ok.iloc[1].tag[3:]))
    rows = []
    for i in picks:
        for params in NEIGHBOURS[i]:
            rows.append(_run(lab, base, params, f"s2-nb{i}"))
            _save([rows[-1]])
    print("stage-2 neighbour trials:", rows)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    lab = Lab.load()
    base = lab.baseline_results
    if cmd == "stage1":
        stage1(lab, base)
    elif cmd == "stage2":
        stage2(lab, base)
    elif cmd == "leak":
        for dates in (("2020-06-15 08:00", "2022-03-01 16:00"), ("2019-09-10 12:00", "2021-05-19 04:00")):
            print("leak_check", dates, lab.leak_check(weights, FROZEN or {}, dates=dates), flush=True)
    elif cmd == "variants":       # per-variant train numbers of the frozen candidate (same trial, not logged again)
        d = lab.evaluate(weights, FAMILY, FROZEN or {}, log=False)
        cols = ["n", "calendar", "train_sharpe", "train_cagr", "train_maxdd", "train_turnover", "train_exposure"]
        t = d[cols].merge(base[["n", "calendar", "train_sharpe", "train_maxdd"]], on=["n", "calendar"], suffixes=("", "_base"))
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
