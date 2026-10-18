"""timeframes - timeframe diversification: blend trend sleeves that live on different bar lengths.

Idea: the live bot's signal is six 4h Donchian systems (entries 50/100/200 bars = 8/17/33 days, exits 10/20
bars = 1.7/3.3 days) under a daily close > EMA50 gate. It is fast: it is flat most of the time (~18% invested)
and its short exits cut big crypto trends early. Trend followers diversify across speeds because a slower
system holds the long trends the fast one keeps exiting, and its errors (late entries, giving back gains)
come at different times from the fast system's errors (whipsaws). So we add sleeves on slower timeframes and
average them with the 4h sleeve into one signal S in [0, 1]:

  b  the live bot's 4h sleeve (lab.baseline_signal()), confirmed by the daily close > EMA50 (gate mode).
  s  a slow 4h sleeve with 12h-equivalent lengths (entries 150/300/600 4h bars = 25/50/100 days, exits
     30/60 bars = 5/10 days), also under the daily gate. (algo.htf has no 12h timeframe, so the 12h system
     is expressed on the 4h grid with three times the lengths.)
  d  a DAILY sleeve: Donchian on daily candles (built with algo.htf.resample from the 4h bars), entries
     20/55/100 days, exits 10/20 days, confirmed by the WEEKLY timeframe (weekly close > weekly EMA20 or
     EMA50, via algo.htf.confirmation = resample + align), in gate mode: it may only enter while the weekly
     confirmation holds and it is closed when the confirmation turns off. Daily values reach the 4h grid with
     algo.htf.align, i.e. only on the 4h bar whose close is the daily close.

Every sleeve therefore keeps a confirmation from a timeframe above its own (daily for 4h sleeves, weekly
for the daily sleeve), so every entry has a daily-or-higher confirmation, as the user requires.

Layer: signal (lab.build(signal=S)). S = sum_i mix_i * sleeve_i / sum_i mix_i ("mean"), or the max over the
chosen sleeves ("max"). Ranking (top 3 by 14-day momentum), sizing (S / max(sum S, 6)) and costs unchanged.
"""
import json
import sys

import numpy as np
import pandas as pd

from algo.htf import align, confirmation, resample, state
from research.harness import METHODS, Lab, show, summary

FAMILY = "timeframes"
S_ENTRIES, S_EXITS = (150, 300, 600), (30, 60)        # 4h bars (= a 12h system with 50/100/200, 10/20)
D_ENTRIES, D_EXITS = (20, 55, 100), (10, 20)          # days
W20, W50 = "close>ema20", "close>ema50"               # weekly confirmations for the daily sleeve

# ----- GRID, declared before any evaluation (at most 12 of the 20 allowed trials) -----
# Stage 1 (6 trials): what each slower sleeve carries alone, and the 50/50 blends with the live 4h sleeve.
GRID = [
    {"mix": {"d": 1}, "wk": W20},
    {"mix": {"d": 1}, "wk": W50},
    {"mix": {"s": 1}},
    {"mix": {"b": 0.5, "d": 0.5}, "wk": W20},
    {"mix": {"b": 0.5, "d": 0.5}, "wk": W50},
    {"mix": {"b": 0.5, "s": 0.5}},
]
# Stage 2 (4 trials), a rule fixed in advance: wk* = the weekly confirmation of the better 50/50 b+d blend
# (stage-1 trials 4/5, by median Sharpe). Then evaluate, all with wk*:
#   the three-timeframe blend b+s+d (1/3 each), the b+d blend at 75/25 and 25/75 (blend-weight plateau),
#   and max(b, d) ("either timeframe is in trend -> full signal").
def stage2(wk: str) -> list:
    return [{"mix": {"b": 1 / 3, "s": 1 / 3, "d": 1 / 3}, "wk": wk},
            {"mix": {"b": 0.75, "d": 0.25}, "wk": wk},
            {"mix": {"b": 0.25, "d": 0.75}, "wk": wk},
            {"mix": {"b": 1, "d": 1}, "wk": wk, "agg": "max"}]
# Stage 3 (<= 2 trials): C = the stage-1/2 trial with the highest median Sharpe among those that pass the
# improvement test (median Sharpe > baseline median AND beats_base >= 0.67). If C exists, re-evaluate it with
# the lengths of its slower sleeve scaled x0.7 and x1.4 (dscale if it uses d, else sscale). C is frozen only
# if both neighbours also have median Sharpe > the baseline's (a plateau) and, for b+d blends, the adjacent
# blend weights are not worse than the baseline. If another passing trial is within 0.05 Sharpe of C and is
# simpler (fewer sleeves; "mean" before "max"), that one is frozen instead (and gets the plateau test).
SCALES = (0.7, 1.4)
MAX_TRIALS = 20
FROZEN = None  # set after running GRID on train; None if nothing beat the baseline
# Outcome (train 2018-11..2022, 10 trials, stage 3 never triggered): no candidate beat the baseline (median
# Sharpe 1.57); none beat it in more than 4 of 9 variants. Sharpe falls monotonically with the daily sleeve's
# weight (b only 1.57, 75/25 1.52, 50/50 1.47, 25/75 1.39, d only 1.26 with weekly EMA50 / 1.43 with EMA20);
# slow 4h sleeve 1.44 alone, 1.53 at 50/50; b+s+d 1.44; max(b, d) 1.39 (CAGR +58% but maxDD -39%, worst -48%).


# ---------------- sleeves ----------------
def _n(x: float, scale: float) -> int:
    return max(2, int(round(x * scale)))


def _daily(lab: Lab) -> dict:
    """Daily candles (open time index) resampled from the 4h bars; a candle is used only after it closes."""
    return lab.memo((FAMILY, "daily"), lambda: resample({"close": lab.close, "high": lab.high, "low": lab.low}, "1d"))


def _weekly_gate_daily(lab: Lab, wk: str) -> pd.DataFrame:
    """Weekly confirmation on the daily grid: a weekly candle counts from the daily bar on whose close it closes."""
    def go():
        return confirmation({"close": lab.close, "high": lab.high, "low": lab.low}, _daily(lab)["close"].index, ("1w",), wk)
    return lab.memo((FAMILY, "wgate", wk), go)


def sleeve_b(lab: Lab) -> pd.DataFrame:
    return lab.baseline_signal()


def sleeve_s(lab: Lab, scale: float = 1.0) -> pd.DataFrame:
    """Slow 4h sleeve (12h-equivalent lengths), entries and positions gated by the daily confirmation."""
    def go():
        g = lab.gate(lab.cfg.confirm_tfs, lab.cfg.confirm)
        tot = 0
        for n in S_ENTRIES:
            for m in S_EXITS:
                e, x = lab.donchian(_n(n, scale), _n(m, scale))
                tot = tot + state(e & g, x | ~g)
        return (tot / (len(S_ENTRIES) * len(S_EXITS))).where(lab.live, 0.0).astype("float32")
    return lab.memo((FAMILY, "s", float(scale)), go)


def sleeve_d(lab: Lab, wk: str = W50, scale: float = 1.0) -> pd.DataFrame:
    """Daily Donchian sleeve, entries and positions gated by the weekly confirmation, mapped to the 4h grid."""
    def go():
        p = _daily(lab)
        c, h, l = p["close"], p["high"], p["low"]
        g = _weekly_gate_daily(lab, wk)
        tot = 0
        for n in D_ENTRIES:
            hi = h.rolling(_n(n, scale)).max().shift()
            for m in D_EXITS:
                e = c > hi
                x = c < l.rolling(_n(m, scale)).min().shift()
                tot = tot + state(e & g, x | ~g)
        daily = tot / (len(D_ENTRIES) * len(D_EXITS))
        return align(daily, lab.index, "1d").fillna(0.0).where(lab.live, 0.0).astype("float32")
    return lab.memo((FAMILY, "d", wk, float(scale)), go)


def layer(lab: Lab, mix: dict, wk: str = W50, dscale: float = 1.0, sscale: float = 1.0, agg: str = "mean") -> pd.DataFrame:
    """S in [0, 1]: a weighted average (or the max) of the chosen timeframe sleeves."""
    get = {"b": lambda: sleeve_b(lab), "s": lambda: sleeve_s(lab, sscale), "d": lambda: sleeve_d(lab, wk, dscale)}
    parts = {k: get[k]() for k, v in mix.items() if v > 0}
    if agg == "max":
        S = None
        for v in parts.values():
            S = v.astype("float32") if S is None else np.maximum(S, v)
    else:
        tot = sum(mix[k] for k in parts)
        S = sum(parts[k].astype("float32") * (mix[k] / tot) for k in parts)
    return S.where(lab.live, 0.0).clip(0, 1).astype("float32")


def weights(lab: Lab, n: int, calendar: str, **params) -> pd.DataFrame:
    return lab.build(n, calendar, signal=lab.memo((FAMILY, json.dumps(params, sort_keys=True)), lambda: layer(lab, **params)))


# ---------------- running the protocol ----------------
CSV = METHODS / f"{FAMILY}_train.csv"


def _key(p: dict) -> str:
    return json.dumps(p, sort_keys=True)


def _load_rows() -> list:
    if not CSV.exists():
        return []
    d = pd.read_csv(CSV)
    return [{**r, "params": json.loads(r["params"])} for r in d.to_dict("records")]


def _run(lab: Lab, base: pd.DataFrame, plist: list, rows: list) -> list:
    done = {_key(r["params"]) for r in rows}
    for params in plist:
        if _key(params) in done:
            continue
        n_used = sum(1 for _ in open(METHODS / "trials.jsonl") if json.loads(_)["name"] == FAMILY)
        if n_used >= MAX_TRIALS:
            raise SystemExit("trial budget exhausted")
        d = lab.evaluate(weights, FAMILY, params)
        print(_key(params), ":", show(d, base), flush=True)
        rows.append({"params": params, **summary(d, base).to_dict()})
        pd.DataFrame([{**r, "params": _key(r["params"])} for r in rows]).to_csv(CSV, index=False)
    return rows


def _passes(r: dict, b: float) -> bool:
    return r["sharpe"] > b and r["beats_base"] >= 0.67 - 1e-9


def _complexity(p: dict) -> tuple:
    return (sum(1 for v in p["mix"].values() if v > 0), p.get("agg", "mean") == "max")


def _neighbours(p: dict) -> list:
    k = "dscale" if p["mix"].get("d", 0) > 0 else "sscale"
    return [{**p, k: s} for s in SCALES]


def main(stage: str) -> None:
    lab = Lab.load()
    base = lab.baseline_results
    b = summary(base)
    print("baseline :", show(base), flush=True)
    rows = _load_rows()
    if stage in ("1", "all"):
        rows = _run(lab, base, GRID, rows)
    if stage in ("2", "all"):
        r4, r5 = (next(r for r in rows if _key(r["params"]) == _key(GRID[i])) for i in (3, 4))
        wk = r4["params"]["wk"] if r4["sharpe"] >= r5["sharpe"] else r5["params"]["wk"]
        print("wk* =", wk, flush=True)
        rows = _run(lab, base, stage2(wk), rows)
    if stage in ("3", "all"):
        cand = [r for r in rows if "dscale" not in r["params"] and "sscale" not in r["params"] and _passes(r, b.sharpe)]
        if not cand:
            print("no stage-1/2 trial passes the improvement test -> FROZEN = None", flush=True)
            return
        best = max(cand, key=lambda r: r["sharpe"])
        simpler = [r for r in cand if r["sharpe"] >= best["sharpe"] - 0.05 and _complexity(r["params"]) < _complexity(best["params"])]
        C = min(simpler, key=lambda r: (_complexity(r["params"]), -r["sharpe"])) if simpler else best
        print("candidate C:", _key(C["params"]), flush=True)
        rows = _run(lab, base, _neighbours(C["params"]), rows)
    for r in rows:
        print(f"{_key(r['params']):80s} sharpe {r['sharpe']:.3f} beats {r['beats_base']:.2f} cagr {r['cagr']:+.2f} "
              f"maxdd {r['maxdd']:.2f} worst {r['worst_dd']:.2f} turn {r['turnover']:.0f} expo {r['exposure']:.2f}")


def diagnostics() -> None:
    """Signal-only facts (no performance): how often each sleeve is long and how much the sleeves agree."""
    lab = Lab.load()
    m = lab.mask(15, "Q-jan")
    tr = (lab.index >= "2018-11-01") & (lab.index < "2023-01-01")
    sl = {"b": sleeve_b(lab), "s": sleeve_s(lab), "d20": sleeve_d(lab, W20), "d50": sleeve_d(lab, W50)}
    flat = {k: v.where(m)[tr].stack() for k, v in sl.items()}
    df = pd.DataFrame(flat)
    print("mean signal in universe (train):\n", df.mean().round(3))
    print("share of coin-bars with signal > 0:\n", (df > 0).mean().round(3))
    print("correlation of sleeve signals:\n", df.corr().round(2))
    ch = {k: (v.where(m)[tr].diff().abs() > 0).sum().sum() / tr.sum() * 6 * 365 / m[tr].sum(axis=1).mean() for k, v in sl.items()}
    print("signal changes per coin per year:", {k: round(v, 1) for k, v in ch.items()})


def leak(params: dict) -> float:
    lab = Lab.load()
    a = lab.leak_check(weights, params)
    c = lab.leak_check(weights, params, dates=("2019-09-10 12:00", "2021-05-19 04:00"))
    print("leak_check:", a, c)
    return max(a, c)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "diag":
        diagnostics()
    elif cmd == "leak":
        leak(json.loads(sys.argv[2]))
    else:
        main(cmd)
