"""Seventh batch: exits and channel geometry of the best bot (breakout/exit lengths, trailing stop, daily confirmation).

Same protocol as indicators3-5 / structure, declared before running:
train 2018-11..2022 (higher median Sharpe, >= 6/9 variants better), holdout A (coins 26-45) and B (coins 46-65) on
2023-01..2025-09 both better, and the neighbouring parameter also beats the baseline on train. The locked year
(2025-10-01..) has been looked at twice already, so it is NOT used here at all: a passing idea goes to the paper demo
as an extra profile, never straight into the bot.

    python -m research.methods.geometry
"""
import json

import numpy as np
import pandas as pd

from algo.htf import state
from research.harness import TRAIN, VARIANTS, Lab, metrics
from research.methods import indicators as I
from research.methods import indicators3 as I3
from research.methods.profile_variants import profile_full


def install(lab):
    orig = I.candidates

    def cands(l):
        v = l.volume.astype(float)
        return {**orig(l), "relvol_2.0x": ("entry", lambda: v > 2.0 * v.rolling(20).mean().shift())}
    I.candidates = cands


def signal(lab, entries=(50, 100, 200), exits=(10, 20), gate=(("1d",), "close>ema50"), xgate=None, chand=None):
    """The best bot's signal (I.signal with BASE_E/BASE_X) with its geometry exposed.
    xgate: an extra (tfs, name) confirmation whose failure also closes positions; chand: (bars, atr mult) trailing stop."""
    key = ("geo", entries, exits, gate, xgate, chand)

    def go():
        cand = I.candidates(lab)
        g = lab.gate(*gate)
        c = lab.close
        poc, vah, _ = profile_full(lab, 180, 24)
        ein, xout = c > vah, c < poc
        for name in I3.BASE_E:
            ein = ein & lab.memo(("cand", name), cand[name][1]).fillna(False).astype(bool)
        for name in I3.BASE_X:
            xout = xout | lab.memo(("cand", name), cand[name][1]).fillna(False).astype(bool)
        if xgate is not None:
            xout = xout | ~lab.gate(*xgate)
        if chand is not None:
            n, mult = chand
            xout = xout | (c < c.rolling(n).max() - mult * I.atr(lab, 14))
        total = 0
        for n in entries:
            for m in exits:
                e, x = lab.donchian(n, m)
                total = total + state(e & g & ein, x | ~g | xout)
        return (total / (len(entries) * len(exits))).where(lab.live, 0.0)
    return lab.memo(key, go)


def make(cfg):
    def fn(lab, n, cal, mask=None):
        kw = {"k": 6, "signal": signal(lab, **cfg)}
        if mask is not None:
            kw["mask"] = mask
        return lab.enforce(lab.build(n, cal, **kw), mask)
    return fn


def run(lab, fn, lo, hi):
    return pd.DataFrame([metrics(lab.pnl(fn(lab, n, cal)), lo, hi) for n, cal in VARIANTS])


def hold(lab, fn, ranks):
    sh = []
    for cal in ("Q-jan", "Q-feb", "Q-mar"):
        m = lab.memo(("hold", ranks, cal), lambda: I.holdout(lab, *ranks, cal))
        sh.append(metrics(lab.pnl(fn(lab, 20, cal, mask=m)), *I3.HOLD)["sharpe"])
    return float(np.median(sh))


G = lambda name, tfs=("1d",): (tfs, name)
IDEAS = {
    "exits_faster_5_10": ({"exits": (5, 10)}, {"exits": (7, 14)}),
    "exits_slower_20_40": ({"exits": (20, 40)}, {"exits": (15, 30)}),
    "entries_faster_20_50_100": ({"entries": (20, 50, 100)}, {"entries": (30, 75, 150)}),
    "entries_slower_100_200_400": ({"entries": (100, 200, 400)}, {"entries": (75, 150, 300)}),
    "trailing_stop_3atr": ({"chand": (50, 3.0)}, {"chand": (50, 3.5)}),
    "trailing_stop_5atr": ({"chand": (100, 5.0)}, {"chand": (100, 6.0)}),
    "daily_gate_ema20": ({"gate": G("close>ema20")}, {"gate": G("ema10>ema30")}),
    "daily_gate_higher_lows": ({"gate": G("higher-lows")}, {"gate": G("donchian20/10")}),
    "plus_weekly_gate": ({"gate": (("1d", "1w"), "close>ema50")}, {"gate": (("1d", "1w"), "close>ema20")}),
    "exit_on_daily_ema20": ({"xgate": G("close>ema20")}, {"xgate": G("ema10>ema30")}),
}


def main():
    lab = Lab.load()
    install(lab)
    ref = I.signal(lab, I3.BASE_E, I3.BASE_X)
    print("same as the best bot:", float((signal(lab) - ref).abs().max().max()), flush=True)
    print("leak:", max(lab.leak_check(lambda l, n, cal, c: make(c)(l, n, cal), {"c": IDEAS[k][0]}, dates=("2021-05-19 04:00",))
                       for k in ("trailing_stop_3atr", "plus_weekly_gate", "exit_on_daily_ema20")), flush=True)
    base = make({})
    bt = run(lab, base, *TRAIN)
    bA, bB = hold(lab, base, (25, 45)), hold(lab, base, (45, 65))
    b = bt.sharpe.median()
    print(f"baseline: train Sh {b:.2f} CAGR {bt.cagr.median():+.0%} DD {bt.maxdd.median():.0%} | A {bA:+.2f} | B {bB:+.2f}", flush=True)
    rows = []
    for name, (cfg, nb) in IDEAS.items():
        fn = make(cfg)
        d = run(lab, fn, *TRAIN)
        beats = (d.sharpe.to_numpy() > bt.sharpe.to_numpy()).mean()
        A, B = hold(lab, fn, (25, 45)), hold(lab, fn, (45, 65))
        plateau = run(lab, make(nb), *TRAIN).sharpe.median() > b
        ok = d.sharpe.median() > b and beats >= 6 / 9 and A > bA and B > bB and plateau
        rows.append({"name": name, "sharpe": d.sharpe.median(), "cagr": d.cagr.median(), "maxdd": d.maxdd.median(), "beats": beats,
                     "A": A, "B": B, "plateau": plateau, "pass_all": ok})
        print(f"{name:28s} train Sh {d.sharpe.median():.2f} CAGR {d.cagr.median():+.0%} DD {d.maxdd.median():.0%} beats {beats:.0%}"
              f" | A {A:+.2f} | B {B:+.2f} | plateau {'ok' if plateau else 'NO'}{'  <== PASS ALL' if ok else ''}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("data/methods/geometry.csv", index=False)
    winners = list(t[t.pass_all].name)
    json.dump({"passed": winners}, open("data/methods/geometry_choice.json", "w"))
    print("\npassed all tests:", winners or "nothing - keep the current best bot")


if __name__ == "__main__":
    main()
