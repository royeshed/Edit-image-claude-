"""Sixth batch: the bot's structure (which coins, how much in each, which signalled coins first) on top of the best bot.

Earlier families (sizing, ranking, universe_*) were researched on the old k=3 bot without the volume profile; this
re-tests the leads and a few new ideas on the current best bot (k=6 + volume profile + ATR expansion + 2x volume +
CMF exit). Declared before running; nothing on/after 2025-10-01 chooses anything:
  1. train 2018-11..2022 (9 variants): higher median Sharpe and >= 6/9 variants better;
  2. holdouts 2023-01..2025-09: sizing/ranking ideas -> coins 26-45 (A) and 46-65 (B), both better;
     universe ideas change the coin list itself, so their holdout is the bot's own list on 2023-01..2025-09
     (median of the 9 variants) - better than the baseline there;
  3. plateau: the neighbouring parameter also beats the baseline on train.

    python -m research.methods.structure
"""
import json

import numpy as np
import pandas as pd

from research.harness import LOCKED, TRAIN, VARIANTS, Lab, metrics
from research.methods import indicators as I
from research.methods import indicators3 as I3
from research.methods.ranking import score
from research.methods.sizing import make_sizing
from research.universe_selection import CALENDARS, rule_scores_all

HOLD = I3.HOLD


def install(lab):
    orig = I.candidates

    def cands(l):
        v = l.volume.astype(float)
        return {**orig(l), "relvol_2.0x": ("entry", lambda: v > 2.0 * v.rolling(20).mean().shift())}
    I.candidates = cands


def base_signal(lab):
    return I.signal(lab, I3.BASE_E, I3.BASE_X)


def uni_score(lab, kind, p):
    f = lab.features
    if kind == "age":
        return f.vol90.where(f.age >= p)
    if kind == "vol30":                      # faster liquidity rank (catches coins getting hot)
        return f.vol30.where(f.age >= 365)
    if kind == "vol_blend":
        g = f.groupby("date")
        return (g.vol30.rank(pct=True) + g.vol90.rank(pct=True)).where(f.age >= 365)
    return lab.memo(("rule", "mature_volume"), lambda: rule_scores_all(f, "mature_volume"))


def make(cfg: dict):
    """cfg -> fn(lab, n, calendar, mask=None) giving weights. Keys: uni, p, refresh, extra_n, k, rank, days, sizing."""
    def fn(lab, n, cal, mask=None):
        kw = {"k": cfg.get("k", 6), "signal": base_signal(lab)}
        if "slots" in cfg:
            kw["slots"] = cfg["slots"]
        if "rank" in cfg:
            kw["rank"] = lab.memo(("rk", cfg["rank"], cfg.get("days", 14)),
                                  lambda: score(lab, cfg["rank"], cfg.get("days", 14), signal=base_signal(lab)))
        if "sizing" in cfg:
            kw["sizing"] = make_sizing(**cfg["sizing"])
        if mask is None and ("uni" in cfg or "refresh" in cfg or "extra_n" in cfg):
            s = uni_score(lab, cfg.get("uni"), cfg.get("p"))
            from research.universe_selection import membership, universe
            months = {**CALENDARS, "bimonthly": (1, 3, 5, 7, 9, 11)}[cfg.get("refresh", cal)]
            mask = membership(universe(lab.features, s, months, n + cfg.get("extra_n", 0)), lab.index, lab.assets).astype(bool)
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
        sh.append(metrics(lab.pnl(fn(lab, 20, cal, mask=m)), *HOLD)["sharpe"])
    return float(np.median(sh))


# name -> (family, main cfg, neighbour cfg)
IDEAS = {
    "universe_age_180d": ("universe", {"uni": "age", "p": 180}, {"uni": "age", "p": 270}),
    "universe_rank_vol30": ("universe", {"uni": "vol30"}, {"uni": "vol_blend"}),
    "universe_monthly_refresh": ("universe", {"refresh": "monthly"}, {"refresh": "bimonthly"}),
    "universe_plus_10_coins": ("universe", {"extra_n": 10}, {"extra_n": 5}),
    "size_inverse_vol_0.7": ("sizing", {"sizing": {"ivol": 0.7}}, {"sizing": {"ivol": 1.0}}),
    "size_full_slot_any_signal": ("sizing", {"sizing": {"power": 0.0}}, {"sizing": {"power": 0.5}}),
    "size_8_coins": ("sizing", {"k": 8, "slots": 8}, {"k": 7, "slots": 7}),
    "size_vol_target_40": ("sizing", {"sizing": {"vt": 0.4}}, {"sizing": {"vt": 0.5}}),
    "size_halve_in_drawdown_15": ("sizing", {"sizing": {"dd": 0.15}}, {"sizing": {"dd": 0.20}}),
    "rank_momentum_7d": ("ranking", {"rank": "mom", "days": 7}, {"rank": "mom", "days": 10}),
    "rank_signal_strength_first": ("ranking", {"rank": "sigstr", "days": 14}, {"rank": "sigstr", "days": 28}),
}


def main():
    lab = Lab.load()
    install(lab)
    base = make({})
    print("leak:", max(lab.leak_check(lambda l, n, cal, c: make(c)(l, n, cal), {"c": IDEAS[k][1]}, dates=("2021-05-19 04:00",))
                       for k in ("universe_rank_vol30", "size_vol_target_40", "rank_signal_strength_first")), flush=True)
    bt, bh = run(lab, base, *TRAIN), run(lab, base, *HOLD)
    bA, bB = hold(lab, base, (25, 45)), hold(lab, base, (45, 65))
    b = bt.sharpe.median()
    print(f"baseline: train Sh {b:.2f} CAGR {bt.cagr.median():+.0%} DD {bt.maxdd.median():.0%} | 2023-25.09 own list Sh {bh.sharpe.median():.2f}"
          f" CAGR {bh.cagr.median():+.0%} DD {bh.maxdd.median():.0%} | A {bA:+.2f} | B {bB:+.2f}", flush=True)
    rows = []
    for name, (fam, cfg, nb) in IDEAS.items():
        fn = make(cfg)
        d, h = run(lab, fn, *TRAIN), run(lab, fn, *HOLD)
        beats = (d.sharpe.to_numpy() > bt.sharpe.to_numpy()).mean()
        A, B = (hold(lab, fn, (25, 45)), hold(lab, fn, (45, 65))) if fam != "universe" else (np.nan, np.nan)
        plateau = run(lab, make(nb), *TRAIN).sharpe.median() > b
        hold_ok = h.sharpe.median() > bh.sharpe.median() if fam == "universe" else (A > bA and B > bB)
        ok = d.sharpe.median() > b and beats >= 6 / 9 and hold_ok and plateau
        rows.append({"name": name, "family": fam, "sharpe": d.sharpe.median(), "cagr": d.cagr.median(), "maxdd": d.maxdd.median(),
                     "beats": beats, "hold_sharpe": h.sharpe.median(), "hold_cagr": h.cagr.median(), "hold_maxdd": h.maxdd.median(),
                     "A": A, "B": B, "plateau": plateau, "pass_all": ok})
        print(f"{name:28s} train Sh {d.sharpe.median():.2f} CAGR {d.cagr.median():+.0%} DD {d.maxdd.median():.0%} beats {beats:.0%}"
              f" | 2023-25.09 Sh {h.sharpe.median():.2f} CAGR {h.cagr.median():+.0%} DD {h.maxdd.median():.0%} | A {A:+.2f} | B {B:+.2f}"
              f" | plateau {'ok' if plateau else 'NO'}{'  <== PASS ALL' if ok else ''}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("data/methods/structure.csv", index=False)
    winners = list(t[t.pass_all].sort_values("sharpe", ascending=False).name)
    print("\npassed all tests:", winners, flush=True)
    chosen, best = {}, b
    for name in winners:
        trial = {**chosen, **IDEAS[name][1]}
        if "sizing" in chosen and "sizing" in IDEAS[name][1]:
            trial["sizing"] = {**chosen["sizing"], **IDEAS[name][1]["sizing"]}
        s = run(lab, make(trial), *TRAIN).sharpe.median()
        print(f"  + {name:26s} train Sh {s:.2f}", flush=True)
        if s > best + 0.02:
            chosen, best = trial, s
    json.dump({"chosen": chosen}, open("data/methods/structure_choice.json", "w"))
    print("\nCHOSEN:", chosen or "nothing - keep the current best bot")


if __name__ == "__main__":
    main()
