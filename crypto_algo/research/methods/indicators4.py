"""Fourth batch: liquidity, chop/range and accumulation (order flow) on top of the current best bot.

Same protocol as indicators3 (last year locked: nothing on/after 2025-10-01 chooses anything):
train 2018-11..2022 (>= 6/9 better, higher median Sharpe), holdout A (coins 26-45) and B (coins 46-65) on
2023-01..2025-09 both better, and a neighbouring parameter must also beat the baseline on train.

    python -m research.methods.indicators4
"""
import json

import numpy as np
import pandas as pd

from research.harness import Lab, show, summary
from research.methods import indicators as I
from research.methods import indicators3 as I3


def new_candidates(lab: Lab, p: float = 1.0) -> dict:
    """p scales every parameter (1.0 = main setting, other values = neighbours for the plateau check)."""
    c, h, l, o = lab.close, lab.high, lab.low, lab.open.astype(float)
    v = lab.volume.astype(float)
    n = lambda x: max(2, int(round(x * p)))
    qv, tb, tr = I3.flow(lab)
    net = (2 * tb / qv - 1) * qv                     # signed taker flow (USDT)
    cvd = net.fillna(0).cumsum()

    def cs_spread():
        b = (np.log(h / l) ** 2) + (np.log(h.shift() / l.shift()) ** 2)
        g = np.log(np.maximum(h, h.shift()) / np.minimum(l, l.shift())) ** 2
        k = 3 - 2 * np.sqrt(2)
        a = (np.sqrt(2 * b) - np.sqrt(b)) / k - np.sqrt(g / k)
        return (2 * (np.exp(a) - 1) / (1 + np.exp(a))).clip(lower=0).rolling(n(42)).mean()

    def sweep_then_break():
        swept = l < l.rolling(n(20)).min().shift()           # took the lows (stops)
        return swept.rolling(n(12)).max().fillna(0).astype(bool)

    def failed_breakouts():
        brk = c > h.rolling(50).max().shift()
        fail = brk.shift(6).fillna(False) & (c < h.shift(6))  # broke out 6 bars ago, now back below that bar's high
        return fail.rolling(n(60)).sum() < 2

    def base_compression():
        rng = h.rolling(n(30)).max() / l.rolling(n(30)).min() - 1
        return rng.shift(1) < rng.rolling(540, min_periods=180).quantile(0.25).shift(1)

    def absorption():
        small = (h - l) < 0.7 * I.atr(lab, 14)
        loud = v > 1.5 * v.rolling(20).mean().shift()
        return (small & loud).rolling(n(30)).sum() >= 3

    def spring():
        low60 = l.rolling(n(60)).min().shift()
        sp = (l < low60) & (c > low60)
        return sp.rolling(n(30)).max().fillna(0).astype(bool)

    def obv_leads():
        ob = (np.sign(lab.ret.fillna(0)) * v).cumsum()
        return ob >= ob.rolling(n(50)).max().shift()

    return {
        "low_spread_cs": ("entry", lambda: cs_spread() < cs_spread().rolling(540, min_periods=180).median()),
        "liquid_now_amihud": ("entry", lambda: (lambda il: il < il.rolling(540, min_periods=180).median())((lab.ret.abs() / qv).rolling(n(42)).mean())),
        "liquidity_sweep_then_breakout": ("entry", lambda: sweep_then_break()),
        "base_compression_before_break": ("entry", lambda: base_compression()),
        "daily_adx_trend_20": ("entry", lambda: I.daily(lab, f"dadx{p}", lambda d: (lambda hh, ll, cc: (lambda up, dn: (lambda atr: I.rma(100 * ((I.rma(up.where((up > dn) & (up > 0), 0.0), 14) - I.rma(dn.where((dn > up) & (dn > 0), 0.0), 14)).abs() / (I.rma(up.where((up > dn) & (up > 0), 0.0), 14) + I.rma(dn.where((dn > up) & (dn > 0), 0.0), 14))), 14))(None))(hh.diff(), -ll.diff()))(d["high"], d["low"], d["close"]) > 20 * p) > 0.5),
        "no_recent_failed_breakouts": ("entry", lambda: failed_breakouts()),
        "cvd_accumulation_divergence": ("entry", lambda: (cvd - cvd.shift(n(30)) > 0) & (c / c.shift(n(30)) - 1 < 0.05)),
        "absorption_recent": ("entry", lambda: absorption()),
        "persistent_taker_buying": ("entry", lambda: (tb.rolling(n(30)).sum() / qv.rolling(n(30)).sum()) > 0.5),
        "obv_new_high_first": ("entry", lambda: obv_leads()),
        "wyckoff_spring_recent": ("entry", lambda: spring()),
        "exit_liquidity_grab_rejection": ("exit", lambda: (h > h.rolling(n(50)).max().shift()) & ((h - np.maximum(c, o)) > 0.6 * (h - l))),
        "exit_distribution_cvd_down_at_highs": ("exit", lambda: (c >= 0.97 * h.rolling(n(50)).max()) & (cvd < cvd.shift(n(30)))),
    }


def main():
    lab = Lab.load()
    I3.install(lab)
    main_c, nb_c = new_candidates(lab, 1.0), new_candidates(lab, 1.3)
    old = I.candidates
    I.candidates = lambda l: {**old(l), **main_c, **{k + "__nb": v for k, v in nb_c.items()}}
    BE, BX = I3.BASE_E, I3.BASE_X
    fn = lambda l, n, cal, entries=(), exits=(): I.weights(l, n, cal, BE + tuple(entries), BX + tuple(exits))
    print("leak:", lab.leak_check(fn, {"entries": [k for k, (t, _) in main_c.items() if t == "entry"],
                                       "exits": [k for k, (t, _) in main_c.items() if t == "exit"]}, dates=("2021-05-19 04:00",)), flush=True)
    base = lab.evaluate(fn, "ind4-base", {}, log=False)
    b = base.train_sharpe.median()
    bA, bB = I3.hold_sharpe(lab, BE, BX, (25, 45), *I3.HOLD), I3.hold_sharpe(lab, BE, BX, (45, 65), *I3.HOLD)
    print(f"baseline: train {show(base)} | A {bA[0]:+.2f} ({bA[1]:+.0%}) | B {bB[0]:+.2f} ({bB[1]:+.0%})", flush=True)
    rows = []
    for name, (kind, _) in main_c.items():
        e, x = ((name,), ()) if kind == "entry" else ((), (name,))
        s = summary(lab.evaluate(fn, "indicators4", {"entries": list(e), "exits": list(x)}), base)
        A = I3.hold_sharpe(lab, BE + e, BX + x, (25, 45), *I3.HOLD)
        B = I3.hold_sharpe(lab, BE + e, BX + x, (45, 65), *I3.HOLD)
        e2, x2 = ((name + "__nb",), ()) if kind == "entry" else ((), (name + "__nb",))
        plateau = summary(lab.evaluate(fn, "indicators4-nb", {"entries": list(e2), "exits": list(x2)}), base).sharpe > b
        ok = s.sharpe > b and s.beats_base >= 6 / 9 and A[0] > bA[0] and B[0] > bB[0] and plateau
        rows.append({"name": name, **s.to_dict(), "A": A[0], "B": B[0], "plateau": plateau, "pass_all": ok})
        print(f"{name:38s} train Sh {s.sharpe:.2f} CAGR {s.cagr:+.0%} DD {s.maxdd:.0%} beats {s.beats_base:.0%} | A {A[0]:+.2f} | B {B[0]:+.2f}"
              f" | plateau {'ok' if plateau else 'NO'}{'  <== PASS ALL' if ok else ''}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("data/methods/indicators4.csv", index=False)
    winners = list(t[t.pass_all].sort_values("sharpe", ascending=False).name)
    print("\npassed all tests:", winners, flush=True)
    chosen, best = [], b
    for name in winners:
        trial = chosen + [name]
        e = [k for k in trial if main_c[k][0] == "entry"]
        x = [k for k in trial if main_c[k][0] == "exit"]
        s = summary(lab.evaluate(fn, "indicators4-combo", {"entries": e, "exits": x}), base)
        print(f"  + {name:36s} train Sh {s.sharpe:.2f} CAGR {s.cagr:+.0%} DD {s.maxdd:.0%}", flush=True)
        if s.sharpe > best + 0.02:
            chosen, best = trial, s.sharpe
    if chosen:
        e = [k for k in chosen if main_c[k][0] == "entry"]
        x = [k for k in chosen if main_c[k][0] == "exit"]
        A = I3.hold_sharpe(lab, BE + tuple(e), BX + tuple(x), (25, 45), *I3.HOLD)
        B = I3.hold_sharpe(lab, BE + tuple(e), BX + tuple(x), (45, 65), *I3.HOLD)
        print(f"COMBINATION {chosen}: train {best:.2f} | A {A[0]:+.2f} vs {bA[0]:+.2f} | B {B[0]:+.2f} vs {bB[0]:+.2f}")
    json.dump({"chosen": chosen}, open("data/methods/indicators4_choice.json", "w"))
    print("\nCHOSEN:", chosen or "nothing - keep the current best bot")


if __name__ == "__main__":
    main()
