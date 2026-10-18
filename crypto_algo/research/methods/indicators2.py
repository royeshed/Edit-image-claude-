"""Second indicator batch on top of the current best bot:
aggressive (k=6) + volume profile (VAH entry, POC exit) + ATR(5)>ATR(20) entries + CMF<-0.1 exits.

Same three tests as research/methods/indicators.py, declared before running: train 2018-11..2022 (higher median
Sharpe, >= 6 of 9 variants better), holdout A (coins 26-45, 2023-2026) and holdout B (coins 46-65, 2023-2026) both
better. Passing indicators are added greedily on train only and the result re-checked on A and B.

    python -m research.methods.indicators2
"""
import json

import numpy as np
import pandas as pd

from research.harness import Lab, show, summary
from research.methods import indicators as I
from research.methods.profile_variants import profile_full

BASE_E, BASE_X = ("vol_expanding",), ("exit_cmf_neg",)


def new_candidates(lab: Lab) -> dict:
    c, h, l, o = lab.close, lab.high, lab.low, lab.open.astype(float)
    v = lab.volume.astype(float)
    a14 = lambda: I.atr(lab, 14)
    er = lambda n: (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()
    btc = c["BTCUSDT"]
    poc = lambda: profile_full(lab, 180, 24)[0]

    def r2(n=50):
        def go():
            y = np.log(c)
            t = pd.Series(np.arange(len(c), dtype=float), index=c.index)
            mt, my = t.rolling(n).mean(), y.rolling(n).mean()
            cov = (y.mul(t, axis=0)).rolling(n).mean() - my.mul(mt, axis=0)
            vt = (t * t).rolling(n).mean() - mt * mt
            vy = (y * y).rolling(n).mean() - my * my
            return (cov.pow(2).div(vt, axis=0)) / vy
        return lab.memo(("r2", n), go)

    return {
        "strong_close_top25": ("entry", lambda: (c - l) / (h - l).replace(0, np.nan) > 0.75),
        "momentum_candle_1atr": ("entry", lambda: (c - o) > a14()),
        "channel_widening": ("entry", lambda: (h.rolling(50).max() - l.rolling(50).min()) > (h.rolling(50).max() - l.rolling(50).min()).shift(6)),
        "relvol_2x": ("entry", lambda: v > 2 * v.rolling(20).mean().shift()),
        "efficiency_ratio_0.3": ("entry", lambda: er(20) > 0.3),
        "trend_r2_0.5": ("entry", lambda: r2() > 0.5),
        "coin_vs_btc_uptrend": ("entry", lambda: (lambda rr: rr > I.ema(rr, 50 * 6))(c.div(btc, axis=0)) | (c.columns == "BTCUSDT")),
        "not_overextended_3atr": ("entry", lambda: (c - I.ema(c, 200)) < 3 * a14() * 4),
        "ema_ribbon_aligned": ("entry", lambda: (I.ema(c, 20) > I.ema(c, 50)) & (I.ema(c, 50) > I.ema(c, 100))),
        "poc_rising": ("entry", lambda: poc() > poc().shift(6 * 7)),
        "higher_lows_4h": ("entry", lambda: l.rolling(20).min() > l.rolling(20).min().shift(20)),
        "low_btc_correlation": ("entry", lambda: c.pct_change(fill_method=None).rolling(180).corr(btc.pct_change(fill_method=None)) < 0.7),
        "btc_above_ema50_daily": ("entry", lambda: pd.DataFrame({k: lab.gate(("1d",), "close>ema50")["BTCUSDT"] for k in c.columns})),
        "exit_volume_climax_wick": ("exit", lambda: (v > 4 * v.rolling(20).mean().shift()) & ((h - np.maximum(c, o)) > 0.5 * (h - l))),
        "exit_efficiency_collapse": ("exit", lambda: er(20) < 0.05),
        "exit_poc_falling": ("exit", lambda: poc() < poc().shift(6 * 7)),
        "exit_close_below_ema20_and_cmf_neg": ("exit", lambda: (c < I.ema(c, 20)) & (I.cmf(lab) < 0)),
        "exit_btc_daily_below_ema50": ("exit", lambda: pd.DataFrame({k: ~lab.gate(("1d",), "close>ema50")["BTCUSDT"] for k in c.columns})),
    }


def main():
    lab = Lab.load()
    old = I.candidates
    I.candidates = lambda lab_: {**old(lab_), **new_candidates(lab_)}
    cand = I.candidates(lab)
    new = new_candidates(lab)
    fn = lambda lab_, n, cal, entries=(), exits=(): I.weights(lab_, n, cal, tuple(BASE_E) + tuple(entries), tuple(BASE_X) + tuple(exits))
    print("leak:", lab.leak_check(fn, {"entries": [k for k, (t, _) in new.items() if t == "entry"], "exits": [k for k, (t, _) in new.items() if t == "exit"]},
                                  dates=("2021-05-19 04:00",)), flush=True)
    base = lab.evaluate(fn, "ind2-base", {}, log=False)
    hs = lambda e=(), x=(), ranks=(25, 45): I.holdout_sharpe(lab, tuple(BASE_E) + tuple(e), tuple(BASE_X) + tuple(x), ranks)
    bA, bB = hs(), hs(ranks=(45, 65))
    b = base.train_sharpe.median()
    print(f"baseline (best bot): train {show(base)} | A {bA[0]:+.2f} ({bA[1]:+.0%}) | B {bB[0]:+.2f} ({bB[1]:+.0%})", flush=True)
    rows = []
    for name, (kind, _) in new.items():
        e, x = ((name,), ()) if kind == "entry" else ((), (name,))
        d = lab.evaluate(fn, "indicators2", {"entries": list(e), "exits": list(x)})
        s = summary(d, base)
        A, B = hs(e, x), hs(e, x, (45, 65))
        ok = s.sharpe > b and s.beats_base >= 6 / 9 and A[0] > bA[0] and B[0] > bB[0]
        rows.append({"name": name, "kind": kind, **s.to_dict(), "A": A[0], "B": B[0], "pass_all": ok})
        print(f"{name:36s} train Sh {s.sharpe:.2f} CAGR {s.cagr:+.0%} DD {s.maxdd:.0%} beats {s.beats_base:.0%} | A {A[0]:+.2f} | B {B[0]:+.2f}"
              f"{'  <== PASS ALL' if ok else ''}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("data/methods/indicators2.csv", index=False)
    winners = list(t[t.pass_all].sort_values("sharpe", ascending=False).name)
    print("\npassed all three tests:", winners, flush=True)
    chosen, best = [], b
    for name in winners:
        trial = chosen + [name]
        e = [n for n in trial if new[n][0] == "entry"]
        x = [n for n in trial if new[n][0] == "exit"]
        s = summary(lab.evaluate(fn, "indicators2-combo", {"entries": e, "exits": x}), base)
        print(f"  + {name:34s} train Sh {s.sharpe:.2f} CAGR {s.cagr:+.0%} DD {s.maxdd:.0%}", flush=True)
        if s.sharpe > best + 0.02:
            chosen, best = trial, s.sharpe
    if chosen:
        e = [n for n in chosen if new[n][0] == "entry"]
        x = [n for n in chosen if new[n][0] == "exit"]
        A, B = hs(e, x), hs(e, x, (45, 65))
        print(f"\nCOMBINATION {chosen}: train Sharpe {best:.2f} | A {A[0]:+.2f} ({A[1]:+.0%}) vs {bA[0]:+.2f} | B {B[0]:+.2f} ({B[1]:+.0%}) vs {bB[0]:+.2f}")
        json.dump({"entries": e, "exits": x}, open("data/methods/indicators2_combo.json", "w"))
    else:
        print("\nno combination beats the current best bot")


if __name__ == "__main__":
    main()
