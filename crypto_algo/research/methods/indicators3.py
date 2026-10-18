"""Third indicator batch, with the last year locked away.

Baseline = current best bot: aggressive (k=6) + volume profile (VAH entry / POC exit) + ATR(5)>ATR(20) + breakout volume
> 2x average + CMF<-0.1 exit.

LOCK: nothing on or after 2025-10-01 is used to choose anything here. Declared before running:
  1. train 2018-11..2022 (bot universe): higher median Sharpe, >= 6 of 9 variants better;
  2. holdout A: coins ranked 26-45, 2023-01..2025-09: higher Sharpe;
  3. holdout B: coins ranked 46-65, 2023-01..2025-09: higher Sharpe;
  4. plateau: the same idea with a neighbouring parameter must also beat the baseline on train.
Passing ideas are added greedily on train only. Then, ONCE (`final`), the chosen model and the baseline are run on
2025-10-01..now (bot universe, 3 calendars + both holdouts). Adopt only if it beats the baseline there too.

    python -m research.methods.indicators3          # selection (locked year untouched)
    python -m research.methods.indicators3 final    # the one-time test on the locked year
"""
import json
import sys

import numpy as np
import pandas as pd

from research.dynamic_universe import UNI
from research.harness import Lab, metrics, show, summary
from research.methods import indicators as I
from research.methods.profile_variants import profile_full

LOCK = pd.Timestamp("2025-10-01", tz="UTC")
HOLD = (pd.Timestamp("2023-01-01", tz="UTC"), LOCK)
BASE_E, BASE_X = ("vol_expanding", "relvol_2.0x"), ("exit_cmf_neg",)


def flow(lab):
    def go():
        qv, tb, tr = {}, {}, {}
        for a in lab.assets:
            f = UNI / "flow" / f"{a.split('#')[0]}_4h.csv"
            if f.exists():
                d = pd.read_csv(f, index_col=0)
                d.index = pd.to_datetime(d.index, utc=True)
                qv[a], tb[a], tr[a] = d.quote_volume, d.tb_quote, d.trades
        mk = lambda x: pd.DataFrame(x).reindex(index=lab.index, columns=lab.assets).where(lab.live)
        return mk(qv), mk(tb), mk(tr)
    return lab.memo(("flow3",), go)


def heikin_daily_green(lab):
    def go():
        from algo.htf import align, resample
        d = resample(lab.bars, "1d")
        hc = (d["open"] + d["high"] + d["low"] + d["close"]) / 4
        ho = hc.copy() * np.nan
        o, c = d["open"].to_numpy(), hc.to_numpy()
        out = np.full(c.shape, np.nan)
        for j in range(c.shape[1]):
            prev = np.nan
            for t in range(c.shape[0]):
                if np.isnan(c[t, j]):
                    continue
                prev = o[t, j] if np.isnan(prev) else (prev + out_c) / 2
                out[t, j] = prev
                out_c = c[t, j]
        ho = pd.DataFrame(out, index=hc.index, columns=hc.columns)
        return align((hc > ho).astype(float), lab.index, "1d") > 0.5
    return lab.memo(("ha_daily",), go)


def new_candidates(lab: Lab) -> dict:
    c, h, l = lab.close, lab.high, lab.low
    v = lab.volume.astype(float)
    a = lambda n: I.atr(lab, n)

    def vwma(n):
        return (c * v).rolling(n).sum() / v.rolling(n).sum()

    def squeeze_release():
        sd = c.rolling(20).std()
        inside = (2 * sd) < (1.5 * a(20))           # Bollinger inside Keltner
        return inside.shift(1).rolling(12).max().fillna(0).astype(bool) & ~inside

    def anchored_vwap():
        low_pos = l.rolling(540, min_periods=100).apply(np.argmin, raw=True)
        tp = (h + l + c) / 3
        pv, vv = (tp * v).cumsum(), v.cumsum()
        k = (540 - 1 - low_pos).fillna(0).astype(int).clip(lower=0)
        idx = np.arange(len(c))[:, None] - k.to_numpy()
        idx = np.clip(idx, 0, len(c) - 1)
        pva, vva = pv.to_numpy(), vv.to_numpy()
        cols = np.arange(c.shape[1])[None, :]
        num = pva - pva[idx, cols]
        den = vva - vva[idx, cols]
        return c > pd.DataFrame(num / np.where(den > 0, den, np.nan), index=c.index, columns=c.columns)

    return {
        "squeeze_release": ("entry", lambda: lab.memo(("sq",), squeeze_release)),
        "daily_close_at_20d_high": ("entry", lambda: (lambda d: I.align_daily(lab, d["close"] >= d["high"].rolling(20).max().shift()))(I_resample(lab))),
        "heikin_ashi_daily_green": ("entry", lambda: heikin_daily_green(lab)),
        "vwma20_above_vwma50": ("entry", lambda: vwma(20) > vwma(50)),
        "range_expansion_bar": ("entry", lambda: (h - l) > 1.5 * a(14)),
        "above_anchored_vwap_90d_low": ("entry", lambda: lab.memo(("avwap",), anchored_vwap)),
        "taker_buy_3bar_0.52": ("entry", lambda: (lambda q, t, _: (t.rolling(3).sum() / q.rolling(3).sum()) > 0.52)(*flow(lab))),
        "big_trade_size": ("entry", lambda: (lambda q, t, n: (q / n) > 1.5 * (q / n).rolling(90).mean().shift())(*flow(lab))),
        "breadth_universe_above_ema50": ("entry", lambda: pd.DataFrame({k: ((c > I.ema(c, 50)).where(lab.live).mean(axis=1) > 0.5) for k in c.columns})),
        "exit_heikin_ashi_daily_red": ("exit", lambda: ~heikin_daily_green(lab)),
        "exit_taker_sell_6bar": ("exit", lambda: (lambda q, t, _: (t.rolling(6).sum() / q.rolling(6).sum()) < 0.45)(*flow(lab))),
        "exit_below_anchored_vwap": ("exit", lambda: ~lab.memo(("avwap",), anchored_vwap)),
        "exit_daily_bearish_engulfing": ("exit", lambda: (lambda d: I.align_daily(lab, (d["close"] < d["open"]) & (d["open"] >= d["close"].shift()) & (d["close"] <= d["open"].shift()) & (d["close"].shift() > d["open"].shift())))(I_resample(lab))),
    }


# neighbouring parameters for the plateau check
NEIGHBOURS = {
    "taker_buy_3bar_0.52": ("entry", lambda lab: (lambda q, t, _: (t.rolling(3).sum() / q.rolling(3).sum()) > 0.51)(*flow(lab))),
    "big_trade_size": ("entry", lambda lab: (lambda q, t, n: (q / n) > 1.3 * (q / n).rolling(90).mean().shift())(*flow(lab))),
    "range_expansion_bar": ("entry", lambda lab: (lab.high - lab.low) > 1.3 * I.atr(lab, 14)),
    "vwma20_above_vwma50": ("entry", lambda lab: (lambda c, v: (c * v).rolling(30).sum() / v.rolling(30).sum() > (c * v).rolling(60).sum() / v.rolling(60).sum())(lab.close, lab.volume.astype(float))),
    "exit_taker_sell_6bar": ("exit", lambda lab: (lambda q, t, _: (t.rolling(6).sum() / q.rolling(6).sum()) < 0.44)(*flow(lab))),
}


def I_resample(lab):
    from algo.htf import resample
    return lab.memo(("daily_panel",), lambda: resample(lab.bars, "1d"))


def _align_daily(lab, x):
    from algo.htf import align
    return align(x.astype(float), lab.index, "1d") > 0.5


I.align_daily = _align_daily


def install(lab):
    old = I.candidates
    v = lab.volume.astype(float)
    extra = {"relvol_2.0x": ("entry", lambda: v > 2.0 * v.rolling(20).mean().shift())}
    extra.update(new_candidates(lab))
    extra.update({f"{k}__nb": (kind, (lambda f=f: f(lab))) for k, (kind, f) in NEIGHBOURS.items()})
    I.candidates = lambda l: {**old(l), **extra}
    return extra


def hold_sharpe(lab, e, x, ranks, lo, hi):
    sh, cg = [], []
    for cal in ("Q-jan", "Q-feb", "Q-mar"):
        m = lab.memo(("hold", ranks, cal), lambda: I.holdout(lab, *ranks, cal))
        w = lab.enforce(lab.build(20, cal, k=6, mask=m, signal=I.signal(lab, tuple(e), tuple(x))), m)
        mt = metrics(lab.pnl(w), lo, hi)
        sh.append(mt["sharpe"]); cg.append(mt["cagr"])
    return float(np.median(sh)), float(np.median(cg))


def main():
    lab = Lab.load()
    extra = install(lab)
    new = new_candidates(lab)
    fn = lambda l, n, cal, entries=(), exits=(): I.weights(l, n, cal, BASE_E + tuple(entries), BASE_X + tuple(exits))
    print("leak:", lab.leak_check(fn, {"entries": [k for k, (t, _) in new.items() if t == "entry"],
                                       "exits": [k for k, (t, _) in new.items() if t == "exit"]}, dates=("2021-05-19 04:00",)), flush=True)
    base = lab.evaluate(fn, "ind3-base", {}, log=False)
    b = base.train_sharpe.median()
    bA, bB = hold_sharpe(lab, BASE_E, BASE_X, (25, 45), *HOLD), hold_sharpe(lab, BASE_E, BASE_X, (45, 65), *HOLD)
    print(f"baseline (best bot): train {show(base)} | A {bA[0]:+.2f} ({bA[1]:+.0%}) | B {bB[0]:+.2f} ({bB[1]:+.0%})   [holdouts end 2025-09-30]", flush=True)
    rows = []
    for name, (kind, _) in new.items():
        e, x = ((name,), ()) if kind == "entry" else ((), (name,))
        s = summary(lab.evaluate(fn, "indicators3", {"entries": list(e), "exits": list(x)}), base)
        A = hold_sharpe(lab, BASE_E + e, BASE_X + x, (25, 45), *HOLD)
        B = hold_sharpe(lab, BASE_E + e, BASE_X + x, (45, 65), *HOLD)
        plateau = True
        if name in NEIGHBOURS:
            nb = name + "__nb"
            e2, x2 = ((nb,), ()) if kind == "entry" else ((), (nb,))
            plateau = summary(lab.evaluate(fn, "indicators3-nb", {"entries": list(e2), "exits": list(x2)}), base).sharpe > b
        ok = s.sharpe > b and s.beats_base >= 6 / 9 and A[0] > bA[0] and B[0] > bB[0] and plateau
        rows.append({"name": name, "kind": kind, **s.to_dict(), "A": A[0], "B": B[0], "plateau": plateau, "pass_all": ok})
        print(f"{name:30s} train Sh {s.sharpe:.2f} CAGR {s.cagr:+.0%} DD {s.maxdd:.0%} beats {s.beats_base:.0%} | A {A[0]:+.2f} | B {B[0]:+.2f}"
              f" | plateau {'ok' if plateau else 'NO'}{'  <== PASS ALL' if ok else ''}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("data/methods/indicators3.csv", index=False)
    winners = list(t[t.pass_all].sort_values("sharpe", ascending=False).name)
    print("\npassed all tests:", winners, flush=True)
    chosen, best = [], b
    for name in winners:
        trial = chosen + [name]
        e = [n for n in trial if new[n][0] == "entry"]
        x = [n for n in trial if new[n][0] == "exit"]
        s = summary(lab.evaluate(fn, "indicators3-combo", {"entries": e, "exits": x}), base)
        print(f"  + {name:28s} train Sh {s.sharpe:.2f} CAGR {s.cagr:+.0%} DD {s.maxdd:.0%}", flush=True)
        if s.sharpe > best + 0.02:
            chosen, best = trial, s.sharpe
    e = [n for n in chosen if new[n][0] == "entry"]
    x = [n for n in chosen if new[n][0] == "exit"]
    json.dump({"entries": e, "exits": x}, open("data/methods/indicators3_choice.json", "w"))
    print("\nCHOSEN (frozen before the locked year):", chosen or "nothing - keep the current best bot")


def final():
    lab = Lab.load()
    install(lab)
    ch = json.load(open("data/methods/indicators3_choice.json"))
    lo, hi = LOCK, pd.Timestamp("2100-01-01", tz="UTC")
    models = {"current best": (BASE_E, BASE_X), "new": (BASE_E + tuple(ch["entries"]), BASE_X + tuple(ch["exits"]))}
    for name, (e, x) in models.items():
        res = []
        for cal in ("Q-jan", "Q-feb", "Q-mar"):
            w = lab.enforce(lab.build(15, cal, k=6, signal=I.signal(lab, e, x)))
            res.append(metrics(lab.pnl(w), lo, hi))
        r = pd.DataFrame(res)
        A, B = hold_sharpe(lab, e, x, (25, 45), lo, hi), hold_sharpe(lab, e, x, (45, 65), lo, hi)
        days = (lab.index[-1] - lo).days
        tot = [(1 + m["cagr"]) ** (days / 365) - 1 for m in res]
        print(f"{name:14s} LOCKED YEAR {lo.date()}..{lab.index[-1].date()}: return {[f'{v:+.0%}' for v in tot]} | maxDD {[f'{v:.0%}' for v in r.maxdd]}"
              f" | Sharpe median {r.sharpe.median():.2f} | holdout A {A[0]:+.2f} B {B[0]:+.2f}")


if __name__ == "__main__":
    final() if sys.argv[1:] == ["final"] else main()
