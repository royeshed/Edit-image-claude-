"""Fifth batch: getting in before the breakout.

Part 1 - staged entry: a starter position (1/3 of a full slot) on an accumulation sign while the daily trend is up;
the bot's normal breakout (all current filters) takes it to a full position. The starter is dropped on a break of
the 10-bar low, a failed daily trend, or if no breakout comes within `timeout` bars.
Part 2 - futures open interest (positioning) as a starter signal, a breakout filter and an exit.

Same protocol as indicators3/4, declared before running (nothing on/after 2025-10-01 chooses anything):
train 2018-11..2022 (>= 6/9 better, higher median Sharpe), holdout A (coins 26-45) and B (coins 46-65) on
2023-01..2025-09 both better, and the neighbouring parameter (all lengths x1.3) must also beat the baseline on train.
Open interest exists only from 2020-09 (BTC) / 2021-12 (most coins), so open-interest ideas are judged on train
2022 instead of 2018-11..2022 (where the data is missing they are neutral: a starter never fires, a filter passes).

    python -m research.methods.indicators5
"""
import json

import numpy as np
import pandas as pd

from research.dynamic_universe import UNI
from research.harness import TRAIN, VARIANTS, Lab, metrics
from research.methods import indicators as I
from research.methods import indicators3 as I3
from research.methods.profile_variants import profile_full

FRAC = 1 / 3
TIMEOUT = 60                      # 10 days of 4h bars
TRAIN_OI = (pd.Timestamp("2022-01-01", tz="UTC"), TRAIN[1])


def oi_panel(lab):
    def go():
        d = {}
        for a in lab.assets:
            f = UNI / "oi" / f"{a.split('#')[0]}_4h.csv"
            if f.exists():
                x = pd.read_csv(f, index_col=0)
                if len(x):
                    x.index = pd.to_datetime(x.index, utc=True)
                    d[a] = x.oi
        return pd.DataFrame(d).reindex(index=lab.index, columns=lab.assets).where(lab.live)
    return lab.memo(("oi",), go)


def starters(lab, p=1.0) -> dict:
    """Accumulation signs (bool, trailing). p scales every length (plateau check)."""
    c, h, l = lab.close, lab.high, lab.low
    v = lab.volume.astype(float)
    n = lambda x: max(2, int(round(x * p)))
    flat = lambda k: (c / c.shift(n(k)) - 1).abs() < 0.05

    def flow():
        qv, tb, _ = I3.flow(lab)
        return qv, tb, ((2 * tb / qv - 1) * qv).fillna(0).cumsum()

    def oi_build():
        oi = oi_panel(lab)
        return (oi / oi.shift(n(30)) - 1 > 0.10) & flat(30)

    def spring():
        low = l.rolling(n(60)).min().shift()
        return ((l < low) & (c > low)).rolling(n(30)).max().fillna(0).astype(bool)

    def sweep_reclaim():
        low = l.rolling(n(20)).min().shift()
        return (l < low).rolling(n(12)).max().fillna(0).astype(bool) & (c > l.rolling(n(20)).min().shift(n(12)))

    def compression():
        hi, lo = h.rolling(n(30)).max(), l.rolling(n(30)).min()
        rng = hi / lo - 1
        tight = rng < rng.rolling(540, min_periods=180).quantile(0.25)
        return tight & (c > (hi + lo) / 2)

    def obv_leads():
        ob = (np.sign(lab.ret.fillna(0)) * v).cumsum()
        return (ob >= ob.rolling(n(50)).max().shift()) & (c < h.rolling(n(50)).max().shift())

    return {
        "start_cvd_accumulation": lambda: (lambda q, t, cvd: (cvd > cvd.shift(n(30))) & flat(30))(*flow()),
        # note: "taker buy share > 50% over 30 bars" is the same condition as CVD rising over 30 bars (identical results)
        "start_taker_buying_flat": lambda: (lambda q, t, _: (t.rolling(n(30)).sum() / q.rolling(n(30)).sum() > 0.5) & flat(30))(*flow()),
        "start_wyckoff_spring": spring,
        "start_sweep_reclaim": sweep_reclaim,
        "start_compression_upper_half": compression,
        "start_obv_leads_price": obv_leads,
        "start_oi_build_flat_price": oi_build,
    }


def oi_filters(lab, p=1.0) -> dict:
    """Open interest as a breakout filter / exit on top of the best bot (missing data = neutral)."""
    n = lambda x: max(2, int(round(x * p)))
    oi = lambda: oi_panel(lab)
    return {
        "oi_rising_6bar": ("entry", lambda: (oi() > oi().shift(n(6))) | oi().isna()),
        "oi_not_crowded": ("entry", lambda: (lambda x: ~(x / x.shift(n(30)) - 1 > 0.5))(oi())),
        "exit_oi_flush_15pct": ("exit", lambda: oi() / oi().rolling(n(6)).max() - 1 < -0.15),
    }


def staged(lab, start: pd.DataFrame, full: pd.DataFrame, frac=FRAC, timeout=TIMEOUT):
    """max(full signal, frac x starter). The starter opens only while flat and the daily trend is up, and ends at the
    breakout hand-off, a close under the 10-bar low, a failed daily trend, or after `timeout` bars."""
    g = lab.gate(("1d",), "close>ema50").to_numpy()
    _, x10 = lab.donchian(50, 10)
    st, f, x = start.fillna(False).to_numpy(bool), full.to_numpy() > 0, x10.to_numpy(bool)
    out = np.zeros(st.shape)
    age = np.full(st.shape[1], -1)
    for t in range(st.shape[0]):                     # vectorised over coins
        on = age >= 0
        age = np.where(on, age + 1, age)
        age = np.where(on & (f[t] | x[t] | ~g[t] | (age > timeout)), -1, age)
        age = np.where((age < 0) & st[t] & g[t] & ~f[t] & ~x[t], 0, age)
        out[t] = np.where(age >= 0, frac, 0.0)
    pre = pd.DataFrame(out, index=full.index, columns=full.columns)
    return np.maximum(full, pre).where(lab.live, 0.0)


def base_signal(lab):
    return I.signal(lab, I3.BASE_E, I3.BASE_X)


def run(lab, sig_fn, lo, hi):
    """Median train Sharpe/CAGR/DD over the 9 variants + per-variant Sharpe."""
    rows = []
    for nn, cal in VARIANTS:
        w = lab.enforce(lab.build(nn, cal, k=6, signal=sig_fn(lab)))
        rows.append(metrics(lab.pnl(w), lo, hi))
    return pd.DataFrame(rows)


def hold(lab, sig_fn, ranks):
    lo, hi = I3.HOLD
    sh = []
    for cal in ("Q-jan", "Q-feb", "Q-mar"):
        m = lab.memo(("hold", ranks, cal), lambda: I.holdout(lab, *ranks, cal))
        w = lab.enforce(lab.build(20, cal, k=6, mask=m, signal=sig_fn(lab)), m)
        sh.append(metrics(lab.pnl(w), lo, hi)["sharpe"])
    return float(np.median(sh))


def main():
    lab = Lab.load()
    filt_main = oi_filters(lab, 1.0)
    orig = I.candidates

    def cands(l):                   # built from the lab passed in, so the look-ahead check sees truncated data
        v = l.volume.astype(float)
        return {**orig(l), "relvol_2.0x": ("entry", lambda: v > 2.0 * v.rolling(20).mean().shift()),
                **oi_filters(l, 1.0), **{k + "__nb": f for k, f in oi_filters(l, 1.3).items()}}
    I.candidates = cands

    def sig_start(name, p=1.0, frac=FRAC):
        def f(l):
            return l.memo(("staged", name, p, frac), lambda: staged(l, l.memo(("start", name, p), starters(l, p)[name]), base_signal(l), frac))
        return f

    def sig_filter(name, kind):
        e, x = ((name,), ()) if kind == "entry" else ((), (name,))
        return lambda l: I.signal(l, I3.BASE_E + e, I3.BASE_X + x)

    # look-ahead check on the new machinery
    lk = lambda l, nn, cal, name: l.build(nn, cal, k=6, signal=sig_start(name)(l))
    worst = max(lab.leak_check(lk, {"name": nm}, dates=("2022-05-19 04:00",)) for nm in ("start_cvd_accumulation", "start_oi_build_flat_price"))
    worst = max(worst, lab.leak_check(lambda l, nn, cal: l.build(nn, cal, k=6, signal=sig_filter("oi_rising_6bar", "entry")(l)), dates=("2022-05-19 04:00",)))
    print("leak:", worst, flush=True)

    bt = {"full": run(lab, base_signal, *TRAIN), "2022": run(lab, base_signal, *TRAIN_OI)}
    bA, bB = hold(lab, base_signal, (25, 45)), hold(lab, base_signal, (45, 65))
    print(f"baseline: train Sh {bt['full'].sharpe.median():.2f} CAGR {bt['full'].cagr.median():+.0%} DD {bt['full'].maxdd.median():.0%}"
          f" | train-2022 Sh {bt['2022'].sharpe.median():.2f} | A {bA:+.2f} | B {bB:+.2f}", flush=True)

    tests = [(nm, "start", sig_start(nm), sig_start(nm, 1.3)) for nm in starters(lab)]
    tests += [(nm, kind, sig_filter(nm, kind), sig_filter(nm + "__nb", kind)) for nm, (kind, _) in filt_main.items()]
    rows = []
    for name, kind, fn, fn_nb in tests:
        per = "2022" if "oi_" in name else "full"
        lo_hi = TRAIN_OI if per == "2022" else TRAIN
        d = run(lab, fn, *lo_hi)
        beats = (d.sharpe.to_numpy() > bt[per].sharpe.to_numpy()).mean()
        sh, b = d.sharpe.median(), bt[per].sharpe.median()
        A, B = hold(lab, fn, (25, 45)), hold(lab, fn, (45, 65))
        plateau = run(lab, fn_nb, *lo_hi).sharpe.median() > b
        ok = sh > b and beats >= 6 / 9 and A > bA and B > bB and plateau
        extra = ""
        if kind == "start":
            h = run(lab, sig_start(name, 1.0, 0.5), *lo_hi).sharpe.median()
            extra = f" | starter 1/2: {h:.2f}"
        rows.append({"name": name, "kind": kind, "train_period": per, "sharpe": sh, "base": b, "cagr": d.cagr.median(), "maxdd": d.maxdd.median(),
                     "beats": beats, "A": A, "B": B, "plateau": plateau, "pass_all": ok})
        print(f"{name:30s} [{per}] train Sh {sh:.2f} (base {b:.2f}) CAGR {d.cagr.median():+.0%} DD {d.maxdd.median():.0%} beats {beats:.0%}"
              f" | A {A:+.2f} | B {B:+.2f} | plateau {'ok' if plateau else 'NO'}{extra}{'  <== PASS ALL' if ok else ''}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("data/methods/indicators5.csv", index=False)
    winners = list(t[t.pass_all].name)
    json.dump({"passed": winners}, open("data/methods/indicators5_choice.json", "w"))
    print("\npassed all tests:", winners or "nothing - keep the current best bot")


if __name__ == "__main__":
    main()
