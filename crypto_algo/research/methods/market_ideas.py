"""Eighth batch: ideas borrowed from well-known trading systems, on top of the best bot.

Baseline = the best bot (aggressive k=6, volume profile VAH entry / POC exit, ATR(5)>ATR(20), breakout volume > 2x the
20-bar average, CMF<-0.1 exit, Donchian 50/100/200 x 10/20 under the daily close > EMA50 gate).

LOCK: the data itself is cut at 2025-10-01 (Lab.truncate), so nothing from the locked year can be read here at all.
Same protocol as indicators3-5 / structure / geometry, declared before running:
  1. train 2018-11..2022 (bot universe): higher median Sharpe and >= 6 of 9 variants better;
  2. holdout A (coins ranked 26-45) and B (coins 46-65), 2023-01..2025-09, median of 3 calendars: both better;
  3. plateau: the neighbouring parameter also beats the baseline on train.
A passing idea goes to the paper demo as an extra profile, never straight into the bot.

Ideas (source -> rule; main / neighbour parameter):
  dip_sleeve_rsi2       Connors RSI(2): while the daily gate holds, buy half a slot when 4h RSI(2) < 10 / 5,
                        sell above the 5-bar SMA, on a 20-bar low or when the gate fails. A second, mean-reverting
                        sleeve for the cash the breakout bot leaves idle.
  pyramid_after_1atr    Turtle pyramiding: half a position on the breakout, the full position once the close has
                        run 1.0 / 1.5 ATR(42) above the entry close.
  core_btc_eth_200d     Faber / 200-day rule: BTC and ETH get a full slot (1/6) while their daily close is above the
                        200 / 150-day SMA, on top of the bot (scaled down if the book is over 100%).
  fear_greed_below_80   Sentiment (alternative.me Fear & Greed, previous day's value): no new entries above 80 / 75.
  skip_weekend_breakouts Thin weekend liquidity: no entries on bars closing Sat/Sun UTC (neighbour: window 8h earlier).
  rvol_same_hour        Equity-style RVOL: breakout volume vs the same 4h slot of the previous 20 days, > 2.0x / 1.7x
                        (replaces the plain 20-bar average).
  strong_close          Breakout bar quality: close in the top 30% / 40% of the bar's range.
  not_overextended      Minervini / "don't chase": no entry when (close - EMA120) / ATR(84) is above its train q80 /
                        q90 at breakout bars (descriptive quantiles, computed before any performance is looked at).
  cooldown_after_loss   Freqtrade protections: after a coin's position closes at a loss, no new entry for 36 / 24 bars.

    python -m research.methods.market_ideas
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from algo.htf import align, resample, state
from research.dynamic_universe import UNI
from research.harness import TRAIN, VARIANTS, Lab, metrics
from research.methods import geometry as G
from research.methods import indicators as I
from research.methods import indicators3 as I3
from research.methods.profile_variants import profile_full

LOCK = I3.LOCK
CORE = ("BTCUSDT", "ETHUSDT")


def locked_lab() -> Lab:
    """The research data with everything from the locked year removed."""
    lab = Lab.load().truncate(LOCK - pd.Timedelta("4h"))
    assert lab.index[-1] < LOCK
    return lab


def fear_greed(lab):
    """Previous day's Fear & Greed value on the 4h grid (the value dated D is used from D+1 00:00 UTC)."""
    def go():
        d = json.load(open(UNI / "fng.json"))["data"]
        s = pd.Series({pd.Timestamp(int(x["timestamp"]), unit="s", tz="UTC"): float(x["value"]) for x in d}).sort_index()
        s.index = s.index + pd.Timedelta("1D")
        s = s[s.index <= lab.index[-1] + pd.Timedelta("4h")]
        # a 4h bar decides at its close (index + 4h), so it may use values stamped up to that time
        v = s.reindex(lab.index + pd.Timedelta("4h"), method="ffill").to_numpy()
        return pd.DataFrame(np.repeat(v[:, None], len(lab.assets), axis=1), index=lab.index, columns=lab.assets)
    return lab.memo(("fng",), go)


def extension(lab):
    return lab.memo(("ext",), lambda: (lab.close - I.ema(lab.close, 120)) / I.atr(lab, 84))


def runs(S):
    """Start flags and run ids of the stretches where S > 0."""
    on = S > 0
    start = on & ~on.shift(fill_value=False)
    return start, start.cumsum().where(on)


def run_cummax(x, rid):
    return pd.DataFrame({c: x[c].groupby(rid[c]).cummax() for c in x.columns}).reindex(index=x.index, columns=x.columns)


def extra_entry(lab, name, p, q):
    c, h, l = lab.close, lab.high, lab.low
    if name == "fng":
        return fear_greed(lab) < p
    if name == "weekend":
        close_t = lab.index + pd.Timedelta("4h") + pd.Timedelta(f"{p}h")
        bad = np.isin(close_t.dayofweek, (5, 6))
        return pd.DataFrame(np.repeat(~bad[:, None], len(lab.assets), axis=1), index=lab.index, columns=lab.assets)
    if name == "clv":
        return (c - l) / (h - l).replace(0, np.nan) >= p
    if name == "ext":
        return extension(lab) < q[p]
    raise KeyError(name)


def signal(lab, extra=None, relvol=("plain", 2.0), cooldown=0, pyramid=0.0, dip=None, q=None):
    key = ("mkt", extra, relvol, cooldown, pyramid, dip)

    def go():
        cand = I.candidates(lab)
        g = lab.gate(("1d",), "close>ema50")
        c = lab.close
        v = lab.volume.astype(float)
        poc, vah, _ = profile_full(lab, 180, 24)
        ein = (c > vah) & lab.memo(("cand", "vol_expanding"), cand["vol_expanding"][1]).fillna(False).astype(bool)
        kind, mult = relvol
        if kind == "plain":
            ein = ein & (v > mult * v.rolling(20).mean().shift())
        else:  # same 4h slot of the previous 20 days
            ein = ein & (v > mult * sum(v.shift(6 * i) for i in range(1, 21)) / 20)
        xout = (c < poc) | lab.memo(("cand", "exit_cmf_neg"), cand["exit_cmf_neg"][1]).fillna(False).astype(bool)
        if extra is not None:
            ein = ein & extra_entry(lab, *extra, q).fillna(False).astype(bool)

        def systems(e_ok):
            total = 0
            for n in (50, 100, 200):
                for m in (10, 20):
                    e, x = lab.donchian(n, m)
                    total = total + state(e & g & e_ok, x | ~g | xout)
            return total / 6

        S = systems(ein)
        if cooldown:
            start, _ = runs(S)
            entry_px = c.where(start).ffill()
            closed = (S == 0) & (S.shift() > 0)
            loss = closed & (c < entry_px)
            block = loss.astype(float).rolling(cooldown, min_periods=1).max() > 0
            S = systems(ein & ~block)
        if pyramid:
            start, rid = runs(S)
            a = I.atr(lab, 42)
            entry_px, entry_atr = c.where(start).ffill(), a.where(start).ffill()
            full = run_cummax(c, rid) >= entry_px + pyramid * entry_atr
            S = S * np.where(full, 1.0, 0.5)
        if dip is not None:
            r = I.rsi(lab, 2)
            de = (r < dip) & g
            dx = (c > c.rolling(5).mean()) | ~g | (c < lab.low.rolling(20).min().shift())
            S = np.maximum(S, 0.5 * state(de, dx))
        return S.where(lab.live, 0.0)
    return lab.memo(key, go)


def core_overlay(lab, n_days):
    def go():
        d = resample(lab.bars, "1d")["close"][[a for a in CORE if a in lab.assets]]
        on = align((d > d.rolling(n_days).mean()).astype(float), lab.index, "1d").fillna(0.0)
        return on.reindex(columns=lab.assets).fillna(0.0) * (1 / 6)
    return lab.memo(("core", n_days), go)


def make(cfg, q=None):
    cfg = dict(cfg)
    core = cfg.pop("core", None)

    def fn(lab, n, cal, mask=None):
        kw = {"k": 6, "signal": signal(lab, q=q, **cfg)}
        if mask is not None:
            kw["mask"] = mask
        w = lab.build(n, cal, **kw)
        m = mask
        if core is not None:
            w = np.maximum(w, core_overlay(lab, core))
            m = (lab.mask(n, cal) if mask is None else mask).copy()
            for a in CORE:
                if a in m.columns:
                    m[a] = True
        return lab.enforce(w, m)
    return fn


IDEAS = {
    "dip_sleeve_rsi2": ({"dip": 10}, {"dip": 5}),
    "pyramid_after_1atr": ({"pyramid": 1.0}, {"pyramid": 1.5}),
    "core_btc_eth_200d": ({"core": 200}, {"core": 150}),
    "fear_greed_below_80": ({"extra": ("fng", 80)}, {"extra": ("fng", 75)}),
    "skip_weekend_breakouts": ({"extra": ("weekend", 0)}, {"extra": ("weekend", 8)}),
    "rvol_same_hour": ({"relvol": ("hour", 2.0)}, {"relvol": ("hour", 1.7)}),
    "strong_close": ({"extra": ("clv", 0.7)}, {"extra": ("clv", 0.6)}),
    "not_overextended": ({"extra": ("ext", "q80")}, {"extra": ("ext", "q90")}),
    "cooldown_after_loss": ({"cooldown": 36}, {"cooldown": 24}),
}


def ext_quantiles(lab):
    """Descriptive only: extension at the baseline's entry bars on train (no returns involved)."""
    S = signal(lab)
    start, _ = runs(S)
    x = extension(lab).where(start)
    x = x[x.index < TRAIN[1]].stack().dropna()
    return {"q80": float(x.quantile(0.8)), "q90": float(x.quantile(0.9))}


def main():
    lab = locked_lab()
    print("data ends", lab.index[-1], flush=True)
    G.install(lab)
    ref = I.signal(lab, I3.BASE_E, I3.BASE_X)
    print("same as the best bot:", float((signal(lab) - ref).abs().max().max()), flush=True)
    q = ext_quantiles(lab)
    print("extension quantiles at entries (train):", {k: round(v, 2) for k, v in q.items()}, flush=True)
    leak = max(lab.leak_check(lambda l, n, cal, c: make(c, q)(l, n, cal), {"c": IDEAS[k][0]}, dates=("2021-05-19 04:00", "2024-03-04 12:00"))
               for k in IDEAS)
    print("leak:", leak, flush=True)
    base = make({}, q)
    bt = G.run(lab, base, *TRAIN)
    bA, bB = G.hold(lab, base, (25, 45)), G.hold(lab, base, (45, 65))
    b = bt.sharpe.median()
    print(f"baseline: train Sh {b:.2f} CAGR {bt.cagr.median():+.0%} DD {bt.maxdd.median():.0%} | A {bA:+.2f} | B {bB:+.2f}", flush=True)
    rows = []
    for name, (cfg, nb) in IDEAS.items():
        fn = make(cfg, q)
        d = G.run(lab, fn, *TRAIN)
        beats = (d.sharpe.to_numpy() > bt.sharpe.to_numpy()).mean()
        A, B = G.hold(lab, fn, (25, 45)), G.hold(lab, fn, (45, 65))
        plateau = G.run(lab, make(nb, q), *TRAIN).sharpe.median() > b
        ok = d.sharpe.median() > b and beats >= 6 / 9 and A > bA and B > bB and plateau
        rows.append({"name": name, "sharpe": d.sharpe.median(), "cagr": d.cagr.median(), "maxdd": d.maxdd.median(), "beats": beats,
                     "A": A, "B": B, "plateau": plateau, "pass_all": ok})
        print(f"{name:24s} train Sh {d.sharpe.median():.2f} CAGR {d.cagr.median():+.0%} DD {d.maxdd.median():.0%} beats {beats:.0%}"
              f" | A {A:+.2f} | B {B:+.2f} | plateau {'ok' if plateau else 'NO'}{'  <== PASS ALL' if ok else ''}", flush=True)
    t = pd.DataFrame(rows)
    Path("data/methods").mkdir(parents=True, exist_ok=True)
    t.to_csv("data/methods/market_ideas.csv", index=False)
    winners = list(t[t.pass_all].name)
    json.dump({"passed": winners, "baseline": {"train": b, "A": bA, "B": bB}}, open("data/methods/market_ideas_choice.json", "w"))
    print("\npassed all tests:", winners or "nothing - keep the current best bot")


if __name__ == "__main__":
    main()
