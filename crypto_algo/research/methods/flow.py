"""Order flow, volume profile and liquidity as entry/exit filters for the aggressive bot (top 6, <= 1/6 per coin).

Data: Binance 4h klines with taker-buy volume (research/flow_data.py). No historical order book exists, so order
flow is measured from aggressive (market-order) buying: TBR = taker-buy quote volume / quote volume.

Rules declared before running:
  * Baseline = the aggressive bot (k=6). Choose on 2018-11..2022 (median of the 9 harness variants): an idea counts
    only if it beats the baseline in >= 6 of 9 variants with a higher median Sharpe.
  * Ideas that pass are combined once (all passing entry filters together, plus passing exits).
  * Holdout, run once for every idea: coins ranked 26-45 (never used to judge the bot), 2023-01..2026-10, 3 calendars.
    Adopt only what passed train AND beats the baseline on the holdout.

    python -m research.methods.flow
"""
import numpy as np
import pandas as pd

from algo.htf import state
from research.dynamic_universe import UNI
from research.harness import LOCKED, TRAIN, Lab, metrics, show, summary
from research.methods.entries import holdout_mask

FAMILY = "flow"
K = 6
ENTRY = {
    "tbr_break": {"tbr_min": 0.55},
    "cvd_trend": {"cvd_bars": 30},
    "above_poc": {"poc": True},
    "above_vah": {"vah": True},
    "liquid_entry": {"illiq_max": 2.0},
    "obv_trend": {"obv": True},
    "trades_surge": {"trades_mult": 1.5},
}
EXIT = {
    "cvd_exit": {"cvd_exit": 12},
    "poc_exit": {"poc_exit": True},
}


def flow_panels(lab: Lab) -> dict:
    def go():
        qv, tb, tr = {}, {}, {}
        for a in lab.assets:
            f = UNI / "flow" / f"{a.split('#')[0]}_4h.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, index_col=0)
            d.index = pd.to_datetime(d.index, utc=True)
            qv[a], tb[a], tr[a] = d.quote_volume, d.tb_quote, d.trades
        mk = lambda x: pd.DataFrame(x).reindex(index=lab.index, columns=lab.assets).where(lab.live)
        return {"qv": mk(qv), "tb": mk(tb), "trades": mk(tr)}
    return lab.memo(("flow_panels",), go)


def volume_profile(lab: Lab, window: int = 180, bins: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The live bot's volume profile (algo/profile.py) on the research panels."""
    from algo.profile import volume_profile as vp
    return lab.memo(("vprofile", window, bins), lambda: vp(lab.high, lab.low, lab.close, lab.volume.astype(float), window, bins))


def signal(lab: Lab, tbr_min=None, cvd_bars=None, poc=False, vah=False, illiq_max=None, obv=False, trades_mult=None,
           cvd_exit=None, poc_exit=False) -> pd.DataFrame:
    key = (FAMILY, tbr_min, cvd_bars, poc, vah, illiq_max, obv, trades_mult, cvd_exit, poc_exit)

    def go():
        g = lab.gate(("1d",), "close>ema50")
        c = lab.close
        fp = flow_panels(lab)
        qv, tb = fp["qv"], fp["tb"]
        net = (2 * tb / qv - 1) * qv                    # signed aggressive flow in USDT
        ein = pd.DataFrame(True, index=c.index, columns=c.columns)
        xout = pd.DataFrame(False, index=c.index, columns=c.columns)
        if tbr_min:
            ein &= (tb / qv) > tbr_min
        if cvd_bars:
            ein &= net.rolling(cvd_bars).sum() > 0
        if poc or vah or poc_exit:
            P, V = volume_profile(lab)
            if poc:
                ein &= c > P
            if vah:
                ein &= c > V
            if poc_exit:
                xout |= c < P
        if illiq_max:
            il = (lab.ret.abs() / qv).rolling(42).mean()
            ein &= il <= illiq_max * il.rolling(540, min_periods=180).median()
        if obv:
            o = (np.sign(lab.ret.fillna(0)) * lab.volume.astype(float)).cumsum()
            ein &= o > o.ewm(span=50, adjust=False).mean()
        if trades_mult:
            t = fp["trades"]
            ein &= t > trades_mult * t.rolling(20).mean().shift()
        if cvd_exit:
            xout |= net.rolling(cvd_exit).sum() < 0
        total = 0
        for n in lab.cfg.entries:
            for m in lab.cfg.exits:
                e, x = lab.donchian(n, m)
                total = total + state(e & g & ein.fillna(False), x | ~g | xout.fillna(False))
        return (total / (len(lab.cfg.entries) * len(lab.cfg.exits))).where(lab.live, 0.0)
    return lab.memo(key, go)


def weights(lab, n, calendar, **p):
    return lab.build(n, calendar, k=K, signal=signal(lab, **p) if p else None)


def final():
    """Confirmation of the adopted rule (enter above VAH, exit below POC) with the live bot's code."""
    lab = Lab.load()
    p = {"vah": True, "poc_exit": True}
    for k in (6, 3):
        fn = lambda lab, n, cal, **q: lab.build(n, cal, k=k, signal=signal(lab, **q) if q else None)
        base = lab.evaluate(fn, f"{FAMILY}-final-base-k{k}", {}, log=False)
        d = lab.evaluate(fn, f"{FAMILY}-final-k{k}", p)
        print(f"k={k} train: base", show(base), "\n          profile", show(d, base))
        lo, hi = LOCKED["oos"]
        rows = []
        for cal in ("Q-jan", "Q-feb", "Q-mar"):
            m = holdout_mask(lab, cal)
            for name, q in (("base", {}), ("profile", p)):
                w = lab.enforce(lab.build(20, cal, k=k, mask=m, signal=signal(lab, **q) if q else None), m)
                rows.append({"idea": name, **metrics(lab.pnl(w), lo, hi)})
        print(f"k={k} holdout:\n", pd.DataFrame(rows).groupby("idea")[["sharpe", "cagr", "maxdd"]].median().round(3).to_string())


def main():
    lab = Lab.load()
    base = lab.evaluate(weights, FAMILY + "-baseline", {}, log=False)
    print("aggressive bot (train):", show(base), flush=True)
    rows, ideas = [], {**ENTRY, **EXIT}
    for name, p in ideas.items():
        d = lab.evaluate(weights, FAMILY, p)
        rows.append({"idea": name, **summary(d, base).to_dict()})
        print(f"{name:14s}", show(d, base), flush=True)
    t = pd.DataFrame(rows).set_index("idea")
    passed = list(t[(t.sharpe > base.train_sharpe.median()) & (t.beats_base >= 6 / 9)].index)
    print("\npassed on 2018-2022:", passed, flush=True)
    if len(passed) > 1:
        combo = {k: v for name in passed for k, v in ideas[name].items()}
        ideas["combined"] = combo
        d = lab.evaluate(weights, FAMILY, combo)
        t.loc["combined"] = summary(d, base)
        print(f"{'combined':14s}", show(d, base), flush=True)
    lo, hi = LOCKED["oos"]
    out = []
    for cal in ("Q-jan", "Q-feb", "Q-mar"):
        m = holdout_mask(lab, cal)
        for name, p in [("aggressive bot", {})] + list(ideas.items()):
            w = lab.enforce(lab.build(20, cal, k=K, mask=m, signal=signal(lab, **p) if p else None), m)
            pn = lab.pnl(w)
            out.append({"idea": name, "cal": cal, **metrics(pn, lo, hi)})
    h = pd.DataFrame(out).groupby("idea")[["sharpe", "cagr", "maxdd"]].median()
    print("\nholdout (coins ranked 26-45, 2023-2026, median of 3 calendars):")
    print(h.round(3).to_string())
    pd.concat([t, h.add_prefix("holdout_")], axis=1).to_csv("data/methods/flow.csv")


if __name__ == "__main__":
    import sys
    final() if sys.argv[1:] == ["final"] else main()
