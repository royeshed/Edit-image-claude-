"""Entry filters on top of the live bot's Donchian breakouts, with a fresh holdout.

2023-2026 on the bot's own 15 coins was already seen in the previous round, so it cannot judge new ideas.
Rules declared before running:
  * Choose on 2018-11..2022 (median of the 9 harness variants): an idea counts only if it beats the current bot
    in >= 6 of 9 variants with a higher median Sharpe.
  * Holdout, run once for every idea: coins ranked 26-45 by the same mature-volume rule (never used to judge the
    bot), 2023-01..2026-10. The chosen idea is adopted only if it also beats the current bot there.

    python -m research.methods.entries
"""
import json

import numpy as np
import pandas as pd

from algo.htf import state
from research.harness import LOCKED, TRAIN, Lab, metrics, show, summary
from research.universe_selection import CALENDARS, membership, rule_scores_all, universe

FAMILY = "entries"
GRID = {
    "volume_confirm": {"vol_mult": 1.5},
    "squeeze": {"squeeze_q": 0.3},
    "adx_daily": {"adx_min": 20},
    "beats_btc": {"rs_days": 30},
    "two_closes": {"confirm_bars": 2},
    "shorter_channels": {"scale": 0.85},
    "gate_ema20": {"gate": "close>ema20"},
    "gate_ema100": {"gate": "close>ema100"},
}


def daily_ema_gate(lab: Lab, n: int) -> pd.DataFrame:
    from algo.htf import align, resample
    def go():
        d = resample(lab.bars, "1d")["close"]
        return align((d > d.ewm(span=n, adjust=False).mean()).astype(float), lab.index, "1d").fillna(0).astype(bool)
    return lab.memo(("ema_gate", n), go)


def daily_adx(lab: Lab, n: int = 14) -> pd.DataFrame:
    from algo.htf import align, resample
    def go():
        d = resample(lab.bars, "1d")
        h, l, c = d["high"], d["low"], d["close"]
        up, dn = h.diff(), -l.diff()
        pdm = up.where((up > dn) & (up > 0), 0.0)
        ndm = dn.where((dn > up) & (dn > 0), 0.0)
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()]).groupby(level=0).max()
        atr = tr.ewm(alpha=1 / n, adjust=False).mean()
        pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr
        ndi = 100 * ndm.ewm(alpha=1 / n, adjust=False).mean() / atr
        adx = (100 * (pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / n, adjust=False).mean()
        return align(adx, lab.index, "1d")
    return lab.memo(("adx", n), go)


def signal(lab: Lab, vol_mult=None, squeeze_q=None, adx_min=None, rs_days=None, confirm_bars=1, scale=1.0, gate="close>ema50"):
    key = (FAMILY, vol_mult, squeeze_q, adx_min, rs_days, confirm_bars, scale, gate)

    def go():
        g = lab.gate(("1d",), "close>ema50") if gate == "close>ema50" else daily_ema_gate(lab, int(gate.split("ema")[1]))
        c = lab.close
        extra = pd.DataFrame(True, index=c.index, columns=c.columns)
        if vol_mult:
            v = lab.volume.astype(float)
            extra &= v > vol_mult * v.rolling(20).mean().shift()
        if squeeze_q:
            bw = (c.rolling(20).std() / c.rolling(20).mean())
            extra &= (bw.rolling(10).min() <= bw.rolling(500, min_periods=200).quantile(squeeze_q))
        if adx_min:
            extra &= daily_adx(lab) > adx_min
        if rs_days:
            r = c / c.shift(rs_days * 6) - 1
            extra &= r.gt(r["BTCUSDT"], axis=0) | (c.columns == "BTCUSDT")
        total = 0
        systems = [(round(n * scale), round(m * scale)) for n in lab.cfg.entries for m in lab.cfg.exits]
        for n, m in systems:
            e = c > lab.high.rolling(n).max().shift()
            if confirm_bars > 1:
                e = e & e.shift().fillna(False)
            x = c < lab.low.rolling(m).min().shift()
            total = total + state(e & g & extra, x | ~g)
        return (total / len(systems)).where(lab.live, 0.0)
    return lab.memo(key, go)


def weights(lab, n, calendar, **p):
    return lab.build(n, calendar, signal=signal(lab, **p) if p else None)


def holdout_mask(lab: Lab, calendar="Q-jan"):
    f = lab.features
    s = rule_scores_all(f, "mature_volume")
    r = s.groupby(f.date).rank(ascending=False)
    s2 = s.where((r > 25) & (r <= 45))
    return membership(universe(f, s2, CALENDARS[calendar], 20), lab.index, lab.assets).astype(bool)


def main():
    lab = Lab.load()
    base = lab.baseline_results
    print("current bot (train):", show(base), flush=True)
    rows = []
    for name, p in GRID.items():
        d = lab.evaluate(weights, FAMILY, p)
        s = summary(d, base)
        rows.append({"idea": name, **s.to_dict()})
        print(f"{name:16s}", show(d, base), flush=True)
    t = pd.DataFrame(rows).set_index("idea")
    passed = t[(t.sharpe > base.train_sharpe.median()) & (t.beats_base >= 6 / 9)]
    print("\npassed on 2018-2022:", list(passed.index))
    # holdout: coins ranked 26-45, 2023-2026, every idea, once
    lo, hi = LOCKED["oos"]
    out = []
    for cal in ("Q-jan", "Q-feb", "Q-mar"):
        m = holdout_mask(lab, cal)
        for name, p in [("current bot", {})] + list(GRID.items()):
            w = lab.enforce(lab.build(20, cal, mask=m, signal=signal(lab, **p) if p else None), m)
            pn = lab.pnl(w)
            out.append({"idea": name, "cal": cal, **metrics(pn, lo, hi), "train_sh": metrics(pn, *TRAIN)["sharpe"]})
    h = pd.DataFrame(out).groupby("idea")[["sharpe", "cagr", "maxdd", "train_sh"]].median()
    print("\nholdout (coins ranked 26-45, 2023-2026, median of 3 calendars):")
    print(h.round(3).to_string())
    pd.concat([t, h.add_prefix("holdout_")], axis=1).to_csv("data/methods/entries.csv")


if __name__ == "__main__":
    main()
