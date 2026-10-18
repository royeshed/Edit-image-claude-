"""Variants of the volume-profile rule on the aggressive bot (k=6). Baseline = aggressive + VAH entry + POC exit.

Rules declared before running: choose on 2018-11..2022 (>= 6 of 9 variants beat the baseline, higher median Sharpe);
holdout once: coins ranked 26-45, 2023-2026, 3 calendars. Adopt only what passes both.

    python -m research.methods.profile_variants
"""
import numpy as np
import pandas as pd

from algo.htf import state
from algo.profile import volume_profile
from research.harness import LOCKED, Lab, metrics, show, summary
from research.methods.entries import holdout_mask

K = 6
GRID = {
    "window_7d": dict(window=42),
    "window_14d": dict(window=84),
    "window_90d": dict(window=540),
    "exit_below_VAL": dict(exit="val"),
    "exit_below_VAH": dict(exit="vah"),
    "bins_12": dict(bins=12),
    "bins_48": dict(bins=48),
    "above_30d_and_90d_VAH": dict(second=540),
}


def profile_full(lab, window, bins):
    """POC, VAH and VAL (value-area low) with the same method as algo.profile."""
    def go():
        return volume_profile(lab.high, lab.low, lab.close, lab.volume.astype(float), window, bins, with_val=True)
    return lab.memo(("pfull", window, bins), go)


def signal(lab, window=180, bins=24, exit="poc", second=None):
    def go():
        g = lab.gate(("1d",), "close>ema50")
        c = lab.close
        poc, vah, val = profile_full(lab, window, bins)
        ein = c > vah
        if second:
            ein &= c > profile_full(lab, second, bins)[1]
        xout = c < {"poc": poc, "val": val, "vah": vah}[exit]
        total = 0
        for n in lab.cfg.entries:
            for m in lab.cfg.exits:
                e, x = lab.donchian(n, m)
                total = total + state(e & g & ein, x | ~g | xout)
        return (total / 6).where(lab.live, 0.0)
    return lab.memo(("pv", window, bins, exit, second), go)


def weights(lab, n, calendar, **p):
    return lab.build(n, calendar, k=K, signal=signal(lab, **p))


def main():
    lab = Lab.load()
    base = lab.evaluate(weights, "pv-base", {}, log=False)
    print("aggressive + profile (train):", show(base), flush=True)
    rows = []
    for name, p in GRID.items():
        d = lab.evaluate(weights, "profile_variants", p)
        rows.append({"idea": name, **summary(d, base).to_dict()})
        print(f"{name:22s}", show(d, base), flush=True)
    t = pd.DataFrame(rows).set_index("idea")
    print("\npassed on 2018-2022:", list(t[(t.sharpe > base.train_sharpe.median()) & (t.beats_base >= 6 / 9)].index), flush=True)
    lo, hi = LOCKED["oos"]
    out = []
    for cal in ("Q-jan", "Q-feb", "Q-mar"):
        m = holdout_mask(lab, cal)
        for name, p in [("baseline", {})] + list(GRID.items()):
            w = lab.enforce(lab.build(20, cal, k=K, mask=m, signal=signal(lab, **p)), m)
            out.append({"idea": name, **metrics(lab.pnl(w), lo, hi)})
    print("\nholdout (coins 26-45, 2023-2026):")
    print(pd.DataFrame(out).groupby("idea")[["sharpe", "cagr", "maxdd"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
