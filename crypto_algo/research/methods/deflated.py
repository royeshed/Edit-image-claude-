"""How much of the bot's Sharpe survives the number of ideas tried (Bailey & Lopez de Prado's Deflated Sharpe Ratio),
and how big an improvement a new idea needs before it is more than the luck of trying many.

Data cut at 2025-10-01 (locked year unread). Bot = the best bot on its live universe (15 coins, quarterly from January).

  DSR  = P(true Sharpe > SR0), SR0 = the best Sharpe expected from N tries of skill-less strategies whose Sharpes
         spread with standard deviation sd (annualised); corrected for skew and fat tails of the 4h returns.
  bar  = the train Sharpe gain that pure noise reaches as the best of N ideas: SE(gain) x E[max of N normals],
         SE(gain) = sqrt(2 (1 - rho) / years), rho = return correlation of the idea with the bot.

    python -m research.methods.deflated
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from research.harness import PPY, TRAIN
from research.methods import geometry as G
from research.methods.indicators3 import HOLD
from research.methods.market_ideas import IDEAS, ext_quantiles, locked_lab, make

EULER = 0.5772156649


def emax(n):
    """Expected maximum of n independent standard normals."""
    return (1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * np.e))


def dsr(r, n_trials, sd_ann):
    r = r.dropna()
    sr = r.mean() / r.std()
    sr0 = sd_ann / np.sqrt(PPY) * emax(n_trials)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    z = (sr - sr0) * np.sqrt(len(r) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return sr * np.sqrt(PPY), sr0 * np.sqrt(PPY), norm.cdf(z), g3, g4


def period(p, lo, hi):
    return p[(p.index >= lo) & (p.index < hi)].fillna(0)


def main():
    lab = locked_lab()
    G.install(lab)
    base = lab.pnl(make({})(lab, 15, "Q-jan"))
    for name, (lo, hi) in (("train 2018-11..2022", TRAIN), ("2023-01..2025-09", HOLD)):
        r = period(base, lo, hi)
        print(f"\n{name}: {len(r) / PPY:.1f} years", flush=True)
        for n in (100, 350):
            for sd in (0.15, 0.30):
                s, s0, p, g3, g4 = dsr(r, n, sd)
                print(f"  N={n:3d} trials, spread {sd:.2f}: Sharpe {s:.2f}, luck alone reaches {s0:.2f}, DSR {p:.1%}"
                      f"  (skew {g3:+.1f}, kurtosis {g4:.0f})", flush=True)
    years = len(period(base, *TRAIN)) / PPY
    rb = period(base, *TRAIN)
    sb = rb.mean() / rb.std() * np.sqrt(PPY)
    q = ext_quantiles(lab)
    print(f"\nnoise bar for a train Sharpe gain ({years:.1f} train years):")
    for name, (cfg, _) in IDEAS.items():
        r = period(lab.pnl(make(cfg, q)(lab, 15, "Q-jan")), *TRAIN)
        rho = r.corr(rb)
        se = np.sqrt(2 * (1 - rho) / years)
        gain = r.mean() / r.std() * np.sqrt(PPY) - sb
        print(f"  {name:24s} gain {gain:+.2f}  corr {rho:.2f}  SE {se:.2f}  z {gain / se:+.1f}  "
              f"bar for 1 idea {1.645 * se:.2f}, best of 140 {emax(140) * se:.2f}", flush=True)


if __name__ == "__main__":
    main()
