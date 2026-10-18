"""Does the live bot's code make the same decisions as the research backtest?

Replays PortfolioTrader (algo/forward.py, simulated clock, paper broker) over the research data and compares it
with the harness's backtest of the same rules (15 coins, quarterly from January), costs matched at 0.12% per side.

    python -m research.live_parity 2025-01-01 2026-10-16
"""
import json
import sys

import numpy as np
import pandas as pd

from algo import forward
from algo.trend_portfolio import PortfolioConfig
from research.dynamic_universe import UNI
from research.harness import Lab, baseline


def main(start: str, end: str) -> None:
    cfg = PortfolioConfig(**json.load(open("config.json"))["portfolio"])
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    r = forward.run(cfg, None, lo, hi, capital=10_000, fee_rate=0.001, slippage=0.0002, local_dir=UNI)
    live = r["equity"].pct_change().dropna()
    lab = Lab.load()
    w = lab.enforce(baseline(lab, cfg.universe_size, cfg.universe_calendar))
    pnl = lab.pnl(w)
    # the paper equity at time t (= candle t-4h closed) includes the return of candle t-4h
    bt = pnl.reindex(live.index - pd.Timedelta("4h")).set_axis(live.index)
    both = pd.DataFrame({"live": live, "backtest": bt}).dropna()
    tot = (1 + both).prod() - 1
    print(f"{len(both)} decisions; total return live {tot.live:+.1%} vs backtest {tot.backtest:+.1%}; "
          f"4h-return correlation {both.corr().iloc[0, 1]:.4f}; mean |diff| {np.abs(both.live - both.backtest).mean():.2e}")
    for R, u in sorted(r["universes"].items()):
        print(R, " ".join(s.split("/")[0] for s in u))
    worst = (both.live - both.backtest).abs().sort_values(ascending=False).head(5)
    print("largest 4h differences:\n", worst.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
