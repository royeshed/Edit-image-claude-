"""Do the live bot's demo profiles make the same decisions as their research versions (research/methods/market_ideas)?

    python -m research.live_parity_profiles 2025-01-01 2025-04-01
"""
import json
import sys

import numpy as np
import pandas as pd

from algo import forward
from algo.trend_portfolio import PortfolioConfig
from research.dynamic_universe import UNI
from research.harness import Lab
from research.methods import geometry as G
from research.methods.market_ideas import make

RESEARCH = {"best": {}, "best_no_weekend": {"extra": ("weekend", 0)}, "best_core_btc_eth": {"core": 200}}


def main(start: str, end: str) -> None:
    raw = json.load(open("config.json"))
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    lab = Lab.load()
    G.install(lab)
    for name, rcfg in RESEARCH.items():
        cfg = PortfolioConfig(**{**raw["portfolio"], **raw["paper"]["profiles"][name]})
        r = forward.run(cfg, None, lo, hi, capital=10_000, fee_rate=0.001, slippage=0.0002, local_dir=UNI)
        live = r["equity"].pct_change().dropna()
        pnl = lab.pnl(make(rcfg)(lab, cfg.universe_size, cfg.universe_calendar))
        bt = pnl.reindex(live.index - pd.Timedelta("4h")).set_axis(live.index)
        both = pd.DataFrame({"live": live, "backtest": bt}).dropna()
        tot = (1 + both).prod() - 1
        print(f"{name:18s} {len(both)} decisions; live {tot.live:+.1%} vs backtest {tot.backtest:+.1%}; "
              f"corr {both.corr().iloc[0, 1]:.4f}; mean |diff| {np.abs(both.live - both.backtest).mean():.2e}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
