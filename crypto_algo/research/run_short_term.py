"""Sweep the long-only short-horizon families on a timeframe. Usage: python -m research.run_short_term 1h 2h 4h"""
import sys

import numpy as np
import pandas as pd

from .lab import BARS_PER_YEAR, SPOT_COST, backtest, load_panel, metrics, period_mask, portfolio
from .short_term import daily_gate, families

PERIODS_15M = {"train": ("2022-01-01", "2024-01-01"), "valid": ("2024-01-01", "2025-01-01"), "test": ("2025-01-01", "2100-01-01")}


def evaluate(pos: pd.DataFrame, r: pd.DataFrame, tf: str, periods=("train", "valid"), cost: float = SPOT_COST) -> dict:
    pnl = portfolio(backtest(pos, r, cost))
    out = {}
    entries = (pos.diff() > 0).where(r.notna())
    for per in periods:
        m = metrics(pnl, tf, per)
        mask = period_mask(pos.index, per)
        years = mask.sum() / BARS_PER_YEAR[tf]
        live_coins = r[mask].notna().any().sum()
        out.update({f"{per}_sh": m["sharpe"], f"{per}_cagr": m["cagr"], f"{per}_dd": m["maxdd"],
                    f"{per}_trades_y": entries[mask].sum().sum() / max(live_coins, 1) / years,
                    f"{per}_exp": pos[mask].where(r[mask].notna()).mean().mean()})
    return out


def sweep(tf: str) -> pd.DataFrame:
    p = load_panel(tf)
    r = p["ret"]
    gate = daily_gate(p, 50)
    rows = [{"tf": tf, "strategy": "buy & hold", "gate": "-", **evaluate(pd.DataFrame(1.0, index=r.index, columns=r.columns), r, tf, cost=0)},
            {"tf": tf, "strategy": "daily-trend gate alone", "gate": "-", **evaluate(gate.astype(float), r, tf)}]
    for name, (fn, kw) in families(tf).items():
        sig = fn(p, **kw)
        for g in ("none", "daily50"):
            pos = sig if g == "none" else sig * gate
            rows.append({"tf": tf, "strategy": name, "gate": g, **evaluate(pos, r, tf)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    frames = [sweep(tf) for tf in sys.argv[1:] or ["1h", "2h", "4h"]]
    d = pd.concat(frames)
    d.to_csv("data/research_short_term.csv", index=False)
    pd.set_option("display.width", 250)
    fmt = {k: "{:.2f}".format for k in d.columns if k.endswith("_sh")} | {k: "{:+.0%}".format for k in d.columns if k.endswith("_cagr")} | \
          {k: "{:.0%}".format for k in d.columns if k.endswith(("_dd", "_exp"))} | {k: "{:.0f}".format for k in d.columns if k.endswith("_trades_y")}
    for tf, g in d.groupby("tf", sort=False):
        print(f"\n===== {tf} (sorted by train Sharpe) =====")
        print(g.sort_values("train_sh", ascending=False).drop(columns="tf").to_string(index=False, formatters=fmt))
