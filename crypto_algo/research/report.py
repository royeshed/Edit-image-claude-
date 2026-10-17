"""Combined-portfolio P&L report across all coins (spot, long only, spot costs).

Strategies:
  breakout  - 4h ensemble of six Donchian systems with a daily close>EMA50 gate (the default bot)
  momentum  - daily ensemble momentum on 1h bars, vol-targeted 50%, cap 1x, decided 13:00 UTC
  combined  - each coin's capital split 50/50 between the two, rebalanced daily
  hold      - buy & hold
Outputs a JSON blob for the report page:  python -m research.report out.json
"""
import json
import sys

import numpy as np
import pandas as pd

from algo.htf import confirmation, confirmed_breakouts
from .final_eval import NEW, NICHES
from .lab import COINS, SPOT_COST, backtest, load_panel
from .run_mtf_confirm import ENS
from .trend import trend_positions

ALL = COINS + NEW
STRATS = ["breakout", "momentum", "combined", "hold"]
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")


def daily(x: pd.DataFrame) -> pd.DataFrame:
    return (1 + x).resample("1D").prod(min_count=1) - 1


def episodes(pos: pd.Series, pnl: pd.Series) -> list[float]:
    """Return of each continuous holding period (position > 0)."""
    held = (pos.shift(1).fillna(0) > 0) | (pos > 0)
    start = held & ~held.shift(1, fill_value=False)
    ep = start.cumsum().where(held)
    g = (1 + pnl.fillna(0)).groupby(ep).prod() - 1
    return g.tolist()


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 30:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / 365
    return {"total": eq.iloc[-1] - 1, "cagr": eq.iloc[-1] ** (1 / years) - 1, "maxdd": (eq / eq.cummax() - 1).min(),
            "sharpe": r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0.0}


def main(out_path: str) -> None:
    p1 = load_panel("1h", ALL)
    p4 = load_panel("4h", ALL)
    pos_b = confirmed_breakouts(p4, ENS, confirmation(p4, p4["close"].index, ("1d",), "close>ema50"), "gate")
    pnl_b = backtest(pos_b, p4["ret"], SPOT_COST)
    pos_m = trend_positions(p1, base=480, hours=(13,), long_only=True, target_vol=0.5, cap=1.0)
    pnl_m = backtest(pos_m, p1["ret"], SPOT_COST)

    d = {"breakout": daily(pnl_b), "momentum": daily(pnl_m), "hold": daily(p1["ret"])}
    d["combined"] = 0.5 * d["breakout"] + 0.5 * d["momentum"]
    idx = d["hold"].index
    d = {k: v.reindex(idx) for k, v in d.items()}
    live = d["hold"].notna()
    for k in d:
        d[k] = d[k].where(live)

    report = {"coins": [], "portfolio": {}, "niches": [], "years": sorted(set(idx.year)), "generated": str(idx[-1].date())}
    # --- per coin
    for c in ALL:
        row = {"coin": c, "niche": next(n for n, cs in NICHES.items() if c in cs), "research": c in COINS,
               "since": str(d["hold"][c].first_valid_index().date())}
        for s in STRATS:
            r = d[s][c].dropna()
            row[s] = {**stats(r), "pnl_usd": 1000 * ((1 + r).prod() - 1),
                      "test": stats(r[r.index >= TEST_START]),
                      "yearly": {int(y): float((1 + g).prod() - 1) for y, g in r.groupby(r.index.year)}}
        for s, pos, pnl in (("breakout", pos_b[c], pnl_b[c]), ("momentum", pos_m[c], pnl_m[c])):
            ep = episodes(pos, pnl)
            wins = [e for e in ep if e > 0]
            row[s]["trades"] = {"count": len(ep), "win_rate": len(wins) / len(ep) if ep else 0,
                                "avg": float(np.mean(ep)) if ep else 0, "best": max(ep) if ep else 0, "worst": min(ep) if ep else 0,
                                "profit_factor": sum(wins) / -sum(e for e in ep if e <= 0) if any(e <= 0 for e in ep) else None}
        report["coins"].append(row)
    # --- portfolio (equal weight across coins trading that day)
    eqs = {}
    for s in STRATS:
        r = d[s].mean(axis=1, skipna=True).dropna()
        eqs[s] = (1 + r).cumprod()
        report["portfolio"][s] = {**stats(r), "test": stats(r[r.index >= TEST_START]),
                                  "yearly": {int(y): float((1 + g).prod() - 1) for y, g in r.groupby(r.index.year)},
                                  "monthly": {k.strftime("%Y-%m"): float(v) for k, v in ((1 + r).resample("ME").prod() - 1).items()}}
    weekly = pd.DataFrame(eqs).resample("W").last()
    report["equity"] = {"dates": [str(x.date()) for x in weekly.index], **{s: [round(float(v), 4) for v in weekly[s]] for s in STRATS}}
    dd = (weekly / weekly.cummax() - 1)
    report["drawdown"] = {s: [round(float(v), 4) for v in dd[s]] for s in STRATS}
    # correlation of daily returns between the two strategies (portfolio level)
    pb, pm = d["breakout"].mean(axis=1), d["momentum"].mean(axis=1)
    report["corr_breakout_momentum"] = float(pb.corr(pm))
    # --- niches
    for niche, cs in NICHES.items():
        row = {"niche": niche, "coins": cs}
        for s in STRATS:
            r = d[s][cs].mean(axis=1, skipna=True).dropna()
            row[s] = {**stats(r), "test": stats(r[r.index >= TEST_START])}
        report["niches"].append(row)
    with open(out_path, "w") as f:
        json.dump(report, f, default=float)
    # console summary
    for s in STRATS:
        x = report["portfolio"][s]
        print(f"{s:<9} total {x['total']:+8.0%} cagr {x['cagr']:+5.0%} dd {x['maxdd']:5.0%} sharpe {x['sharpe']:.2f} | 2025-26 {x['test']['total']:+5.0%} dd {x['test']['maxdd']:5.0%}")
    print("corr breakout/momentum:", round(report["corr_breakout_momentum"], 2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/report.json")
