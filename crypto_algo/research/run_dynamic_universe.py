"""Backtest the portfolio bot on point-in-time universes vs the fixed hindsight list.

    python -m research.run_dynamic_universe plan   # print universes and the 4h symbols needed
    python -m research.run_dynamic_universe run    # backtest, then a sweep over list size and refresh dates
"""
import json
import sys

import numpy as np
import pandas as pd

from algo.htf import confirmation, confirmed_breakouts
from algo.trend_portfolio import PortfolioConfig
from .allocation import run
from .dynamic_universe import UNI, load_bars, load_daily, membership, universes
from .lab import SPOT_COST, metrics

FIXED23 = ["BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL", "AVAX", "LINK", "LTC", "DOT", "TRX", "XLM", "BCH", "ATOM", "UNI",
           "NEAR", "AAVE", "SHIB", "ETC", "FIL", "HBAR", "PEPE"]
FIXED = [f"{c}USDT" for c in FIXED23]
CONFIGS = [(f, n) for f in ("yearly", "quarterly", "monthly") for n in (15, 23, 40)]
# Robustness: same rule with other list sizes and refresh dates moved off January.
SWEEP = [(f, n) for f in ("yearly", "YS-APR", "YS-JUL", "YS-OCT", "quarterly", "QS-FEB", "QS-MAR", "monthly")
         for n in (10, 15, 20, 23, 30, 40, 50, 60)]


def plan():
    close_d, vol_d, spans = load_daily()
    need = set(FIXED)
    univ = {}
    for f, n in dict.fromkeys(CONFIGS + SWEEP):
        u = universes(vol_d, f, n)
        univ[f"{f}-{n}"] = {str(k.date()): v for k, v in u.items()}
        for v in u.values():
            need |= {spans[a][0] for a in v}
    y23 = universes(vol_d, "yearly", 23)
    univ["fixed2020-23"] = {"2020-01-01": y23[pd.Timestamp("2020-01-01", tz="UTC")]}  # chosen once, never updated
    (UNI / "universes.json").write_text(json.dumps(univ))
    print(len(need), "symbols need 4h bars")
    print(" ".join(sorted(need)))
    for y in ("2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01", "2026-01-01"):
        print(y, univ["yearly-23"].get(y))


def weights(s: pd.DataFrame, mom: pd.DataFrame, k: int, slots: int) -> pd.DataFrame:
    m = mom.where(s > 0)
    sk = s.where(m.rank(axis=1, ascending=False, method="first") <= k, 0)
    return sk.div(np.maximum(sk.sum(axis=1), slots), axis=0)


def go():
    cfg = PortfolioConfig(**json.load(open("config.json"))["portfolio"])
    close_d, vol_d, spans = load_daily()
    univ = {k: {pd.Timestamp(d, tz="UTC"): v for d, v in u.items()} for k, u in json.loads((UNI / "universes.json").read_text()).items()}
    assets = sorted({a for u in univ.values() for v in u.values() for a in v} | set(FIXED))
    p = load_bars([a for a in assets if a in spans], spans)
    r, c = p["ret"], p["close"]
    live = r.notna()
    systems = [(n, m) for n in cfg.entries for m in cfg.exits]
    S = confirmed_breakouts(p, systems, confirmation(p, r.index, cfg.confirm_tfs, cfg.confirm), cfg.confirm_mode).where(live, 0)
    MOM = c / c.shift(cfg.rank_days * 6) - 1
    rows, pnls = [], {}

    def add(name, mask):
        s = S.where(mask > 0, 0)
        w = weights(s, MOM, cfg.top_k, cfg.max_slots)
        pnl = run(w, r, SPOT_COST)
        hold = (r.where(mask > 0)).mean(axis=1, skipna=True).fillna(0)
        row = {"universe": name}
        for per in ("train", "valid", "test", None):
            m = metrics(pnl, "4h", per); tag = per or "all"
            row.update({f"{tag}_cagr": m["cagr"], f"{tag}_dd": m["maxdd"], f"{tag}_sh": m["sharpe"]})
        hm = metrics(hold, "4h")
        p20, h20 = pnl[pnl.index >= "2020-01-01"], hold[hold.index >= "2020-01-01"]
        m20, mh20 = metrics(p20, "4h"), metrics(h20, "4h")
        row.update({"hold_cagr": hm["cagr"], "hold_dd": hm["maxdd"], "final_$": 10_000 * (1 + pnl).prod(),
                    "s2020_cagr": m20["cagr"], "s2020_dd": m20["maxdd"], "s2020_sh": m20["sharpe"],
                    "s2020_hold_cagr": mh20["cagr"], "s2020_hold_dd": mh20["maxdd"]})
        rows.append(row)
        pnls[name], pnls[name + " hold"] = pnl, hold

    fixed_cols = [a for a in FIXED if a in S.columns]
    fixed_mask = np.broadcast_to(S.columns.isin(fixed_cols), S.shape).astype(float)
    add("FIXED 23 (hindsight list)", pd.DataFrame(fixed_mask, index=S.index, columns=S.columns))
    keys = ["fixed2020-23"] + [f"{f}-{n}" for f, n in dict.fromkeys(CONFIGS + SWEEP)]
    for key in keys:
        add(key, membership(univ[key], S.index, S.columns))
    d = pd.DataFrame(rows)
    d.to_csv(UNI / "results.csv", index=False)
    pd.DataFrame(pnls).to_csv(UNI / "pnl_4h.csv")
    pd.set_option("display.width", 250)
    fmt = {k: "{:+.0%}".format for k in d.columns if k.endswith("_cagr")} | {k: "{:.0%}".format for k in d.columns if k.endswith("_dd")} | \
          {k: "{:.2f}".format for k in d.columns if k.endswith("_sh")} | {"final_$": "${:,.0f}".format}
    main = ["FIXED 23 (hindsight list)", "fixed2020-23"] + [f"{f}-{n}" for f, n in CONFIGS]
    print(d[d.universe.isin(main)].to_string(index=False, formatters=fmt))
    sw = d[d.universe.isin([f"{f}-{n}" for f, n in SWEEP])].copy()
    sw[["freq", "n"]] = sw.universe.str.rsplit("-", n=1, expand=True)
    sw["n"] = sw.n.astype(int)
    for col in ("all_cagr", "all_dd", "all_sh", "s2020_cagr", "test_cagr"):
        t = sw.pivot(index="freq", columns="n", values=col).reindex([f for f, _ in dict.fromkeys((f, 0) for f, _ in SWEEP)])
        f = "{:.2f}".format if col.endswith("_sh") else "{:+.0%}".format
        print(f"\n{col}\n" + t.to_string(float_format=f))
    print("\nsweep spread:", sw[["all_cagr", "all_dd", "all_sh", "s2020_cagr", "test_cagr"]].describe().loc[["min", "25%", "50%", "75%", "max"]].round(2).to_string())


if __name__ == "__main__":
    plan() if sys.argv[1] == "plan" else go()
