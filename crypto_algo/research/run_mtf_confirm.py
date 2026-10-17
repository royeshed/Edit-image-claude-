"""Does higher-timeframe confirmation improve long-only spot breakout systems?

    python -m research.run_mtf_confirm
"""
import itertools

import pandas as pd

from .lab import COINS, SPOT_COST, load_panel
from .mtf_confirm import align, confirmations, confirmed_breakouts, resample
from .run_short_term import evaluate

ENS = [(50, 10), (50, 20), (100, 10), (100, 20), (200, 10), (200, 20)]  # in 4h bars
BASES = {
    "4h ensemble":       ("4h", ENS,                              [("1d",), ("1w",), ("1d", "1w")]),
    "4h fast 20/10":     ("4h", [(20, 10)],                       [("1d",), ("1w",), ("1d", "1w")]),
    "2h ensemble":       ("2h", [(n * 2, m * 2) for n, m in ENS], [("4h",), ("1d",), ("4h", "1d")]),
    "1h ensemble":       ("1h", [(n * 4, m * 4) for n, m in ENS], [("4h",), ("1d",), ("4h", "1d")]),
    "1h fast 50/20":     ("1h", [(50, 20)],                       [("4h",), ("1d",), ("4h", "1d")]),
}


def run(coins=COINS) -> pd.DataFrame:
    p1 = load_panel("1h", coins)
    panels = {tf: resample(p1, tf) for tf in ("1h", "2h", "4h", "1d", "1w")}
    conf = {tf: confirmations(panels[tf]) for tf in ("4h", "1d", "1w")}
    rows = []
    for base, (tf, systems, htf_sets) in BASES.items():
        p = panels[tf]
        r = p["ret"]
        rows.append({"base": base, "confirm_tf": "-", "confirm": "none", "mode": "-",
                     **evaluate(confirmed_breakouts(p, systems, None, "entry"), r, tf, cost=SPOT_COST)})
        for htfs, (cname, mode) in itertools.product(htf_sets, itertools.product(conf["1d"], ("entry", "gate"))):
            c = None
            for h in htfs:
                a = align(conf[h][cname].astype(float), r.index, h).fillna(0).astype(bool)
                c = a if c is None else (c & a)
            rows.append({"base": base, "confirm_tf": "+".join(htfs), "confirm": cname, "mode": mode,
                         **evaluate(confirmed_breakouts(p, systems, c, mode), r, tf, cost=SPOT_COST)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = run()
    d.to_csv("data/research_mtf_confirm.csv", index=False)
    base = d[d.confirm == "none"].set_index("base")
    d["d_train_sh"] = d.train_sh - d.base.map(base.train_sh)
    d["d_valid_sh"] = d.valid_sh - d.base.map(base.valid_sh)
    d["d_train_dd"] = d.train_dd - d.base.map(base.train_dd)
    d["d_valid_dd"] = d.valid_dd - d.base.map(base.valid_dd)
    pd.set_option("display.width", 250)
    fmt = {k: "{:+.2f}".format for k in ("d_train_sh", "d_valid_sh")} | {k: "{:.2f}".format for k in ("train_sh", "valid_sh")} | \
          {k: "{:+.0%}".format for k in ("train_cagr", "valid_cagr", "d_train_dd", "d_valid_dd")} | {k: "{:.0%}".format for k in ("train_dd", "valid_dd")} | \
          {k: "{:.0f}".format for k in ("train_trades_y",)}
    cols = ["base", "confirm_tf", "confirm", "mode", "train_sh", "valid_sh", "d_train_sh", "d_valid_sh", "train_cagr", "valid_cagr", "train_dd", "valid_dd", "train_trades_y"]
    print("=== Baselines (no confirmation) ===")
    print(d[d.confirm == "none"][cols].to_string(index=False, formatters=fmt))
    c = d[d.confirm != "none"]
    both = c[(c.d_train_sh > 0) & (c.d_valid_sh > 0)]
    print(f"\nconfirmed variants: {len(c)} | improve Sharpe in BOTH periods: {len(both)} ({len(both)/len(c):.0%}) | "
          f"improve in train only: {((c.d_train_sh > 0) & (c.d_valid_sh <= 0)).sum()} | median d_sharpe train {c.d_train_sh.median():+.2f} valid {c.d_valid_sh.median():+.2f}")
    print("\nMedian Sharpe change by factor (train / valid):")
    for f in ("base", "confirm_tf", "confirm", "mode"):
        g = c.groupby(f)[["d_train_sh", "d_valid_sh"]].median()
        print(f"  {f}: " + ", ".join(f"{a}={b.d_train_sh:+.2f}/{b.d_valid_sh:+.2f}" for a, b in g.iterrows()))
    print("\n=== Top 20 by train Sharpe among confirmed variants ===")
    print(c.sort_values("train_sh", ascending=False).head(20)[cols].to_string(index=False, formatters=fmt))
