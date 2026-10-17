"""Held-out evaluation of the spot strategy chosen in the short-timeframe study (run once).

    python -m research.final_eval_breakout
"""
import pandas as pd

from .final_eval import NEW, NICHES
from .lab import COINS, PERIODS, SPOT_COST, backtest, load_panel, metrics, portfolio
from .short_term import breakout

ENTRIES, EXITS = (50, 100, 200), (10, 20)  # fixed before opening the test period


def positions(p: dict) -> pd.DataFrame:
    return sum(breakout(p, n, m) for n in ENTRIES for m in EXITS) / (len(ENTRIES) * len(EXITS))


def evaluate(coins):
    p = load_panel("4h", coins)
    r = p["ret"]
    strat = portfolio(backtest(positions(p), r, SPOT_COST))
    hold = portfolio(backtest(pd.DataFrame(1.0, index=r.index, columns=r.columns).where(r.notna()), r, cost=0))
    rows = []
    for name, s in (("strategy", strat), ("buy & hold", hold)):
        row = {"": name}
        for per in list(PERIODS) + ["all"]:
            m = metrics(s, "4h", None if per == "all" else per)
            row.update({f"{per} sharpe": m["sharpe"], f"{per} cagr": m["cagr"], f"{per} maxdd": m["maxdd"]})
        rows.append(row)
    return pd.DataFrame(rows), strat, hold


def main():
    for label, coins in (("ORIGINAL 8", COINS), ("15 UNSEEN COINS", NEW), ("ALL 23", COINS + NEW)):
        table, strat, hold = evaluate(coins)
        print(f"\n=== {label} ===\n" + table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        if label == "ORIGINAL 8":
            t = strat[strat.index >= PERIODS["test"][0]]
            yearly = (1 + t).groupby(t.index.year).prod() - 1
            th = hold[hold.index >= PERIODS["test"][0]]
            yh = (1 + th).groupby(th.index.year).prod() - 1
            print("by year (strategy / hold):", {y: f"{a:+.0%} / {b:+.0%}" for (y, a), b in zip(yearly.items(), yh)})
    print("\n=== By niche, held-out 2025-2026 ===")
    for niche, coins in NICHES.items():
        table, *_ = evaluate(coins)
        s, h = table.iloc[0], table.iloc[1]
        print(f"{niche:<18} strategy sharpe {s['test sharpe']:5.2f} cagr {s['test cagr']:+6.0%} dd {s['test maxdd']:5.0%} | "
              f"hold cagr {h['test cagr']:+6.0%} dd {h['test maxdd']:5.0%}")


if __name__ == "__main__":
    main()
