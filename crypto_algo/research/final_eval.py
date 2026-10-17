"""Final evaluation of the chosen strategy, including the held-out 2025-2026 period.

Run once, after all choices were made on 2018-2024:
    python -m research.final_eval
"""
import numpy as np
import pandas as pd

from .lab import COINS, PERIODS, backtest, load_panel, metrics, period_mask, portfolio
from .trend import trend_positions

NEW = ["LINK", "LTC", "DOT", "TRX", "XLM", "BCH", "ATOM", "UNI", "NEAR", "AAVE", "SHIB", "ETC", "FIL", "HBAR", "PEPE"]
NICHES = {
    "Store of value": ["BTC"],
    "Smart-contract L1": ["ETH", "SOL", "AVAX", "ADA", "DOT", "NEAR", "ATOM", "TRX", "ETC"],
    "Payments": ["XRP", "XLM", "LTC", "BCH", "HBAR"],
    "Memes": ["DOGE", "SHIB", "PEPE"],
    "DeFi / oracles": ["LINK", "UNI", "AAVE"],
    "Exchange / other": ["BNB", "FIL"],
}
CHOSEN = dict(base=24 * 20, hours=(13,), long_only=True, target_vol=0.5, cap=1.0)  # fixed before opening the test period


def evaluate(coins: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    p = load_panel("1h", coins)
    r = p["ret"]
    pos = trend_positions(p, **CHOSEN)
    strat = portfolio(backtest(pos, r))
    hold = portfolio(backtest(pd.DataFrame(1.0, index=r.index, columns=r.columns).where(r.notna()), r, cost=0))
    rows = []
    for name, series in (("strategy", strat), ("buy & hold", hold)):
        row = {"": name}
        for per in list(PERIODS) + ["all"]:
            m = metrics(series, "1h", None if per == "all" else per)
            row[f"{per} sharpe"] = m["sharpe"]
            row[f"{per} cagr"] = m["cagr"]
            row[f"{per} maxdd"] = m["maxdd"]
        rows.append(row)
    return pd.DataFrame(rows), strat, hold


def main() -> None:
    fmt = lambda d: d.to_string(index=False, float_format=lambda v: f"{v:.2f}")
    for label, coins in (("ORIGINAL 8 (research universe)", COINS), ("15 UNSEEN COINS", NEW), ("ALL 23", COINS + NEW)):
        table, strat, hold = evaluate(coins)
        print(f"\n=== {label} ===\n{fmt(table)}")
        if label == "ALL 23":
            pd.DataFrame({"strategy": (1 + strat).cumprod(), "buy_hold": (1 + hold).cumprod()}).resample("1D").last().to_csv(
                "data/final_equity.csv")
    print("\n=== By niche (2025-2026 held-out period) ===")
    for niche, coins in NICHES.items():
        table, *_ = evaluate(coins)
        s, h = table.iloc[0], table.iloc[1]
        print(f"{niche:<18} strategy sharpe {s['test sharpe']:5.2f} cagr {s['test cagr']:+6.0%} dd {s['test maxdd']:5.0%} | "
              f"hold sharpe {h['test sharpe']:5.2f} cagr {h['test cagr']:+6.0%} dd {h['test maxdd']:5.0%}")


if __name__ == "__main__":
    main()
