"""P&L report data for the live portfolio bot as configured in config.json (breakout, top-k, risk dial).

    python -m research.report_bot <out_dir>      # writes the CSV files the report page reads
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from algo.htf import confirmation, confirmed_breakouts
from algo.trend_portfolio import PortfolioConfig
from .final_eval import NICHES
from .lab import SPOT_COST, load_panel
from .allocation import run

START_CASH = 10_000
TEST_START = "2025-01-01"
HE_NICHE = {"Store of value": "שמירת ערך", "Smart-contract L1": "בלוקצ'יין L1", "Payments": "תשלומים", "Memes": "מימים",
            "DeFi / oracles": "DeFi ואורקלים", "Exchange / other": "בורסה ואחרים"}


def bot_weights(p: dict, cfg: PortfolioConfig) -> pd.DataFrame:
    r, c = p["ret"], p["close"]
    live = r.notna()
    systems = [(n, m) for n in cfg.entries for m in cfg.exits]
    conf = confirmation(p, r.index, cfg.confirm_tfs, cfg.confirm) if cfg.confirm_tfs else None
    s = confirmed_breakouts(p, systems, conf, cfg.confirm_mode).where(live, 0)
    if cfg.top_k:
        mom = (c / c.shift(cfg.rank_days * 6) - 1).where(s > 0)
        s = s.where(mom.rank(axis=1, ascending=False, method="first") <= cfg.top_k, 0)
    slots = cfg.max_slots or live.sum(axis=1)
    return s.div(np.maximum(s.sum(axis=1), slots), axis=0)


def curve(pnl: pd.Series) -> pd.Series:
    return (1 + pnl).cumprod() * START_CASH


def drawdown_info(eq: pd.Series) -> dict:
    dd = eq / eq.cummax() - 1
    t = dd.idxmin()
    pk = eq[:t].idxmax()
    under = eq < eq.cummax()
    runs = (under != under.shift()).cumsum()[under]
    longest = int(under.groupby(runs).size().max() / 6) if under.any() else 0
    return {"maxdd": dd.min(), "peak": eq[pk], "peak_date": str(pk.date()), "trough": eq[t], "trough_date": str(t.date()), "underwater_days": longest}


def main(out: str) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pcfg = PortfolioConfig(**json.load(open("config.json"))["portfolio"])
    coins = [s.split("/")[0] for s in pcfg.symbols]
    p = load_panel("4h", coins)
    r, c = p["ret"], p["close"]
    w = bot_weights(p, pcfg)
    pnl = run(w, r, SPOT_COST)
    eqs = {"bot": curve(pnl), "hold": curve(r.mean(axis=1, skipna=True).fillna(0)), "btc": curve(r["BTC"].fillna(0))}
    years = sorted(set(pnl.index.year))
    yrs_len = len(pnl) / 2190

    def write(name, rows):
        with open(out / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

    # summary: one row per series
    summary = []
    for key, label in (("bot", "הבוט (17% למטבע)"), ("hold", "קנה והחזק – כל המטבעות"), ("btc", "קנה והחזק – ביטקוין")):
        eq = eqs[key]
        d = drawdown_info(eq)
        t = eq[eq.index >= TEST_START]
        test_start_val = eq[eq.index < TEST_START].iloc[-1]
        summary.append({"series": key, "label": label, "start": START_CASH, "end": round(eq.iloc[-1]), "multiple": round(eq.iloc[-1] / START_CASH, 2),
                        "cagr": round((eq.iloc[-1] / START_CASH) ** (1 / yrs_len) - 1, 4), "maxdd": round(d["maxdd"], 4),
                        "dd_peak": round(d["peak"]), "dd_peak_date": d["peak_date"], "dd_trough": round(d["trough"]), "dd_trough_date": d["trough_date"],
                        "test_change": round(t.iloc[-1] / test_start_val - 1, 4)})
    write("summary.csv", summary)

    # weekly equity and drawdown in dollars
    weekly = pd.DataFrame(eqs).resample("W").last()
    weekly = pd.concat([pd.DataFrame({k: [START_CASH] for k in eqs}, index=[pnl.index[0]]), weekly])
    dd = weekly / weekly.cummax() - 1
    write("equity.csv", [{"date": str(i.date()), **{k: round(row[k]) for k in eqs}, **{f"dd_{k}": round(dd.loc[i, k], 4) for k in eqs}}
                         for i, row in weekly.iterrows()])

    # yearly table
    rows, bal = [], {k: START_CASH for k in eqs}
    for y in years:
        row = {"year": y}
        for k in eqs:
            e = eqs[k][str(y)]
            start, end = bal[k], e.iloc[-1]
            row.update({f"{k}_start": round(start), f"{k}_end": round(end), f"{k}_change": round(end / start - 1, 4),
                        f"{k}_pnl": round(end - start), f"{k}_dd": round((e / e.cummax() - 1).min(), 4)})
            bal[k] = end
        rows.append(row)
    write("yearly.csv", rows)

    # per-coin attribution: each bar's account P&L split by coin (fees charged to the coin traded)
    eq_prev = eqs["bot"].shift(1).fillna(START_CASH)
    contrib = (w.shift(1) * r.fillna(0) - SPOT_COST * w.diff().abs().fillna(w.abs())).mul(eq_prev, axis=0)
    year_start = {y: (START_CASH if i == 0 else eqs["bot"][str(y - 1)].iloc[-1]) for i, y in enumerate(years)}
    held_prev = w.shift(1).fillna(0) > 0
    coin_rows, cy_rows = [], []
    for col in coins:
        ep_id = (held_prev[col] & ~held_prev[col].shift(1, fill_value=False)).cumsum().where(held_prev[col])
        eps = ((1 + r[col].fillna(0)).groupby(ep_id).prod() - 1).tolist()
        listed = c[col].first_valid_index()
        niche = next(n for n, cs in NICHES.items() if col in cs)
        coin_rows.append({"coin": col, "niche": HE_NICHE[niche], "profit": round(contrib[col].sum()),
                          "profit_test": round(contrib[col][TEST_START:].sum()), "time_held": round(float((w[col] > 0)[r[col].notna()].mean()), 4),
                          "trades": len(eps), "win_rate": round(float(np.mean([e > 0 for e in eps])), 4) if eps else "",
                          "best_trade": round(max(eps), 4) if eps else "", "worst_trade": round(min(eps), 4) if eps else "",
                          "coin_change": round(c[col].ffill().iloc[-1] / c.loc[listed, col] - 1, 4), "since": str(listed.date())})
        for y in years:
            v = contrib[col][str(y)]
            if r[col][str(y)].notna().any():
                cy_rows.append({"coin": col, "year": y, "pp": round(v.sum() / year_start[y], 4), "usd": round(v.sum())})
    write("coins.csv", coin_rows)
    write("coin_yearly.csv", cy_rows)

    total_profit = sum(x["profit"] for x in coin_rows)
    niche_rows = []
    for niche, cs in NICHES.items():
        prof = sum(x["profit"] for x in coin_rows if x["coin"] in cs)
        niche_rows.append({"niche": HE_NICHE[niche], "coins": " · ".join(cs), "profit": prof, "share": round(prof / total_profit, 4),
                           "profit_test": sum(x["profit_test"] for x in coin_rows if x["coin"] in cs)})
    write("niches.csv", niche_rows)

    yearly_bot = [x["bot_change"] for x in rows]
    m = (1 + pnl).resample("ME").prod() - 1
    worst_i, best_i = int(np.argmin(yearly_bot)), int(np.argmax(yearly_bot))
    facts = {"start_cash": START_CASH, "data_end": str(pnl.index[-1].date()), "test_start": TEST_START, "coins_total": len(coins),
             "top_k": pcfg.top_k, "max_per_coin": round(1 / pcfg.max_slots, 4), "max_invested": round(pcfg.top_k / pcfg.max_slots, 4),
             "avg_invested": round(float(w.sum(axis=1).mean()), 4), "fee_per_side": SPOT_COST, "rank_days": pcfg.rank_days,
             "years_up": sum(v > 0 for v in yearly_bot), "years_down": sum(v < 0 for v in yearly_bot), "years_total": len(years),
             "worst_year": years[worst_i], "worst_year_change": round(yearly_bot[worst_i], 4),
             "best_year": years[best_i], "best_year_change": round(yearly_bot[best_i], 4),
             "months_up": round(float((m > 0).mean()), 4), "best_month": round(float(m.max()), 4), "worst_month": round(float(m.min()), 4),
             "underwater_days": drawdown_info(eqs["bot"])["underwater_days"],
             "coins_profitable": sum(x["profit"] > 0 for x in coin_rows),
             "top_coin": max(coin_rows, key=lambda x: x["profit"])["coin"], "top_coin_profit": max(x["profit"] for x in coin_rows),
             "hold_years_down": sum(x["hold_change"] < 0 for x in rows)}
    write("facts.csv", [{"key": k, "value": v} for k, v in facts.items()])
    print(json.dumps({"summary": summary, "facts": facts}, ensure_ascii=False, default=str, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/report_bot")
