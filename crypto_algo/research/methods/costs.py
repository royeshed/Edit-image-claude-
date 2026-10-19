"""What trading costs take from the best bot, and what cheaper execution or interest on idle cash would add.

Nothing here chooses a rule, so there is nothing to overfit; still, the data is cut at 2025-10-01 (locked year unread).
Bot = the best bot (market_ideas.make({})), 9 universe variants, median reported.

Cost scenarios, per side:
  now        0.12%  (taker 0.10% + 0.02% slippage, what every backtest so far assumed)
  bnb        0.095% (fees paid in BNB: 0.075% + slippage)
  limit      limit order at the decision bar's close, valid for the next 4h bar. It fills if the next bar trades
             0.05% through the price (maker 0.075% with BNB, no slippage); otherwise the order is done at market
             on the next close (0.095%) and the bot misses that bar's move (bought one bar late / sold one bar late).
             Run for buys only, sells only and both.
  zero       no cost at all (the ceiling).
Cash yield: interest on the uninvested part of equity at 3% and 5% a year.

    python -m research.methods.costs
"""
import numpy as np
import pandas as pd

from research.harness import PPY, TRAIN, VARIANTS, metrics
from research.methods import geometry as G
from research.methods.indicators3 import HOLD
from research.methods.market_ideas import locked_lab, make

THROUGH = 0.0005
MAKER, TAKER = 0.00075, 0.00095


def pnl(lab, w, cost=0.0012, limit=(), yield_=0.0):
    ret = lab.ret.fillna(0)
    d = w.diff().fillna(w)
    gross = (w.shift(1) * ret).sum(axis=1)
    if not limit:
        fees = cost * d.abs().sum(axis=1)
    else:
        c = lab.close
        nxt_low, nxt_high = lab.low.shift(-1), lab.high.shift(-1)
        buy, sell = d.clip(lower=0), (-d).clip(lower=0)
        fill_b = (nxt_low < c * (1 - THROUGH)) if "buy" in limit else pd.DataFrame(True, index=w.index, columns=w.columns)
        fill_s = (nxt_high > c * (1 + THROUGH)) if "sell" in limit else pd.DataFrame(True, index=w.index, columns=w.columns)
        rate_b = np.where(fill_b, MAKER if "buy" in limit else TAKER, TAKER)
        rate_s = np.where(fill_s, MAKER if "sell" in limit else TAKER, TAKER)
        fees = (buy * rate_b).sum(axis=1) + (sell * rate_s).sum(axis=1)
        nxt = ret.shift(-1)
        # a missed order changes the holding during the next bar only
        late = (-(buy.where(~fill_b, 0.0)) + sell.where(~fill_s, 0.0)) * nxt
        gross = gross + late.sum(axis=1).shift(1).fillna(0)
    out = gross - fees
    if yield_:
        out = out + yield_ / PPY * (1 - w.shift(1).sum(axis=1)).clip(lower=0)
    return out


def fill_rates(lab, w):
    d = w.diff()
    c = lab.close
    b, s = d > 1e-9, d < -1e-9
    fb = (lab.low.shift(-1) < c * (1 - THROUGH))[b].stack().mean()
    fs = (lab.high.shift(-1) > c * (1 + THROUGH))[s].stack().mean()
    return fb, fs


SCEN = {
    "now 0.12%": dict(cost=0.0012),
    "BNB 0.095%": dict(cost=TAKER),
    "BNB + limit buys": dict(limit=("buy",)),
    "BNB + limit sells": dict(limit=("sell",)),
    "BNB + limit both": dict(limit=("buy", "sell")),
    "zero cost": dict(cost=0.0),
    "now + 3% on cash": dict(cost=0.0012, yield_=0.03),
    "now + 5% on cash": dict(cost=0.0012, yield_=0.05),
    "BNB + 5% on cash": dict(cost=TAKER, yield_=0.05),
}


def main():
    lab = locked_lab()
    G.install(lab)
    fn = make({})
    ws = {v: fn(lab, *v) for v in VARIANTS}
    tr = [metrics(lab.pnl(w), *TRAIN, w) for w in ws.values()]
    ho = [metrics(lab.pnl(w), *HOLD, w) for w in ws.values()]
    for name, per in (("train 2018-11..2022", tr), ("2023-01..2025-09", ho)):
        t = np.median([m["turnover"] for m in per])
        e = np.median([m["exposure"] for m in per])
        print(f"{name}: traded {t:.0f}x equity/yr (both sides), invested {e:.0%} on average, "
              f"fees at 0.12% = {0.0012 * t:.1%} of equity/yr", flush=True)
    fb, fs = zip(*(fill_rates(lab, w) for w in ws.values()))
    print(f"limit fill rate next bar: buys {np.median(fb):.0%}, sells {np.median(fs):.0%}\n", flush=True)
    rows = []
    for name, kw in SCEN.items():
        r = {"scenario": name}
        for per, (lo, hi) in (("train", TRAIN), ("2023-25", HOLD)):
            ms = [metrics(pnl(lab, w, **kw), lo, hi) for w in ws.values()]
            r[f"{per} CAGR"] = np.median([m["cagr"] for m in ms])
            r[f"{per} Sharpe"] = np.median([m["sharpe"] for m in ms])
            r[f"{per} DD"] = np.median([m["maxdd"] for m in ms])
        rows.append(r)
        print(f"{name:20s} train {r['train CAGR']:+.1%} Sh {r['train Sharpe']:.2f} DD {r['train DD']:.0%} | "
              f"2023-25.09 {r['2023-25 CAGR']:+.1%} Sh {r['2023-25 Sharpe']:.2f} DD {r['2023-25 DD']:.0%}", flush=True)
    pd.DataFrame(rows).to_csv("data/methods/costs.csv", index=False)


if __name__ == "__main__":
    main()
