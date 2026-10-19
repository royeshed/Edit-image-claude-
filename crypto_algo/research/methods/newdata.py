"""Is the edge real? The best bot, unchanged, on data no decision was ever made on. Nothing is tuned here.

All data ends before 2025-10-01 (the locked year stays unread).

  coins  coins ranked 66-85 by the universe rule (never used by any batch; A was 26-45, B 46-65),
         2018-11..2025-09, 3 calendars, vs equal-weight buy-and-hold of the same coins and vs BTC.
  okx    the bot's own universe, but prices and volumes from OKX instead of Binance (coins OKX lists only),
         against the same run on Binance data restricted to the same coins.
  etf    other markets: 38 ETFs (stock indices, countries, sectors, bonds, commodities, currencies), daily bars,
         2008..2025-09. The bot's code in bars (50/100/200 x 10/20 Donchian, volume profile 180 bars, ...), the
         confirmation one timeframe up (weekly close > weekly EMA50), top 6 by 84-bar momentum, at most 1/6 each,
         0.05% per side. Also the core alone (breakout + weekly gate + ranking, no volume rules).

    python -m research.methods.newdata coins|okx|etf
"""
import json
import sys

import numpy as np
import pandas as pd

from algo.trend_portfolio import PortfolioConfig
from research.harness import ROOT, TRAIN, Lab
from research.methods import geometry as G
from research.methods import indicators as I
from research.methods.indicators3 import HOLD, LOCK
from research.methods.market_ideas import locked_lab, make

CALS = ("Q-jan", "Q-feb", "Q-mar")
ALL = (TRAIN[0], LOCK)


def stats(r, ppy):
    r = r.fillna(0)
    eq = (1 + r).cumprod()
    yrs = len(r) / ppy
    return {"cagr": eq.iloc[-1] ** (1 / yrs) - 1, "sharpe": r.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else 0.0,
            "maxdd": (eq / eq.cummax() - 1).min()}


def cut(r, lo, hi):
    return r[(r.index >= lo) & (r.index < hi)]


def line(name, rs, ppy, periods):
    out = []
    for lo, hi in periods:
        s = pd.DataFrame([stats(cut(r, lo, hi), ppy) for r in rs]).median()
        out.append(f"{s.cagr:+6.0%} Sh {s.sharpe:5.2f} DD {s.maxdd:4.0%}")
    print(f"  {name:30s} " + " | ".join(out), flush=True)


def eq_weight(lab, m):
    w = m & lab.live
    return (lab.ret.fillna(0) * w.shift(1).fillna(False)).sum(axis=1) / w.shift(1).fillna(False).sum(axis=1).replace(0, np.nan)


def coins():
    lab = locked_lab()
    G.install(lab)
    fn = make({})
    periods = (TRAIN, HOLD, ALL)
    print("coins ranked 66-85 (never used) | train 2018-11..2022 | 2023-01..2025-09 | whole", flush=True)
    masks = [I.holdout(lab, 65, 85, cal) for cal in CALS]
    line("best bot", [lab.pnl(fn(lab, 20, cal, mask=m)) for cal, m in zip(CALS, masks)], 2190, periods)
    line("equal-weight buy & hold", [eq_weight(lab, m).fillna(0) for m in masks], 2190, periods)
    line("BTC buy & hold", [lab.ret["BTCUSDT"]], 2190, periods)
    print("bot's own universe (15 coins), for reference:", flush=True)
    line("best bot", [lab.pnl(fn(lab, 15, cal)) for cal in CALS], 2190, periods)


def okx():
    lab = locked_lab()
    G.install(lab)
    d = ROOT / "data" / "okx"
    frames = {}
    for a in lab.assets:
        f = d / f"{a.split('#')[0]}_4h.csv"
        if f.exists():
            x = pd.read_csv(f, index_col=0)
            if len(x):
                x.index = pd.to_datetime(x.index, utc=True)
                # same listing span as the Binance asset (a relisted symbol is a different coin)
                span = lab.close[a].dropna().index
                frames[a] = x[(x.index >= span[0]) & (x.index <= span[-1])]
    have = list(frames)
    print(len(have), "coins with OKX candles", flush=True)
    pan = {k: pd.DataFrame({a: f[k] for a, f in frames.items()}).reindex(index=lab.index, columns=lab.assets) for k in
           ("open", "high", "low", "close", "volume")}
    ret = pan["close"].ffill().pct_change(fill_method=None).where(pan["close"].notna())
    ox = Lab(close=pan["close"], high=pan["high"], low=pan["low"], open=pan["open"], volume=pan["volume"].astype("float32"),
             ret=ret, daily_close=lab.daily_close, daily_volume=lab.daily_volume, features=lab.features, cfg=lab.cfg)
    G.install(ox)
    on_okx = pan["close"].notna()
    fn = make({})
    periods = (TRAIN, HOLD)
    print("same coins, two data sources | train 2018-11..2022 | 2023-01..2025-09", flush=True)
    for name, L in (("Binance data", lab), ("OKX data", ox)):
        rs = [L.pnl(fn(L, 15, cal, mask=lab.mask(15, cal) & on_okx)) for cal in CALS]
        line(name, rs, 2190, periods)
    m = lab.mask(15, "Q-jan")
    print(f"  share of universe coin-bars covered by OKX: {float((m & on_okx).sum().sum() / m.sum().sum()):.0%}", flush=True)


def etf():
    d = ROOT / "data" / "etf"
    frames = {f.stem: pd.read_csv(f, index_col=0, parse_dates=True) for f in sorted(d.glob("*.csv"))}
    idx = pd.bdate_range("2005-01-03", "2025-09-30", tz="UTC")
    pan = {k: pd.DataFrame({t: f[k] for t, f in frames.items()}).reindex(idx) for k in ("open", "high", "low", "close", "volume")}
    pan = {k: v.ffill(limit=3) if k != "volume" else v.fillna(0) for k, v in pan.items()}  # market holidays
    ret = pan["close"].pct_change(fill_method=None)
    cfg = PortfolioConfig(**json.load(open(ROOT / "config.json"))["portfolio"])
    lab = Lab(close=pan["close"], high=pan["high"], low=pan["low"], open=pan["open"], volume=pan["volume"].astype("float32"),
              ret=ret, daily_close=pan["close"], daily_volume=pan["volume"], features=pd.DataFrame(), cfg=cfg)
    G.install(lab)
    m = pan["close"].notna()
    cost = 0.0005
    wk = (("1w",), "close>ema50")
    full = G.signal(lab, gate=wk)

    def core():
        from algo.htf import state
        g = lab.gate(*wk)
        tot = 0
        for n in (50, 100, 200):
            for k in (10, 20):
                e, x = lab.donchian(n, k)
                tot = tot + state(e & g, x | ~g)
        return (tot / 6).where(lab.live, 0.0)

    periods = ((pd.Timestamp("2008-01-01", tz="UTC"), pd.Timestamp("2017-01-01", tz="UTC")),
               (pd.Timestamp("2017-01-01", tz="UTC"), LOCK), (pd.Timestamp("2008-01-01", tz="UTC"), LOCK))
    print("38 ETFs, daily | 2008-2016 | 2017-2025.09 | whole", flush=True)
    for name, S in (("best bot (as is, in bars)", full), ("core: breakout + weekly gate", core())):
        w = lab.enforce(lab.build(len(lab.assets), "", mask=m, signal=S, k=6), m)
        line(name, [lab.pnl(w, cost)], 252, periods)
        print(f"    invested {w.sum(axis=1)[w.index >= periods[0][0]].mean():.0%} on average", flush=True)
    mm = m.shift(1, fill_value=False).astype(float)
    line("equal weight all ETFs", [(ret.fillna(0) * mm).sum(axis=1) / mm.sum(axis=1).replace(0, np.nan)], 252, periods)
    line("SPY buy & hold", [ret["SPY"]], 252, periods)
    line("60% SPY / 40% IEF", [0.6 * ret["SPY"] + 0.4 * ret["IEF"]], 252, periods)


if __name__ == "__main__":
    {"coins": coins, "okx": okx, "etf": etf}[sys.argv[1]]()
