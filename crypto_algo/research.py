"""Parameter research: optimize on 2017-2018 (train), report 2019 (test) separately.

Usage: python research.py   (needs data/coinbase_BTC-USD_{1h,4h}.csv)
"""
import itertools, sys, json
from dataclasses import replace
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from algo.backtest import run_backtest, PERIODS_PER_YEAR
from algo.config import Config
from algo.data import load_csv

SPLIT = pd.Timestamp("2019-01-01", tz="UTC")

def seg(res, tf, start=None, end=None):
    eq = res.equity
    eq = eq[(eq.index >= start) if start is not None else slice(None)]
    eq = eq[(eq.index < end)] if end is not None else eq
    r = eq.pct_change().dropna()
    ppy = PERIODS_PER_YEAR[tf]
    trades = [t for t in res.trades if (start is None or t.entry_time >= start) and (end is None or t.entry_time < end)]
    dd = (eq / eq.cummax() - 1).min()
    ret = eq.iloc[-1] / eq.iloc[0] - 1
    return dict(ret=ret, dd=dd, sharpe=r.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else 0,
                trades=len(trades), calmar=ret / -dd if dd < 0 else 0)

def configs(base):
    for name in ("ema_rsi", "donchian"):
        variants = ([dict(rsi_min=45, rsi_max=70), dict(rsi_min=0, rsi_max=100)] if name == "ema_rsi" else
                    [dict(donchian_entry=e, donchian_exit=x) for e in (20, 55) for x in (10, 20)])
        for v, rr, trail, stop in itertools.product(variants, (0, 2.5, 4), (2, 3, 4), (2, 3)):
            yield replace(base, strategy=replace(base.strategy, name=name, **v),
                          risk=replace(base.risk, risk_per_trade=0.02, reward_risk=rr, trailing_atr_mult=trail, atr_stop_mult=stop))

def label(c):
    s, r = c.strategy, c.risk
    v = f"rsi{'on' if s.rsi_min else 'off'}" if s.name == "ema_rsi" else f"{s.donchian_entry}/{s.donchian_exit}"
    return f"{s.name:<8} {v:<6} tp={r.reward_risk:<3} trail={r.trailing_atr_mult} stop={r.atr_stop_mult}"

if __name__ == "__main__":
    rows = []
    for tf in ("1h", "4h"):
        df = load_csv(f"data/coinbase_BTC-USD_{tf}.csv")
        base = Config.load("config.json"); base.timeframe = tf
        for c in configs(base):
            res = run_backtest(df, c)
            tr, te = seg(res, tf, end=SPLIT), seg(res, tf, start=SPLIT)
            rows.append(dict(tf=tf, cfg=label(c), **{f"tr_{k}": v for k, v in tr.items()}, **{f"te_{k}": v for k, v in te.items()}))
    out = pd.DataFrame(rows)
    out.to_csv("data/research_grid.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    fmt = out.copy()
    for k in ("tr_ret", "tr_dd", "te_ret", "te_dd"): fmt[k] = fmt[k].map("{:.1%}".format)
    for k in ("tr_sharpe", "te_sharpe", "tr_calmar", "te_calmar"): fmt[k] = fmt[k].map("{:.2f}".format)
    for tf in ("1h", "4h"):
        sub = out[(out.tf == tf) & (out.tr_trades >= 15)].sort_values("tr_sharpe", ascending=False)
        print(f"\n=== {tf}: top 12 by TRAIN sharpe (selection uses train only) ===")
        print(fmt.loc[sub.index[:12], ["cfg", "tr_ret", "tr_dd", "tr_sharpe", "tr_trades", "te_ret", "te_dd", "te_sharpe", "te_trades"]].to_string(index=False))
        print(f"  share of {tf} configs profitable in test: {(out[out.tf==tf].te_ret > 0).mean():.0%}, median test ret {out[out.tf==tf].te_ret.median():.1%}")
        for n in ("ema_rsi", "donchian"):
            s = out[(out.tf == tf) & out.cfg.str.startswith(n)]
            print(f"  {n}: median train sharpe {s.tr_sharpe.median():.2f}, median test sharpe {s.te_sharpe.median():.2f}")
