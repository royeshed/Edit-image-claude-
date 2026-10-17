"""ICT multi-timeframe research: train on 2022-2024, test on 2025 onward.

Needs 5m data, e.g.:
  python -c "from algo.data import load_history; load_history('binance', 'BTC/USDT', '5m', 1750)"
Usage: python research_ict.py BTC ETH
"""
import sys, itertools; sys.path.insert(0, ".")
from dataclasses import replace
import numpy as np, pandas as pd
from algo.config import Config
from algo.data import load_csv
from algo.backtest import run_backtest
from algo.mtf import resample

SPLIT = pd.Timestamp("2025-01-01", tz="UTC")
_cache = {}

def data(coin, tf):
    k = (coin, tf)
    if k not in _cache:
        d5 = load_csv(f"data/binance_{coin}-USDT_5m_1750d.csv")
        _cache[k] = d5 if tf == "5m" else resample(d5, tf)
    return _cache[k]

def cfg_for(tf, fee=0.0005, slip=0.0002, **strat):
    c = Config.load("config.json")
    risk = strat.pop("risk", {})
    return replace(c, timeframe=tf, strategy=replace(c.strategy, name="ict_mtf", **strat),
                   risk=replace(c.risk, **{**dict(risk_per_trade=0.01, reward_risk=2.0, trailing_atr_mult=0,
                                                 max_position_pct=3.0, max_daily_loss=0.05), **risk}),
                   costs=replace(c.costs, fee_rate=fee, slippage=slip))

def evaluate(coin, tf, period="train", **kw):
    df = data(coin, tf)
    df = df[df.index < SPLIT] if period == "train" else df[df.index >= SPLIT - pd.Timedelta(days=30)]
    out = {}
    for label, fee, slip in (("gross", 0, 0), ("net", 0.0005, 0.0002)):
        r = run_backtest(df, cfg_for(tf, fee, slip, **kw))
        eq = r.equity if period == "train" else r.equity[r.equity.index >= SPLIT]
        tr = r.trades if period == "train" else [t for t in r.trades if t.entry_time >= SPLIT]
        wins = sum(t.pnl > 0 for t in tr)
        gp = sum(t.pnl for t in tr if t.pnl > 0); gl = -sum(t.pnl for t in tr if t.pnl <= 0)
        out[label] = dict(ret=eq.iloc[-1] / eq.iloc[0] - 1, dd=(eq / eq.cummax() - 1).min(), n=len(tr),
                          win=wins / len(tr) if tr else 0, pf=gp / gl if gl else np.inf)
    return out

def show(name, o):
    g, n = o["gross"], o["net"]
    print(f"{name:<44} n={n['n']:>5} win={n['win']:>4.0%} | gross PF {g['pf']:>4.2f} ret {g['ret']:>+8.0%} | "
          f"net PF {n['pf']:>4.2f} ret {n['ret']:>+7.0%} dd {n['dd']:>5.0%}", flush=True)


BEST = dict(bias_tf="4h", setup_tf="4h", trigger="fvg", use_killzone=True, swing_n=3,
            risk=dict(reward_risk=3.0, trailing_atr_mult=0))

if __name__ == "__main__":
    for coin in sys.argv[1:] or ["BTC", "ETH"]:
        for period in ("train", "test"):
            show(f"{coin} 1h/4h ICT {period}", evaluate(coin, "1h", period, **{k: dict(v) if isinstance(v, dict) else v
                                                                              for k, v in BEST.items()}))
