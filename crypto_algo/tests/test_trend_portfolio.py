import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.trend_portfolio import PaperPortfolioBroker, PortfolioConfig, PortfolioTrader, target_weights


def panel(n=3000, seed=0, drifts=(0.003, -0.003, 0.0)):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({f"C{i}/USDT": 100 * np.exp(np.cumsum(rng.normal(d, 0.01, n))) for i, d in enumerate(drifts)}, index=idx)


def test_weights_long_only_capped_and_trend_following():
    closes = panel()
    cfg = PortfolioConfig(symbols=list(closes.columns))
    w = target_weights(closes, cfg)
    assert (w >= 0).all() and (w <= cfg.cap / len(w) + 1e-12).all()
    assert w["C0/USDT"] > 0          # steady uptrend -> long
    assert w["C1/USDT"] == 0         # steady downtrend -> flat, never short


def test_weights_scale_inversely_with_volatility():
    closes = panel(drifts=(0.003, 0.003, 0.0))
    closes["C1/USDT"] = 100 * (closes["C0/USDT"] / 100) ** 3  # same trend, triple volatility
    cfg = PortfolioConfig(symbols=list(closes.columns), cap=10)
    w = target_weights(closes, cfg)
    assert w["C1/USDT"] == pytest.approx(w["C0/USDT"] / 3, rel=0.05)


def test_trader_rebalances_once_per_day_at_decision_hour(tmp_path):
    closes = panel()
    cfg = PortfolioConfig(symbols=list(closes.columns), state_file=str(tmp_path / "s.json"))

    class Replay(PortfolioTrader):
        def bars(self):
            return {"close": self._closes, "high": self._closes, "low": self._closes}

    t = Replay(cfg, None, PaperPortfolioBroker(10_000, 0.0005, 0.0002))
    decisions = 0
    for ts in closes.index[2000:2100]:
        t._closes = closes.loc[:ts]
        decisions += t.step()
        t._closes = closes.loc[:ts]
        assert not t.step()  # a second poll on the same candle never trades again
    assert decisions == sum(ts.hour == cfg.decide_hour for ts in closes.index[2000:2100])
    cash, held = t.broker.balances(cfg.symbols)
    assert cash >= 0 and all(q >= 0 for q in held.values())
    assert held["C0/USDT"] > 0 and held["C1/USDT"] == 0
    # State survives a restart.
    t2 = Replay(cfg, None, PaperPortfolioBroker(0, 0.0005, 0.0002))
    assert t2.broker.balances(cfg.symbols) == (cash, held)


def test_breakout_weights_follow_channels():
    from algo.trend_portfolio import breakout_weights

    closes = panel(n=1500)
    cfg = PortfolioConfig(strategy="breakout", symbols=list(closes.columns))
    w = breakout_weights({"close": closes, "high": closes, "low": closes}, cfg)
    assert (w >= 0).all() and (w <= 1 / len(w) + 1e-12).all()
    assert w["C0/USDT"] == pytest.approx(1 / 3)   # strong uptrend: all 6 systems long
    assert w["C1/USDT"] == 0                       # downtrend: flat


def test_htf_confirmation_blocks_entries_against_the_daily_structure():
    from algo.trend_portfolio import breakout_weights

    idx = pd.date_range("2024-01-01", periods=1200, freq="4h", tz="UTC")
    rng = np.random.default_rng(1)
    up = 100 * np.exp(np.cumsum(rng.normal(0.004, 0.01, len(idx))))
    closes = pd.DataFrame({"A/USDT": up, "B/USDT": up[::-1].copy()}, index=idx)
    bars = {"close": closes, "high": closes * 1.002, "low": closes * 0.998}
    plain = PortfolioConfig(strategy="breakout", symbols=list(closes.columns))
    conf = PortfolioConfig(strategy="breakout", symbols=list(closes.columns), confirm_tfs=("1d",), confirm_mode="gate")
    assert breakout_weights(bars, conf)["A/USDT"] > 0            # uptrend confirmed by daily higher lows
    assert breakout_weights(bars, conf)["B/USDT"] == 0
    assert (breakout_weights(bars, conf) <= breakout_weights(bars, plain) + 1e-12).all()  # confirmation only removes exposure


def test_htf_alignment_never_uses_an_unclosed_daily_candle():
    from algo.htf import align, resample

    idx = pd.date_range("2024-01-01", periods=60, freq="4h", tz="UTC")
    closes = pd.DataFrame({"X": np.arange(60, dtype=float)}, index=idx)
    d = resample({"close": closes, "high": closes, "low": closes}, "1d")["close"]
    a = align(d, idx, "1d")
    # The 4h bar labelled 20:00 closes at midnight: first bar that may see that day's close.
    assert a.loc["2024-01-01 20:00", "X"] == closes.loc["2024-01-01 20:00", "X"]
    assert np.isnan(a.loc["2024-01-01 16:00", "X"])


def test_top_k_holds_only_the_strongest_trending_coins():
    from algo.trend_portfolio import breakout_weights

    idx = pd.date_range("2024-01-01", periods=1200, freq="4h", tz="UTC")
    t = np.arange(len(idx))
    closes = pd.DataFrame({f"C{i}/USDT": 100 * np.exp(t * d) for i, d in enumerate((0.004, 0.003, 0.002, 0.001, -0.002))}, index=idx)
    bars = {"close": closes, "high": closes, "low": closes}
    cfg = PortfolioConfig(strategy="breakout", symbols=list(closes.columns), top_k=2, max_slots=4)
    w = breakout_weights(bars, cfg)
    assert set(w[w > 0].index) == {"C0/USDT", "C1/USDT"}       # the two strongest uptrends
    assert w.max() <= 1 / 4 + 1e-12 and w.sum() <= 1 + 1e-12   # risk dial caps each coin
    plain = breakout_weights(bars, PortfolioConfig(strategy="breakout", symbols=list(closes.columns)))
    assert plain["C3/USDT"] == pytest.approx(1 / 5)            # default: fixed 1/N sleeves
