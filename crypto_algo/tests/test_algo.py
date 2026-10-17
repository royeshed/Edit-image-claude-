import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.backtest import run_backtest
from algo.broker import PaperBroker
from algo.config import Config, RiskConfig
from algo.data import synthetic_ohlcv
from algo.indicators import atr, rsi
from algo.live import Trader
from algo.risk import Position, position_size, trail_stop


@pytest.fixture
def df():
    return synthetic_ohlcv(bars=3000, seed=1)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 100, dtype=float))
    assert rsi(up).dropna().eq(100).all()
    r = rsi(synthetic_ohlcv(500)["close"]).dropna()
    assert r.between(0, 100).all()


def test_atr_positive(df):
    assert (atr(df).dropna() > 0).all()


def test_position_size_risks_fixed_fraction():
    cfg = RiskConfig(risk_per_trade=0.01, atr_stop_mult=2, reward_risk=2, max_position_pct=1.0)
    qty, stop, tp = position_size(10_000, 100, 1, cfg)
    assert stop == 98 and tp == 104
    assert qty * (100 - stop) == pytest.approx(100)  # 1% of equity


def test_position_size_capped():
    cfg = RiskConfig(risk_per_trade=0.5, max_position_pct=0.5)
    qty, *_ = position_size(10_000, 100, 0.01, cfg)
    assert qty * 100 == pytest.approx(5_000)


def test_trailing_stop_only_moves_up():
    cfg = RiskConfig(trailing_atr_mult=2)
    p = Position(1, 100, 95, 120, highest=100)
    trail_stop(p, 110, 1, cfg)
    assert p.stop == 108
    trail_stop(p, 105, 1, cfg)
    assert p.stop == 108


def test_backtest_runs_and_is_consistent(df):
    res = run_backtest(df, Config())
    assert len(res.equity) == len(df)
    assert res.equity.iloc[-1] == pytest.approx(Config().initial_capital + sum(t.pnl for t in res.trades))
    for t in res.trades:
        assert t.exit_time >= t.entry_time


@pytest.mark.parametrize("name", ["ema_rsi", "donchian", "ict_mtf"])
def test_backtest_has_no_lookahead(df, name):
    """Changing future bars must not change decisions made before them."""
    cut = 2000
    cfg = Config()
    cfg.strategy.name = name
    cfg.strategy.bias_tf = cfg.strategy.setup_tf = "4h"
    full = run_backtest(df, cfg)
    altered = df.copy()
    altered.iloc[cut:, :4] *= 1.5
    part = run_backtest(altered, cfg)
    pd.testing.assert_series_equal(full.equity.iloc[: cut - 1], part.equity.iloc[: cut - 1])


def test_take_profit_can_be_disabled():
    qty, stop, tp = position_size(10_000, 100, 1, RiskConfig(reward_risk=0))
    assert qty > 0 and tp == float("inf")


def test_donchian_trades(df):
    cfg = Config()
    cfg.strategy.name = "donchian"
    assert run_backtest(df, cfg).stats["trades"] > 0


def test_fees_reduce_returns(df):
    cheap = run_backtest(df, Config())
    pricey_cfg = Config()
    pricey_cfg.costs.fee_rate = 0.01
    pricey = run_backtest(df, pricey_cfg)
    assert pricey.stats["final_equity"] < cheap.stats["final_equity"]


class FakeExchange:
    """Replays a DataFrame one candle at a time."""

    def __init__(self, df, start):
        self.df, self.i = df, start

    def fetch_ohlcv(self, symbol, timeframe, limit=500):
        window = self.df.iloc[max(0, self.i - limit + 1): self.i + 1]
        return [[int(ts.timestamp() * 1000), *row] for ts, row in zip(window.index, window.to_numpy())]

    def fetch_ticker(self, symbol):
        return {"last": float(self.df["close"].iloc[self.i])}


def test_paper_trader_end_to_end(df, tmp_path):
    cfg = Config(state_file=str(tmp_path / "state.json"))
    ex = FakeExchange(df, start=600)
    broker = PaperBroker(cfg.initial_capital, cfg.costs.fee_rate, cfg.costs.slippage)
    trader = Trader(cfg, ex, broker)
    entered = False
    for i in range(600, len(df)):
        ex.i = i
        trader.step()
        entered |= trader.position is not None
        cash, base = broker.balance()
        assert cash >= -1e-9 and base >= -1e-12
    assert entered
    # Restart from saved state keeps the same balances.
    restored = Trader(cfg, ex, PaperBroker(0, cfg.costs.fee_rate, cfg.costs.slippage))
    assert restored.broker.balance() == broker.balance()


def test_replay_matches_backtest_roughly():
    from algo.replay import replay

    df = synthetic_ohlcv(bars=1500, seed=3)
    cfg = Config()
    cfg.strategy.name = "donchian"
    eq = replay(df, cfg)
    bt = run_backtest(df, cfg).equity.loc[eq.index]
    assert abs(eq.iloc[-1] / bt.iloc[-1] - 1) < 0.05


def test_mtf_align_has_no_lookahead():
    from algo.mtf import align, resample

    df = synthetic_ohlcv(bars=600, freq="5min", seed=4)
    h1 = align(resample(df, "1h")["close"], df.index, "1h")
    # The 1h bar opening 10:00 closes at 11:00, i.e. with the 5m bar labelled 10:55.
    t = df.index[df.index.hour == 10][0].normalize() + pd.Timedelta(hours=10)
    assert pd.isna(h1.get(t + pd.Timedelta(minutes=50))) or h1[t + pd.Timedelta(minutes=50)] != df.loc[t + pd.Timedelta(minutes=55), "close"]
    assert h1[t + pd.Timedelta(minutes=55)] == df.loc[t + pd.Timedelta(minutes=55), "close"]
    altered = df.copy()
    altered.iloc[400:, :4] *= 2
    h1b = align(resample(altered, "1h")["close"], df.index, "1h")
    pd.testing.assert_series_equal(h1.iloc[:400], h1b.iloc[:400])


def test_short_trade_profits_when_price_falls():
    from algo.backtest import run_backtest as rb

    idx = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
    close = np.concatenate([np.full(250, 100.0), np.linspace(100, 80, 50)])
    df = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1.0}, index=idx)

    class Short:
        warmup = 0

        def compute(self, d):
            out = d.copy()
            out["atr"] = 1.0
            out["short_entry"] = False
            out.iloc[249, out.columns.get_loc("short_entry")] = True
            return out

    import algo.backtest as bt
    orig = bt.make_strategy
    bt.make_strategy = lambda cfg: Short()
    try:
        cfg = Config()
        cfg.risk.reward_risk = 0
        cfg.risk.trailing_atr_mult = 0
        cfg.risk.atr_stop_mult = 5
        res = rb(df, cfg)
    finally:
        bt.make_strategy = orig
    assert len(res.trades) == 1 and res.trades[0].side == -1
    assert res.trades[0].pnl > 0 and res.equity.iloc[-1] > cfg.initial_capital


def test_limit_entry_fills_only_when_price_trades_through():
    import algo.backtest as bt

    idx = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
    close = np.full(60, 100.0)
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0}, index=idx)
    df.iloc[20, df.columns.get_loc("low")] = 97.0  # only bar 20 reaches the 97.5 limit

    def strat(limit):
        class S:
            warmup = 0

            def compute(self, d):
                out = d.copy()
                out["atr"] = 1.0
                out["long_entry"] = False
                out.iloc[10, out.columns.get_loc("long_entry")] = True
                out["limit_long"], out["stop_long"] = limit, 90.0
                out.attrs["limit_bars"] = 15
                return out
        return S()

    orig = bt.make_strategy
    try:
        cfg = Config()
        cfg.risk.reward_risk = 0
        cfg.risk.trailing_atr_mult = 0
        bt.make_strategy = lambda c: strat(97.5)
        hit = bt.run_backtest(df, cfg)
        bt.make_strategy = lambda c: strat(96.0)  # never reached
        miss = bt.run_backtest(df, cfg)
    finally:
        bt.make_strategy = orig
    assert len(hit.trades) == 1 and hit.trades[0].entry_price == 97.5 and hit.trades[0].entry_time == idx[20]
    assert miss.trades == []
