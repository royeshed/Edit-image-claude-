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


def test_backtest_has_no_lookahead(df):
    """Changing future bars must not change decisions made before them."""
    cut = 2000
    full = run_backtest(df, Config())
    altered = df.copy()
    altered.iloc[cut:, :4] *= 1.5
    part = run_backtest(altered, Config())
    pd.testing.assert_series_equal(full.equity.iloc[: cut - 1], part.equity.iloc[: cut - 1])


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
