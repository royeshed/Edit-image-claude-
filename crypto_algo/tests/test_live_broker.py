"""LiveBroker + Trader against a fake ccxt exchange (no network, no real orders)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.broker import LiveBroker, OrderRejected
from algo.config import Config
from algo.data import synthetic_ohlcv
from algo.live import Trader


class FakeCcxt:
    def __init__(self, df, i, quote=10_000.0):
        self.df, self.i = df, i
        self.free = {"USDT": quote, "BTC": 0.0}
        self.orders: dict[str, dict] = {}
        self.fail_next_buy = 0
        self._id = 0

    # market data
    def load_markets(self):
        pass

    def market(self, symbol):
        return {"base": "BTC", "quote": "USDT", "limits": {"amount": {"min": 1e-5}, "cost": {"min": 10}}}

    def fetch_ohlcv(self, symbol, timeframe, limit=500):
        w = self.df.iloc[max(0, self.i - limit + 1): self.i + 1]
        return [[int(ts.timestamp() * 1000), *row] for ts, row in zip(w.index, w.to_numpy())]

    def fetch_ticker(self, symbol):
        return {"last": self.price}

    @property
    def price(self):
        return float(self.df["close"].iloc[self.i])

    # account / orders
    def amount_to_precision(self, symbol, qty):
        return f"{qty:.5f}"

    def price_to_precision(self, symbol, p):
        return f"{p:.2f}"

    def fetch_balance(self):
        return {"free": dict(self.free)}

    def _new(self, **o):
        self._id += 1
        o["id"] = str(self._id)
        self.orders[o["id"]] = o
        return o

    def create_market_buy_order(self, symbol, qty):
        if self.fail_next_buy:
            self.fail_next_buy -= 1
            raise ConnectionError("exchange timeout")
        self.free["USDT"] -= qty * self.price
        self.free["BTC"] += qty
        return self._new(side="buy", filled=qty, average=self.price, status="closed")

    def create_market_sell_order(self, symbol, qty):
        assert qty <= self.free["BTC"] + 1e-9, "selling more than we hold"
        self.free["BTC"] -= qty
        self.free["USDT"] += qty * self.price
        return self._new(side="sell", filled=qty, average=self.price, status="closed")

    def create_order(self, symbol, type_, side, qty, price, params):
        return self._new(side=side, type=type_, qty=qty, limit=price, trigger=params["stopLossPrice"], status="open")

    def cancel_order(self, oid, symbol):
        if self.orders[oid]["status"] != "open":
            raise RuntimeError("order not open")
        self.orders[oid]["status"] = "canceled"

    def fetch_order(self, oid, symbol):
        return self.orders[oid]

    def trigger_stop(self, oid):
        o = self.orders[oid]
        self.free["BTC"] -= o["qty"]
        self.free["USDT"] += o["qty"] * o["trigger"]
        o["status"] = "closed"

    def open_stops(self):
        return [o for o in self.orders.values() if o.get("trigger") and o["status"] == "open"]


@pytest.fixture
def setup(tmp_path):
    df = synthetic_ohlcv(bars=3000, seed=1)
    cfg = Config(state_file=str(tmp_path / "state.json"))
    cfg.strategy.name = "donchian"
    ex = FakeCcxt(df, 600)
    return cfg, ex


def run_until(trader, ex, cond, end=3000):
    while ex.i < end - 1:
        ex.i += 1
        trader.step()
        if cond():
            return True
    return False


def test_entry_places_exchange_stop_and_trailing_moves_it(setup):
    cfg, ex = setup
    t = Trader(cfg, ex, LiveBroker(ex, cfg.symbol))
    assert run_until(t, ex, lambda: t.position is not None)
    stops = ex.open_stops()
    assert len(stops) == 1 and stops[0]["id"] == t.position.stop_order_id
    assert float(stops[0]["trigger"]) == pytest.approx(t.position.stop, rel=1e-4)
    first = t.position.stop_order_id
    assert run_until(t, ex, lambda: t.position is None or t.position.stop_order_id != first)
    assert len(ex.open_stops()) <= 1  # the old stop was cancelled, never two at once


def test_exit_cancels_stop_and_sells_everything(setup):
    cfg, ex = setup
    t = Trader(cfg, ex, LiveBroker(ex, cfg.symbol))
    for _ in range(5):
        assert run_until(t, ex, lambda: t.position is not None)
        assert run_until(t, ex, lambda: t.position is None)
        assert ex.open_stops() == []
        assert ex.free["BTC"] == pytest.approx(0, abs=1e-5)


def test_stop_filled_on_exchange_while_bot_down(setup):
    cfg, ex = setup
    t = Trader(cfg, ex, LiveBroker(ex, cfg.symbol))
    assert run_until(t, ex, lambda: t.position is not None)
    ex.trigger_stop(t.position.stop_order_id)  # bot is "down"; exchange stop fires
    restarted = Trader(cfg, ex, LiveBroker(ex, cfg.symbol))  # restored from state file
    assert restarted.position is not None
    restarted.step()
    assert restarted.position is None


def test_transient_entry_failure_is_retried(setup):
    cfg, ex = setup
    t = Trader(cfg, ex, LiveBroker(ex, cfg.symbol))
    ex.fail_next_buy = 1
    assert run_until(t, ex, lambda: t.pending_entry is not None)
    assert t.position is None  # first attempt failed, signal kept
    t.step()  # next poll, same candle
    assert t.position is not None and t.pending_entry is None


def test_below_minimum_order_is_rejected_not_retried(setup):
    cfg, ex = setup
    ex.free["USDT"] = 5.0  # below the 10 USDT minimum
    broker = LiveBroker(ex, cfg.symbol)
    with pytest.raises(OrderRejected):
        broker.market_buy(0.0001, ex.price)
    t = Trader(cfg, ex, broker)
    run_until(t, ex, lambda: False, end=1500)
    assert t.position is None and t.pending_entry is None
    assert not [o for o in ex.orders.values() if o["side"] == "buy"]


def test_fee_taken_in_base_coin(setup):
    cfg, ex = setup
    orig = ex.create_market_buy_order

    def buy_with_fee(symbol, qty):
        o = orig(symbol, qty)
        ex.free["BTC"] -= qty * 0.001  # 0.1% fee deducted from the coin received
        return o

    ex.create_market_buy_order = buy_with_fee
    t = Trader(cfg, ex, LiveBroker(ex, cfg.symbol))
    assert run_until(t, ex, lambda: t.position is not None)
    assert t.position.qty <= ex.free["BTC"] + 1e-9
    assert float(ex.open_stops()[0]["qty"]) <= ex.free["BTC"] + 1e-9
    assert run_until(t, ex, lambda: t.position is None)  # exits without overselling
