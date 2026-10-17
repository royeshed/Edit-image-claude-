"""Live / paper trading loop.

Every `poll_seconds` it checks the latest price against the stop and take-profit.
When a new candle closes it recomputes the strategy on closed candles only and acts
on the signal. In live mode the stop also rests on the exchange, so the position is
protected while the bot is down. State is persisted so the bot can be restarted
without losing track of an open position.
"""
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import Config
from .broker import OrderRejected
from .data import to_frame
from .risk import Position, position_size, trail_stop
from .strategy import make_strategy

log = logging.getLogger(__name__)


class Trader:
    def __init__(self, cfg: Config, exchange, broker):
        self.cfg = cfg
        self.ex = exchange
        self.broker = broker
        self.strategy = make_strategy(cfg.strategy)
        self.state_path = Path(cfg.state_file)
        self.position: Position | None = None
        self.last_candle: str | None = None
        self.day: str | None = None
        self.day_start_equity: float | None = None
        self.pending_entry: dict | None = None  # entry signal not yet filled (retried within its candle)
        self._load_state()

    # ---- state -------------------------------------------------------------
    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        s = json.loads(self.state_path.read_text())
        self.position = Position(**s["position"]) if s.get("position") else None
        self.last_candle = s.get("last_candle")
        self.day = s.get("day")
        self.day_start_equity = s.get("day_start_equity")
        self.pending_entry = s.get("pending_entry")
        if hasattr(self.broker, "cash") and "paper_cash" in s:
            self.broker.cash, self.broker.base_qty = s["paper_cash"], s["paper_base"]
        log.info("Restored state: position=%s", self.position)

    def _save_state(self) -> None:
        s = {
            "position": asdict(self.position) if self.position else None,
            "last_candle": self.last_candle,
            "day": self.day,
            "day_start_equity": self.day_start_equity,
            "pending_entry": self.pending_entry,
        }
        if hasattr(self.broker, "cash"):
            s["paper_cash"], s["paper_base"] = self.broker.cash, self.broker.base_qty
        self.state_path.write_text(json.dumps(s, indent=2, default=str))

    # ---- helpers -----------------------------------------------------------
    def _closed_candles(self) -> pd.DataFrame:
        limit = self.strategy.warmup * 3
        df = to_frame(self.ex.fetch_ohlcv(self.cfg.symbol, self.cfg.timeframe, limit=limit))
        return df.iloc[:-1]  # the last candle is still forming

    def _equity(self, price: float) -> float:
        cash, base = self.broker.balance()
        return cash + base * price

    def _halted(self, price: float) -> bool:
        # Day of the latest closed candle (equals the wall-clock day live, and keeps replays correct).
        now = pd.Timestamp(self.last_candle) if self.last_candle else pd.Timestamp.now(tz="UTC")
        today = now.date().isoformat()
        equity = self._equity(price)
        if today != self.day:
            self.day, self.day_start_equity = today, equity
        return equity <= self.day_start_equity * (1 - self.cfg.risk.max_daily_loss)

    def _exit(self, price: float, reason: str) -> None:
        if self.position.stop_order_id:
            self.broker.cancel_order(self.position.stop_order_id)
            self.position.stop_order_id = None
        _, base = self.broker.balance()
        try:
            qty, fill = self.broker.market_sell(min(self.position.qty, base), price)
        except OrderRejected as e:  # nothing (or only dust) left, e.g. the exchange stop already sold it
            log.warning("EXIT (%s): nothing to sell (%s); clearing position", reason, e)
            self.position = None
            return
        pnl = qty * (fill - self.position.entry_price)
        log.info("EXIT (%s) qty=%.6f fill=%.2f pnl≈%.2f", reason, qty, fill, pnl)
        self.position = None

    def _sync_exchange_stop(self) -> None:
        """Detect a stop that filled on the exchange (e.g. while the bot was down)."""
        if self.position and self.position.stop_order_id and self.broker.order_filled(self.position.stop_order_id):
            log.warning("Exchange stop %s filled; position closed", self.position.stop_order_id)
            self.position = None

    def _move_exchange_stop(self) -> None:
        self._sync_exchange_stop()  # don't replace a stop that has just filled
        if not self.position:
            return
        if self.position.stop_order_id:
            self.broker.cancel_order(self.position.stop_order_id)
        self.position.stop_order_id = self.broker.place_stop(self.position.qty, self.position.stop)

    def _try_entry(self, price: float) -> None:
        if self._halted(price):
            log.warning("Daily loss limit reached; skipping entry")
            self.pending_entry = None
            return
        cash, _ = self.broker.balance()
        qty, stop, tp = position_size(cash, price, self.pending_entry["atr"], self.cfg.risk)
        try:
            if qty <= 0:
                raise OrderRejected("position size is zero")
            filled, fill = self.broker.market_buy(qty, price)
        except OrderRejected as e:
            log.warning("Entry skipped: %s", e)
            self.pending_entry = None
            return
        except Exception:
            log.exception("Entry order failed; will retry next poll")
            return
        self.pending_entry = None
        if filled > 0:
            # Some exchanges take the fee out of the coin bought; track what we actually hold.
            _, base = self.broker.balance()
            held = min(filled, base) if base > 0 else filled
            self.position = Position(held, fill, stop, tp, self.last_candle, highest=fill)
            self.position.stop_order_id = self.broker.place_stop(held, stop)
            log.info("ENTRY qty=%.6f fill=%.2f stop=%.2f tp=%.2f", held, fill, stop, tp)

    # ---- main loop ---------------------------------------------------------
    def step(self) -> None:
        price = float(self.ex.fetch_ticker(self.cfg.symbol)["last"])

        self._sync_exchange_stop()
        if self.position:
            if price <= self.position.stop:
                self._exit(price, "stop")
            elif price >= self.position.take_profit:
                self._exit(price, "take_profit")

        df = self._closed_candles()
        candle = str(df.index[-1])
        if candle != self.last_candle:
            self.last_candle = candle
            sig = self.strategy.compute(df).iloc[-1]
            log.info("Candle %s close=%.2f rsi=%.1f entry=%s exit=%s",
                     candle, sig["close"], sig["rsi"], sig["entry"], sig["exit"])
            self.pending_entry = None  # a signal is only valid for the candle that produced it
            if self.position:
                old_stop = self.position.stop
                trail_stop(self.position, sig["high"], sig["atr"], self.cfg.risk)
                if sig["exit"]:
                    self._exit(price, "signal")
                elif self.position.stop > old_stop:
                    self._move_exchange_stop()
            elif sig["entry"]:
                self.pending_entry = {"candle": candle, "atr": float(sig["atr"])}

        if self.pending_entry and self.position is None:
            self._try_entry(price)

        self._halted(price)  # roll the day even when idle
        self._save_state()
        log.debug("price=%.2f equity=%.2f position=%s", price, self._equity(price), self.position)

    def run(self) -> None:
        log.info("Trading %s %s with %s", self.cfg.symbol, self.cfg.timeframe, type(self.broker).__name__)
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                raise
            except Exception:
                log.exception("Step failed; retrying next poll")
            time.sleep(self.cfg.poll_seconds)
