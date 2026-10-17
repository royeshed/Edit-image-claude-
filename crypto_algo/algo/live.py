"""Live / paper trading loop.

Every `poll_seconds` it checks the latest price against the stop and take-profit.
When a new candle closes it recomputes the strategy on closed candles only and acts
on the signal. State is persisted so the bot can be restarted without losing track
of an open position.
"""
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import Config
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
        if hasattr(self.broker, "cash") and "paper_cash" in s:
            self.broker.cash, self.broker.base_qty = s["paper_cash"], s["paper_base"]
        log.info("Restored state: position=%s", self.position)

    def _save_state(self) -> None:
        s = {
            "position": asdict(self.position) if self.position else None,
            "last_candle": self.last_candle,
            "day": self.day,
            "day_start_equity": self.day_start_equity,
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
        today = pd.Timestamp.now(tz="UTC").date().isoformat()
        equity = self._equity(price)
        if today != self.day:
            self.day, self.day_start_equity = today, equity
        return equity <= self.day_start_equity * (1 - self.cfg.risk.max_daily_loss)

    def _exit(self, price: float, reason: str) -> None:
        qty, fill = self.broker.market_sell(self.position.qty, price)
        pnl = qty * (fill - self.position.entry_price)
        log.info("EXIT (%s) qty=%.6f fill=%.2f pnl≈%.2f", reason, qty, fill, pnl)
        self.position = None

    # ---- main loop ---------------------------------------------------------
    def step(self) -> None:
        price = float(self.ex.fetch_ticker(self.cfg.symbol)["last"])

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
            if self.position:
                trail_stop(self.position, sig["high"], sig["atr"], self.cfg.risk)
                if sig["exit"]:
                    self._exit(price, "signal")
            elif sig["entry"]:
                if self._halted(price):
                    log.warning("Daily loss limit reached; skipping entry")
                else:
                    cash, _ = self.broker.balance()
                    qty, stop, tp = position_size(cash, price, sig["atr"], self.cfg.risk)
                    if qty > 0:
                        filled, fill = self.broker.market_buy(qty, price)
                        if filled > 0:
                            self.position = Position(filled, fill, stop, tp, candle, highest=fill)
                            log.info("ENTRY qty=%.6f fill=%.2f stop=%.2f tp=%.2f", filled, fill, stop, tp)

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
