"""Multi-coin, long-only spot portfolios selected by the research in research/.

Two strategies, both rebalanced with market orders to per-coin weights
(weight = size / number_of_coins, i.e. a fraction of total equity):

"momentum" (1h candles, once a day at the close of the `decide_hour` UTC candle)
    signal = mean of sign(return over L) for L in base * (1/4, 1/2, 1, 2, 4), floored at 0
    size   = signal * target_vol / realised_vol(30 days), capped at `cap`

"breakout" (4h candles, at every candle close)
    size   = share of the Donchian systems (entry N-bar high, exit M-bar low) that are long,
             for N in `entries` x M in `exits` -> 0, 1/6, ..., 1
    Optional higher-timeframe confirmation (`confirm_tfs`, e.g. ["1d"] or ["1d", "1w"]): a system
    may only enter while `confirm` holds on every listed timeframe (closed candles only); with
    `confirm_mode` "gate" it also exits when the confirmation fails.
    Capital allocation: with `top_k` > 0 only the k coins with the best `rank_days` return among
    those whose signal is on are held; weights are size / max(sum of held sizes, `max_slots`), so
    `max_slots` is the risk dial (each coin gets at most 1/max_slots of equity). With top_k = 0 and
    max_slots = 0 every coin keeps a fixed 1/N sleeve.

There are no stops: risk is controlled by sizing and by going to cash when trends end.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .broker import OrderRejected
from .data import fetch_ohlcv
from .htf import confirmation, confirmed_breakouts

log = logging.getLogger(__name__)

HOURS_PER_YEAR = 8760


@dataclass
class PortfolioConfig:
    strategy: str = "momentum"     # "momentum" or "breakout"
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
                                                         "DOGE/USDT", "SOL/USDT", "AVAX/USDT"])
    base_hours: int = 480          # 20 days; the ensemble uses 5, 10, 20, 40 and 80 days
    decide_hour: int = 13          # UTC hour whose candle close triggers the daily decision
    target_vol: float = 0.5        # annualised volatility each coin's sleeve is scaled to
    cap: float = 1.0               # max sleeve size (1.0 = no leverage, spot-compatible)
    vol_hours: int = 720           # realised-volatility window (30 days)
    min_trade_usd: float = 10.0    # skip rebalancing trades smaller than this
    entries: tuple = (50, 100, 200)  # breakout: entry channel lengths (4h bars)
    exits: tuple = (10, 20)          # breakout: exit channel lengths (4h bars)
    confirm_tfs: tuple = ()          # breakout: higher-timeframe confirmation, e.g. ("1d",) or ("1d", "1w"); () = off
    confirm: str = "higher-lows"     # higher-lows | close>ema20 | close>ema50 | ema10>ema30 | donchian20/10 | momentum
    confirm_mode: str = "entry"      # "entry" (filter entries) or "gate" (also exit when confirmation fails)
    top_k: int = 0                   # breakout: hold only the k strongest coins in a trend (0 = all)
    rank_days: int = 14              # breakout: momentum window used to rank coins for top_k
    max_slots: float = 0             # breakout: risk dial, max weight per coin = 1/max_slots (0 = number of coins)
    state_file: str = "portfolio_state.json"

    @property
    def timeframe(self) -> str:
        return "4h" if self.strategy == "breakout" else "1h"

    @property
    def history_hours(self) -> int:
        if self.strategy == "breakout":
            hours = (max(self.entries) * 4 + 50) * 4  # enough 4h bars for the channel state to settle
            if "1w" in self.confirm_tfs:
                hours = max(hours, 24 * 7 * 60)       # ~60 weekly candles for weekly swing structure
            return hours
        return self.base_hours * 4 + 24


def ensemble_signal(close: pd.DataFrame, base: int, mults=(0.25, 0.5, 1, 2, 4)) -> pd.DataFrame:
    return sum(np.sign(close / close.shift(int(base * m)) - 1) for m in mults) / len(mults)


def breakout_weights(bars: dict[str, pd.DataFrame], cfg: PortfolioConfig) -> pd.Series:
    """Weights from 4h close/high/low panels whose last row is the just-closed candle."""
    c = bars["close"]
    systems = [(n, m) for n in cfg.entries for m in cfg.exits]
    confirm = confirmation(bars, c.index, cfg.confirm_tfs, cfg.confirm) if cfg.confirm_tfs else None
    size = confirmed_breakouts(bars, systems, confirm, cfg.confirm_mode).iloc[-1]
    live = c.iloc[-1].notna()
    size = size.where(live, 0.0)
    if cfg.top_k:
        bars_back = cfg.rank_days * 6  # 4h bars
        mom = (c.iloc[-1] / c.iloc[-1 - bars_back] - 1).where(size > 0) if len(c) > bars_back else size * np.nan
        keep = mom.rank(ascending=False, method="first") <= cfg.top_k
        size = size.where(keep, 0.0)
    slots = cfg.max_slots or live.sum()
    return size / max(size.sum(), slots)


def target_weights(closes: pd.DataFrame, cfg: PortfolioConfig) -> pd.Series:
    """Weights (fraction of equity) from hourly closes whose last row is the decision candle."""
    signal = ensemble_signal(closes, cfg.base_hours).clip(lower=0).iloc[-1]
    vol = closes.pct_change(fill_method=None).iloc[-cfg.vol_hours:].std() * np.sqrt(HOURS_PER_YEAR)
    size = (signal * cfg.target_vol / vol).clip(lower=0, upper=cfg.cap).fillna(0)
    live = closes.iloc[-1].notna()
    return (size / live.sum()).where(live, 0.0)


class PaperPortfolioBroker:
    def __init__(self, cash: float, fee_rate: float, slippage: float):
        self.cash, self.fee_rate, self.slippage = cash, fee_rate, slippage
        self.holdings: dict[str, float] = {}

    def balances(self, symbols) -> tuple[float, dict[str, float]]:
        return self.cash, {s: self.holdings.get(s, 0.0) for s in symbols}

    def market_buy(self, symbol: str, qty: float, price: float) -> None:
        fill = price * (1 + self.slippage)
        qty = min(qty, self.cash / (fill * (1 + self.fee_rate)))
        self.cash -= qty * fill * (1 + self.fee_rate)
        self.holdings[symbol] = self.holdings.get(symbol, 0.0) + qty
        log.info("PAPER BUY  %-10s %.6f @ %.4f", symbol, qty, fill)

    def market_sell(self, symbol: str, qty: float, price: float) -> None:
        fill = price * (1 - self.slippage)
        qty = min(qty, self.holdings.get(symbol, 0.0))
        self.cash += qty * fill * (1 - self.fee_rate)
        self.holdings[symbol] = self.holdings.get(symbol, 0.0) - qty
        log.info("PAPER SELL %-10s %.6f @ %.4f", symbol, qty, fill)


class LivePortfolioBroker:
    """Spot market orders on a ccxt exchange. Only constructed when the user explicitly opts in."""

    def __init__(self, exchange, quote: str = "USDT"):
        self.ex, self.quote = exchange, quote
        self.ex.load_markets()

    def balances(self, symbols) -> tuple[float, dict[str, float]]:
        free = self.ex.fetch_balance()["free"]
        return float(free.get(self.quote) or 0), {s: float(free.get(self.ex.market(s)["base"]) or 0) for s in symbols}

    def _qty(self, symbol: str, qty: float, price: float) -> float:
        qty = float(self.ex.amount_to_precision(symbol, qty))
        limits = self.ex.market(symbol).get("limits", {})
        if qty <= 0 or qty < ((limits.get("amount") or {}).get("min") or 0) or qty * price < ((limits.get("cost") or {}).get("min") or 0):
            raise OrderRejected(f"{symbol} {qty} below exchange minimum")
        return qty

    def market_buy(self, symbol: str, qty: float, price: float) -> None:
        order = self.ex.create_market_buy_order(symbol, self._qty(symbol, qty, price))
        log.warning("LIVE BUY  %s -> %s", symbol, order.get("id"))

    def market_sell(self, symbol: str, qty: float, price: float) -> None:
        order = self.ex.create_market_sell_order(symbol, self._qty(symbol, qty, price))
        log.warning("LIVE SELL %s -> %s", symbol, order.get("id"))


class PortfolioTrader:
    def __init__(self, cfg: PortfolioConfig, exchange, broker):
        self.cfg, self.ex, self.broker = cfg, exchange, broker
        self.state_path = Path(cfg.state_file)
        self.last_decision: str | None = None
        if self.state_path.exists():
            s = json.loads(self.state_path.read_text())
            self.last_decision = s.get("last_decision")
            if isinstance(broker, PaperPortfolioBroker) and "paper_cash" in s:
                broker.cash, broker.holdings = s["paper_cash"], s["paper_holdings"]

    def _save(self) -> None:
        s = {"last_decision": self.last_decision}
        if isinstance(self.broker, PaperPortfolioBroker):
            s["paper_cash"], s["paper_holdings"] = self.broker.cash, self.broker.holdings
        self.state_path.write_text(json.dumps(s, indent=2))

    def bars(self) -> dict[str, pd.DataFrame]:
        """close/high/low panels of closed candles for all symbols."""
        since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=self.cfg.history_hours + 8)
        frames = {s: fetch_ohlcv(self.ex, s, self.cfg.timeframe, since) for s in self.cfg.symbols}
        return {k: pd.DataFrame({s: f[k] for s, f in frames.items()}).iloc[:-1]  # last candle is still forming
                for k in ("close", "high", "low")}

    def rebalance(self, weights: pd.Series, prices: pd.Series) -> None:
        cash, held = self.broker.balances(self.cfg.symbols)
        equity = cash + sum(held[s] * prices[s] for s in self.cfg.symbols)
        delta = {s: weights[s] * equity / prices[s] - held[s] for s in self.cfg.symbols}
        log.info("Rebalance: equity %.2f, targets %s", equity, {s: f"{w:.1%}" for s, w in weights.items() if w > 0})
        # Sells first so their proceeds fund the buys.
        for s in sorted(delta, key=lambda k: delta[k]):
            notional = abs(delta[s]) * prices[s]
            if notional < self.cfg.min_trade_usd:
                continue
            try:
                (self.broker.market_sell if delta[s] < 0 else self.broker.market_buy)(s, abs(delta[s]), prices[s])
            except OrderRejected as e:
                log.warning("Skipped %s: %s", s, e)

    def step(self) -> bool:
        bars = self.bars()
        closes = bars["close"]
        last = closes.index[-1]
        if str(last) == self.last_decision:
            return False
        if self.cfg.strategy == "breakout":
            weights = breakout_weights(bars, self.cfg)
        else:
            if last.hour != self.cfg.decide_hour:
                return False
            weights = target_weights(closes, self.cfg)
        self.rebalance(weights, closes.iloc[-1])
        self.last_decision = str(last)
        self._save()
        return True

    def run(self, poll_seconds: int = 300) -> None:
        when = "every 4h candle close" if self.cfg.strategy == "breakout" else f"daily at the {self.cfg.decide_hour:02d}:00 UTC candle close"
        log.info("%s portfolio on %d coins, rebalancing %s", self.cfg.strategy, len(self.cfg.symbols), when)
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                raise
            except Exception:
                log.exception("Step failed; retrying next poll")
            time.sleep(poll_seconds)
