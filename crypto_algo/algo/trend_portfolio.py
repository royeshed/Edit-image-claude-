"""Multi-coin, long-only spot portfolios selected by the research in research/.

Two strategies, both rebalanced with market orders to per-coin weights
(weight = size / number_of_coins, i.e. a fraction of total equity):

"momentum" (1h candles, once a day at the close of the `decide_hour` UTC candle)
    signal = mean of sign(return over L) for L in base * (1/4, 1/2, 1, 2, 4), floored at 0
    size   = signal * target_vol / realised_vol(30 days), capped at `cap`

"breakout" (4h candles, at every candle close)
    size   = share of the Donchian systems (entry N-bar high, exit M-bar low) that are long,
             for N in `entries` x M in `exits` -> 0, 1/6, ..., 1

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
    state_file: str = "portfolio_state.json"

    @property
    def timeframe(self) -> str:
        return "4h" if self.strategy == "breakout" else "1h"

    @property
    def history_hours(self) -> int:
        if self.strategy == "breakout":
            return (max(self.entries) * 4 + 50) * 4  # enough 4h bars for the channel state to settle
        return self.base_hours * 4 + 24


def ensemble_signal(close: pd.DataFrame, base: int, mults=(0.25, 0.5, 1, 2, 4)) -> pd.DataFrame:
    return sum(np.sign(close / close.shift(int(base * m)) - 1) for m in mults) / len(mults)


def breakout_state(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, n: int, m: int) -> pd.DataFrame:
    """1 from a close above the prior n-bar high until a close below the prior m-bar low."""
    entry = close > high.rolling(n).max().shift()
    exit_ = close < low.rolling(m).min().shift()
    s = pd.DataFrame(np.where(exit_, 0.0, np.where(entry, 1.0, np.nan)), index=close.index, columns=close.columns)
    return s.ffill().fillna(0.0)


def breakout_weights(bars: dict[str, pd.DataFrame], cfg: PortfolioConfig) -> pd.Series:
    """Weights from 4h close/high/low panels whose last row is the just-closed candle."""
    c, h, l = bars["close"], bars["high"], bars["low"]
    size = sum(breakout_state(c, h, l, n, m) for n in cfg.entries for m in cfg.exits) / (len(cfg.entries) * len(cfg.exits))
    live = c.iloc[-1].notna()
    return (size.iloc[-1] / live.sum()).where(live, 0.0)


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
