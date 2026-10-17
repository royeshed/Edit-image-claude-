"""Multi-coin trend portfolio: the strategy selected by the research in research/.

Once a day, at the close of the `decide_hour` UTC candle, every coin gets a weight:

    signal = mean of sign(return over L) for L in base * (1/4, 1/2, 1, 2, 4), floored at 0 (long only)
    size   = signal * target_vol / realised_vol(30 days), capped at `cap`
    weight = size / number_of_coins           (fraction of total equity)

and the account is rebalanced to those weights with market orders. There are no
stops: risk is controlled by volatility scaling and by going to cash when trends end.
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
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
                                                         "DOGE/USDT", "SOL/USDT", "AVAX/USDT"])
    base_hours: int = 480          # 20 days; the ensemble uses 5, 10, 20, 40 and 80 days
    decide_hour: int = 13          # UTC hour whose candle close triggers the daily decision
    target_vol: float = 0.5        # annualised volatility each coin's sleeve is scaled to
    cap: float = 1.0               # max sleeve size (1.0 = no leverage, spot-compatible)
    vol_hours: int = 720           # realised-volatility window (30 days)
    min_trade_usd: float = 10.0    # skip rebalancing trades smaller than this
    state_file: str = "portfolio_state.json"

    @property
    def history_hours(self) -> int:
        return self.base_hours * 4 + 24


def ensemble_signal(close: pd.DataFrame, base: int, mults=(0.25, 0.5, 1, 2, 4)) -> pd.DataFrame:
    return sum(np.sign(close / close.shift(int(base * m)) - 1) for m in mults) / len(mults)


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

    def closes(self) -> pd.DataFrame:
        """Hourly closes of closed candles for all symbols."""
        since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=self.cfg.history_hours + 2)
        cols = {s: fetch_ohlcv(self.ex, s, "1h", since)["close"] for s in self.cfg.symbols}
        df = pd.DataFrame(cols)
        return df.iloc[:-1]  # last candle is still forming

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
        closes = self.closes()
        last = closes.index[-1]
        if last.hour != self.cfg.decide_hour or str(last) == self.last_decision:
            return False
        self.rebalance(target_weights(closes, self.cfg), closes.iloc[-1])
        self.last_decision = str(last)
        self._save()
        return True

    def run(self, poll_seconds: int = 300) -> None:
        log.info("Trend portfolio on %d coins, deciding daily at %02d:00 UTC candle close", len(self.cfg.symbols), self.cfg.decide_hour)
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                raise
            except Exception:
                log.exception("Step failed; retrying next poll")
            time.sleep(poll_seconds)
