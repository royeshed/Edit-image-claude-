"""Execution backends: a paper broker that simulates fills, and a live broker over ccxt."""
import logging

log = logging.getLogger(__name__)


class PaperBroker:
    def __init__(self, cash: float, fee_rate: float, slippage: float, base_qty: float = 0.0):
        self.cash = cash
        self.base_qty = base_qty
        self.fee_rate = fee_rate
        self.slippage = slippage

    def balance(self) -> tuple[float, float]:
        return self.cash, self.base_qty

    def market_buy(self, qty: float, price: float) -> tuple[float, float]:
        fill = price * (1 + self.slippage)
        qty = min(qty, self.cash / (fill * (1 + self.fee_rate)))
        self.cash -= qty * fill * (1 + self.fee_rate)
        self.base_qty += qty
        log.info("PAPER BUY %.6f @ %.2f", qty, fill)
        return qty, fill

    def market_sell(self, qty: float, price: float) -> tuple[float, float]:
        fill = price * (1 - self.slippage)
        qty = min(qty, self.base_qty)
        self.cash += qty * fill * (1 - self.fee_rate)
        self.base_qty -= qty
        log.info("PAPER SELL %.6f @ %.2f", qty, fill)
        return qty, fill


class LiveBroker:
    """Places real market orders. Only constructed when the user explicitly opts in."""

    def __init__(self, exchange, symbol: str):
        self.ex = exchange
        self.symbol = symbol
        self.ex.load_markets()
        market = self.ex.market(symbol)
        self.base, self.quote = market["base"], market["quote"]

    def balance(self) -> tuple[float, float]:
        bal = self.ex.fetch_balance()
        return float(bal["free"].get(self.quote, 0)), float(bal["free"].get(self.base, 0))

    def _fill(self, order: dict, fallback_price: float) -> tuple[float, float]:
        order = self.ex.fetch_order(order["id"], self.symbol) if order.get("filled") is None else order
        return float(order.get("filled") or 0), float(order.get("average") or fallback_price)

    def market_buy(self, qty: float, price: float) -> tuple[float, float]:
        qty = float(self.ex.amount_to_precision(self.symbol, qty))
        order = self.ex.create_market_buy_order(self.symbol, qty)
        log.warning("LIVE BUY %s %s -> order %s", qty, self.symbol, order.get("id"))
        return self._fill(order, price)

    def market_sell(self, qty: float, price: float) -> tuple[float, float]:
        qty = float(self.ex.amount_to_precision(self.symbol, qty))
        order = self.ex.create_market_sell_order(self.symbol, qty)
        log.warning("LIVE SELL %s %s -> order %s", qty, self.symbol, order.get("id"))
        return self._fill(order, price)
