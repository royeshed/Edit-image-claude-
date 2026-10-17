"""Execution backends: a paper broker that simulates fills, and a live broker over ccxt."""
import logging

log = logging.getLogger(__name__)


class OrderRejected(Exception):
    """The order can never succeed as sized (below exchange minimums, etc.). Don't retry."""


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

    # Paper stops are enforced by the trading loop, not resting orders.
    def place_stop(self, qty: float, stop: float) -> str | None:
        return None

    def cancel_order(self, order_id: str) -> None:
        pass

    def order_filled(self, order_id: str) -> bool:
        return False


class LiveBroker:
    """Places real orders. Only constructed when the user explicitly opts in."""

    STOP_LIMIT_BUFFER = 0.01  # stop-limit sells may fill down to 1% below the trigger

    def __init__(self, exchange, symbol: str):
        self.ex = exchange
        self.symbol = symbol
        self.ex.load_markets()
        self.market = self.ex.market(symbol)
        self.base, self.quote = self.market["base"], self.market["quote"]

    def balance(self) -> tuple[float, float]:
        bal = self.ex.fetch_balance()
        return float(bal["free"].get(self.quote) or 0), float(bal["free"].get(self.base) or 0)

    def _check_limits(self, qty: float, price: float) -> float:
        qty = float(self.ex.amount_to_precision(self.symbol, qty))
        limits = self.market.get("limits", {})
        min_amount = (limits.get("amount") or {}).get("min") or 0
        min_cost = (limits.get("cost") or {}).get("min") or 0
        if qty <= 0 or qty < min_amount or qty * price < min_cost:
            raise OrderRejected(f"{qty} {self.base} (~{qty * price:.2f} {self.quote}) is below the exchange minimum "
                                f"(amount {min_amount}, cost {min_cost})")
        return qty

    def _fill(self, order: dict, fallback_price: float) -> tuple[float, float]:
        if order.get("filled") is None:
            order = self.ex.fetch_order(order["id"], self.symbol)
        return float(order.get("filled") or 0), float(order.get("average") or fallback_price)

    def market_buy(self, qty: float, price: float) -> tuple[float, float]:
        qty = self._check_limits(qty, price)
        order = self.ex.create_market_buy_order(self.symbol, qty)
        log.warning("LIVE BUY %s %s -> order %s", qty, self.symbol, order.get("id"))
        return self._fill(order, price)

    def market_sell(self, qty: float, price: float) -> tuple[float, float]:
        qty = self._check_limits(qty, price)
        order = self.ex.create_market_sell_order(self.symbol, qty)
        log.warning("LIVE SELL %s %s -> order %s", qty, self.symbol, order.get("id"))
        return self._fill(order, price)

    def place_stop(self, qty: float, stop: float) -> str | None:
        """Rest a stop-loss on the exchange so the position is protected even if the bot dies.

        Uses ccxt's unified `stopLossPrice` as a stop-limit. Returns None if the exchange
        refuses, in which case the trading loop keeps enforcing the stop itself.
        """
        try:
            qty = float(self.ex.amount_to_precision(self.symbol, qty))
            limit = float(self.ex.price_to_precision(self.symbol, stop * (1 - self.STOP_LIMIT_BUFFER)))
            trigger = float(self.ex.price_to_precision(self.symbol, stop))
            order = self.ex.create_order(self.symbol, "limit", "sell", qty, limit, {"stopLossPrice": trigger})
            log.warning("LIVE STOP %s %s @ %s (limit %s) -> order %s", qty, self.symbol, trigger, limit, order.get("id"))
            return order.get("id")
        except Exception:
            log.exception("Exchange refused the stop order; the bot will enforce the stop itself")
            return None

    def cancel_order(self, order_id: str) -> None:
        try:
            self.ex.cancel_order(order_id, self.symbol)
        except Exception as e:  # already filled or cancelled
            log.info("Cancel %s: %s", order_id, e)

    def order_filled(self, order_id: str) -> bool:
        return self.ex.fetch_order(order_id, self.symbol).get("status") == "closed"
