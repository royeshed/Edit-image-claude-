"""Position sizing and stop management."""
from dataclasses import dataclass

from .config import RiskConfig


@dataclass
class Position:
    qty: float
    entry_price: float
    stop: float
    take_profit: float
    entry_time: object = None
    highest: float = 0.0


def position_size(equity: float, price: float, atr_value: float, cfg: RiskConfig) -> tuple[float, float, float]:
    """Size so that hitting the ATR stop loses `risk_per_trade` of equity.

    Returns (qty, stop, take_profit). qty is 0 if the trade can't be sized.
    """
    stop_dist = cfg.atr_stop_mult * atr_value
    if stop_dist <= 0 or price <= stop_dist:
        return 0.0, 0.0, 0.0
    qty = equity * cfg.risk_per_trade / stop_dist
    qty = min(qty, equity * cfg.max_position_pct / price)
    return qty, price - stop_dist, price + cfg.reward_risk * stop_dist


def trail_stop(pos: Position, high: float, atr_value: float, cfg: RiskConfig) -> None:
    """Ratchet the stop up behind the highest price seen. Never moves it down."""
    pos.highest = max(pos.highest, high)
    if cfg.trailing_atr_mult > 0 and atr_value > 0:
        pos.stop = max(pos.stop, pos.highest - cfg.trailing_atr_mult * atr_value)
