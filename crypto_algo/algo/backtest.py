"""Bar-by-bar backtester for a long-only spot strategy.

No look-ahead: signals are computed on bar i's close and filled at bar i+1's open.
Stops/targets are checked against each bar's high/low; if both are touched in the
same bar the stop is assumed to fill first (conservative).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .risk import Position, position_size, trail_stop
from .strategy import make_strategy

PERIODS_PER_YEAR = {"1m": 525_600, "3m": 175_200, "5m": 105_120, "15m": 35_040, "30m": 17_520,
                    "1h": 8_760, "4h": 2_190, "1d": 365}


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    reason: str
    side: int = 1

    @property
    def return_pct(self) -> float:
        return self.side * (self.exit_price / self.entry_price - 1)


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade]
    stats: dict = field(default_factory=dict)


def _signals(data: pd.DataFrame) -> dict[str, np.ndarray]:
    """Long/short entry and exit arrays. Long-only strategies provide just `entry` / `exit`."""
    false = np.zeros(len(data), dtype=bool)
    get = lambda k, default: data[k].fillna(False).to_numpy(dtype=bool) if k in data else default
    long_entry = get("long_entry", get("entry", false))
    long_exit = get("long_exit", get("exit", false))
    nan = np.full(len(data), np.nan)
    return {
        "long_entry": long_entry, "long_exit": long_exit,
        "short_entry": get("short_entry", false), "short_exit": get("short_exit", false),
        "stop_long": data["stop_long"].to_numpy() if "stop_long" in data else nan,
        "stop_short": data["stop_short"].to_numpy() if "stop_short" in data else nan,
        # Optional limit entry prices; NaN means enter at market on the next open.
        "limit_long": data["limit_long"].to_numpy() if "limit_long" in data else nan,
        "limit_short": data["limit_short"].to_numpy() if "limit_short" in data else nan,
        "limit_bars": int(data.attrs.get("limit_bars", 1)),
    }


def run_backtest(df: pd.DataFrame, cfg: Config) -> BacktestResult:
    """Long and short, one position at a time, sized by risk to the stop.

    Accounting is margin-style (like a futures account at 1x): fees are paid from cash
    and equity is cash plus the open position's unrealised P&L. For a long-only strategy
    this is equivalent to holding the coin.

    A signal with a limit price rests as a limit order for `limit_bars` bars and fills
    (maker fee, no slippage) only if price trades through it. Take-profits are resting
    limits too; stops and signal exits are market orders (taker fee plus slippage).
    """
    strat = make_strategy(cfg.strategy)
    data = strat.compute(df)
    fee, slip = cfg.costs.fee_rate, cfg.costs.slippage
    maker = cfg.costs.maker_fee if cfg.costs.maker_fee is not None else fee
    sig = _signals(data)
    limit_order = None  # (side, price, stop, expiry_bar)

    cash = cfg.initial_capital
    pos: Position | None = None
    trades: list[Trade] = []
    equity = np.empty(len(data))
    in_market = np.zeros(len(data), dtype=bool)
    pending_side, pending_stop, pending_exit = 0, np.nan, False

    day, day_start_equity, halted = None, cash, False

    o, h, l, c = (data[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr_v = data["atr"].to_numpy()
    idx = data.index

    def mark(price: float) -> float:
        return cash + (pos.side * pos.qty * (price - pos.entry_price) if pos else 0.0)

    def close_position(i: int, raw_price: float, reason: str) -> None:
        nonlocal cash, pos
        is_limit = reason == "take_profit"
        price = raw_price if is_limit else raw_price * (1 - slip * pos.side)
        exit_fee = pos.qty * price * (maker if is_limit else fee)
        gross = pos.side * pos.qty * (price - pos.entry_price)
        cash += gross - exit_fee
        trades.append(Trade(pos.entry_time, idx[i], pos.entry_price, price, pos.qty,
                            gross - exit_fee - pos.entry_fee, reason, pos.side))
        pos = None

    def open_position(i: int, side: int, price: float, stop: float, fee_rate: float) -> None:
        nonlocal cash, pos
        dist = side * (price - stop)
        if dist <= 0:
            return
        equity_now = mark(price)
        qty = min(equity_now * cfg.risk.risk_per_trade / dist, equity_now * cfg.risk.max_position_pct / price)
        tp = price + side * cfg.risk.reward_risk * dist if cfg.risk.reward_risk > 0 else side * np.inf
        paid = qty * price * fee_rate
        cash -= paid
        pos = Position(qty, price, stop, tp, idx[i], highest=price, side=side, entry_fee=paid)

    for i in range(len(data)):
        # Daily loss kill switch.
        if idx[i].date() != day:
            day = idx[i].date()
            day_start_equity = mark(o[i])
            halted = False

        # 1. Fill orders decided on the previous close, at this bar's open.
        if pending_exit and pos:
            close_position(i, o[i], "signal")
        if pending_side and pos is None and not halted:
            side = pending_side
            price = o[i] * (1 + slip * side)
            stop = pending_stop if np.isfinite(pending_stop) else price - side * cfg.risk.atr_stop_mult * atr_v[i - 1]
            open_position(i, side, price, stop, fee)
        pending_side, pending_stop, pending_exit = 0, np.nan, False

        # Resting limit entry: fills if this bar trades through it (at the open if it gaps past).
        filled_now = False
        if limit_order and pos is None:
            side, lim, stop, expiry = limit_order
            if i > expiry or halted:
                limit_order = None
            elif (side == 1 and l[i] <= lim) or (side == -1 and h[i] >= lim):
                fill = min(o[i], lim) if side == 1 else max(o[i], lim)
                open_position(i, side, fill, stop, maker)
                limit_order, filled_now = None, pos is not None

        # 2. Intrabar stop / take-profit (stop assumed first if both are touched).
        if pos:
            # On the bar a limit fills we can't know the intrabar order, so only the stop counts.
            if pos.side == 1:
                if l[i] <= pos.stop:
                    close_position(i, min(o[i], pos.stop) if not filled_now else pos.stop, "stop")
                elif h[i] >= pos.take_profit and not filled_now:
                    close_position(i, max(o[i], pos.take_profit), "take_profit")
            else:
                if h[i] >= pos.stop:
                    close_position(i, max(o[i], pos.stop) if not filled_now else pos.stop, "stop")
                elif l[i] <= pos.take_profit and not filled_now:
                    close_position(i, min(o[i], pos.take_profit), "take_profit")
            if pos:
                trail_stop(pos, h[i], atr_v[i], cfg.risk, low=l[i])

        equity[i] = mark(c[i])
        in_market[i] = pos is not None
        if equity[i] <= day_start_equity * (1 - cfg.risk.max_daily_loss):
            halted = True

        # 3. Decide on this bar's close.
        if pos:
            if (pos.side == 1 and sig["long_exit"][i]) or (pos.side == -1 and sig["short_exit"][i]):
                pending_exit = True
        elif not halted:
            for side, key in ((1, "long"), (-1, "short")):
                if sig[f"{key}_entry"][i]:
                    lim = sig[f"limit_{key}"][i]
                    if np.isfinite(lim):
                        limit_order = (side, lim, sig[f"stop_{key}"][i], i + sig["limit_bars"])
                    else:
                        pending_side, pending_stop = side, sig[f"stop_{key}"][i]
                    break

    if pos:
        close_position(len(data) - 1, c[-1], "end")
        equity[-1] = cash

    eq = pd.Series(equity, index=idx, name="equity")
    stats = compute_stats(eq, trades, data["close"], cfg)
    stats["exposure"] = float(in_market.mean())
    return BacktestResult(eq, trades, stats)


def compute_stats(eq: pd.Series, trades: list[Trade], close: pd.Series, cfg: Config) -> dict:
    rets = eq.pct_change().dropna()
    ppy = PERIODS_PER_YEAR.get(cfg.timeframe, 8_760)
    years = len(eq) / ppy
    total = eq.iloc[-1] / eq.iloc[0] - 1
    drawdown = eq / eq.cummax() - 1
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    return {
        "total_return": total,
        "cagr": (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan"),
        "buy_and_hold": close.iloc[-1] / close.iloc[0] - 1,
        "max_drawdown": drawdown.min(),
        "sharpe": rets.mean() / rets.std() * np.sqrt(ppy) if rets.std() > 0 else 0.0,
        "trades": len(trades),
        "longs": sum(t.side == 1 for t in trades),
        "shorts": sum(t.side == -1 for t in trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": sum(wins) / -sum(losses) if losses and sum(losses) < 0 else float("inf"),
        "avg_trade": float(np.mean([t.return_pct for t in trades])) if trades else 0.0,
        "avg_hold_hours": float(np.mean([(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]))
        if trades else 0.0,
        "final_equity": eq.iloc[-1],
    }


def format_stats(stats: dict) -> str:
    pct = {"total_return", "cagr", "buy_and_hold", "max_drawdown", "win_rate", "avg_trade", "exposure"}
    lines = []
    for k, v in stats.items():
        if k in pct:
            lines.append(f"{k:>15}: {v:8.2%}")
        elif isinstance(v, float):
            lines.append(f"{k:>15}: {v:10.2f}")
        else:
            lines.append(f"{k:>15}: {v}")
    return "\n".join(lines)
