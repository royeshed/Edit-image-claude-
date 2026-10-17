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
from .strategy import EmaRsiTrendStrategy

PERIODS_PER_YEAR = {"1m": 525_600, "5m": 105_120, "15m": 35_040, "30m": 17_520,
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

    @property
    def return_pct(self) -> float:
        return self.exit_price / self.entry_price - 1


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade]
    stats: dict = field(default_factory=dict)


def run_backtest(df: pd.DataFrame, cfg: Config) -> BacktestResult:
    strat = EmaRsiTrendStrategy(cfg.strategy)
    data = strat.compute(df)
    fee, slip = cfg.costs.fee_rate, cfg.costs.slippage

    cash = cfg.initial_capital
    pos: Position | None = None
    trades: list[Trade] = []
    equity = np.empty(len(data))
    in_market = np.zeros(len(data), dtype=bool)
    pending_entry = pending_exit = False

    day, day_start_equity, halted = None, cash, False

    o, h, l, c = (data[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr_v = data["atr"].to_numpy()
    entries, exits = data["entry"].to_numpy(), data["exit"].to_numpy()
    idx = data.index

    def close_position(i: int, raw_price: float, reason: str) -> None:
        nonlocal cash, pos
        price = raw_price * (1 - slip)
        proceeds = pos.qty * price * (1 - fee)
        cost = pos.qty * pos.entry_price * (1 + fee)
        cash += proceeds
        trades.append(Trade(pos.entry_time, idx[i], pos.entry_price, price, pos.qty, proceeds - cost, reason))
        pos = None

    for i in range(len(data)):
        # Daily loss kill switch.
        if idx[i].date() != day:
            day = idx[i].date()
            day_start_equity = cash + (pos.qty * o[i] if pos else 0.0)
            halted = False

        # 1. Fill orders decided on the previous close, at this bar's open.
        if pending_exit and pos:
            close_position(i, o[i], "signal")
        if pending_entry and pos is None and not halted:
            price = o[i] * (1 + slip)
            qty, stop, tp = position_size(cash, price, atr_v[i - 1], cfg.risk)
            qty = min(qty, cash / (price * (1 + fee)))
            if qty > 0:
                cash -= qty * price * (1 + fee)
                pos = Position(qty, price, stop, tp, idx[i], highest=price)
        pending_entry = pending_exit = False

        # 2. Intrabar stop / take-profit.
        if pos:
            if l[i] <= pos.stop:
                close_position(i, min(o[i], pos.stop), "stop")
            elif h[i] >= pos.take_profit:
                close_position(i, max(o[i], pos.take_profit), "take_profit")
            else:
                trail_stop(pos, h[i], atr_v[i], cfg.risk)

        equity[i] = cash + (pos.qty * c[i] if pos else 0.0)
        in_market[i] = pos is not None
        if equity[i] <= day_start_equity * (1 - cfg.risk.max_daily_loss):
            halted = True

        # 3. Decide on this bar's close.
        if pos and exits[i]:
            pending_exit = True
        elif pos is None and entries[i] and not halted:
            pending_entry = True

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
