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

Universe: `universe` = "fixed" trades `symbols`; "mature_volume" re-chooses `universe_size` coins on every
refresh date of `universe_calendar` (algo/universe.py). Coins that leave the universe are sold; coins that
join are bought only when their own signal is on.

There are no stops: risk is controlled by sizing and by going to cash when trends end.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from . import universe as uni
from .broker import OrderRejected
from .data import fetch_ohlcv
from .htf import confirmation, confirmed_breakouts
from .profile import atr_panel, cmf_panel, volume_profile

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
    volume_profile: bool = False     # breakout: enter only above the 30-day value-area high, exit below its POC
    atr_expansion: bool = False      # breakout: enter only while ATR(5) > ATR(20) (volatility expanding)
    cmf_exit: float | None = None    # breakout: exit when Chaikin money flow(20) falls below this (e.g. -0.1)
    universe: str = "fixed"          # "fixed" (trade `symbols`) or "mature_volume" (re-chosen each refresh, algo/universe.py)
    universe_size: int = 15
    universe_min_age: int = 365      # days of uninterrupted daily history a coin needs to be chosen
    universe_calendar: str = "Q-jan"  # refresh on the 1st of Jan/Apr/Jul/Oct, 00:00 UTC
    state_file: str = "portfolio_state.json"

    @property
    def timeframe(self) -> str:
        return "4h" if self.strategy == "breakout" else "1h"

    @property
    def history_hours(self) -> int:
        if self.strategy == "breakout":
            hours = (max(self.entries) * 4 + 50) * 4  # enough 4h bars for the channel state to settle
            if "1d" in self.confirm_tfs:
                hours = max(hours, 24 * 365)          # a daily EMA50 forgets its starting value after ~a year
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
    entry_ok = exit_now = None
    if cfg.volume_profile:
        poc, vah = volume_profile(bars["high"], bars["low"], c, bars["volume"])
        entry_ok, exit_now = c > vah, c < poc
    if cfg.atr_expansion:
        exp = atr_panel(bars["high"], bars["low"], c, 5) > atr_panel(bars["high"], bars["low"], c, 20)
        entry_ok = exp if entry_ok is None else entry_ok & exp
    if cfg.cmf_exit is not None:
        out = cmf_panel(bars["high"], bars["low"], c, bars["volume"]) < cfg.cmf_exit
        exit_now = out if exit_now is None else exit_now | out
    size = confirmed_breakouts(bars, systems, confirm, cfg.confirm_mode, entry_ok, exit_now).iloc[-1]
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
        self.fills: list[dict] = []   # every simulated fill, for reports

    def balances(self, symbols) -> tuple[float, dict[str, float]]:
        return self.cash, {s: self.holdings.get(s, 0.0) for s in symbols}

    def market_buy(self, symbol: str, qty: float, price: float) -> None:
        fill = price * (1 + self.slippage)
        qty = min(qty, self.cash / (fill * (1 + self.fee_rate)))
        self.cash -= qty * fill * (1 + self.fee_rate)
        self.holdings[symbol] = self.holdings.get(symbol, 0.0) + qty
        self.fills.append({"symbol": symbol, "side": "buy", "qty": qty, "price": fill, "fee": qty * fill * self.fee_rate})
        log.info("PAPER BUY  %-10s %.6f @ %.4f", symbol, qty, fill)

    def market_sell(self, symbol: str, qty: float, price: float) -> None:
        fill = price * (1 - self.slippage)
        qty = min(qty, self.holdings.get(symbol, 0.0))
        self.cash += qty * fill * (1 - self.fee_rate)
        self.holdings[symbol] = self.holdings.get(symbol, 0.0) - qty
        if self.holdings[symbol] <= 1e-12:
            del self.holdings[symbol]
        self.fills.append({"symbol": symbol, "side": "sell", "qty": qty, "price": fill, "fee": qty * fill * self.fee_rate})
        log.info("PAPER SELL %-10s %.6f @ %.4f", symbol, qty, fill)


class LivePortfolioBroker:
    """Spot market orders on a ccxt exchange. Only constructed when the user explicitly opts in."""

    def __init__(self, exchange, quote: str = "USDT"):
        self.ex, self.quote = exchange, quote
        self.ex.load_markets()

    def balances(self, symbols) -> tuple[float, dict[str, float]]:
        free = self.ex.fetch_balance()["free"]
        return float(free.get(self.quote) or 0), {s: float(free.get(self.ex.market(s)["base"]) or 0) for s in symbols}

    def coins_usd(self, min_usd: float) -> dict[str, float]:
        """Every coin in the account (free, in orders or locked) worth at least `min_usd`, valued in the quote."""
        out = {}
        for cur, qty in self.ex.fetch_balance()["total"].items():
            sym = f"{cur}/{self.quote}"
            if cur == self.quote or not qty or sym not in self.ex.markets:
                continue
            value = float(qty) * float(self.ex.fetch_ticker(sym)["last"] or 0)
            if value >= min_usd:
                out[sym] = value
        return out

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
    tradable_only = True  # universe choice: only pairs trading now (a replay of the past also sees delisted ones)
    tradable_at = None    # replay: fn(symbol, R) -> was the pair trading at R (see algo/forward.py)

    def __init__(self, cfg: PortfolioConfig, exchange, broker):
        self.cfg, self.ex, self.broker = cfg, exchange, broker
        self.state_path = Path(cfg.state_file)
        self.last_decision: str | None = None
        self.universe_date: str | None = None
        self.universe: list[str] = list(cfg.symbols)
        self.positions: list[str] = []  # symbols the bot holds, so coins that left the universe are still sold
        if cfg.universe != "fixed" and cfg.strategy != "breakout":
            raise ValueError("a changing universe is only implemented for the breakout strategy")
        s = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.last_decision = s.get("last_decision")
        self.universe_date = s.get("universe_date")
        self.account_checked = s.get("account_checked", False)   # live: coins already in the account were reviewed
        self.universe = s.get("universe", self.universe)
        if isinstance(broker, PaperPortfolioBroker) and "paper_cash" in s:
            broker.cash, broker.holdings = s["paper_cash"], s["paper_holdings"]
        # A state from the fixed-list version (or no state) does not list positions: look at every coin the bot
        # may have bought; those still held are kept and sold if they are no longer in the universe.
        self.positions = s["positions"] if "positions" in s else \
            list(dict.fromkeys([*cfg.symbols, *self.universe, *s.get("paper_holdings", {})]))

    def _save(self) -> None:
        s = {"last_decision": self.last_decision, "universe_date": self.universe_date, "universe": self.universe,
             "positions": self.positions, "account_checked": self.account_checked}
        if isinstance(self.broker, PaperPortfolioBroker):
            s["paper_cash"], s["paper_holdings"] = self.broker.cash, self.broker.holdings
        self.state_path.write_text(json.dumps(s, indent=2))

    def now(self) -> pd.Timestamp:
        return pd.Timestamp(self.ex.milliseconds(), unit="ms", tz="UTC")

    def symbols_for(self, candle: pd.Timestamp) -> list[str]:
        """The universe governing the decision on the 4h candle opening at `candle`."""
        if self.cfg.universe == "fixed":
            return list(self.cfg.symbols)
        R = uni.refresh_date(candle, self.cfg.universe_calendar)
        if self.universe_date != str(R):
            try:
                chosen = uni.choose(self.ex, R, self.cfg.universe_size, self.cfg.universe_min_age, self.tradable_only,
                                    tradable_at=self.tradable_at)
            except Exception:
                if not self.universe_date or self.tradable_at is not None:   # a replay must never fall back
                    raise
                # keep trading the previous universe and try again on the next candle
                log.exception("Universe for %s not chosen; keeping %s's for now", R.date(), self.universe_date[:10])
                return self.universe
            self.universe, self.universe_date = chosen, str(R)
            log.info("Universe for %s: %s", R.date(), " ".join(self.universe))
            self._save()
        return self.universe

    def bars(self, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """close/high/low panels of closed candles for `symbols` (default: the configured ones)."""
        symbols = list(self.cfg.symbols) if symbols is None else symbols
        since = self.now() - pd.Timedelta(hours=self.cfg.history_hours + 8)
        frames = {}
        for s in symbols:
            for attempt in range(3):
                try:
                    frames[s] = fetch_ohlcv(self.ex, s, self.cfg.timeframe, since)
                    break
                except ccxt.OperationFailed:          # timeout, rate limit, exchange down: never decide on partial data
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
                except (ccxt.BadSymbol, RuntimeError) as e:  # unknown pair or no candles: a halted or delisted market
                    log.warning("No %s candles for %s: %s", self.cfg.timeframe, s, e)
                    break
        return {k: pd.DataFrame({s: f[k] for s, f in frames.items()}).iloc[:-1]  # last candle is still forming
                for k in ("close", "high", "low", "volume")}

    def rebalance(self, weights: pd.Series, prices: pd.Series, frozen: set[str] = frozenset()) -> None:
        """Trade toward `weights` (fractions of equity). Coins in `frozen` (no candle on the decision bar,
        e.g. a halted market) count in equity at their last price but are not traded."""
        symbols = [s for s in weights.index if pd.notna(prices.get(s))]
        cash, held = self.broker.balances(symbols)
        equity = cash + sum(held[s] * prices[s] for s in symbols)
        delta = {s: weights[s] * equity / prices[s] - held[s] for s in symbols if s not in frozen}
        log.info("Rebalance: equity %.2f, targets %s", equity, {s: f"{w:.1%}" for s, w in weights.items() if w > 0})
        # Sells first (largest first) so their proceeds fund the buys.
        retry = None
        for s in sorted(delta, key=lambda k: delta[k] * prices[k]):
            notional = abs(delta[s]) * prices[s]
            if notional < self.cfg.min_trade_usd:
                continue
            try:
                (self.broker.market_sell if delta[s] < 0 else self.broker.market_buy)(s, abs(delta[s]), prices[s])
            except ccxt.OperationFailed as e:  # timeout, rate limit, exchange down: retry the candle at the next poll
                log.warning("Order for %s failed, will retry: %s", s, e)
                retry = retry or e
            except Exception as e:  # rejected for good (halted market, minimum size...): must not block the others
                log.warning("Skipped %s: %s", s, e)
        if retry:
            raise retry

    def step(self) -> bool:
        if self.cfg.universe != "fixed":
            return self._step_universe()
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

    def _step_universe(self) -> bool:
        """Breakout portfolio on a universe that changes on refresh dates."""
        step = pd.Timedelta(self.cfg.timeframe)
        candle = self.now().floor(step) - step            # the candle that has just closed
        if str(candle) == self.last_decision:
            return False
        universe = self.symbols_for(candle)
        bars = self.bars(list(dict.fromkeys(universe + self.positions)))
        closes = bars["close"]
        if closes.empty or closes.index[-1] != candle:  # data not there yet: try again on the next poll
            return False
        live = [s for s in universe if s in closes.columns]
        weights = breakout_weights({k: v[live] for k, v in bars.items()}, self.cfg).reindex(closes.columns).fillna(0.0)
        prices = closes.ffill().iloc[-1]
        self.rebalance(weights, prices, frozen=set(closes.columns[closes.iloc[-1].isna()]))
        _, held = self.broker.balances(list(closes.columns))
        unpriced = [s for s in self.positions if s not in closes.columns]   # keep tracking coins we could not price
        self.positions = unpriced + [s for s in closes.columns if held.get(s, 0) * prices[s] >= self.cfg.min_trade_usd]
        self.last_decision = str(candle)
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
