"""Forward paper trading that cannot be tuned after the fact.

The rules are frozen in git before the start date (tag `paper-v1`). Each run drives the live bot's own code,
PortfolioTrader.step(), with a simulated clock, one 4h close at a time from the start date to now, through
PaperPortfolioBroker (fees, slippage, $10 minimum trade). Every decision sees only candles that had closed at
that moment, so the ledger is exactly what a paper bot running since the start date would show (fills at the
candle close), and anyone can re-run it to check.

    python main.py portfolio-forward                     # every profile in config.json "paper"
    python main.py portfolio-forward --start 2025-01-01  # the same engine over the past (a backtest, not paper)
"""
import json
import logging
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .data import fetch_ohlcv
from .trend_portfolio import PaperPortfolioBroker, PortfolioConfig, PortfolioTrader

log = logging.getLogger(__name__)
MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
PAPER_TAG = "paper-v3"   # the git tag that freezes the paper rules (code and config.json)


class HistoricalExchange:
    """Binance market data as it was at a simulated moment `now`: nothing after `now` is ever served."""

    rateLimit = 0

    def __init__(self, exchange, history_start: pd.Timestamp, local_dir: str | Path | None = None):
        self.ex, self.history_start = exchange, history_start
        self.local = Path(local_dir) if local_dir else None
        self.now: pd.Timestamp | None = None
        self._series: dict[tuple[str, str], pd.DataFrame] = {}
        self._info = None

    def milliseconds(self) -> int:
        return int(self.now.timestamp() * 1000)

    def parse_timeframe(self, tf: str) -> int:
        return MS[tf] // 1000

    def publicGetExchangeInfo(self) -> dict:
        if self._info is None:
            if self.local:
                syms = json.loads((self.local / "candidates.json").read_text())
                self._info = {"symbols": [{"symbol": s, "baseAsset": s[:-4], "quoteAsset": "USDT", "status": "TRADING"} for s in syms]}
            else:
                self._info = self.ex.publicGetExchangeInfo()
        return self._info

    def _full(self, symbol: str, tf: str) -> pd.DataFrame:
        """All candles of `symbol` from history_start, fetched once (open, high, low, close, volume[, quote_volume])."""
        key = (symbol, tf)
        if key not in self._series:
            if self.local:
                f = self.local / f"{symbol.replace('/', '')}_{tf}.csv"
                d = pd.read_csv(f, index_col=0) if f.exists() else pd.DataFrame(columns=["open", "high", "low", "close", "volume", "quote_volume"])
                d.index = pd.to_datetime(d.index, utc=True)
            else:
                d = self._download(symbol, tf)
            self._series[key] = d[d.index >= self.history_start - pd.Timedelta(days=1100 if tf == "1d" else 0)]
        return self._series[key]

    def _download(self, symbol: str, tf: str, retries: int = 4) -> pd.DataFrame:
        """Full history from the exchange; network errors are retried, then raised (never a silent gap)."""
        import ccxt

        for attempt in range(retries):
            try:
                return self._daily_from_exchange(symbol) if tf == "1d" else fetch_ohlcv(self.ex, symbol, tf, self.history_start)
            except RuntimeError:              # the pair has no candles in this window
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "quote_volume"],
                                    index=pd.DatetimeIndex([], tz="UTC"))
            except ccxt.OperationFailed:
                if attempt == retries - 1:
                    raise
                time.sleep(2 * 2 ** attempt)

    def _daily_from_exchange(self, symbol: str) -> pd.DataFrame:
        rows, cursor = [], int((self.history_start - pd.Timedelta(days=1100)).timestamp() * 1000)
        while True:
            batch = self.ex.publicGetKlines({"symbol": symbol.replace("/", ""), "interval": "1d", "startTime": cursor, "limit": 1000})
            if not batch:
                break
            rows += batch
            if len(batch) < 1000:
                break
            cursor = int(batch[-1][0]) + MS["1d"]
        d = pd.DataFrame([[int(r[0]), *map(float, (r[1], r[2], r[3], r[4], r[5], r[7]))] for r in rows],
                         columns=["ts", "open", "high", "low", "close", "volume", "quote_volume"])
        d.index = pd.to_datetime(d.pop("ts"), unit="ms", utc=True)
        return d[~d.index.duplicated()]

    def _arrays(self, symbol: str, tf: str) -> np.ndarray:
        key = ("arr", symbol, tf)
        if key not in self._series:
            d = self._full(symbol, tf)
            self._series[key] = np.column_stack([d.index.as_unit("ms").asi8, d[["open", "high", "low", "close", "volume"]].to_numpy(float)])
        return self._series[key]

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        a = self._arrays(symbol, timeframe)
        lo = np.searchsorted(a[:, 0], since or 0, side="left")
        hi = np.searchsorted(a[:, 0], self.milliseconds(), side="right")    # may include the forming candle
        rows = a[lo:min(hi, lo + limit)].tolist()
        return [[int(r[0]), *r[1:]] for r in rows]

    def traded_at(self, symbol: str, R: pd.Timestamp) -> bool:
        """Was the pair trading just before R (a 4h candle in the 12 hours before R)? Stands in for the exchange
        status the live bot reads at R."""
        d = self._full(symbol, "4h")
        return bool(((d.index >= R - pd.Timedelta("12h")) & (d.index < R)).any())

    def publicGetKlines(self, params: dict) -> list:
        end = params["endTime"]
        if end >= self.milliseconds():
            raise ValueError("asked for candles from the future")
        d = self._full(params["symbol"][:-4] + "/" + params["symbol"][-4:], params["interval"])
        d = d[d.index.as_unit("ms").asi8 <= end].iloc[-params.get("limit", 500):]
        return [[int(t.timestamp() * 1000), r.open, r.high, r.low, r.close, r.volume, 0, r.quote_volume]
                for t, r in d.iterrows()]


def run(cfg: PortfolioConfig, exchange, start: pd.Timestamp, end: pd.Timestamp | None = None, capital: float = 10_000,
        fee_rate: float = 0.001, slippage: float = 0.0005, local_dir=None, hx: HistoricalExchange | None = None) -> dict:
    """Replay the bot from `start` to `end` (default: the last closed 4h candle). Returns the paper ledger.
    Pass the same `hx` to several runs to share downloads (its history_start must cover every config)."""
    step = pd.Timedelta(cfg.timeframe)
    end = end or pd.Timestamp.now(tz="UTC").floor(step)
    hx = hx or HistoricalExchange(exchange, start - pd.Timedelta(hours=cfg.history_hours + 48), local_dir)
    broker = PaperPortfolioBroker(capital, fee_rate, slippage)
    with tempfile.TemporaryDirectory() as tmp:
        trader = PortfolioTrader(replace(cfg, state_file=str(Path(tmp) / "state.json")), hx, broker)
        trader.tradable_only = False       # a coin delisted later was still tradable back then...
        trader.tradable_at = hx.traded_at  # ...but only if it was trading at the refresh date
        equity, fills, universes = {}, [], {}
        for t in pd.date_range(start.ceil(step), end, freq=step):
            hx.now = t + pd.Timedelta(minutes=1)
            n = len(broker.fills)
            if not trader.step():
                continue
            candle = pd.Timestamp(trader.last_decision)
            universes.setdefault(trader.universe_date, list(trader.universe))
            for f in broker.fills[n:]:
                fills.append({"time": t, **f})
            prices = {s: hx._full(s, cfg.timeframe)["close"].loc[:candle].iloc[-1] for s in broker.holdings}
            equity[t] = broker.cash + sum(q * prices[s] for s, q in broker.holdings.items())
    eq = pd.Series(list(equity.values()), index=pd.DatetimeIndex(list(equity), tz="UTC"), name="equity", dtype=float)
    btc = pd.Series(dtype=float)
    if len(eq):
        b = hx._full("BTC/USDT", cfg.timeframe)["close"].reindex(eq.index - step, method="ffill")
        btc = pd.Series(b.values / b.values[0] * capital, index=eq.index)
    return {"equity": eq, "fills": pd.DataFrame(fills), "universes": universes, "cash": broker.cash,
            "holdings": dict(broker.holdings), "btc": btc}


def is_official(config_path: str | Path, custom_start: bool, start: pd.Timestamp) -> tuple[bool, str]:
    """(official, commit): the paper record is only a run of the commit tagged PAPER_TAG, with no change to the
    code or config.json (tracked or not), with that config.json, from its own start date, tagged before that date."""
    repo = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout.strip()
    try:
        head, tagged = git("rev-parse", "HEAD"), git("rev-parse", "-q", "--verify", f"{PAPER_TAG}^{{commit}}")
        changed = git("status", "--porcelain", "--", "algo", "main.py", "config.json")
        tagged_at = git("log", "-1", "--format=%cI", tagged) if tagged else ""
    except OSError:
        return False, "unknown"
    own_config = Path(config_path).resolve() == (repo / "config.json").resolve()
    before_start = bool(tagged_at) and pd.Timestamp(tagged_at).tz_convert("UTC") <= start
    return bool(tagged) and head == tagged and not changed and own_config and not custom_start and before_start, head or "unknown"


def summary(r: dict, capital: float) -> dict:
    eq = r["equity"]
    if eq.empty:
        return {"days": 0}
    days = (eq.index[-1] - eq.index[0]).total_seconds() / 86400 + 4 / 24
    total = eq.iloc[-1] / capital - 1
    fills = r["fills"]
    return {"from": str(eq.index[0]), "to": str(eq.index[-1]), "days": round(days, 1), "start_usd": capital,
            "end_usd": round(eq.iloc[-1], 2), "return": total,
            "annualised": (1 + total) ** (365 / days) - 1 if days >= 365 else None,   # meaningless over shorter spans
            "max_drawdown": float((eq / eq.cummax() - 1).min()), "trades": len(fills),
            "fees_usd": round(float(fills.fee.sum()), 2) if len(fills) else 0.0,
            "btc_hold_return": r["btc"].iloc[-1] / capital - 1, "holdings": sorted(r["holdings"]),
            "universe": r["universes"][max(r["universes"])] if r["universes"] else []}
