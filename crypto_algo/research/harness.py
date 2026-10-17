"""Shared research harness for improving the bot without fooling ourselves.

One cached data load, one pipeline the bot follows, one scorer, and a lockbox:

    universe -> signal (entry/exit state, with higher-timeframe confirmation) -> regime multiplier
             -> keep coins in the universe -> rank and keep the top k -> size (<= 1/6 per coin, <= 100% total)
             -> smooth (turnover control) -> P&L after 0.12% per side

Every candidate is scored on the same 9 variants (15 +- 5 coins chosen by the "mature volume" rule,
refreshed quarterly from January, February or March) and judged by its median variant.
`evaluate` returns 2018-11..2022 (train) numbers only. The 2023-2026 numbers stay locked until the
one-time final evaluation runs with LOCKBOX=open, so nothing can be tuned on them.

    python -m research.harness cache      # build data/cache/lab.pkl (once)
    python -m research.harness baseline   # train numbers of the current bot on the 9 variants
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from algo.htf import confirmation, state
from algo.trend_portfolio import PortfolioConfig
from .dynamic_universe import UNI, load_bars, load_daily
from .lab import BARS_PER_YEAR, SPOT_COST
from .universe_selection import CALENDARS, load_features, membership, rule_scores_all, universe

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
METHODS = ROOT / "data" / "methods"
MAX_WEIGHT = 1 / 6          # the user's risk choice: at most ~17% of equity in one coin
PPY = BARS_PER_YEAR["4h"]
TRAIN = (pd.Timestamp("2018-11-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC"))
LOCKED = {"valid": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
          "test": (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2100-01-01", tz="UTC")),
          "oos": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2100-01-01", tz="UTC"))}
VARIANTS = [(n, cal) for n in (10, 15, 20) for cal in ("Q-jan", "Q-feb", "Q-mar")]


def lockbox_open() -> bool:
    return os.environ.get("LOCKBOX") == "open"


def metrics(pnl: pd.Series, lo: pd.Timestamp, hi: pd.Timestamp, w: pd.DataFrame | None = None) -> dict:
    r = pnl[(pnl.index >= lo) & (pnl.index < hi)].fillna(0)
    eq = (1 + r).cumprod()
    years = len(r) / PPY
    sd = r.std()
    out = {"cagr": eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else -1.0,
           "sharpe": r.mean() / sd * np.sqrt(PPY) if sd > 0 else 0.0,
           "maxdd": (eq / eq.cummax() - 1).min(),
           "vol": sd * np.sqrt(PPY)}
    if w is not None:
        ww = w[(w.index >= lo) & (w.index < hi)]
        out["turnover"] = ww.diff().abs().sum(axis=1).mean() * PPY      # times equity traded per year
        out["exposure"] = ww.sum(axis=1).mean()                           # average fraction invested
    return out


@dataclass
class Lab:
    """Panels on the 4h grid for every coin that ever enters the candidate pool, plus monthly features."""
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    open: pd.DataFrame
    volume: pd.DataFrame
    ret: pd.DataFrame
    daily_close: pd.DataFrame
    daily_volume: pd.DataFrame
    features: pd.DataFrame
    cfg: PortfolioConfig
    _cache: dict = field(default_factory=dict, repr=False)

    # ---------- loading ----------
    @classmethod
    def load(cls) -> "Lab":
        f = CACHE / "lab.pkl"
        if not f.exists():
            build_cache()
        d = pd.read_pickle(f)
        cfg = PortfolioConfig(**json.load(open(ROOT / "config.json"))["portfolio"])
        return cls(**d, cfg=cfg)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def assets(self) -> pd.Index:
        return self.close.columns

    @property
    def bars(self) -> dict:
        return {"open": self.open, "high": self.high, "low": self.low, "close": self.close, "volume": self.volume, "ret": self.ret}

    @cached_property
    def live(self) -> pd.DataFrame:
        """True while a coin is trading (has a return on this bar)."""
        return self.ret.notna()

    def memo(self, key, fn: Callable):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # ---------- building blocks ----------
    def gate(self, tfs=("1d",), name: str = "close>ema50") -> pd.DataFrame:
        """Higher-timeframe confirmation (bool), visible only after the HTF candle has closed."""
        return self.memo(("gate", tuple(tfs), name), lambda: confirmation(self.bars, self.index, tuple(tfs), name))

    def donchian(self, n: int, m: int) -> pd.DataFrame:
        """(entry, exit) booleans: close above the prior n-bar high / below the prior m-bar low."""
        def go():
            c = self.close
            return c > self.high.rolling(n).max().shift(), c < self.low.rolling(m).min().shift()
        return self.memo(("donchian", n, m), go)

    def baseline_signal(self) -> pd.DataFrame:
        """The live bot's signal: mean of 6 Donchian systems, entries and positions gated by daily close > EMA50."""
        def go():
            g = self.gate(self.cfg.confirm_tfs, self.cfg.confirm)
            total = 0
            for n in self.cfg.entries:
                for m in self.cfg.exits:
                    e, x = self.donchian(n, m)
                    total = total + state(e & g, x | ~g)
            return (total / (len(self.cfg.entries) * len(self.cfg.exits))).where(self.live, 0.0)
        return self.memo("baseline_signal", go)

    def momentum(self, days: int = 14) -> pd.DataFrame:
        return self.memo(("mom", days), lambda: self.close / self.close.shift(days * 6) - 1)

    def mask(self, n: int, calendar: str, rule: str = "mature_volume", score: pd.Series | None = None) -> pd.DataFrame:
        """Universe membership (bool) on the 4h grid. `score` (aligned to self.features) overrides `rule`."""
        def go():
            s = score if score is not None else self.memo(("rule", rule), lambda: rule_scores_all(self.features, rule))
            u = universe(self.features, s, CALENDARS[calendar], n)
            return membership(u, self.index, self.assets).astype(bool)
        key = ("mask", n, calendar, rule) if score is None else None
        return go() if key is None else self.memo(key, go)

    def known(self, R: pd.Timestamp) -> pd.Series:
        """Feature rows whose 3-month outcome (fwd_*) had fully ended by R: the only rows a model may learn from at R."""
        return self.features.date + pd.DateOffset(months=3) <= R

    # ---------- the pipeline ----------
    def build(self, n: int, calendar: str, *, mask: pd.DataFrame | None = None, signal: pd.DataFrame | None = None,
              regime: pd.Series | pd.DataFrame | None = None, rank: pd.DataFrame | None = None, k: int | None = None,
              slots: float | None = None, sizing: Callable | None = None, smoothing: Callable | None = None) -> pd.DataFrame:
        """Weights at each 4h close (earning the next bar). Any layer left as None uses the live bot's choice."""
        m = self.mask(n, calendar) if mask is None else mask
        S = self.baseline_signal() if signal is None else signal
        if regime is not None:
            S = S.mul(regime, axis=0) if isinstance(regime, pd.Series) else S * regime
        S = S.where(m & self.live, 0.0).clip(0, 1)
        k = self.cfg.top_k if k is None else k
        slots = self.cfg.max_slots if slots is None else slots
        R = self.momentum(self.cfg.rank_days) if rank is None else rank
        if k:
            keep = R.where(S > 0).rank(axis=1, ascending=False, method="first") <= k
            S = S.where(keep, 0.0)
        w = S.div(np.maximum(S.sum(axis=1), slots), axis=0)
        if sizing is not None:
            w = sizing(self, w, S)
        if smoothing is not None:
            w = smoothing(self, w)
        return self.enforce(w, m)

    def enforce(self, w: pd.DataFrame, m: pd.DataFrame | None = None) -> pd.DataFrame:
        """Spot, long-only, no leverage: 0 <= w <= 1/6, sum <= 1, nothing outside the universe or in dead coins."""
        w = w.reindex(index=self.index, columns=self.assets).fillna(0.0).clip(lower=0.0, upper=MAX_WEIGHT)
        w = w.where(self.live, 0.0)
        if m is not None:
            w = w.where(m, 0.0)
        tot = w.sum(axis=1)
        return w.div(np.maximum(tot, 1.0), axis=0)

    def pnl(self, w: pd.DataFrame, cost: float = SPOT_COST) -> pd.Series:
        """Weights decided at bar t's close earn bar t+1; every change pays `cost` per side."""
        gross = (w.shift(1) * self.ret.fillna(0)).sum(axis=1)
        fees = cost * w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
        return gross - fees

    # ---------- scoring ----------
    def evaluate(self, fn: Callable, name: str, params: dict | None = None, variants=VARIANTS, log: bool = True) -> pd.DataFrame:
        """fn(lab, n, calendar, **params) -> weights. One row per variant: train metrics (+ locked periods if open)."""
        params = params or {}
        rows = []
        for n, cal in variants:
            w = self.enforce(fn(self, n, cal, **params))
            p = self.pnl(w)
            row = {"name": name, "params": json.dumps(params, sort_keys=True), "n": n, "calendar": cal}
            row.update({f"train_{k}": v for k, v in metrics(p, *TRAIN, w).items()})
            if lockbox_open():
                for per, (lo, hi) in LOCKED.items():
                    row.update({f"{per}_{k}": v for k, v in metrics(p, lo, hi, w).items()})
            rows.append(row)
        d = pd.DataFrame(rows)
        if log:
            METHODS.mkdir(parents=True, exist_ok=True)
            with open(METHODS / "trials.jsonl", "a") as fh:
                fh.write(json.dumps({"name": name, "params": params, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                     **summary(d).to_dict()}, default=float) + "\n")
        return d

    # ---------- look-ahead detector ----------
    def truncate(self, T: pd.Timestamp) -> "Lab":
        """The world as seen at the close of the 4h bar opening at T: later bars, unfinished daily candles,
        feature rows dated after T and outcomes that had not ended by T are all removed."""
        cut = lambda df: df[df.index <= T]
        d_cut = lambda df: df[df.index + pd.Timedelta("1D") <= T + pd.Timedelta("4h")]
        f = self.features[self.features.date <= T].copy()
        f.loc[f.date + pd.DateOffset(months=3) > T, ["fwd_bot", "fwd_hold", "fwd_dead"]] = np.nan
        return Lab(close=cut(self.close), high=cut(self.high), low=cut(self.low), open=cut(self.open),
                   volume=cut(self.volume), ret=cut(self.ret), daily_close=d_cut(self.daily_close),
                   daily_volume=d_cut(self.daily_volume), features=f, cfg=self.cfg)

    def leak_check(self, fn: Callable, params: dict | None = None, n: int = 15, calendar: str = "Q-jan",
                   dates=("2020-06-15 08:00", "2022-03-01 16:00")) -> float:
        """Max |weight difference| up to T between running on all data and on data cut at T. Must be 0."""
        params = params or {}
        full = self.enforce(fn(self, n, calendar, **params))
        worst = 0.0
        for T in map(lambda x: pd.Timestamp(x, tz="UTC"), dates):
            t = self.truncate(T)
            part = t.enforce(fn(t, n, calendar, **params))
            diff = (full.loc[:T] - part.loc[:T]).abs().to_numpy()
            worst = max(worst, float(np.nanmax(diff)) if diff.size else 0.0)
        return worst

    @cached_property
    def baseline_results(self) -> pd.DataFrame:
        return self.evaluate(baseline, "baseline", log=False)


def baseline(lab: Lab, n: int, calendar: str) -> pd.DataFrame:
    return lab.build(n, calendar)


def summary(d: pd.DataFrame, base: pd.DataFrame | None = None) -> pd.Series:
    """Median variant (and worst drawdown); with `base`, the share of variants that beat it on Sharpe."""
    s = {"sharpe": d.train_sharpe.median(), "cagr": d.train_cagr.median(), "maxdd": d.train_maxdd.median(),
         "worst_dd": d.train_maxdd.min(), "turnover": d.train_turnover.median(), "exposure": d.train_exposure.median()}
    if base is not None:
        b = base.set_index(["n", "calendar"]).train_sharpe
        s["beats_base"] = (d.set_index(["n", "calendar"]).train_sharpe > b.reindex(d.set_index(["n", "calendar"]).index)).mean()
    if lockbox_open() and "oos_sharpe" in d:
        for per in LOCKED:
            s[f"{per}_sharpe"], s[f"{per}_cagr"], s[f"{per}_maxdd"] = d[f"{per}_sharpe"].median(), d[f"{per}_cagr"].median(), d[f"{per}_maxdd"].median()
    return pd.Series(s)


def show(d: pd.DataFrame, base: pd.DataFrame | None = None) -> str:
    s = summary(d, base)
    parts = [f"Sharpe {s.sharpe:.2f}", f"CAGR {s.cagr:+.0%}", f"maxDD {s.maxdd:.0%} (worst {s.worst_dd:.0%})",
             f"turnover {s.turnover:.0f}x/yr", f"invested {s.exposure:.0%}"]
    if "beats_base" in s:
        parts.append(f"beats baseline in {s.beats_base:.0%} of variants")
    return " | ".join(parts)


def build_cache() -> None:
    """Load every pool coin's 4h bars once and store them with the monthly features."""
    feats = load_features()
    _, _, spans = load_daily()
    close_d, vol_d, _ = load_daily()
    assets = sorted(set(feats.asset))
    p = load_bars(assets, spans)
    CACHE.mkdir(parents=True, exist_ok=True)
    d = {k: p[k] for k in ("close", "high", "low", "open", "volume", "ret")}
    d["open"], d["volume"] = d["open"].astype("float32"), d["volume"].astype("float32")
    keep = [a for a in assets if a in close_d.columns]
    d.update(daily_close=close_d[keep], daily_volume=vol_d[keep], features=feats)
    pd.to_pickle(d, CACHE / "lab.pkl")
    print("cached", d["close"].shape, "4h bars,", feats.date.nunique(), "monthly feature dates")


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "cache":
        build_cache()
    elif cmd == "baseline":
        lab = Lab.load()
        d = lab.baseline_results
        pd.set_option("display.width", 200)
        print(d.drop(columns=["params"]).round(3).to_string(index=False))
        print("\nbaseline:", show(d))


if __name__ == "__main__":
    main()
