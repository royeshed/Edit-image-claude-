import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.trend_portfolio import PortfolioConfig
from research.harness import MAX_WEIGHT, Lab


def _lab():
    idx = pd.date_range("2018-06-01", "2023-06-01", freq="4h", tz="UTC")
    rng = np.random.default_rng(3)
    close = pd.DataFrame(np.exp(np.cumsum(rng.normal(0.0002, 0.02, (len(idx), 4)), axis=0)), index=idx, columns=list("ABCD"))
    close.loc["2021-01-01":, "D"] = np.nan  # D dies
    ret = close.pct_change(fill_method=None)
    daily = close.resample("1D").last()
    feats = pd.DataFrame({"date": pd.Timestamp("2019-01-01", tz="UTC"), "asset": list("ABCD"), "fwd_bot": 0.0, "fwd_hold": 0.0, "fwd_dead": 0.0})
    cfg = PortfolioConfig(strategy="breakout", confirm_tfs=("1d",), confirm="close>ema50", confirm_mode="gate", top_k=3, max_slots=6)
    return Lab(close=close, high=close * 1.01, low=close * 0.99, open=close, volume=close * 0 + 1, ret=ret,
               daily_close=daily, daily_volume=daily * 0 + 1, features=feats, cfg=cfg)


def test_enforce_keeps_spot_long_only_limits():
    lab = _lab()
    w = pd.DataFrame(0.9, index=lab.index, columns=lab.assets)
    w["B"] = -0.5
    e = lab.enforce(w)
    assert (e >= 0).all().all() and (e <= MAX_WEIGHT + 1e-12).all().all() and (e.sum(axis=1) <= 1 + 1e-12).all()
    assert (e.loc["2021-06-01":, "D"] == 0).all()


def test_evaluate_hides_2023_onwards_unless_lockbox_open():
    lab = _lab()
    os.environ.pop("LOCKBOX", None)
    fn = lambda lab, n, cal: lab.build(n, cal, mask=lab.live)
    d = lab.evaluate(fn, "t", log=False, variants=[(15, "Q-jan")])
    assert not any(c.startswith(("oos_", "valid_", "test_")) for c in d.columns)


def test_leak_check_passes_causal_and_catches_lookahead():
    lab = _lab()
    causal = lambda lab, n, cal: lab.build(n, cal, mask=lab.live)
    peek = lambda lab, n, cal: lab.build(n, cal, mask=lab.live, signal=(lab.close.shift(-6) > lab.close).astype(float))
    dates = ("2020-06-15 08:00",)
    assert lab.leak_check(causal, dates=dates) == 0
    assert lab.leak_check(peek, dates=dates) > 0
