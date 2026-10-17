import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dynamic_universe import membership as slow_membership
from research.universe_features import base, daily_features, eligible
from research.universe_selection import FEATURES, membership, model_scores, universe


def test_binance_tickers_map_to_coinmarketcap():
    assert base("1000PEPEUSDT") == ("PEPE", 1000.0)
    assert base("IOTAUSDT") == ("MIOTA", 1.0)
    assert base("BTCUSDT") == ("BTC", 1.0)


def test_daily_features_ignore_everything_after_the_cutoff():
    idx = pd.date_range("2020-01-01", "2021-06-01", freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    close = pd.DataFrame({"A": np.exp(np.cumsum(rng.normal(0, 0.03, len(idx)))), "B": 1.0 + rng.normal(0, 0.001, len(idx))}, index=idx)
    vol = pd.DataFrame({"A": 1e6, "B": 5e6}, index=idx)
    D = pd.Timestamp("2021-01-31", tz="UTC")
    f1 = daily_features(close, vol, D)
    close2, vol2 = close.copy(), vol.copy()
    close2.loc[D + pd.Timedelta("1D"):] *= 10
    vol2.loc[D + pd.Timedelta("1D"):] = 1e12
    pd.testing.assert_frame_equal(f1, daily_features(close2, vol2, D))
    f1["not_coin"] = False
    assert list(f1.index[eligible(f1)]) == ["A"]  # B never moves: stablecoin-like


def _panel(months=40, coins=30, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range(pd.Timestamp("2018-01-01", tz="UTC"), periods=months, freq="MS"):
        for i in range(coins):
            row = {"date": d, "asset": f"C{i}", "fwd_bot": rng.normal()}
            row.update({c: rng.normal() for c in FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def test_model_only_learns_from_outcomes_that_had_ended():
    f = _panel()
    s1 = model_scores(f)
    R = pd.Timestamp("2020-06-01", tz="UTC")
    g = f.copy()
    late = g.date + pd.DateOffset(months=3) > R  # outcomes still unknown on R
    g.loc[late, "fwd_bot"] = 1e6 * g.loc[late, FEATURES[0]]  # a perfect future signal, if it leaked
    s2 = model_scores(g)
    now = f.date == R
    assert s1[now].notna().all()
    np.testing.assert_allclose(s1[now].values, s2[now].values)


def test_universe_takes_top_scores_on_refresh_months_only():
    f = _panel(months=6, coins=5)
    score = pd.Series(f.asset.str[1:].astype(int), index=f.index)  # C4 best, C3 next
    u = universe(f, score, (1, 4), 2)
    assert sorted(u) == [pd.Timestamp("2018-01-01", tz="UTC"), pd.Timestamp("2018-04-01", tz="UTC")]
    assert u[pd.Timestamp("2018-01-01", tz="UTC")] == ["C4", "C3"]


def test_fast_membership_matches_reference():
    idx = pd.date_range("2021-01-01", "2021-06-01", freq="4h", tz="UTC")
    u = {pd.Timestamp("2021-01-01", tz="UTC"): ["A", "C"], pd.Timestamp("2021-03-01", tz="UTC"): ["B", "X"]}
    np.testing.assert_array_equal(membership(u, idx, ["A", "B", "C"]).values, slow_membership(u, idx, ["A", "B", "C"]).values)
