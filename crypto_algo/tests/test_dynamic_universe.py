import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dynamic_universe import membership, segments, universes


def test_universe_uses_only_past_volume():
    idx = pd.date_range("2019-06-01", "2021-03-01", freq="D", tz="UTC")
    vol = pd.DataFrame({"A": 100.0, "B": 50.0, "C": 10.0}, index=idx)
    vol.loc["2021-01-01":, "C"] = 1e9          # C explodes on and after the refresh date
    u = universes(vol, "yearly", n=2, start="2020-01-01")
    assert u[pd.Timestamp("2021-01-01", tz="UTC")] == ["A", "B"]   # not known yet on Jan 1
    vol.loc["2020-12-01":, "C"] = 1e9
    u = universes(vol, "yearly", n=2, start="2020-01-01")
    assert u[pd.Timestamp("2021-01-01", tz="UTC")][0] == "C"        # known by then


def test_new_listing_needs_history_and_dead_coins_drop_out():
    idx = pd.date_range("2020-01-01", "2021-12-31", freq="D", tz="UTC")
    vol = pd.DataFrame({"OLD": 10.0, "NEW": np.nan, "DEAD": 99.0}, index=idx)
    vol.loc["2020-11-15":, "NEW"] = 500.0      # listed 47 days before the refresh
    vol.loc["2020-10-01":, "DEAD"] = np.nan     # stopped trading
    u = universes(vol, "yearly", n=3, start="2021-01-01", min_days=60)
    assert u[pd.Timestamp("2021-01-01", tz="UTC")] == ["OLD"]


def test_relisting_after_a_long_gap_is_a_new_segment():
    idx = pd.DatetimeIndex(list(pd.date_range("2022-01-01", "2022-05-13", tz="UTC")) + list(pd.date_range("2022-05-31", "2022-07-01", tz="UTC")))
    seg = segments(idx)
    assert len(seg) == 2 and seg[1][0] == pd.Timestamp("2022-05-31", tz="UTC")


def test_membership_follows_refresh_dates():
    idx = pd.date_range("2021-01-01", "2021-03-01", freq="4h", tz="UTC")
    u = {pd.Timestamp("2021-01-01", tz="UTC"): ["A"], pd.Timestamp("2021-02-01", tz="UTC"): ["B"]}
    m = membership(u, idx, ["A", "B"])
    assert m.loc["2021-01-31 20:00", "A"] == 1 and m.loc["2021-02-01 00:00", "A"] == 0 and m.loc["2021-02-01 00:00", "B"] == 1
