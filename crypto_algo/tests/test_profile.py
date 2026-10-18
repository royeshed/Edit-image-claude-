import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.profile import volume_profile


def _bars(n=400, seed=0):
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    c = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))}, index=idx)
    return c * 1.005, c * 0.995, c, pd.DataFrame({"A": rng.uniform(1, 10, n)}, index=idx)


def test_profile_uses_only_closed_days_and_never_the_future():
    h, l, c, v = _bars()
    poc, vah = volume_profile(h, l, c, v)
    T = c.index[300]
    p2, v2 = volume_profile(h.loc[:T], l.loc[:T], c.loc[:T], v.loc[:T])
    pd.testing.assert_frame_equal(poc.loc[:T], p2)
    pd.testing.assert_frame_equal(vah.loc[:T], v2)
    changes = poc["A"].dropna().diff().fillna(1) != 0
    assert set((poc["A"].dropna()[changes].index + pd.Timedelta("4h")).hour) <= {0}   # new values only at day closes


def test_poc_sits_where_the_volume_traded():
    h, l, c, v = _bars()
    v.iloc[:] = 1.0
    v.iloc[200:260] = 1000.0                       # heavy trading in one stretch
    poc, vah = volume_profile(h, l, c, v)
    t = c.index[263]                               # a day close within the window after the heavy stretch
    t = poc["A"].loc[:c.index[300]].last_valid_index()
    heavy = c["A"].iloc[200:260]
    assert heavy.min() * 0.97 <= poc["A"].loc[t] <= heavy.max() * 1.03
    assert vah["A"].loc[t] >= poc["A"].loc[t]


def test_atr_and_cmf_are_trailing_and_bounded():
    from algo.profile import atr_panel, cmf_panel
    h, l, c, v = _bars()
    a, m = atr_panel(h, l, c, 14), cmf_panel(h, l, c, v)
    T = c.index[250]
    pd.testing.assert_frame_equal(a.loc[:T], atr_panel(h.loc[:T], l.loc[:T], c.loc[:T], 14))
    pd.testing.assert_frame_equal(m.loc[:T], cmf_panel(h.loc[:T], l.loc[:T], c.loc[:T], v.loc[:T]))
    assert (a.dropna() > 0).all().all() and m.dropna().abs().max().max() <= 1
