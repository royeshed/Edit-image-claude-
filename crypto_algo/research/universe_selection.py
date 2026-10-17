"""Which coins should the bot trade next period? A test of selection rules without hindsight.

Protocol, fixed before looking at any result:
  * Candidates on each date: eligible coins in the top 100 by 90-day volume or by market cap
    (research.universe_features). Every feature uses only data from before the refresh date.
  * Rules come from a short list with an economic reason each (size, liquidity, maturity,
    attention, trend), plus a walk-forward model and random picks as the benchmark to beat.
  * Each rule runs on 32 variants: 4 list sizes x 8 refresh calendars (monthly, quarterly from
    Jan/Feb/Mar, yearly from Jan/Apr/Jul/Oct). A rule is judged by its median variant, never its best.
  * The choice is made on 2018-2022 only; 2023-2024 and 2025-2026 are reported afterwards.

    python -m research.universe_selection ic      # does any feature predict next quarter's bot P&L?
    python -m research.universe_selection rules   # portfolio backtests of every rule and variant
"""
import json
import sys

import numpy as np
import pandas as pd

from algo.trend_portfolio import PortfolioConfig
from .allocation import run
from .dynamic_universe import UNI, load_bars, load_daily
from .lab import SPOT_COST, metrics
from .run_dynamic_universe import weights
from .universe_features import coin_bot_returns  # noqa: F401  (same rules, used by `ic`)

FEATURES = ["vol30", "vol90", "vol365", "volmed90", "mcap", "turnover", "age", "mom90", "mom365", "volat90", "amihud90", "ath_dd"]
SIZES = (10, 15, 20, 25)
CALENDARS = {"monthly": range(1, 13), "Q-jan": (1, 4, 7, 10), "Q-feb": (2, 5, 8, 11), "Q-mar": (3, 6, 9, 12),
             "Y-jan": (1,), "Y-apr": (4,), "Y-jul": (7,), "Y-oct": (10,)}
TRAIN_END, VALID_END = pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")
MODEL_MIN_MONTHS = 24


def load_features() -> pd.DataFrame:
    f = pd.read_csv(UNI / "features.csv", parse_dates=["date"])
    f["date"] = f.date.dt.tz_convert("UTC") if f.date.dt.tz is not None else f.date.dt.tz_localize("UTC")
    return f


# ---------- 1. information coefficient ----------

def spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    return a[ok].rank().corr(b[ok].rank())


def ic() -> None:
    f = load_features().dropna(subset=["fwd_bot"])
    rows = []
    for d, g in f.groupby("date"):
        for col in FEATURES:
            x = g[col]
            if x.notna().sum() < 20:
                continue
            rows.append({"date": d, "feature": col,
                         "ic_bot": spearman(x, g.fwd_bot), "ic_hold": spearman(x, g.fwd_hold), "ic_dead": spearman(x, g.fwd_dead)})
    d = pd.DataFrame(rows)
    d["period"] = np.where(d.date < TRAIN_END, "train", np.where(d.date < VALID_END, "valid", "test"))
    out = []
    for (feat, per), g in d.groupby(["feature", "period"]):
        q = g.iloc[::3]  # every third month: forward windows do not overlap, so the t-stat is honest
        out.append({"feature": feat, "period": per, "ic_bot": g.ic_bot.mean(), "t_bot": q.ic_bot.mean() / q.ic_bot.std() * np.sqrt(len(q)),
                    "hit_bot": (g.ic_bot > 0).mean(), "ic_hold": g.ic_hold.mean(), "ic_dead": g.ic_dead.mean()})
    t = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    for col in ("ic_bot", "t_bot", "hit_bot", "ic_hold", "ic_dead"):
        print(f"\n{col}\n" + t.pivot(index="feature", columns="period", values=col)[["train", "valid", "test"]].round(2).to_string())

    # what the bot earned on a coin in the next 3 months, by its rank on each size measure
    f["vol_rank"] = f.groupby("date").vol90.rank(ascending=False)
    f["mcap_rank"] = f.groupby("date").mcap.rank(ascending=False)
    bins = [0, 10, 20, 40, 70, 200]
    for col in ("vol_rank", "mcap_rank"):
        f["bucket"] = pd.cut(f[col], bins, labels=["1-10", "11-20", "21-40", "41-70", "71+"])
        f["period"] = np.where(f.date < TRAIN_END, "train", "valid+test")
        g = f.groupby(["period", "bucket"], observed=True).agg(bot=("fwd_bot", "median"), hold=("fwd_hold", "median"),
                                                               dead=("fwd_dead", "mean"), n=("asset", "size"))
        print(f"\nmedian 3-month result by {col}\n" + g.round(3).to_string())


# ---------- 2. selection rules ----------

def rank_pct(g: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    return g[col].rank(pct=True, ascending=ascending)


def rule_scores(g: pd.DataFrame, rule: str) -> pd.Series:
    """Higher score = chosen first. Only columns known on the refresh date."""
    vr = g.vol90.rank(ascending=False)
    if rule == "volume90":
        return g.vol90
    if rule == "volume365":
        return g.vol365
    if rule == "median_volume":
        return g.volmed90
    if rule == "mcap":
        return g.mcap
    if rule == "mcap_liquid":       # biggest coins that also trade in size
        return g.mcap.where(vr <= 50)
    if rule == "size_blend":        # market cap and volume agree
        return rank_pct(g, "mcap") + rank_pct(g, "vol90")
    if rule == "mature_volume":     # volume, but only coins with a year of history
        return g.vol90.where(g.age >= 365)
    if rule == "liquidity":         # least price impact per dollar traded (Amihud)
        return -g.amihud90
    if rule == "trend_year":        # strongest 12-month trend among the 50 most traded
        return g.mom365.where(vr <= 50)
    if rule == "attention":         # hottest turnover (volume / market cap) among the top 50 by market cap
        return g.turnover.where(g.mcap.rank(ascending=False) <= 50)
    if rule == "quality":           # added after the 2018-2022 IC: big, old, calm coins
        return rank_pct(g, "mcap") + rank_pct(g, "age") + rank_pct(g, "volat90", ascending=False)
    raise KeyError(rule)


RULES = ["volume90", "volume365", "median_volume", "mcap", "mcap_liquid", "size_blend", "mature_volume", "liquidity",
         "trend_year", "attention", "quality"]


def model_scores(f: pd.DataFrame, alpha: float = 50.0) -> pd.Series:
    """Walk-forward ridge on within-date percentile ranks of the features -> rank of next-quarter bot P&L.

    At refresh date R it learns only from dates whose 3-month outcome had fully ended by R.
    """
    X = f.groupby("date")[FEATURES].rank(pct=True).fillna(0.5) - 0.5
    y = f.groupby("date").fwd_bot.rank(pct=True) - 0.5
    out = pd.Series(np.nan, index=f.index)
    dates = sorted(f.date.unique())
    outcome_end = f.date + pd.DateOffset(months=3)
    for R in dates:
        known = outcome_end <= R
        if f.loc[known, "date"].nunique() < MODEL_MIN_MONTHS:
            continue
        tr = known & y.notna()
        A, b = X[tr].values, y[tr].values
        beta = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ b)
        now = f.date == R
        out[now] = X[now].values @ beta
    return out


def membership(univ: dict, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """Same as dynamic_universe.membership, vectorised: 1 while an asset is in the latest chosen list."""
    pos = {c: i for i, c in enumerate(columns)}
    m = np.zeros((len(index), len(columns)))
    dates = sorted(univ)
    for i, d in enumerate(dates):
        lo = index.searchsorted(d)
        hi = index.searchsorted(dates[i + 1]) if i + 1 < len(dates) else len(index)
        m[lo:hi, [pos[c] for c in univ[d] if c in pos]] = 1.0
    return pd.DataFrame(m, index=index, columns=columns)


def universe(f: pd.DataFrame, score: pd.Series, months, n: int, start=None) -> dict:
    u = {}
    for R, g in f.groupby("date"):
        if R.month not in months or (start is not None and R < start):
            continue
        s = score[g.index].dropna()
        if len(s):
            u[R] = list(g.loc[s.sort_values(ascending=False).index[:n], "asset"])
    return u


def rules() -> None:
    cfg = PortfolioConfig(**json.load(open("config.json"))["portfolio"])
    f = load_features()
    close_d, vol_d, spans = load_daily()
    p = load_bars(sorted(set(f.asset)), spans)
    r, c = p["ret"], p["close"]
    from algo.htf import confirmation, confirmed_breakouts
    systems = [(a, b) for a in cfg.entries for b in cfg.exits]
    S = confirmed_breakouts(p, systems, confirmation(p, r.index, cfg.confirm_tfs, cfg.confirm), cfg.confirm_mode).where(r.notna(), 0)
    MOM = c / c.shift(cfg.rank_days * 6) - 1
    model_start = None

    scores = {name: rule_scores_all(f, name) for name in RULES}
    scores["model"] = model_scores(f)
    model_start = f.date[scores["model"].notna()].min()
    print("walk-forward model makes its first pick on", model_start.date())

    rng = np.random.default_rng(7)
    rows, pnls = [], {}

    def evaluate(name, cal, n, u):
        mask = membership(u, S.index, S.columns)
        pnl = run(weights(S.where(mask > 0, 0), MOM, cfg.top_k, cfg.max_slots), r, SPOT_COST)
        row = {"rule": name, "calendar": cal, "n": n}
        for tag, lo, hi in (("train", "2018", "2023"), ("valid", "2023", "2025"), ("test", "2025", "2100"),
                            ("s2020", "2020", "2100"), ("oos", "2023", "2100")):
            m = metrics(pnl[(pnl.index >= lo) & (pnl.index < hi)], "4h")
            row.update({f"{tag}_cagr": m["cagr"], f"{tag}_dd": m["maxdd"], f"{tag}_sh": m["sharpe"]})
        rows.append(row)
        return pnl

    for name, score in scores.items():
        for cal, months in CALENDARS.items():
            for n in SIZES:
                pnls[f"{name}|{cal}|{n}"] = evaluate(name, cal, n, universe(f, score, months, n))
        print("done", name, flush=True)
    # benchmark: n coins drawn at random from the 2n most traded, 10 draws per variant
    for cal, months in CALENDARS.items():
        for n in SIZES:
            for k in range(10):
                noise = pd.Series(rng.random(len(f)), index=f.index)
                score = noise.where(f.groupby("date").vol90.rank(ascending=False) <= 2 * n)
                evaluate(f"random{k}", cal, n, universe(f, score, months, n))
    print("done random", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(UNI / "selection_results.csv", index=False)
    pd.DataFrame(pnls).to_csv(UNI / "selection_pnl_4h.csv")
    summarize(d)


def rule_scores_all(f: pd.DataFrame, rule: str) -> pd.Series:
    return pd.concat([rule_scores(g, rule) for _, g in f.groupby("date")]).reindex(f.index)


def summarize(d: pd.DataFrame) -> None:
    d = d.copy()
    d["group"] = np.where(d.rule.str.startswith("random"), "RANDOM", d.rule)
    cols = ["train_sh", "train_cagr", "train_dd", "valid_sh", "valid_cagr", "test_sh", "test_cagr", "oos_sh", "oos_cagr", "oos_dd",
            "s2020_sh", "s2020_cagr", "s2020_dd"]
    med = d.groupby("group")[cols].median().sort_values("train_sh", ascending=False)
    worst = d.groupby("group")[["train_dd", "oos_dd"]].min().add_prefix("worst_")
    beat = d[d.group != "RANDOM"].merge(d[d.group == "RANDOM"].groupby(["calendar", "n"])[["train_sh", "oos_sh"]].median()
                                        .add_suffix("_rand").reset_index(), on=["calendar", "n"])
    beat = beat.assign(train_beats=beat.train_sh > beat.train_sh_rand, oos_beats=beat.oos_sh > beat.oos_sh_rand) \
        .groupby("group")[["train_beats", "oos_beats"]].mean()
    t = med.join(worst).join(beat)
    pd.set_option("display.width", 260)
    fmt = {k: "{:+.0%}".format for k in t.columns if k.endswith("_cagr")} | {k: "{:.0%}".format for k in t.columns if k.endswith("_dd")} | \
          {k: "{:.2f}".format for k in t.columns if k.endswith("_sh")} | {k: "{:.0%}".format for k in ("train_beats", "oos_beats")}
    print("\nmedian over 32 variants (train = 2018-2022, oos = 2023-2026); *_beats = share of variants above random")
    print(t.to_string(formatters=fmt))
    print("\nrank correlation, train Sharpe vs out-of-sample Sharpe across rules:",
          round(spearman(med.drop("RANDOM").train_sh, med.drop("RANDOM").oos_sh), 2))
    for col in ("train_sh", "oos_sh"):
        print(f"\n{col} by list size\n" + d.groupby(["group", "n"])[col].median().unstack().round(2).to_string())


if __name__ == "__main__":
    {"ic": ic, "rules": rules, "summary": lambda: summarize(pd.read_csv(UNI / "selection_results.csv"))}[sys.argv[1]]()
