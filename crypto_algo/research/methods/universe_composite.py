"""universe_composite - which 10-20 coins the bot trades each quarter.

Idea: the live bot trades the n coins with the highest 90-day mean USDT volume among coins with >= 1 year of
Binance history ("mature_volume"). Volume alone is a noisy proxy for "a coin the trend system trades well": a
90-day mean is inflated by a few pump days, and on 2018-2022 a cross-sectional study of next-quarter bot P&L
per coin found older, bigger, calmer and more liquid coins did better (Spearman IC age +0.11, volat90 -0.10,
mcap +0.08, amihud90 -0.08, volmed90 +0.07, vol90 +0.06, turnover -0.06, momentum ~0). This family keeps the
rest of the bot fixed (signal with the daily close > EMA50 confirmation, top-3 by 14-day momentum, 1/6 slots)
and changes only WHICH coins are eligible, through a score per (refresh date, coin):

  * blend    - (1 - a) * pct-rank(vol90) + a * composite, composites of within-date percentile ranks among
               the eligible coins: quality = age, mcap, -volat90 ("big, old, calm"); all7 = the 7 IC-signed
               features equally weighted; ic7 = the same 7 weighted by |IC| (NOTE: those IC numbers were
               measured on 2018-2022 = train, so ic7 is partly in-sample by construction; ic_wf is its honest
               version); liquidity = vol90, volmed90, -amihud90 (robust liquidity, less pump-sensitive);
               size = mcap (index-style: market cap with a volume anchor).
  * ic_wf    - walk-forward IC weights: at refresh date R, each feature's weight is its mean within-date
               Spearman IC vs next-quarter bot P&L over the dates whose 3-month outcome had ended by R
               (lab.known logic), estimated on the whole candidate pool; vol90 until min_dates such dates exist.
  * rho      - correlation-aware greedy pick: walk down the base order, skip a coin whose 90-day daily-return
               RESIDUAL correlation (beta-adjusted to the equal-weight return of the eligible coins, candles
               closed by R) with an already accepted coin exceeds rho; skipped coins go to the back of the queue
               (so n coins are still picked when the pool allows). Raw correlations are 0.7-0.9 for almost every
               pair and regime-driven (0.9+ in crashes), so a raw threshold would only bind in crashes; the
               residual removes the common crypto factor and leaves "moves together" clusters (forks, same
               ecosystem).
  * floor    - drawdown-from-ATH filter: coins with ath_dd < floor (lost 90%+: dying projects whose rallies are
               mostly dead-cat bounces) are moved behind every other eligible coin (fallback fill, not removal).
  * min_age  - maturity threshold of the hygiene filter (baseline 365 days of Binance history).
NaN score = not eligible (younger than min_age, as in the baseline). lab.universe takes the top n by score on
each refresh date of the calendar, i.e. min(n, #eligible) coins, exactly like the baseline (in 2018-2019 fewer
than n coins have a year of Binance USDT history: 2-17, so until mid-2019/2020 every variant holds all of them).

Sector caps with CoinMarketCap tags were considered and NOT run: tags are multi-label (smart-contracts,
platform, defi, ... overlap on most large caps; ETH/ADA/DOGE all carry "mineable"), cover only ~40% of the
mature top-20 before mid-2020, and would need a hand-made primary-sector mapping chosen with hindsight. The
residual-correlation greedy pick is the data-driven version of the same idea ("do not pick many coins that move
together").

Layer: universe -> lab.build(n, calendar, mask=lab.mask(n, calendar, score=score_layer(lab, **FROZEN))).

    python3 -m research.methods.universe_composite stage1
    python3 -m research.methods.universe_composite stage2
    python3 -m research.methods.universe_composite final
"""
import json
import sys

import numpy as np
import pandas as pd

from research.harness import METHODS, Lab, show, summary

FAMILY = "universe_composite"
BASE_AGE = 365
FEATS = ["age", "mcap", "volat90", "amihud90", "volmed90", "vol90", "turnover"]
IC_STATIC = {"age": 0.11, "volat90": -0.10, "mcap": 0.08, "amihud90": -0.08, "volmed90": 0.07, "vol90": 0.06, "turnover": -0.06}
COMPS = {  # "-x" = lower is better
    "quality": {"age": 1, "mcap": 1, "-volat90": 1},
    "all7": {"age": 1, "mcap": 1, "-volat90": 1, "-amihud90": 1, "volmed90": 1, "vol90": 1, "-turnover": 1},
    "ic7": {("-" if v < 0 else "") + k: abs(v) for k, v in IC_STATIC.items()},
    "liquidity": {"vol90": 1, "volmed90": 1, "-amihud90": 1},
    "size": {"mcap": 1},
    # single-factor / reduced composites, used only as stage-2 neighbours or attribution
    "age": {"age": 1},
    "calm": {"-volat90": 1},
    "liq_med": {"vol90": 1, "volmed90": 1},
    "liq_amihud": {"vol90": 1, "-amihud90": 1},
}

# ----- GRID, declared before any evaluation (descriptive facts used to set it: no P&L was computed) -----
# * mature (age >= 365) coins per date on train: 2 (2018-10), 5 (2019-01), 15 (2019-07), 17 (2020-01), 31 (2020-04),
#   66 (2020-10), 99 (2022-01): composites can only differ from the baseline from ~2019-07 (n=10) / 2020-04 (n=20).
# * top-15 overlap with the baseline at quarterly dates 2019-10..2022-10: quality a=0.5 keeps 53-100% of the coins,
#   all7 53-100%, liquidity 73-100%, pure quality (a=1) 40-93%  -> the composites are real but moderate changes.
# * residual corr (beta to equal-weight pool) of a top-20 coin with its most correlated higher-ranked coin: > 0.5 for
#   0-40% of coins (median ~10%), > 0.4 for 6-55%  -> rho = 0.5 skips roughly one coin in ten.
# * ath_dd < -0.9 among the top-15 by volume: 0-6 coins in bear phases (2019-10, 2020-01, 2022-07), 0 in 2021.
GRID = [
    {"base": "blend", "comp": "quality", "a": 0.5},     # big, old, calm + volume anchor (the IC story)
    {"base": "blend", "comp": "all7", "a": 1.0},        # all 7 IC-signed features, equal weights
    {"base": "blend", "comp": "ic7", "a": 1.0},         # all 7, |IC| weights (in-sample weights, see docstring)
    {"base": "ic_wf", "min_dates": 12},                 # walk-forward IC weights (honest version of ic7)
    {"base": "blend", "comp": "liquidity", "a": 1.0},   # robust liquidity: mean, median volume, Amihud
    {"base": "blend", "comp": "size", "a": 0.5},        # index-style: market cap + volume
    {"base": "vol90", "rho": 0.5},                      # residual-correlation greedy de-duplication
    {"base": "vol90", "floor": -0.9},                   # dying coins (-90% from ATH) to the back
    {"base": "vol90", "min_age": 180},                  # half a year of history is enough
    {"base": "vol90", "min_age": 545},                  # 1.5 years of history
]
# Stage 2 (<= 10 more trials, total <= 20), rule fixed in advance:
#   PASS = stage-1 trials with median Sharpe > baseline median Sharpe (1.57) AND beats_base >= 0.67.
#   If PASS is empty: stop, FROZEN = None.
#   Else, for the best PASS (highest median Sharpe) evaluate its two neighbours along its own parameter:
#     blend quality a 0.5 -> 0.25, 0.75;  all7 / ic7 a 1.0 -> 0.5, 0.75;  ic_wf min_dates 12 -> 6, 24;
#     liquidity -> liq_med, liq_amihud;  size a 0.5 -> 0.25, 0.75;  rho 0.5 -> 0.4, 0.6;  floor -0.9 -> -0.8, -0.95;
#     min_age 180 -> 120, 270;  min_age 545 -> 450, 730.
#   It is frozen only if both neighbours have median Sharpe >= the baseline's (plateau). If not, repeat once for the
#   next-best PASS. For a multi-factor winner also run its simplest single-factor reductions (<= 2 trials, e.g.
#   quality -> age, calm at the same a); within 0.05 Sharpe the simpler one is preferred.
#   If two PASS ideas of different mechanisms (score vs filter) both have a plateau, run ONE stacked trial and keep the
#   stack only if it beats the best single idea by > 0.05 median Sharpe and itself passes.
# Stage 1 outcome (train medians of the 9 variants; baseline Sharpe 1.57, CAGR +52%, maxDD -26%, worst -38%):
#   quality a=.5 1.44 (beats 3/9) | all7 1.52 (3/9) | ic7 1.51 (3/9) | ic_wf 1.44 (4/9) | liquidity 1.41 (2/9)
#   size a=.5 1.54 (6/9, CAGR +56%) | rho .5 1.35 (1/9, worst DD -33%) | floor -.9 1.48 (2/9)
#   min_age 180 1.28 (0/9) | min_age 545 1.41 (3/9)
# PASS is empty (no median above 1.57), so by the rule declared above stage 2 is not run and nothing is frozen.
# (Closest: size a=.5, below the baseline median; paired daily-return t-stat vs baseline per variant median 0.47.)
STAGE2 = []
FROZEN = None  # set after stage 2 per the rule above; None if nothing beat the baseline
# lab.leak_check(weights, p) at 2019-09-10, 2020-06-15, 2021-05-19, 2022-03-01 for p = size blend, ic_wf, rho 0.5
LAYER_LEAK_CHECK = 0.0
NOTES = (
    "No improvement: 10 of 20 allowed trials, none passed (median Sharpe > 1.57 and beats >= 6/9). Stage 2 was not run, "
    "as the rule declared before running says. Closest was the index-style blend 0.5*rank(vol90)+0.5*rank(mcap): median "
    "Sharpe 1.54, beats 6/9, CAGR +56%; its paired daily-return t-stat vs the baseline has a median of 0.47 across "
    "variants, which is noise. "
    "WHAT DID NOT WORK AND WHY: (1) The IC-favoured composites (quality = age+mcap+calm, all7, ic7 with |IC| weights, "
    "walk-forward ic_wf) scored 1.44-1.52 and beat the baseline in only 3-4 of 9 variants. Even ic7 lost, although its "
    "weights come from the 2018-2022 IC study itself (in-sample). They swap the young, volatile new leaders "
    "(MATIC, AVAX, LUNA, FTM, SAND, MANA, NEAR, ATOM in 2020-2022) for old, calm laggards (LTC, ETC, EOS, VET, TRX, "
    "NEO, XMR). The per-coin IC measures the AVERAGE coin's standalone bot P&L, where calm coins whipsaw less. The "
    "portfolio keeps only the top 3 by momentum and earns its money from the right tail: the big trends of new "
    "leaders. So cross-sectional IC on single-coin P&L is the wrong target for picking this bot's universe. "
    "(2) A robust-liquidity composite (mean + median volume, Amihud) scored 1.41: smoothing out volume bursts removed "
    "the coins attracting attention, and those are the ones that trend. "
    "(3) A correlation-aware greedy pick (skip a coin whose 90-day residual correlation to an accepted coin is > 0.5) "
    "scored 1.35, the worst score-based rule. It did lower the worst drawdown (-33% vs -38%), but it diversifies away "
    "the hot sector, and the trend system wants to be in it. "
    "(4) Moving coins down 90%+ from their all-time high to the back scored 1.48. The filter binds only in bear "
    "phases, where such coins supply the first breakouts of a recovery. "
    "(5) Age: 180 days scored 1.28 (0/9 variants: young coins add whipsaws), 545 days 1.41 (too few coins in "
    "2019-2020, and the new leaders drop out). The existing 1-year threshold sits at the top of the 180/365/545 "
    "profile. "
    "(6) Sector caps from CoinMarketCap tags were not run. The tags are multi-label (smart-contracts/platform/defi on "
    "most large caps; ETH, ADA and DOGE all carry 'mineable') and cover about 40% of the mature top 20 before "
    "mid-2020, so they would need a hand-made sector map drawn up with hindsight. "
    "STRUCTURAL CAVEAT: few coins have a year of Binance USDT history before mid-2019 (2 to 17 at each refresh), so "
    "until 2019-07 (n=10) or 2020-04 (n=20) every variant holds all of them and universe rules cannot differ. Train "
    "evidence on universe selection therefore rests on about 3 years (one bull, one bear). The baseline's own 9 "
    "variants already range from 1.31 to 1.66 Sharpe just from n and the calendar, so swapping a few coins moves "
    "Sharpe by +-0.15 by chance. "
    "Leak checks: score-level truncation diffs are 0.0 for all 10 layers. lab.leak_check on the weights is 0.0 at "
    "the 4 dates (2019-09-10, 2020-06-15, 2021-05-19, 2022-03-01) for the size blend, ic_wf and the residual-correlation "
    "layers. Recommendation: keep mature_volume. score_layer(lab, **params) plugs into "
    "lab.mask(n, cal, score=...) so it can be stacked with other families later."
)


# ---------- the layer ----------

def _pct(f: pd.DataFrame, key: str, elig: pd.Series) -> pd.Series:
    """Within-date percentile rank among eligible coins (higher = better); neutral 0.5 where the feature is missing."""
    asc = not key.startswith("-")
    x = f[key.lstrip("-")].where(elig)
    return x.groupby(f.date).rank(pct=True, ascending=asc).where(elig).fillna(0.5).where(elig)


def _blend(f: pd.DataFrame, elig: pd.Series, comp: str, a: float) -> pd.Series:
    w = COMPS[comp]
    c = sum(wt * _pct(f, k, elig) for k, wt in w.items()) / sum(w.values())
    return (1 - a) * _pct(f, "vol90", elig) + a * c


def _ic_wf(lab: Lab, elig: pd.Series, min_dates: int) -> pd.Series:
    """Score = sum of feature pct-ranks weighted by their mean past IC (dates whose 3-month outcome had ended by R)."""
    f = lab.features
    pool = f.age >= 60  # IC estimated on the whole candidate pool (more coins per date)
    ics = {}
    for d, g in f[pool & f.fwd_bot.notna()].groupby("date"):
        if len(g) < 15:
            continue
        y = g.fwd_bot.rank()
        ics[d] = {c: g[c].rank().corr(y) for c in FEATS}
    ics = pd.DataFrame(ics).T.sort_index()
    ranks = {c: _pct(f, c, elig) for c in FEATS}
    out = _pct(f, "vol90", elig)  # fallback before enough history
    for R, idx in f[elig].groupby("date").groups.items():
        known = ics[ics.index + pd.DateOffset(months=3) <= R]
        if len(known) < min_dates:
            continue
        w = known.mean().fillna(0.0)
        out[idx] = sum(w[c] * ranks[c][idx] for c in FEATS)
    return out


def _corr_order(lab: Lab, s: pd.Series, rho: float, window: int = 90) -> pd.Series:
    """Greedy de-duplication on 90-day residual daily-return correlation; returns an order score (higher = earlier)."""
    f = lab.features
    lr = lab.memo((FAMILY, "logret_d"), lambda: np.log(lab.daily_close).diff())
    out = pd.Series(np.nan, index=s.index)
    ok = s.notna()
    for R, idx in s[ok].groupby(f.date[ok]).groups.items():
        ss = s[idx].sort_values(ascending=False, kind="mergesort")
        assets = f.asset[ss.index].tolist()
        win = lr.loc[(lr.index <= R - pd.Timedelta("1D")) & (lr.index > R - pd.Timedelta(f"{window + 1}D")), assets]
        accepted, skipped = [], []
        if len(win) >= window // 2 and len(assets) > 1:
            mkt = win.mean(axis=1)
            dm = mkt - mkt.mean()
            beta = (win - win.mean()).mul(dm, axis=0).sum() / (dm ** 2).sum()
            res = win - np.outer(mkt.values, beta.values)
            C = res.corr(min_periods=(2 * window) // 3).to_numpy()
            C = np.where(np.isnan(C), -1.0, C)
            for i in range(len(assets)):
                (skipped if accepted and C[i, accepted].max() > rho else accepted).append(i)
        else:
            accepted = list(range(len(assets)))
        order = accepted + skipped
        out[ss.index[order]] = np.arange(len(order), 0, -1, dtype=float)
    return out


def score_layer(lab: Lab, base: str = "vol90", comp: str | None = None, a: float = 0.0, min_dates: int = 12,
                min_age: int = BASE_AGE, floor: float | None = None, rho: float | None = None) -> pd.Series:
    """Universe score aligned to lab.features.index (higher = chosen first, NaN = not eligible).
    Uses only feature rows dated <= R (built from candles closed before R) and, for ic_wf, outcomes ended by R."""
    f = lab.features
    elig = f.age >= min_age
    if base == "vol90":
        s = f.vol90.where(elig).groupby(f.date).rank(pct=True)
    elif base == "blend":
        s = _blend(f, elig, comp, a)
    elif base == "ic_wf":
        s = _ic_wf(lab, elig, min_dates)
    else:
        raise KeyError(base)
    s = s.where(elig)
    if floor is not None:      # tier: coins above the ATH-drawdown floor first, the rest only as fallback
        s = s + 10.0 * (f.ath_dd >= floor).astype(float)
    if rho is not None:
        s = _corr_order(lab, s, rho)
    return s.where(elig)


def weights(lab: Lab, n: int, calendar: str, **params) -> pd.DataFrame:
    s = lab.memo((FAMILY, json.dumps(params, sort_keys=True)), lambda: score_layer(lab, **params))
    return lab.build(n, calendar, mask=lab.mask(n, calendar, score=s))


# ---------- running ----------

def run(grid: list, tag: str) -> None:
    lab = Lab.load()
    base = lab.baseline_results
    print("baseline :", show(base), flush=True)
    rows = []
    for params in grid:
        d = lab.evaluate(weights, FAMILY, params)
        print(params, ":", show(d, base), flush=True)
        print("   per variant:", d.set_index(["n", "calendar"]).train_sharpe.round(2).to_dict(), flush=True)
        rows.append({"stage": tag, "params": json.dumps(params, sort_keys=True), **summary(d, base).to_dict()})
    out = METHODS / f"{FAMILY}_train.csv"
    prev = pd.read_csv(out) if out.exists() else pd.DataFrame()
    pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(out, index=False)


def final() -> None:
    lab = Lab.load()
    base = lab.baseline_results
    trials = pd.read_csv(METHODS / f"{FAMILY}_train.csv")
    out = {"family": FAMILY, "idea": __doc__.split("\n\n")[1].replace("\n", " "),
           "layer": "universe: lab.build(n, calendar, mask=lab.mask(n, calendar, score=score_layer(lab, **FROZEN)))",
           "grid": GRID + STAGE2, "n_trials": int(len(trials)), "frozen": FROZEN,
           "baseline_summary": summary(base).to_dict(),
           "trials": trials.to_dict(orient="records")}
    if FROZEN is not None:
        d = lab.evaluate(weights, FAMILY, FROZEN, log=False)
        out["frozen_summary"] = summary(d, base).to_dict()
        out["leak_check"] = max(lab.leak_check(weights, FROZEN),
                                lab.leak_check(weights, FROZEN, dates=("2019-09-10 12:00", "2021-05-19 04:00")))
        out["verdict"] = "improvement"
    else:
        out["frozen_summary"] = None
        out["leak_check"] = LAYER_LEAK_CHECK  # no frozen candidate: max over the layers checked (see NOTES)
        out["verdict"] = "no_improvement"
    out["notes"] = NOTES
    with open(METHODS / f"{FAMILY}.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "trials"}, indent=1, default=float))
    return out


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "stage1":
        run(GRID, "stage1")
    elif cmd == "stage2":
        run(STAGE2, "stage2")
    elif cmd == "final":
        final()


if __name__ == "__main__":
    main()
