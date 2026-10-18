"""universe_ml - walk-forward machine-learning ranking of which coins the bot trades each quarter.

Idea: the live bot trades the n coins with the highest 90-day USDT volume among coins with >= 1 year of Binance
history ("mature_volume"). Volume is a proxy for "a coin the trend system trades well". Here a model learns that
mapping directly: at every monthly date R it is trained ONLY on feature rows whose 3-month outcome window had fully
ended by R (lab.known(R)), and it predicts, for the rows dated R, how well each coin will do over the next quarter.
The universe = the top n eligible coins by prediction. Features = within-date percentile ranks (centred, missing ->
neutral) of the 14 point-in-time coin features (vol30, vol90, vol365, volmed90, age, mom90, mom365, volat90,
amihud90, ath_dd, range90, mcap, cmc_rank, turnover). Targets: within-date rank of fwd_bot (the bot's own
next-quarter P&L on that coin), of fwd_hold (buy & hold), or membership of the top tercile of fwd_bot (the bot's
P&L is skewed: a few big trends make the year, so ranking the winners may matter more than the median).
Models (small, fixed, regularised hyper-parameters, no tuning loop): ridge (as research/universe_selection.py
model_scores), LightGBM regression / binary classification, sklearn HistGradientBoosting (second GBM
implementation as a robustness check), an equal-rank ensemble, and a hybrid that keeps the top 2n coins by
mature_volume and lets the model choose n of them (liquidity stays a hard prior, the model only re-ranks).

Fixed design choices (decided before any evaluation):
  * eligibility stays the baseline hygiene rule: age >= 365 days (NaN score otherwise), so the bot never trades a
    coin with < 1 year of history and the Donchian-200 / EMA50 machinery has data;
  * training population = the whole candidate pool (all feature rows, age >= 60 days, ~80 coins per date): the
    mature subset alone has only ~200 known rows when the model starts; percentile ranks are taken over the same
    pool on every date, so train and predict see the same feature scale;
  * a model needs >= 24 distinct known monthly dates (first prediction 2020-03-01); before that the score is the
    mature_volume rank, so every variant trades from 2018-11 exactly like the baseline until 2020-03.
  * the bot's other layers are untouched: baseline signal with the daily close > EMA50 confirmation on entries and
    open positions, top-3 by 14-day momentum, weight = signal / max(sum, 6) (<= 1/6 per coin, <= 50% invested).

Layer: universe -> lab.build(n, calendar, mask=lab.mask(n, calendar, score=score_layer(lab, n, **FROZEN))).
(the score depends on n only for the hybrid, which pre-filters the top 2n by volume.)

    python3 -m research.methods.universe_ml check    # no-peek unit check + train-only diagnostics (no P&L)
    python3 -m research.methods.universe_ml stage1
    python3 -m research.methods.universe_ml stage2
    python3 -m research.methods.universe_ml final
"""
import json
import sys

import numpy as np
import pandas as pd

from research.harness import METHODS, TRAIN, Lab, show, summary

FAMILY = "universe_ml"
FEATS = ["vol30", "vol90", "vol365", "volmed90", "age", "mom90", "mom365", "volat90", "amihud90", "ath_dd", "range90",
         "mcap", "cmc_rank", "turnover"]
MIN_AGE = 365            # baseline hygiene: >= 1 year of Binance history
MIN_MONTHS = 24          # known monthly dates needed before a model is used
# Small, fixed, regularised GBM settings (not tuned): shallow trees, slow learning, big leaves, row/column bagging.
LGB = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=200, subsample=0.7,
           subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, random_state=0, deterministic=True,
           force_col_wise=True, n_jobs=2, verbose=-1)
HGB = dict(max_iter=200, learning_rate=0.03, max_depth=3, max_leaf_nodes=7, min_samples_leaf=200, l2_regularization=10.0,
           early_stopping=False, random_state=0)
ENS = [("ridge", "bot"), ("lgbm", "bot"), ("lgbm", "top")]   # equal-rank ensemble members

# ----- GRID, declared before any evaluation (no P&L or IC computed when it was written) -----
GRID = [
    {"model": "ridge", "target": "bot"},                  # reference: universe_selection.model_scores (alpha 50)
    {"model": "ridge", "target": "hold"},                 # learn buy & hold instead of the bot's own P&L
    {"model": "ridge", "target": "top"},                  # linear probability of the top tercile of fwd_bot
    {"model": "lgbm", "target": "bot"},                   # non-linear: interactions (e.g. momentum x liquidity)
    {"model": "lgbm", "target": "top"},                   # GBM binary classifier: top tercile of fwd_bot
    {"model": "hgb", "target": "bot"},                    # sklearn GBM: must agree with lgbm if the GBM signal is real
    {"model": "ens"},                                     # mean within-date rank of ridge_bot, lgbm_bot, lgbm_top
    {"model": "ridge", "target": "bot", "hybrid": 2.0},   # top 2n by mature_volume, re-ranked by the model
    {"model": "lgbm", "target": "bot", "hybrid": 2.0},
    {"model": "ens", "hybrid": 2.0},
]
# Stage 2 (<= 6 more trials; total <= 16 <= 20), rule fixed in advance:
#   PASS = stage-1 trials with median Sharpe > baseline median Sharpe AND beats_base >= 0.67.
#   If PASS is empty: stop, FROZEN = None (stage 2 is not run).
#   Else for the best PASS (highest median Sharpe) evaluate its two neighbours:
#     ridge alpha 50 -> 10, 250;  lgbm/hgb -> n_estimators 100, 400 (weaker / stronger fit);
#     ens -> ens without lgbm_top, ens without ridge_bot;  hybrid 2.0 -> 1.5, 3.0 (on the same model).
#   It is frozen only if both neighbours have median Sharpe >= the baseline's (plateau); otherwise repeat once for
#   the next-best PASS. Within 0.05 median Sharpe the simpler candidate wins (ridge < lgbm < ens; hybrid counts as
#   simpler than the pure model because it changes fewer coins).
STAGE2 = []     # not run: stage-1 PASS set was empty (rule above)
FROZEN = None   # no stage-1 trial passed -> no improvement
NOTES = (
    "Verdict: no improvement; keep mature_volume. 10 stage-1 trials (all declared up front), none passed: median "
    "Sharpe across the 9 variants 1.20-1.54 vs baseline 1.57, and beats_base at most 4/9 (needed >= 6/9). Best were "
    "lgbm_bot hybrid-2n 1.54 (4/9), ridge_bot 1.51 (3/9), hgb_bot 1.50 (4/9). By the pre-declared rule stage 2 "
    "(neighbour/plateau tests) was not run, so the family used 10 of its 20 trials. "
    "What we learned: (1) The 14 point-in-time features predict next-quarter bot P&L only weakly. Walk-forward "
    "Spearman IC among the eligible coins (age >= 365), dates 2020-03..2022-10, was +0.025 for mature_volume itself, "
    "+0.043 for ridge_bot, +0.069 for ridge_hold and +0.027 for ridge_top. The gradient-boosting models were at or below "
    "zero (lgbm_bot -0.005, lgbm_top -0.016, hgb_bot -0.009) even with shallow, slow, heavily regularised trees. "
    "t-stats on non-overlapping quarters were <= 1.6, so with about 32 monthly cross-sections there is not enough "
    "signal for non-linear models. They fit noise. (2) Pure models replace 35-65% of the volume-chosen coins (top-15 "
    "overlap 0.34-0.63). That makes results depend heavily on the refresh calendar: for example lgbm_bot ranged "
    "0.87-2.02 Sharpe across the 9 variants, and every GBM model did best on the Q-feb calendar. That is luck in the "
    "refresh dates, not skill, and it pulls the median down. (3) ridge_bot, the reference linear model, cut drawdowns "
    "(median maxDD -24% vs -26%, worst variant -28% vs -38%) at the same CAGR (+52%), but its median Sharpe was lower "
    "(1.51) and it beat the baseline in only 3/9 variants. That is a possible risk-only effect, not an improvement, and "
    "it was not stress-tested. (4) The buy-and-hold target (ridge_hold) had the best IC but the worst small universes "
    "(n=10 Sharpe 0.77-1.14). It favours hot, volatile coins that the trend bot does not trade well. (5) The hybrid "
    "(keep the top 2n by volume, re-rank with the model) is the smallest change and the closest to the baseline, but "
    "it is still not better: the liquidity rank already holds most of what these features can predict. (6) The "
    "ensemble did not help (1.20 pure, 1.39 hybrid), because averaging weak and noisy members does not create signal. "
    "Mechanics: a model needs 24 known monthly dates, so predictions start 2020-03-01. Before that every variant is "
    "exactly the baseline (in 2018-2019 there are too few mature coins for selection to matter anyway). No-peek unit "
    "check: replacing fwd_* with noise for rows not yet known at R (R = 2020-06-01, 2021-07-01, 2022-10-01) changed no "
    "score dated <= R (max difference 0.0) for ridge bot/hold/top, lgbm bot/top, hgb bot and the ens hybrid. "
    "lab.leak_check on the reference ridge_bot layer is in leak_check. The layer (score_layer) plugs into "
    "lab.mask(score=...) and can be stacked later, but nothing here justifies replacing the live universe rule.")


# ---------- the model ----------

def _design(lab: Lab) -> tuple[np.ndarray, dict]:
    """Within-date percentile ranks of the features (centred, missing -> 0) and the three targets.
    Rows of different dates never mix, so a row's X and y depend only on rows with the same date."""
    f = lab.features

    def go():
        g = f.groupby("date")
        X = (g[FEATS].rank(pct=True) - 0.5).fillna(0.0).to_numpy(dtype=np.float64)
        rb = g.fwd_bot.rank(pct=True)
        y = {"bot": (rb - 0.5).to_numpy(), "hold": (g.fwd_hold.rank(pct=True) - 0.5).to_numpy(),
             "top": (rb > 2 / 3).astype(float).where(rb.notna()).to_numpy()}
        return X, y
    return lab.memo((FAMILY, "design"), go)


def _fit_predict(model: str, target: str, A: np.ndarray, b: np.ndarray, Xp: np.ndarray, alpha: float, trees: int) -> np.ndarray:
    if model == "ridge":
        mu = b.mean()
        beta = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (b - mu))
        return Xp @ beta
    if model == "lgbm":
        import lightgbm as lgb
        if target == "top":
            m = lgb.LGBMClassifier(**{**LGB, "n_estimators": trees}).fit(A, b.astype(int))
            return m.predict_proba(Xp)[:, 1]
        return lgb.LGBMRegressor(**{**LGB, "n_estimators": trees}).fit(A, b).predict(Xp)
    if model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        if target == "top":
            m = HistGradientBoostingClassifier(**{**HGB, "max_iter": trees}).fit(A, b.astype(int))
            return m.predict_proba(Xp)[:, 1]
        return HistGradientBoostingRegressor(**{**HGB, "max_iter": trees}).fit(A, b).predict(Xp)
    raise KeyError(model)


def predict(lab: Lab, model: str = "ridge", target: str = "bot", alpha: float = 50.0, trees: int = 200) -> pd.Series:
    """Walk-forward predictions aligned to lab.features.index (NaN where < MIN_MONTHS known dates).
    At each monthly date R the model is fit on rows with lab.known(R) (outcome window ended by R) and applied to
    the rows dated R only."""
    def go():
        f = lab.features
        X, ys = _design(lab)
        y = ys[target]
        out = np.full(len(f), np.nan)
        dates = f.date.to_numpy()
        for R in sorted(f.date.unique()):
            R = pd.Timestamp(R)
            tr = lab.known(R).to_numpy() & ~np.isnan(y)
            if pd.unique(dates[tr]).size < MIN_MONTHS:
                continue
            now = (f.date == R).to_numpy()
            out[now] = _fit_predict(model, target, X[tr], y[tr], X[now], alpha, trees)
        return pd.Series(out, index=f.index)
    return lab.memo((FAMILY, "pred", model, target, float(alpha), int(trees)), go)


def _rank_elig(s: pd.Series, f: pd.DataFrame, elig: pd.Series) -> pd.Series:
    return s.where(elig).groupby(f.date).rank(pct=True)


def model_score(lab: Lab, model: str, target: str | None, alpha: float, trees: int, members=None) -> pd.Series:
    """Model score per row (higher = better), NaN before the model has MIN_MONTHS of known outcomes."""
    f = lab.features
    elig = f.age >= MIN_AGE
    if model == "ens":
        mem = ENS if members is None else [tuple(m.split("_")) for m in members]
        parts = [_rank_elig(predict(lab, m, t, alpha, trees), f, elig) for m, t in mem]
        s = sum(parts) / len(parts)      # NaN while any member is unavailable (they all start together)
        return s
    return predict(lab, model, target, alpha, trees)


def score_layer(lab: Lab, n: int, model: str = "ridge", target: str | None = "bot", alpha: float = 50.0,
                trees: int = 200, hybrid: float | None = None, members=None) -> pd.Series:
    """Universe score aligned to lab.features.index (higher = chosen first, NaN = not eligible).
    Model dates: model score (optionally only among the top hybrid*n by 90-day volume). Earlier dates: mature_volume."""
    f = lab.features
    elig = f.age >= MIN_AGE
    vol = _rank_elig(f.vol90, f, elig)                      # mature_volume, as a within-date pct rank
    ms = _rank_elig(model_score(lab, model, target, alpha, trees, members), f, elig)
    have = ms.notna().groupby(f.date).transform("any")      # dates on which the model is available
    if hybrid is not None:
        k = int(np.ceil(hybrid * n))
        top = f.vol90.where(elig).groupby(f.date).rank(ascending=False, method="first") <= k
        ms = ms.where(top)
    s = ms.where(have, vol)
    return s.where(elig)


def weights(lab: Lab, n: int, calendar: str, **params) -> pd.DataFrame:
    key = (FAMILY, "score", n if params.get("hybrid") else None, json.dumps(params, sort_keys=True))
    s = lab.memo(key, lambda: score_layer(lab, n, **params))
    return lab.build(n, calendar, mask=lab.mask(n, calendar, score=s))


# ---------- checks and train-only diagnostics ----------

def no_peek_check(lab: Lab, params: dict, n: int = 15, dates=("2020-06-01", "2021-07-01", "2022-10-01")) -> float:
    """Unit check: scrambling fwd_* of every row NOT yet known at R must not change any score dated <= R.
    Returns the max |score difference| over rows dated <= R (must be 0.0)."""
    rng = np.random.default_rng(0)
    full = score_layer(lab, n, **params)
    worst = 0.0
    for R in map(lambda x: pd.Timestamp(x, tz="UTC"), dates):
        f2 = lab.features.copy()
        unk = ~lab.known(R)
        for c in ("fwd_bot", "fwd_hold", "fwd_dead"):
            f2.loc[unk, c] = rng.normal(0, 1, unk.sum())
        lab2 = Lab(close=lab.close, high=lab.high, low=lab.low, open=lab.open, volume=lab.volume, ret=lab.ret,
                   daily_close=lab.daily_close, daily_volume=lab.daily_volume, features=f2, cfg=lab.cfg)
        s2 = score_layer(lab2, n, **params)
        upto = lab.features.date <= R
        a, b = full[upto].to_numpy(), s2[upto].to_numpy()
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            return float("inf")
        ok = ~np.isnan(a)
        worst = max(worst, float(np.abs(a[ok] - b[ok]).max()) if ok.any() else 0.0)
    return worst


def diagnostics(lab: Lab) -> pd.DataFrame:
    """Train-only, no P&L: mean within-date Spearman IC of each GRID score vs next-quarter fwd_bot among eligible
    coins (dates whose outcome ended before 2023), and mean overlap of the top-15 with mature_volume's top-15."""
    f = lab.features
    elig = f.age >= MIN_AGE
    in_train = (f.date + pd.DateOffset(months=3) <= TRAIN[1]) & (f.date >= pd.Timestamp("2020-03-01", tz="UTC"))
    vol = f.vol90.where(elig)
    rows = []
    for params in [{"model": "mature_volume"}] + GRID:
        s = vol if params["model"] == "mature_volume" else score_layer(lab, 15, **params)
        ic, ov = [], []
        for R, g in f[in_train & elig].groupby("date"):
            x, y = s[g.index], g.fwd_bot
            ok = x.notna() & y.notna()
            if ok.sum() >= 10:
                ic.append(x[ok].rank().corr(y[ok].rank()))
            top = set(x.dropna().sort_values(ascending=False).index[:15])
            base = set(vol[g.index].dropna().sort_values(ascending=False).index[:15])
            ov.append(len(top & base) / max(len(base), 1))
        q = pd.Series(ic).iloc[::3]
        rows.append({"params": json.dumps(params, sort_keys=True), "ic_mean": np.mean(ic),
                     "ic_t_nonoverlap": q.mean() / q.std() * np.sqrt(len(q)), "ic_hit": np.mean(np.array(ic) > 0),
                     "top15_overlap": np.mean(ov)})
    return pd.DataFrame(rows)


# ---------- running ----------

def check() -> None:
    lab = Lab.load()
    # every base model once, plus the composite (ensemble + hybrid) that is built from them
    specs = [p for p in GRID if p["model"] != "ens" and "hybrid" not in p] + [{"model": "ens", "hybrid": 2.0}]
    for params in specs:
        print("no-peek", params, no_peek_check(lab, params), flush=True)
    pd.set_option("display.width", 200)
    print(diagnostics(lab).round(3).to_string(index=False))


def run(grid: list, tag: str) -> None:
    lab = Lab.load()
    base = lab.baseline_results
    print("baseline :", show(base), flush=True)
    print("   per variant:", base.set_index(["n", "calendar"]).train_sharpe.round(2).to_dict(), flush=True)
    rows = []
    for params in grid:
        d = lab.evaluate(weights, FAMILY, params)
        print(params, ":", show(d, base), flush=True)
        print("   per variant:", d.set_index(["n", "calendar"]).train_sharpe.round(2).to_dict(), flush=True)
        rows.append({"stage": tag, "params": json.dumps(params, sort_keys=True), **summary(d, base).to_dict()})
    out = METHODS / f"{FAMILY}_train.csv"
    prev = pd.read_csv(out) if out.exists() else pd.DataFrame()
    pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(out, index=False)


def final() -> dict:
    lab = Lab.load()
    base = lab.baseline_results
    trials = pd.read_csv(METHODS / f"{FAMILY}_train.csv")
    out = {"family": FAMILY, "idea": " ".join(__doc__.split("\n\n")[1].split()),
           "layer": "universe: lab.build(n, calendar, mask=lab.mask(n, calendar, score=score_layer(lab, n, **FROZEN)))",
           "grid": GRID + STAGE2, "n_trials": int(len(trials)), "frozen": FROZEN,
           "baseline_summary": summary(base).to_dict(), "trials": trials.to_dict(orient="records")}
    if FROZEN is not None:
        d = lab.evaluate(weights, FAMILY, FROZEN, log=False)
        out["frozen_summary"] = summary(d, base).to_dict()
        out["leak_check"] = max(lab.leak_check(weights, FROZEN),
                                lab.leak_check(weights, FROZEN, dates=("2019-09-10 12:00", "2021-05-19 04:00")))
        out["no_peek_check"] = no_peek_check(lab, FROZEN)
        out["verdict"] = "improvement"
    else:
        out["frozen_summary"] = None
        # the leak check is still run on the reference model so the layer is certified causal for later stacking
        ref = GRID[0]
        out["leak_check"] = max(lab.leak_check(weights, ref),
                                lab.leak_check(weights, ref, dates=("2019-09-10 12:00", "2021-05-19 04:00")))
        out["leak_check_params"] = ref
        out["verdict"] = "no_improvement"
    out["notes"] = NOTES
    with open(METHODS / f"{FAMILY}.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "trials"}, indent=1, default=float))
    return out


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "check":
        check()
    elif cmd == "stage1":
        run(GRID, "stage1")
    elif cmd == "stage2":
        run(STAGE2, "stage2")
    elif cmd == "final":
        final()


if __name__ == "__main__":
    main()
