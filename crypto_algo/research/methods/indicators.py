"""A broad indicator library as entry filters / exit triggers for the best bot (aggressive k=6 + volume profile).

Many trials raise the odds of a lucky pass, so the bar is three independent tests, declared before running:
  1. train 2018-11..2022 on the bot's universe: higher median Sharpe and beats the baseline in >= 6 of 9 variants;
  2. holdout A: coins ranked 26-45, 2023-2026 (median of 3 calendars) - higher Sharpe than the baseline;
  3. holdout B: coins ranked 46-65, 2023-2026 - higher Sharpe than the baseline.
Indicators passing all three are then combined greedily (on train only), and the combination re-checked on A and B.

    python -m research.methods.indicators
"""
import json

import numpy as np
import pandas as pd

from algo.htf import align, resample, state
from research.harness import LOCKED, Lab, metrics, show, summary
from research.methods.profile_variants import profile_full
from research.universe_selection import CALENDARS, membership, rule_scores_all, universe

K = 6


# ---------------- indicator building blocks (4h bars; everything trailing) ----------------
def ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def rma(x, n):
    return x.ewm(alpha=1 / n, adjust=False).mean()


def _tr(lab):
    c1 = lab.close.shift()
    return pd.concat([lab.high - lab.low, (lab.high - c1).abs(), (lab.low - c1).abs()]).groupby(level=0).max()


def tr(lab):
    return lab.memo(("tr",), lambda: _tr(lab).reindex(lab.index))


def atr(lab, n=14):
    return lab.memo(("atr", n), lambda: rma(tr(lab), n))


def rsi(lab, n=14):
    def go():
        d = lab.close.diff()
        return 100 - 100 / (1 + rma(d.clip(lower=0), n) / rma((-d).clip(lower=0), n))
    return lab.memo(("rsi", n), go)


def dmi(lab, n=14):
    def go():
        up, dn = lab.high.diff(), -lab.low.diff()
        pdm = up.where((up > dn) & (up > 0), 0.0)
        ndm = dn.where((dn > up) & (dn > 0), 0.0)
        a = atr(lab, n)
        pdi, ndi = 100 * rma(pdm, n) / a, 100 * rma(ndm, n) / a
        adx = rma(100 * (pdi - ndi).abs() / (pdi + ndi), n)
        return pdi, ndi, adx
    return lab.memo(("dmi", n), go)


def supertrend_up(lab, n=10, mult=3.0):
    def go():
        hl2 = (lab.high + lab.low) / 2
        a = atr(lab, n)
        up, dn, c = (hl2 - mult * a).to_numpy(), (hl2 + mult * a).to_numpy(), lab.close.to_numpy()
        out = np.zeros(c.shape, dtype=bool)
        for j in range(c.shape[1]):
            lb, ub, trend = np.nan, np.nan, False
            for t in range(c.shape[0]):
                if np.isnan(c[t, j]) or np.isnan(up[t, j]):
                    continue
                lb = up[t, j] if np.isnan(lb) or c[t - 1, j] < lb else max(up[t, j], lb)
                ub = dn[t, j] if np.isnan(ub) or c[t - 1, j] > ub else min(dn[t, j], ub)
                trend = True if c[t, j] > ub else False if c[t, j] < lb else trend
                out[t, j] = trend
        return pd.DataFrame(out, index=lab.index, columns=lab.assets)
    return lab.memo(("st", n, mult), go)


def daily(lab, fn_key, fn):
    """A daily-candle indicator, aligned so a value is used only after its candle closed."""
    def go():
        d = resample(lab.bars, "1d")
        return align(fn(d).astype(float), lab.index, "1d")
    return lab.memo(("daily", fn_key), go)


def mfi(lab, n=14):
    def go():
        tp = (lab.high + lab.low + lab.close) / 3
        mf = tp * lab.volume.astype(float)
        pos = mf.where(tp.diff() > 0, 0.0).rolling(n).sum()
        neg = mf.where(tp.diff() < 0, 0.0).rolling(n).sum()
        return 100 - 100 / (1 + pos / neg)
    return lab.memo(("mfi", n), go)


def cmf(lab, n=20):
    def go():
        rng = (lab.high - lab.low).replace(0, np.nan)
        mfm = ((lab.close - lab.low) - (lab.high - lab.close)) / rng
        v = lab.volume.astype(float)
        return (mfm * v).rolling(n).sum() / v.rolling(n).sum()
    return lab.memo(("cmf", n), go)


def chop(lab, n=14):
    def go():
        rng = lab.high.rolling(n).max() - lab.low.rolling(n).min()
        return 100 * np.log10(tr(lab).rolling(n).sum() / rng) / np.log10(n)
    return lab.memo(("chop", n), go)


def vwap(lab, n=180):
    def go():
        tp = (lab.high + lab.low + lab.close) / 3
        v = lab.volume.astype(float)
        return (tp * v).rolling(n).sum() / v.rolling(n).sum()
    return lab.memo(("vwap", n), go)


def ichimoku_above_cloud(lab):
    def go():
        h, l, c = lab.high, lab.low, lab.close
        tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
        kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
        a = ((tenkan + kijun) / 2).shift(26)
        b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
        return c > np.maximum(a, b)
    return lab.memo(("ichi",), go)


def psar_up(lab, step=0.02, mx=0.2):
    def go():
        h, l = lab.high.to_numpy(), lab.low.to_numpy()
        out = np.zeros(h.shape, dtype=bool)
        for j in range(h.shape[1]):
            up, sar, ep, af, started = True, np.nan, np.nan, step, False
            for t in range(h.shape[0]):
                if np.isnan(h[t, j]):
                    continue
                if not started:
                    sar, ep, started = l[t, j], h[t, j], True
                    continue
                sar = sar + af * (ep - sar)
                if up:
                    if l[t, j] < sar:
                        up, sar, ep, af = False, ep, l[t, j], step
                    elif h[t, j] > ep:
                        ep, af = h[t, j], min(af + step, mx)
                else:
                    if h[t, j] > sar:
                        up, sar, ep, af = True, ep, h[t, j], step
                    elif l[t, j] < ep:
                        ep, af = l[t, j], min(af + step, mx)
                out[t, j] = up
        return pd.DataFrame(out, index=lab.index, columns=lab.assets)
    return lab.memo(("psar",), go)


def kama_up(lab, n=10):
    def go():
        c = lab.close
        er = (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()
        sc = (er * (2 / 3 - 2 / 31) + 2 / 31) ** 2
        cv, sv = c.to_numpy(), sc.to_numpy()
        k = np.full(cv.shape, np.nan)
        for j in range(cv.shape[1]):
            prev = np.nan
            for t in range(cv.shape[0]):
                if np.isnan(cv[t, j]) or np.isnan(sv[t, j]):
                    continue
                prev = cv[t, j] if np.isnan(prev) else prev + sv[t, j] * (cv[t, j] - prev)
                k[t, j] = prev
        K_ = pd.DataFrame(k, index=c.index, columns=c.columns)
        return K_ > K_.shift(6)
    return lab.memo(("kama", n), go)


# ---------------- the candidates: name -> (kind, function(lab) -> bool DataFrame) ----------------
def candidates(lab: Lab) -> dict:
    c, h, l = lab.close, lab.high, lab.low
    v = lab.volume.astype(float)
    ret = lab.ret
    out = {
        # trend / momentum confirmation at entry
        "rsi_above_50": ("entry", lambda: rsi(lab) > 50),
        "rsi_not_overbought_80": ("entry", lambda: rsi(lab) < 80),
        "macd_hist_pos": ("entry", lambda: (lambda m: m - ema(m, 9))(ema(c, 12) - ema(c, 26)) > 0),
        "adx4h_above_25": ("entry", lambda: dmi(lab)[2] > 25),
        "dmi_plus_above_minus": ("entry", lambda: dmi(lab)[0] > dmi(lab)[1]),
        "aroon_up_70": ("entry", lambda: c.rolling(25).apply(np.argmax, raw=True) * 4 >= 70),
        "cci_pos": ("entry", lambda: ((lambda tp: (tp - tp.rolling(20).mean()) / (0.015 * (tp - tp.rolling(20).mean()).abs().rolling(20).mean()))((h + l + c) / 3)) > 0),
        "stoch_k_above_50": ("entry", lambda: 100 * (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min()) > 50),
        "roc20_pos": ("entry", lambda: c / c.shift(20) - 1 > 0),
        "trix_pos": ("entry", lambda: ema(ema(ema(c, 15), 15), 15).pct_change(fill_method=None) > 0),
        "ichimoku_above_cloud": ("entry", lambda: ichimoku_above_cloud(lab)),
        "supertrend_up": ("entry", lambda: supertrend_up(lab)),
        "psar_up": ("entry", lambda: psar_up(lab)),
        "hull_slope_up": ("entry", lambda: (lambda hma: hma > hma.shift(1))(
            (2 * c.ewm(span=10).mean() - c.ewm(span=20).mean()).ewm(span=4).mean())),
        "kama_up": ("entry", lambda: kama_up(lab)),
        "vortex_plus": ("entry", lambda: (h - l.shift()).abs().rolling(14).sum() > (l - h.shift()).abs().rolling(14).sum()),
        "elder_bull_power": ("entry", lambda: h - ema(c, 13) > 0),
        "above_vwap_30d": ("entry", lambda: c > vwap(lab)),
        "near_52w_high": ("entry", lambda: c >= 0.9 * h.rolling(2190, min_periods=500).max()),
        "daily_rsi_above_55": ("entry", lambda: daily(lab, "drsi55", lambda d: (lambda x: 100 - 100 / (1 + rma(x.clip(lower=0), 14) / rma((-x).clip(lower=0), 14)))(d["close"].diff()) > 55) > 0.5),
        "daily_macd_pos": ("entry", lambda: daily(lab, "dmacd", lambda d: ema(d["close"], 12) - ema(d["close"], 26) > 0) > 0.5),
        "weekly_close_above_ema10": ("entry", lambda: (lambda w: align((w > ema(w, 10)).astype(float), lab.index, "1w") > 0.5)(resample(lab.bars, "1w")["close"])),
        # volatility state
        "calm_atr_below_median": ("entry", lambda: (atr(lab) / c) < (atr(lab) / c).rolling(540, min_periods=180).median()),
        "vol_expanding": ("entry", lambda: atr(lab, 5) > atr(lab, 20)),
        "bb_percent_b_above_0.8": ("entry", lambda: (c - (c.rolling(20).mean() - 2 * c.rolling(20).std())) / (4 * c.rolling(20).std()) > 0.8),
        "keltner_breakout": ("entry", lambda: c > ema(c, 20) + 2 * atr(lab, 20)),
        "trending_chop_below_50": ("entry", lambda: chop(lab) < 50),
        "low_ulcer": ("entry", lambda: ((100 * (c / c.rolling(14).max() - 1)) ** 2).rolling(14).mean() ** 0.5 < 5),
        # volume
        "cmf_pos": ("entry", lambda: cmf(lab) > 0),
        "mfi_50_80": ("entry", lambda: (mfi(lab) > 50) & (mfi(lab) < 80)),
        "ad_line_rising": ("entry", lambda: (lambda ad: ad > ema(ad, 20))(((((c - l) - (h - c)) / (h - l).replace(0, np.nan)).fillna(0) * v).cumsum())),
        "volume_osc_pos": ("entry", lambda: ema(v, 5) > ema(v, 20)),
        "force_index_pos": ("entry", lambda: ema(ret.fillna(0) * v, 13) > 0),
        "vpt_rising": ("entry", lambda: (lambda vpt: vpt > ema(vpt, 20))((ret.fillna(0) * v).cumsum())),
        "ease_of_movement_pos": ("entry", lambda: ((((h + l) / 2).diff()) * (h - l) / v.replace(0, np.nan)).rolling(14).mean() > 0),
        # exits
        "exit_rsi_above_85": ("exit", lambda: rsi(lab) > 85),
        "exit_supertrend_flip": ("exit", lambda: ~supertrend_up(lab)),
        "exit_psar_flip": ("exit", lambda: ~psar_up(lab)),
        "exit_below_vwap_30d": ("exit", lambda: c < vwap(lab)),
        "exit_macd_cross_down": ("exit", lambda: (lambda m: m < ema(m, 9))(ema(c, 12) - ema(c, 26))),
        "exit_chop_above_61": ("exit", lambda: chop(lab) > 61.8),
        "exit_below_kama_trend": ("exit", lambda: ~kama_up(lab)),
        "exit_cmf_neg": ("exit", lambda: cmf(lab) < -0.1),
        "exit_ichimoku_below_cloud": ("exit", lambda: ~ichimoku_above_cloud(lab)),
    }
    return out


def signal(lab: Lab, entries=(), exits=()):
    key = ("ind", tuple(sorted(entries)), tuple(sorted(exits)))

    def go():
        cand = candidates(lab)
        g = lab.gate(("1d",), "close>ema50")
        c = lab.close
        poc, vah, _ = profile_full(lab, 180, 24)
        ein, xout = c > vah, c < poc
        for name in entries:
            ein = ein & lab.memo(("cand", name), cand[name][1]).fillna(False).astype(bool)
        for name in exits:
            xout = xout | lab.memo(("cand", name), cand[name][1]).fillna(False).astype(bool)
        total = 0
        for n in lab.cfg.entries:
            for m in lab.cfg.exits:
                e, x = lab.donchian(n, m)
                total = total + state(e & g & ein, x | ~g | xout)
        return (total / 6).where(lab.live, 0.0)
    return lab.memo(key, go)


def weights(lab, n, calendar, entries=(), exits=()):
    return lab.build(n, calendar, k=K, signal=signal(lab, tuple(entries), tuple(exits)))


def holdout(lab, lo_rank, hi_rank, calendar):
    f = lab.features
    s = rule_scores_all(f, "mature_volume")
    r = s.groupby(f.date).rank(ascending=False)
    u = universe(f, s.where((r > lo_rank) & (r <= hi_rank)), CALENDARS[calendar], hi_rank - lo_rank)
    return membership(u, lab.index, lab.assets).astype(bool)


def holdout_sharpe(lab, entries=(), exits=(), ranks=(25, 45)):
    lo, hi = LOCKED["oos"]
    sh, cg = [], []
    for cal in ("Q-jan", "Q-feb", "Q-mar"):
        m = lab.memo(("hold", ranks, cal), lambda: holdout(lab, *ranks, cal))
        w = lab.enforce(lab.build(20, cal, k=K, mask=m, signal=signal(lab, tuple(entries), tuple(exits))), m)
        mt = metrics(lab.pnl(w), lo, hi)
        sh.append(mt["sharpe"]); cg.append(mt["cagr"])
    return float(np.median(sh)), float(np.median(cg))


def main():
    lab = Lab.load()
    cand = candidates(lab)
    print(len(cand), "candidates")
    base = lab.evaluate(weights, "ind-base", {}, log=False)
    bA, bB = holdout_sharpe(lab), holdout_sharpe(lab, ranks=(45, 65))
    b = base.train_sharpe.median()
    print(f"baseline: train {show(base)} | holdout A Sharpe {bA[0]:.2f} CAGR {bA[1]:+.0%} | B {bB[0]:.2f} {bB[1]:+.0%}", flush=True)
    rows = []
    for name, (kind, _) in cand.items():
        p = {"entries": [name]} if kind == "entry" else {"exits": [name]}
        d = lab.evaluate(weights, "indicators", p)
        s = summary(d, base)
        row = {"name": name, "kind": kind, **s.to_dict(), "pass_train": bool(s.sharpe > b and s.beats_base >= 6 / 9)}
        A, B = holdout_sharpe(lab, **p), holdout_sharpe(lab, ranks=(45, 65), **p)
        row.update(A_sharpe=A[0], A_cagr=A[1], B_sharpe=B[0], B_cagr=B[1], pass_A=A[0] > bA[0], pass_B=B[0] > bB[0])
        row["pass_all"] = row["pass_train"] and row["pass_A"] and row["pass_B"]
        rows.append(row)
        print(f"{name:28s} train Sh {s.sharpe:.2f} CAGR {s.cagr:+.0%} DD {s.maxdd:.0%} beats {s.beats_base:.0%} | A {A[0]:+.2f} | B {B[0]:+.2f}"
              f"{'  <== PASS ALL' if row['pass_all'] else ''}", flush=True)
        pd.DataFrame(rows).to_csv("data/methods/indicators.csv", index=False)
    t = pd.DataFrame(rows)
    winners = list(t[t.pass_all].sort_values("sharpe", ascending=False).name)
    print("\npassed all three tests:", winners, flush=True)
    # greedy combination on train only
    chosen, best = [], b
    for name in winners:
        trial = chosen + [name]
        p = {"entries": [n for n in trial if cand[n][0] == "entry"], "exits": [n for n in trial if cand[n][0] == "exit"]}
        d = lab.evaluate(weights, "indicators-combo", p)
        s = summary(d, base)
        print(f"  + {name:26s} train Sh {s.sharpe:.2f} CAGR {s.cagr:+.0%} DD {s.maxdd:.0%}", flush=True)
        if s.sharpe > best + 0.02:
            chosen, best = trial, s.sharpe
    if chosen:
        p = {"entries": [n for n in chosen if cand[n][0] == "entry"], "exits": [n for n in chosen if cand[n][0] == "exit"]}
        A, B = holdout_sharpe(lab, **p), holdout_sharpe(lab, ranks=(45, 65), **p)
        print(f"\nCOMBINATION {chosen}: train Sharpe {best:.2f} | A {A[0]:+.2f} ({A[1]:+.0%}) vs {bA[0]:+.2f} | B {B[0]:+.2f} ({B[1]:+.0%}) vs {bB[0]:+.2f}")
        json.dump({"chosen": chosen, "params": p}, open("data/methods/indicators_combo.json", "w"))


if __name__ == "__main__":
    main()
