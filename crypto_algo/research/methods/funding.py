"""Perpetual funding as a filter for the spot bot (baseline: aggressive k=6 + volume profile VAH entry / POC exit).

High funding = leveraged longs crowded and paying; often near local tops. Funding paid at T is known at T, so it is
used from the 4h bar that closes at T. Coins without a perpetual keep the baseline rule (no filter).
Rules declared before running: choose on 2018-11..2022 (>= 6 of 9 variants beat the baseline with higher median
Sharpe); holdout once on coins ranked 26-45, 2023-2026; adopt only what passes both.

    python -m research.methods.funding
"""
import pandas as pd

from algo.htf import state
from research.dynamic_universe import UNI
from research.harness import LOCKED, Lab, metrics, show, summary
from research.methods.entries import holdout_mask
from research.methods.profile_variants import profile_full

K = 6
GRID = {
    "skip_crowded_0.05%": dict(max_fund=0.0005),
    "skip_above_own_p90": dict(own_q=0.9),
    "exit_extreme_0.1%": dict(exit_fund=0.001),
    "market_hot_no_entries": dict(btc_max=0.0005),
    "only_neutral_or_negative": dict(max_fund=0.0001),
}


def funding_panel(lab: Lab) -> pd.DataFrame:
    """24h average funding (per 8h) on the 4h grid, NaN where the coin has no perpetual yet."""
    def go():
        cols = {}
        for a in lab.assets:
            f = UNI / "funding" / f"{a.split('#')[0]}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            if d.empty:
                continue
            t = pd.to_datetime(d.time, utc=True, format="mixed").dt.floor("h")
            s = pd.Series(d.rate.values, index=t).groupby(level=0).last()
            s = s.rolling("24h").mean()
            s.index = s.index.floor("4h") - pd.Timedelta("4h")     # the bar whose close is at (or after) the payment
            cols[a] = s.groupby(level=0).last()
        return pd.DataFrame(cols).reindex(index=lab.index, columns=lab.assets).ffill(limit=12)
    return lab.memo(("funding",), go)


def signal(lab, max_fund=None, own_q=None, exit_fund=None, btc_max=None):
    def go():
        g = lab.gate(("1d",), "close>ema50")
        c = lab.close
        poc, vah, _ = profile_full(lab, 180, 24)
        F = funding_panel(lab)
        ein = c > vah
        xout = c < poc
        if max_fund is not None:
            ein &= ~(F > max_fund)
        if own_q is not None:
            ein &= ~(F > F.rolling(540, min_periods=180).quantile(own_q))
        if exit_fund is not None:
            xout |= F > exit_fund
        if btc_max is not None:
            hot = F["BTCUSDT"].rolling(42, min_periods=6).mean() > btc_max
            ein = ein.mul(~hot, axis=0).astype(bool)
        total = 0
        for n in lab.cfg.entries:
            for m in lab.cfg.exits:
                e, x = lab.donchian(n, m)
                total = total + state(e & g & ein, x | ~g | xout)
        return (total / 6).where(lab.live, 0.0)
    return lab.memo(("fund", max_fund, own_q, exit_fund, btc_max), go)


def weights(lab, n, calendar, **p):
    return lab.build(n, calendar, k=K, signal=signal(lab, **p))


def main():
    lab = Lab.load()
    F = funding_panel(lab)
    print("coins with funding data:", int(F.notna().any().sum()), "| median funding (per 8h):", float(F.stack().median()))
    print("leak check:", lab.leak_check(weights, dict(max_fund=0.0005, own_q=0.9, exit_fund=0.001, btc_max=0.0005)))
    base = lab.evaluate(weights, "funding-base", {}, log=False)
    print("aggressive + profile (train):", show(base), flush=True)
    rows = []
    for name, p in GRID.items():
        d = lab.evaluate(weights, "funding", p)
        rows.append({"idea": name, **summary(d, base).to_dict()})
        print(f"{name:26s}", show(d, base), flush=True)
    t = pd.DataFrame(rows).set_index("idea")
    print("\npassed on 2018-2022:", list(t[(t.sharpe > base.train_sharpe.median()) & (t.beats_base >= 6 / 9)].index), flush=True)
    lo, hi = LOCKED["oos"]
    out = []
    for cal in ("Q-jan", "Q-feb", "Q-mar"):
        m = holdout_mask(lab, cal)
        for name, p in [("baseline", {})] + list(GRID.items()):
            w = lab.enforce(lab.build(20, cal, k=K, mask=m, signal=signal(lab, **p)), m)
            out.append({"idea": name, **metrics(lab.pnl(w), lo, hi)})
    print("\nholdout (coins 26-45, 2023-2026):")
    print(pd.DataFrame(out).groupby("idea")[["sharpe", "cagr", "maxdd"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
