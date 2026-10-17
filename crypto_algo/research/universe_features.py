"""Point-in-time features for choosing which coins the bot trades, and what each coin did afterwards.

One row per (refresh date R, asset). Every feature uses only data available before R: Binance
daily candles that closed by R, and the CoinMarketCap snapshot of the day before R. The forward
columns (what the coin did in the next 3 months) exist only to evaluate the rules.

    python -m research.universe_features pool   # assets that ever enter a candidate pool (need 4h bars)
    python -m research.universe_features build  # data/universe/features.csv
"""
import json
import re
import sys

import numpy as np
import pandas as pd

from algo.htf import confirmation, confirmed_breakouts
from algo.trend_portfolio import PortfolioConfig
from .dynamic_universe import UNI, load_bars, load_daily
from .lab import SPOT_COST

ALIAS = {"IOTA": "MIOTA", "BCHABC": "BCH", "BCHSV": "BSV", "BCC": "BCH", "BTTC": "BTT", "GXS": "GXC", "BEAMX": "BEAM",
         "VELODROME": "VELO", "BROCCOLI714": "BROCCOLI"}
SCALE = {"1000000": 1e6, "1000": 1e3, "1M": 1e6}
NOT_COINS = {"stablecoin", "tokenized-gold", "tokenized-stock", "tokenized-stocks", "wrapped-tokens", "tokenized-commodities"}
POOL = 100  # candidates per date: the top 100 by volume or by market cap
HORIZON = pd.DateOffset(months=3)


def base(sym: str) -> tuple[str, float]:
    """Binance base asset -> (CoinMarketCap ticker, price multiplier), e.g. 1000PEPE -> (PEPE, 1000)."""
    b = sym[:-4]
    m = re.match(r"^(1000000|1000|1M)(.+)$", b)
    b, k = (m.group(2), SCALE[m.group(1)]) if m else (b, 1.0)
    return ALIAS.get(b, b), k


def match_cmc(close: pd.DataFrame, spans: dict) -> pd.DataFrame:
    """Market cap per (snapshot date, asset): same ticker and a price within 25% of Binance's that day."""
    cmc = pd.read_csv(UNI / "cmc.csv")
    cmc["symbol"] = cmc.symbol.astype(str).str.upper()
    tags = cmc.dropna(subset=["tags"]).drop_duplicates("id", keep="last").set_index("id").tags  # static asset type only
    rows = []
    for d, g in cmc.groupby("date"):
        D = pd.Timestamp(d, tz="UTC")
        if D not in close.index:
            continue
        px = close.loc[D - pd.Timedelta("1D"): D]
        bysym = dict(list(g.groupby("symbol")))
        for a in px.columns[px.notna().any()]:
            t, k = base(spans[a][0])
            cand = bysym.get(t)
            if cand is None:
                continue
            ok = pd.Series(False, index=cand.index)
            for p in px[a].dropna():
                r = cand.price / (p / k)
                ok |= (r > 0.8) & (r < 1.25)
            if ok.any():
                c = cand[ok].sort_values("mcap", ascending=False).iloc[0]
                kind = set(str(tags.get(c.id, "")).split(","))
                rows.append({"date": D, "asset": a, "mcap": c.mcap, "cmc_rank": c.cmcRank, "cmc_id": c.id,
                             "not_coin": bool(kind & NOT_COINS)})
    return pd.DataFrame(rows)


def daily_features(close: pd.DataFrame, vol: pd.DataFrame, D: pd.Timestamp) -> pd.DataFrame:
    """Features from Binance daily candles up to and including day D (all closed by R = D + 1 day)."""
    c, v = close[close.index <= D], vol[vol.index <= D]
    lr = np.log(c.ffill()).diff()
    last = c.iloc[-1]

    def back(days):
        return c.ffill().iloc[-days - 1] if len(c) > days else pd.Series(np.nan, index=c.columns)

    f = pd.DataFrame({
        "vol30": v.iloc[-30:].mean(), "vol90": v.iloc[-90:].mean(), "vol365": v.iloc[-365:].mean(),
        "volmed90": v.iloc[-90:].median(),
        "age": c.notna().sum(),
        "mom90": last / back(90) - 1, "mom365": last / back(365) - 1,
        "volat90": lr.iloc[-90:].std(),
        "amihud90": (lr.abs() / v).iloc[-90:].mean() * 1e6,
        "ath_dd": last / c.max() - 1,
        "range90": c.iloc[-90:].max() / c.iloc[-90:].min(),
        "trading": c.iloc[-5:].notna().any(),
    })
    return f


def eligible(f: pd.DataFrame, min_days: int = 60) -> pd.Series:
    """Fixed hygiene rules, decided before any testing: tradable now, 60+ days old, a real coin."""
    stable_like = f.range90 < 1.10  # price barely moved for 3 months: stablecoin, gold token
    return f.trading & (f.age >= min_days) & ~stable_like & ~f.not_coin.fillna(False).astype(bool)


def coin_bot_returns(p: dict, cfg: PortfolioConfig) -> pd.DataFrame:
    """4h return of running the bot's entry/exit rules on each coin alone, after costs."""
    r = p["ret"]
    systems = [(n, m) for n in cfg.entries for m in cfg.exits]
    S = confirmed_breakouts(p, systems, confirmation(p, r.index, cfg.confirm_tfs, cfg.confirm), cfg.confirm_mode).where(r.notna(), 0)
    return S.shift(1) * r.fillna(0) - SPOT_COST * S.diff().abs().fillna(S)


def refresh_dates(close: pd.DataFrame) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp("2018-01-01", tz="UTC"), close.index[-1], freq="MS"))


def snapshot_features(close, vol, spans, cmc) -> pd.DataFrame:
    out = []
    by_date = dict(list(cmc.groupby("date")))
    for R in refresh_dates(close):
        D = R - pd.Timedelta("1D")
        f = daily_features(close, vol, D)
        m = by_date.get(D, pd.DataFrame(columns=["asset", "mcap", "cmc_rank", "not_coin"])).set_index("asset")
        f = f.join(m[["mcap", "cmc_rank", "not_coin"]])
        f["turnover"] = f.vol90 / f.mcap
        f = f[eligible(f)]
        f.insert(0, "date", R)
        out.append(f.rename_axis("asset").reset_index())
    return pd.concat(out, ignore_index=True)


def pool(feat: pd.DataFrame) -> pd.DataFrame:
    """Per date, the candidates any rule may choose from: top POOL by 90-day volume or by market cap."""
    feat = feat.copy()
    feat["vol_rank"] = feat.groupby("date").vol90.rank(ascending=False, method="first")
    feat["mcap_rank"] = feat.groupby("date").mcap.rank(ascending=False, method="first")
    return feat[(feat.vol_rank <= POOL) | (feat.mcap_rank <= POOL)]


def forward(feat: pd.DataFrame, bot4h: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """What each candidate did in the 3 months after R: bot alone, buy & hold, and whether it stopped trading."""
    feat = feat.copy()
    cf = close.ffill()
    for col in ("fwd_bot", "fwd_hold", "fwd_dead"):
        feat[col] = np.nan
    for R, g in feat.groupby("date"):
        E = R + HORIZON
        if E > close.index[-1]:
            continue
        a = [x for x in g.asset if x in bot4h.columns]
        win = bot4h.loc[(bot4h.index >= R) & (bot4h.index < E), a]
        feat.loc[g.index[g.asset.isin(a)], "fwd_bot"] = ((1 + win).prod() - 1).reindex(g.asset[g.asset.isin(a)]).values
        p0 = cf.loc[R - pd.Timedelta("1D"), g.asset]
        p1 = cf.loc[E - pd.Timedelta("1D"), g.asset]
        feat.loc[g.index, "fwd_hold"] = (p1 / p0 - 1).values
        feat.loc[g.index, "fwd_dead"] = close.loc[E - pd.Timedelta("5D"): E - pd.Timedelta("1D"), g.asset].isna().all().astype(float).values
    return feat


def main() -> None:
    close, vol, spans = load_daily()
    cmc = match_cmc(close, spans)
    cmc.to_csv(UNI / "cmc_matched.csv", index=False)
    feat = pool(snapshot_features(close, vol, spans, cmc))
    if sys.argv[1] == "pool":
        need = sorted({spans[a][0] for a in feat.asset})
        print(len(need), "symbols;", sum(not (UNI / f"{s}_4h.csv").exists() for s in need), "missing 4h")
        print(" ".join(s for s in need if not (UNI / f"{s}_4h.csv").exists()))
        return
    cfg = PortfolioConfig(**json.load(open("config.json"))["portfolio"])
    p = load_bars(sorted(set(feat.asset)), spans)
    feat = forward(feat, coin_bot_returns(p, cfg), close)
    feat.to_csv(UNI / "features.csv", index=False)
    print(len(feat), "rows,", feat.date.nunique(), "dates,", feat.asset.nunique(), "assets")


if __name__ == "__main__":
    main()
