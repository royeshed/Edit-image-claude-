"""Futures open interest from Binance's public archive (data.binance.vision daily "metrics" files, 5-minute rows).

Stored per 4h candle (the last reading inside the candle). Archive starts 2020-09 for BTC, later for most coins.

    python -m research.oi_data COINS_FILE    # data/universe/oi/{SYM}_4h.csv: oi (coins), oi_usd, taker_ls (taker buy/sell ratio)
"""
import io
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from research.dynamic_universe import UNI

OUT = UNI / "oi"
BASE = "https://data.binance.vision/data/futures/um/daily/metrics/{s}/{s}-metrics-{d}.zip"
DAYS = [f"{d:%Y-%m-%d}" for d in pd.date_range("2020-08-01", pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta("2D"))]
S = requests.Session()
S.verify = os.environ.get("REQUESTS_CA_BUNDLE", True)
S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))


def get(sym: str, day: str):
    for attempt in range(3):
        try:
            r = S.get(BASE.format(s=sym, d=day), timeout=30)
            break
        except requests.RequestException:
            if attempt == 2:
                return None
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        return pd.read_csv(z.open(z.namelist()[0]))


def exists(sym: str, day: str) -> bool:
    return S.head(BASE.format(s=sym, d=day), timeout=30).status_code == 200


def first_day(sym: str):
    """Binary search for the first archived day (listings are contiguous from their start)."""
    lo, hi = 0, len(DAYS) - 1
    if not exists(sym, DAYS[hi]):
        # delisted: find any day with data by probing monthly
        probe = [i for i in range(0, len(DAYS), 30) if exists(sym, DAYS[i])]
        if not probe:
            return None
        hi = probe[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if exists(sym, DAYS[mid]):
            hi = mid
        else:
            lo = mid + 1
    return lo if exists(sym, DAYS[lo]) else None


def coin(spot: str):
    f = OUT / f"{spot}_4h.csv"
    if f.exists():
        return
    base = spot[:-4]
    for fut in (spot, f"1000{base}USDT", f"1000000{base}USDT"):
        i0 = first_day(fut)
        if i0 is None:
            continue
        mult = 1000 if fut.startswith("1000") and not spot.startswith("1000") else 1
        with ThreadPoolExecutor(16) as ex:
            parts = [d for d in ex.map(lambda d: get(fut, d), DAYS[i0:]) if d is not None]
        d = pd.concat(parts)
        d.index = pd.to_datetime(d.create_time, utc=True)
        d = d.sort_index()
        d = d[~d.index.duplicated()]
        bar = d.index.floor("4h")
        out = pd.DataFrame({"oi": pd.to_numeric(d.sum_open_interest, errors="coerce") * mult,
                            "oi_usd": pd.to_numeric(d.sum_open_interest_value, errors="coerce"),
                            "taker_ls": pd.to_numeric(d.sum_taker_long_short_vol_ratio, errors="coerce")}).groupby(bar).last()
        out.index.name = "time"
        out.to_csv(f)
        return
    pd.DataFrame(columns=["oi", "oi_usd", "taker_ls"]).rename_axis("time").to_csv(f)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    coins = open(sys.argv[1]).read().split()
    with ThreadPoolExecutor(14) as ex:
        for i, _ in enumerate(ex.map(coin, coins)):
            if i % 10 == 0:
                print(i, len(coins), flush=True)
    print("done")
