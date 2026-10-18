"""Perpetual-futures funding rates from Binance's public archive (data.binance.vision), one CSV per coin.

    python -m research.funding_data COINS_FILE    # data/universe/funding/{SYM}.csv: time, rate (per 8h-equivalent)
"""
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import os

from research.dynamic_universe import UNI

OUT = UNI / "funding"
BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"
MONTHS = [f"{d:%Y-%m}" for d in pd.date_range("2019-09-01", "2026-09-01", freq="MS")]
S = requests.Session()
S.verify = os.environ.get("REQUESTS_CA_BUNDLE", True)


def month(sym: str, m: str):
    r = S.get(BASE.format(s=sym, m=m), timeout=30)
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        d = pd.read_csv(z.open(z.namelist()[0]))
    if "calc_time" not in d:  # very old files have no header
        d = pd.read_csv(io.BytesIO(zipfile.ZipFile(io.BytesIO(r.content)).read(z.namelist()[0])), header=None,
                        names=["calc_time", "funding_interval_hours", "last_funding_rate"])
    return d


def coin(spot: str):
    f = OUT / f"{spot}.csv"
    if f.exists():
        return
    base = spot[:-4]
    for fut in (spot, f"1000{base}USDT", f"1000000{base}USDT"):
        parts = [d for d in (month(fut, m) for m in MONTHS) if d is not None]
        if parts:
            d = pd.concat(parts)
            d["time"] = pd.to_datetime(pd.to_numeric(d.calc_time, errors="coerce"), unit="ms", utc=True)
            hrs = pd.to_numeric(d.get("funding_interval_hours", 8), errors="coerce").fillna(8)
            d["rate"] = pd.to_numeric(d.last_funding_rate, errors="coerce") * 8 / hrs   # per 8h, comparable across intervals
            d[["time", "rate"]].dropna().sort_values("time").to_csv(f, index=False)
            return
    pd.DataFrame(columns=["time", "rate"]).to_csv(f, index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    coins = open(sys.argv[1]).read().split()
    with ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(coin, coins)):
            if i % 20 == 0:
                print(i, len(coins), flush=True)
    print("done")
