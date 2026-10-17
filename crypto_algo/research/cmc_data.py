"""Point-in-time market caps: CoinMarketCap's historical listings, one snapshot per month end.

    python -m research.cmc_data   # data/universe/cmc.csv

A snapshot dated D is CoinMarketCap's view at the end of day D, so a universe chosen at the
start of month M uses the snapshot of the last day of month M-1 (nothing from M itself).
"""
import json
import os
import time

import pandas as pd
import requests

from .dynamic_universe import UNI

URL = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listings/historical"
FIELDS = ["id", "symbol", "slug", "cmcRank", "circulatingSupply", "dateAdded"]


def snapshot(session: requests.Session, date: pd.Timestamp, limit: int = 1000) -> pd.DataFrame:
    r = session.get(URL, params={"date": date.strftime("%Y-%m-%d"), "limit": limit, "start": 1, "convertId": 2781}, timeout=60)
    r.raise_for_status()
    rows = []
    for x in r.json()["data"]:
        q = x["quotes"][0]
        rows.append({k: x.get(k) for k in FIELDS} | {"price": q["price"], "mcap": q["marketCap"], "cmc_volume": q["volume24h"],
                                                     "tags": ",".join(x.get("tags") or [])})
    d = pd.DataFrame(rows)
    d.insert(0, "date", date.strftime("%Y-%m-%d"))
    return d


def main() -> None:
    s = requests.Session()
    s.verify = os.environ.get("REQUESTS_CA_BUNDLE", True)
    out = UNI / "cmc"
    out.mkdir(parents=True, exist_ok=True)
    for d in pd.date_range("2017-12-31", pd.Timestamp.now(tz="UTC").tz_localize(None), freq="ME"):
        f = out / f"{d:%Y-%m-%d}.csv"
        if f.exists():
            continue
        for attempt in range(4):
            try:
                snapshot(s, d).to_csv(f, index=False)
                break
            except Exception as e:  # rate limit or network hiccup
                print(d.date(), "retry", e, flush=True)
                time.sleep(5 * 2 ** attempt)
        time.sleep(1)
    all_ = pd.concat([pd.read_csv(f) for f in sorted(out.glob("*.csv"))])
    all_.to_csv(UNI / "cmc.csv", index=False)
    print(len(all_), "rows,", all_.date.nunique(), "snapshots")


if __name__ == "__main__":
    main()
