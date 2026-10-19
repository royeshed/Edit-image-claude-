"""4h candles from OKX for the coins the bot's universe ever held (a second price/volume source, same coins).

    python -m research.okx_data   # data/okx/{SYMBOL}_4h.csv for every coin in the 20-coin universes (3 calendars)

Only candles before 2025-10-01 are kept: the locked year stays unread.
"""
import time

import pandas as pd
import requests

from research.harness import ROOT, Lab
from research.methods.indicators3 import LOCK

OUT = ROOT / "data" / "okx"
URL = "https://www.okx.com/api/v5/market/history-candles"


def candles(s: requests.Session, inst: str) -> pd.DataFrame:
    rows, after = [], int(LOCK.timestamp() * 1000)
    while True:
        for attempt in range(5):
            try:
                r = s.get(URL, params={"instId": inst, "bar": "4H", "after": after, "limit": 100}, timeout=30).json()
                break
            except Exception:
                time.sleep(2 ** attempt)
        data = r.get("data") or []
        if not data:
            break
        rows += data
        after = int(data[-1][0])
        time.sleep(0.11)
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows).iloc[:, :8]
    d.columns = ["ts", "open", "high", "low", "close", "volume", "vol_ccy", "quote_volume"]
    d["ts"] = pd.to_datetime(d.ts.astype("int64"), unit="ms", utc=True)
    d = d.set_index("ts").sort_index().astype(float)
    d = d[~d.index.duplicated()]
    return d[d.index < LOCK]


def coins() -> list[str]:
    lab = Lab.load().truncate(LOCK - pd.Timedelta("4h"))
    held = set()
    for cal in ("Q-jan", "Q-feb", "Q-mar"):
        m = lab.mask(20, cal)
        held |= set(m.columns[m.any()])
    return sorted({a.split("#")[0] for a in held})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    cs = coins()
    print(len(cs), "coins", flush=True)
    for i, sym in enumerate(cs):
        f = OUT / f"{sym}_4h.csv"
        if f.exists():
            continue
        d = candles(s, f"{sym[:-4]}-USDT")
        d.to_csv(f)
        print(i, sym, len(d), d.index[0] if len(d) else "-", flush=True)
    print("done")


if __name__ == "__main__":
    main()
