"""Order-flow data: taker-buy volume per 4h candle (the part of volume bought with market orders).

    python -m research.flow_data    # data/universe/flow/{SYM}_4h.csv for every coin in the research cache
"""
import time

import pandas as pd

from algo.data import make_exchange
from research.dynamic_universe import UNI

OUT = UNI / "flow"


def klines(ex, symbol: str, start_ms: int) -> pd.DataFrame:
    rows, cursor = [], start_ms
    while True:
        batch = ex.publicGetKlines({"symbol": symbol, "interval": "4h", "startTime": cursor, "limit": 1000})
        if not batch:
            break
        rows += batch
        if len(batch) < 1000:
            break
        cursor = int(batch[-1][0]) + 14_400_000
    d = pd.DataFrame([[int(r[0]), float(r[7]), float(r[10]), int(r[8])] for r in rows], columns=["ts", "quote_volume", "tb_quote", "trades"])
    d.index = pd.to_datetime(d.pop("ts"), unit="ms", utc=True)
    return d[~d.index.duplicated()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_pickle(UNI.parent / "cache" / "lab.pkl")
    syms = sorted({c.split("#")[0] for c in lab["close"].columns})
    ex = make_exchange("binance")
    start = int(pd.Timestamp("2017-08-01", tz="UTC").timestamp() * 1000)
    for i, s in enumerate(syms):
        f = OUT / f"{s}_4h.csv"
        if f.exists():
            continue
        for attempt in range(4):
            try:
                klines(ex, s, start).to_csv(f)
                break
            except Exception as e:
                print(s, "retry", e, flush=True); time.sleep(2 ** attempt)
        if i % 50 == 0:
            print(i, len(syms), s, flush=True)
    print("done")


if __name__ == "__main__":
    main()
