"""Download a survivorship-free coin universe from Binance (delisted pairs included).

    python -m research.universe_data candidates      # list USDT pairs, minus stablecoins / leveraged tokens
    python -m research.universe_data daily           # 1d klines (with quote volume) for every candidate
    python -m research.universe_data bars 4h SYM...  # 4h klines for the given symbols

Binance keeps serving historical klines for symbols that no longer trade, so coins that died
(LUNA in 2022, FTT, XMR...) are part of the history the backtest can choose from.
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

from algo.data import make_exchange

OUT = Path(__file__).resolve().parents[1] / "data" / "universe"
STABLE = {"USDC", "BUSD", "TUSD", "PAX", "USDP", "DAI", "FDUSD", "USDS", "USDSB", "USDSOLD", "UST", "USTC", "AEUR", "EUR",
          "GBP", "AUD", "EURI", "XUSD", "USD1", "BFUSD", "RLUSD", "PYUSD", "USDE", "SUSD", "BKRW", "IDRT", "BIDR", "TRY", "BRL",
          "PAXG", "WBTC", "WBETH", "BETH", "BNSOL", "USDT"}
COLS = ["ts", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "tb_base", "tb_quote", "ignore"]


def candidates(ex) -> list[str]:
    syms = [s for s in ex.publicGetExchangeInfo()["symbols"] if s["quoteAsset"] == "USDT"]
    bases = {s["baseAsset"] for s in syms}
    out = []
    for s in syms:
        b = s["baseAsset"]
        lev = re.fullmatch(r"(.+?)(UP|DOWN|BULL|BEAR)", b)
        if b in STABLE or (lev and lev.group(1) in bases):
            continue
        out.append(s["symbol"])
    return sorted(set(out))


def klines(ex, symbol: str, interval: str, start_ms: int) -> pd.DataFrame:
    rows, cursor = [], start_ms
    step = {"1d": 86_400_000, "4h": 14_400_000}[interval]
    while True:
        batch = ex.publicGetKlines({"symbol": symbol, "interval": interval, "startTime": cursor, "limit": 1000})
        if not batch:
            break
        rows += batch
        if len(batch) < 1000:
            break
        cursor = int(batch[-1][0]) + step
    df = pd.DataFrame(rows, columns=COLS)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    df = df.set_index("ts")[["open", "high", "low", "close", "volume", "quote_volume"]].astype(float)
    return df[~df.index.duplicated()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ex = make_exchange("binance")
    start = int(pd.Timestamp("2017-08-01", tz="UTC").timestamp() * 1000)
    cmd = sys.argv[1]
    if cmd == "candidates":
        c = candidates(ex)
        (OUT / "candidates.json").write_text(json.dumps(c))
        print(len(c), "candidates")
    elif cmd == "daily":
        c = json.loads((OUT / "candidates.json").read_text())
        for i, sym in enumerate(c):
            f = OUT / f"{sym}_1d.csv"
            if f.exists():
                continue
            for attempt in range(3):
                try:
                    df = klines(ex, sym, "1d", start)
                    df.to_csv(f)
                    break
                except Exception as e:
                    print(sym, "retry", e, flush=True); time.sleep(2 ** attempt)
            if i % 50 == 0:
                print(f"{i}/{len(c)} {sym} {len(df)} days", flush=True)
    elif cmd == "bars":
        tf, symbols = sys.argv[2], sys.argv[3:]
        for i, sym in enumerate(symbols):
            f = OUT / f"{sym}_{tf}.csv"
            if f.exists():
                continue
            for attempt in range(3):
                try:
                    klines(ex, sym, tf, start).to_csv(f)
                    break
                except Exception as e:
                    print(sym, "retry", e, flush=True); time.sleep(2 ** attempt)
            if i % 20 == 0:
                print(f"{i}/{len(symbols)} {sym}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
