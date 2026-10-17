"""Command line entry point.

  python main.py backtest --days 365            # download history and backtest
  python main.py backtest --synthetic           # offline test on generated data
  python main.py optimize --days 730            # grid search on train, verify on test
  python main.py scan --days 2000               # many coins x timeframes in one table
  python main.py replay --csv data/x.csv        # run the live loop over history (paper)
  python main.py paper                          # live market data, simulated orders
  python main.py live --i-understand-the-risk   # REAL orders (needs API keys in env)
"""
import argparse
import itertools
import logging
import os
import sys
from dataclasses import replace

from algo.backtest import format_stats, run_backtest
from algo.config import Config
from algo.data import load_csv, load_history, make_exchange, synthetic_ohlcv


def get_data(args, cfg: Config):
    if args.synthetic:
        return synthetic_ohlcv()
    if args.csv:
        return load_csv(args.csv)
    return load_history(cfg.exchange, cfg.symbol, cfg.timeframe, args.days)


def cmd_backtest(args, cfg: Config) -> None:
    df = get_data(args, cfg)
    res = run_backtest(df, cfg)
    print(f"{cfg.symbol} {cfg.timeframe}  {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}  ({len(df)} bars)\n")
    print(format_stats(res.stats))
    if args.trades:
        print()
        for t in res.trades:
            print(f"{t.entry_time:%Y-%m-%d %H:%M}  {t.exit_time:%Y-%m-%d %H:%M}  "
                  f"{t.entry_price:10.2f} → {t.exit_price:10.2f}  {t.return_pct:+7.2%}  {t.reason}")
    if args.plot:
        res.equity.to_csv(args.plot)
        print(f"\nEquity curve written to {args.plot}")


SCAN_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT"]
SCAN_TIMEFRAMES = ["1h", "4h", "1d"]


def cmd_scan(args, cfg: Config) -> None:
    """Backtest the configured strategy, unchanged, on every symbol x timeframe."""
    symbols = args.symbols.split(",") if args.symbols else SCAN_SYMBOLS
    timeframes = args.timeframes.split(",") if args.timeframes else SCAN_TIMEFRAMES
    print(f"{'symbol':<11}{'tf':<4}{'from':<12}{'final':>9}{'return':>9}{'B&H':>9}{'maxDD':>8}{'B&H DD':>8}"
          f"{'trades':>7}{'win':>5}{'PF':>6}{'hold':>7}")
    for symbol in symbols:
        for tf in timeframes:
            c = replace(cfg, symbol=symbol, timeframe=tf)
            try:
                df = load_history(c.exchange, symbol, tf, args.days)
            except Exception as e:
                print(f"{symbol:<11}{tf:<4}error: {e}")
                continue
            s = run_backtest(df, c).stats
            bh_dd = (df["close"] / df["close"].cummax() - 1).min()
            h = s["avg_hold_hours"]
            print(f"{symbol:<11}{tf:<4}{df.index[0]:%Y-%m-%d}  {s['final_equity']:>9,.0f}{s['total_return']:>9.1%}"
                  f"{s['buy_and_hold']:>9.0%}{s['max_drawdown']:>8.1%}{bh_dd:>8.0%}{s['trades']:>7}{s['win_rate']:>5.0%}"
                  f"{s['profit_factor']:>6.2f}{(f'{h:.0f}h' if h < 48 else f'{h / 24:.1f}d'):>7}")


def cmd_replay(args, cfg: Config) -> None:
    from algo.replay import replay

    df = get_data(args, cfg)
    logging.getLogger("algo").setLevel(logging.WARNING)
    eq = replay(df, cfg)
    bt = run_backtest(df, cfg).equity.loc[eq.index]
    print(f"live loop (paper): {eq.iloc[-1]:>10,.0f}  {eq.iloc[-1] / eq.iloc[0] - 1:+.1%}")
    print(f"backtest:          {bt.iloc[-1]:>10,.0f}  {bt.iloc[-1] / bt.iloc[0] - 1:+.1%}")
    print("Small differences are expected: the live loop sees one price per poll, the backtest uses intrabar highs/lows.")


def cmd_optimize(args, cfg: Config) -> None:
    """Grid search on the first 70% of data, then report the top configs on the unseen 30%."""
    df = get_data(args, cfg)
    split = int(len(df) * 0.7)
    train, test = df.iloc[:split], df.iloc[split - 300:]  # keep warmup bars for indicators
    grid = {
        "ema_fast": [10, 20, 30],
        "ema_slow": [50, 100],
        "atr_stop_mult": [1.5, 2.0, 3.0],
        "reward_risk": [1.5, 2.5, 4.0],
    }
    results = []
    for fast, slow, stop, rr in itertools.product(*grid.values()):
        c = replace(cfg, strategy=replace(cfg.strategy, ema_fast=fast, ema_slow=slow),
                    risk=replace(cfg.risk, atr_stop_mult=stop, reward_risk=rr))
        s = run_backtest(train, c).stats
        if s["trades"] >= 10:
            results.append((s["sharpe"], (fast, slow, stop, rr), c, s))
    results.sort(key=lambda r: r[0], reverse=True)
    print(f"{'fast':>5}{'slow':>5}{'stop':>6}{'rr':>5} | {'train sharpe':>12}{'train ret':>10} | {'test sharpe':>11}{'test ret':>9}{'test dd':>9}")
    for sharpe, params, c, s in results[:10]:
        t = run_backtest(test, c).stats
        print(f"{params[0]:>5}{params[1]:>5}{params[2]:>6}{params[3]:>5} | {sharpe:>12.2f}{s['total_return']:>10.1%} | "
              f"{t['sharpe']:>11.2f}{t['total_return']:>9.1%}{t['max_drawdown']:>9.1%}")
    print("\nPick parameters that hold up on the TEST columns, not just the best train score.")


def cmd_trade(args, cfg: Config, live: bool) -> None:
    from algo.broker import LiveBroker, PaperBroker
    from algo.live import Trader

    if live:
        if not args.i_understand_the_risk:
            sys.exit("Refusing to trade real money without --i-understand-the-risk")
        key, secret = os.environ.get("EXCHANGE_API_KEY"), os.environ.get("EXCHANGE_API_SECRET")
        if not (key and secret):
            sys.exit("Set EXCHANGE_API_KEY and EXCHANGE_API_SECRET (trade-only keys, no withdrawals)")
        exchange = make_exchange(cfg.exchange, key, secret)
        broker = LiveBroker(exchange, cfg.symbol)
    else:
        exchange = make_exchange(cfg.exchange)
        broker = PaperBroker(cfg.initial_capital, cfg.costs.fee_rate, cfg.costs.slippage)
        cfg = replace(cfg, state_file="paper_" + cfg.state_file)
    Trader(cfg, exchange, broker).run()


def main() -> None:
    p = argparse.ArgumentParser(description="Crypto algo trading bot")
    p.add_argument("command", choices=["backtest", "optimize", "scan", "replay", "paper", "live"])
    p.add_argument("--config", default="config.json")
    p.add_argument("--symbol", help="override config symbol, e.g. ETH/USDT")
    p.add_argument("--timeframe", help="override config timeframe, e.g. 4h")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--symbols", help="scan: comma-separated symbols")
    p.add_argument("--timeframes", help="scan: comma-separated timeframes")
    p.add_argument("--csv", help="backtest on a local OHLCV CSV instead of downloading")
    p.add_argument("--synthetic", action="store_true", help="use generated data (offline)")
    p.add_argument("--trades", action="store_true", help="print every trade")
    p.add_argument("--plot", metavar="FILE", help="write the equity curve to CSV")
    p.add_argument("--i-understand-the-risk", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config.load(args.config)
    if args.symbol:
        cfg = replace(cfg, symbol=args.symbol)
    if args.timeframe:
        cfg = replace(cfg, timeframe=args.timeframe)

    if args.command == "backtest":
        cmd_backtest(args, cfg)
    elif args.command == "optimize":
        cmd_optimize(args, cfg)
    elif args.command == "scan":
        cmd_scan(args, cfg)
    elif args.command == "replay":
        cmd_replay(args, cfg)
    else:
        cmd_trade(args, cfg, live=args.command == "live")


if __name__ == "__main__":
    main()
