import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo import universe as uni
from algo.forward import HistoricalExchange
from algo.trend_portfolio import PaperPortfolioBroker, PortfolioConfig, PortfolioTrader

UNI = Path(__file__).resolve().parents[1] / "data" / "universe"
T = lambda s: pd.Timestamp(s, tz="UTC")


def daily(start, end, vol=1e6, seed=0, drift=0.0, flat=False):
    idx = pd.date_range(T(start), T(end), freq="D")
    rng = np.random.default_rng(seed)
    close = np.ones(len(idx)) if flat else 100 * np.exp(np.cumsum(rng.normal(drift, 0.04, len(idx))))
    return pd.DataFrame({"close": close, "quote_volume": vol}, index=idx)


def test_refresh_date_is_the_latest_quarter_start():
    assert uni.refresh_date(T("2026-10-18 04:00")) == T("2026-10-01")
    assert uni.refresh_date(T("2026-10-01 00:00")) == T("2026-10-01")
    assert uni.refresh_date(T("2026-09-30 20:00")) == T("2026-07-01")
    assert uni.refresh_date(T("2026-02-10"), "Q-jan") == T("2026-01-01")
    assert uni.refresh_date(T("2026-01-10"), "Q-mar") == T("2025-12-01")


def test_score_rules_age_trading_stable_and_past_only():
    R = T("2025-01-01")
    old = daily("2023-01-01", "2025-06-01", vol=5e6)
    assert uni.score(old, R) == pytest.approx(5e6)
    young = daily("2024-06-01", "2025-06-01", vol=9e9)
    assert np.isnan(uni.score(young, R))                       # < 365 days of history at R
    dead = daily("2022-01-01", "2024-11-01")
    assert np.isnan(uni.score(dead, R))                        # stopped trading before R
    stable = daily("2022-01-01", "2025-06-01", vol=9e9, flat=True)
    assert np.isnan(uni.score(stable, R))                      # price never moved: stablecoin-like
    relisted = pd.concat([daily("2021-01-01", "2023-12-01"), daily("2024-03-01", "2025-06-01", seed=1)])
    assert np.isnan(uni.score(relisted, R))                    # a 3-month gap restarts the history
    spike = old.copy()
    spike.loc[R:, "quote_volume"] = 1e12                       # volume on/after R is not known at R
    assert uni.score(spike, R) == pytest.approx(5e6)


def test_select_ranks_by_volume_and_skips_ineligible():
    R = T("2025-01-01")
    d = {"A/USDT": daily("2022-01-01", "2025-02-01", vol=3e6), "B/USDT": daily("2022-01-01", "2025-02-01", vol=7e6, seed=2),
         "C/USDT": daily("2024-09-01", "2025-02-01", vol=9e9), "D/USDT": daily("2022-01-01", "2025-02-01", vol=5e6, seed=3)}
    assert uni.select(d, R, size=2) == ["B/USDT", "D/USDT"]


def test_coins_only():
    bases = {"BTC", "BTCUP", "ETH", "USDC", "NVDAB"}
    assert uni.is_coin("BTC", bases) and uni.is_coin("ETH", bases)
    assert not uni.is_coin("BTCUP", bases) and not uni.is_coin("USDC", bases) and not uni.is_coin("NVDAB", bases)


class FakeExchange:
    """4h and daily candles for a few coins; `now` is the simulated clock (as in algo.forward)."""

    def __init__(self, coins: dict[str, pd.DataFrame]):
        self.coins = coins
        self.hx = HistoricalExchange(None, T("2020-01-01"))
        self.hx._full = self._full
        self.hx._info = {"symbols": [{"symbol": s.replace("/", ""), "baseAsset": s.split("/")[0], "quoteAsset": "USDT",
                                      "status": "TRADING"} for s in coins]}

    def _full(self, symbol, tf):
        d4 = self.coins[symbol]
        if tf == "4h":
            return d4
        return pd.DataFrame({"open": d4.open.resample("1D").first(), "high": d4.high.resample("1D").max(),
                             "low": d4.low.resample("1D").min(), "close": d4.close.resample("1D").last(),
                             "volume": d4.volume.resample("1D").sum(), "quote_volume": d4.quote_volume.resample("1D").sum()}).dropna()


def bars4h(start, end, drift, vol, seed):
    idx = pd.date_range(T(start), T(end), freq="4h")
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, len(idx))))
    return pd.DataFrame({"open": c, "high": c * 1.002, "low": c * 0.998, "close": c, "volume": 1.0, "quote_volume": vol / 6}, index=idx)


def test_trader_trades_only_the_universe_and_sells_coins_that_leave(tmp_path):
    # UP trends all along; DOWN falls. UP's volume collapses after July, so the Oct universe drops it.
    up = bars4h("2023-01-01", "2025-12-31", 0.001, 5e6, 1)
    up.loc[T("2025-07-01"):, "quote_volume"] = 1.0
    coins = {"UP/USDT": up, "DOWN/USDT": bars4h("2023-01-01", "2025-12-31", -0.001, 4e6, 2),
             "BTC/USDT": bars4h("2023-01-01", "2025-12-31", 0.0, 3e6, 3)}
    fx = FakeExchange(coins)
    cfg = PortfolioConfig(strategy="breakout", universe="mature_volume", universe_size=2, confirm_tfs=("1d",), confirm="close>ema50",
                          confirm_mode="gate", top_k=3, max_slots=6, state_file=str(tmp_path / "s.json"))
    t = PortfolioTrader(cfg, fx.hx, PaperPortfolioBroker(10_000, 0.001, 0.0005))
    t.tradable_only = False
    held_up_in_q3 = False
    for ts in pd.date_range(T("2025-07-01"), T("2025-10-03"), freq="4h"):
        fx.hx.now = ts + pd.Timedelta(minutes=1)
        t.step()
        if ts < T("2025-10-01"):
            assert t.universe == ["UP/USDT", "DOWN/USDT"]
            held_up_in_q3 |= t.broker.holdings.get("UP/USDT", 0) > 0
            assert "BTC/USDT" not in t.broker.holdings          # never outside the universe
    assert held_up_in_q3
    assert t.universe_date == str(T("2025-10-01")) and "UP/USDT" not in t.universe
    assert t.broker.holdings.get("UP/USDT", 0) == 0               # sold once it left the universe
    w = {s: q for s, q in t.broker.holdings.items()}
    t2 = PortfolioTrader(cfg, fx.hx, PaperPortfolioBroker(0, 0.001, 0.0005))  # state survives a restart
    assert t2.universe == t.universe and t2.broker.holdings == w and t2.positions == t.positions


def test_historical_exchange_never_serves_the_future():
    coins = {"UP/USDT": bars4h("2024-01-01", "2025-01-01", 0.001, 1e6, 1)}
    fx = FakeExchange(coins)
    fx.hx.now = T("2024-06-01 08:01")
    rows = fx.hx.fetch_ohlcv("UP/USDT", "4h", since=0)
    assert pd.Timestamp(rows[-1][0], unit="ms", tz="UTC") == T("2024-06-01 08:00")   # the forming candle, nothing later
    with pytest.raises(ValueError):
        fx.hx.publicGetKlines({"symbol": "UPUSDT", "interval": "1d", "endTime": int(T("2024-07-01").timestamp() * 1000)})


@pytest.mark.skipif(not (UNI / "features.csv").exists(), reason="research data not downloaded")
def test_live_universe_matches_the_research_rule():
    from research.universe_selection import CALENDARS, load_features, rule_scores_all, universe
    feats = load_features()
    research = universe(feats, rule_scores_all(feats, "mature_volume"), CALENDARS["Q-jan"], 15)
    d = {}
    for f in UNI.glob("*_1d.csv"):
        x = pd.read_csv(f, index_col=0)
        if len(x) and uni.is_coin(f.name[:-11], set()):
            x.index = pd.to_datetime(x.index, utc=True)
            d[f.name[:-7]] = x[["close", "quote_volume"]]
    for R in [T("2019-07-01"), T("2021-01-01"), T("2022-07-01"), T("2024-10-01"), T("2026-10-01")]:
        assert uni.select(d, R) == [a.split("#")[0] for a in research[R]]


class FlakyKlines:
    """An exchange whose daily-candle requests fail for some symbols."""

    def __init__(self, error_for: dict[str, Exception]):
        import ccxt  # noqa: F401
        self.error_for = error_for
        self.symbols = ["AAA", "BBB", "CCC"]

    def publicGetExchangeInfo(self):
        return {"symbols": [{"symbol": f"{b}USDT", "baseAsset": b, "quoteAsset": "USDT", "status": "TRADING"} for b in self.symbols]}

    def publicGetKlines(self, params):
        sym = params["symbol"]
        if sym in self.error_for:
            raise self.error_for[sym]
        d = daily("2023-01-01", "2024-12-31", vol={"AAAUSDT": 3e6, "BBBUSDT": 2e6, "CCCUSDT": 1e6}[sym], seed=len(sym))
        return [[int(t.timestamp() * 1000), 0, 0, 0, r.close, 0, 0, r.quote_volume] for t, r in d.iterrows()]


def test_choose_never_returns_a_universe_built_from_partial_data():
    import ccxt
    R = T("2025-01-01")
    assert uni.choose(FlakyKlines({}), R, size=2, pause=0) == ["AAA/USDT", "BBB/USDT"]
    assert uni.choose(FlakyKlines({"AAAUSDT": ccxt.BadSymbol("gone")}), R, size=2, pause=0) == ["BBB/USDT", "CCC/USDT"]
    with pytest.raises(RuntimeError):
        uni.choose(FlakyKlines({"AAAUSDT": ccxt.RequestTimeout("slow")}), R, size=2, pause=0)


def test_failed_refresh_keeps_the_previous_universe(tmp_path):
    import ccxt
    cfg = PortfolioConfig(strategy="breakout", universe="mature_volume", universe_size=2, state_file=str(tmp_path / "s.json"))
    ex = FlakyKlines({})
    ex.milliseconds = lambda: int(T("2025-01-01 04:01").timestamp() * 1000)
    t = PortfolioTrader(cfg, ex, PaperPortfolioBroker(10_000, 0.001, 0.0))
    t.universe, t.universe_date = ["OLD/USDT"], str(T("2024-10-01"))
    ex.error_for = {"BBBUSDT": ccxt.NetworkError("down")}
    import algo.universe as u
    orig = u.choose
    u.choose = lambda *a, **k: orig(*a, **{**k, "pause": 0})
    try:
        assert t.symbols_for(T("2025-01-01 00:00")) == ["OLD/USDT"] and t.universe_date == str(T("2024-10-01"))
        ex.error_for = {}
        assert t.symbols_for(T("2025-01-01 04:00")) == ["AAA/USDT", "BBB/USDT"] and t.universe_date == str(T("2025-01-01"))
    finally:
        u.choose = orig


def test_one_rejected_order_does_not_block_the_others(tmp_path):
    class Picky(PaperPortfolioBroker):
        def market_sell(self, symbol, qty, price):
            if symbol == "HALT/USDT":
                raise RuntimeError("Market is closed.")
            super().market_sell(symbol, qty, price)

    b = Picky(1_000, 0.001, 0.0)
    b.holdings = {"HALT/USDT": 10.0, "OK/USDT": 10.0}
    t = PortfolioTrader(PortfolioConfig(strategy="breakout", universe="mature_volume", state_file=str(tmp_path / "s.json")), None, b)
    prices = pd.Series({"HALT/USDT": 100.0, "OK/USDT": 100.0, "NEW/USDT": 50.0})
    t.rebalance(pd.Series({"HALT/USDT": 0.0, "OK/USDT": 0.0, "NEW/USDT": 1 / 6}), prices)
    assert "OK/USDT" not in b.holdings and b.holdings["HALT/USDT"] == 10.0 and b.holdings["NEW/USDT"] > 0
    b.holdings["HALT/USDT"] = 10.0
    n = len(b.fills)
    t.rebalance(pd.Series({"HALT/USDT": 0.0, "NEW/USDT": 1 / 6}), prices, frozen={"HALT/USDT"})
    assert all(f["symbol"] != "HALT/USDT" for f in b.fills[n:])   # a coin without a current candle is not traded


def test_state_from_the_fixed_list_version_still_sells_old_coins(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"last_decision": "2025-06-30 16:00:00+00:00", "paper_cash": 5_000.0,
                             "paper_holdings": {"OLD/USDT": 3.0}}))
    cfg = PortfolioConfig(strategy="breakout", universe="mature_volume", symbols=["OLD/USDT", "BTC/USDT"], state_file=str(f))
    t = PortfolioTrader(cfg, None, PaperPortfolioBroker(0, 0.001, 0.0))
    assert "OLD/USDT" in t.positions and "BTC/USDT" in t.positions
    with pytest.raises(ValueError):
        PortfolioTrader(replace(cfg, strategy="momentum"), None, PaperPortfolioBroker(0, 0.001, 0.0))


def test_forward_before_the_start_is_an_empty_ledger():
    from algo import forward
    coins = {"BTC/USDT": bars4h("2024-01-01", "2025-01-01", 0.0, 1e6, 1)}
    fx = FakeExchange(coins)
    cfg = PortfolioConfig(strategy="breakout", universe="mature_volume", confirm_tfs=("1d",), confirm="close>ema50")
    orig = forward.HistoricalExchange
    forward.HistoricalExchange = lambda *a, **k: fx.hx
    try:
        r = forward.run(cfg, None, T("2025-03-01"), T("2025-02-01"))
    finally:
        forward.HistoricalExchange = orig
    assert r["equity"].empty and forward.summary(r, 10_000) == {"days": 0}
