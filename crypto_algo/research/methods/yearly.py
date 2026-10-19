"""Why the bot did less in 2023-25 than in 2018-22: a year-by-year look at the market and the bot. Nothing is tuned.

Data cut at 2025-10-01 (locked year unread). Bot = the best bot on its live universe (15 coins, quarterly from January).
Per calendar year: the bot, BTC, the bot's 15 coins held equally (alts), how often the bot was invested, how many
breakout trades it took and how many made money, and how often a universe coin's daily trend gate was on.

    python -m research.methods.yearly
"""
import numpy as np
import pandas as pd

from research.methods import geometry as G
from research.methods.market_ideas import locked_lab, make


def trades(w, ret):
    """Per coin, the compounded return of each stretch it was held (sign = won or lost)."""
    held = w.shift(1).fillna(0) > 0
    out = []
    for c in w.columns:
        h, r = held[c], ret[c].fillna(0)
        rid = (h & ~h.shift(fill_value=False)).cumsum().where(h)
        if rid.notna().any():
            g = (1 + r[h]).groupby(rid[h]).prod() - 1
            first = r[h].groupby(rid[h]).apply(lambda x: x.index[0])
            out.append(pd.DataFrame({"ret": g.values, "start": first.values}))
    return pd.concat(out)


def main():
    lab = locked_lab()
    G.install(lab)
    m = lab.mask(15, "Q-jan")
    w = make({})(lab, 15, "Q-jan")
    pnl = lab.pnl(w)
    ret = lab.ret
    alts = (ret.fillna(0) * m.shift(1, fill_value=False)).sum(axis=1) / m.shift(1, fill_value=False).sum(axis=1).replace(0, np.nan)
    gate = lab.gate(("1d",), "close>ema50")
    t = trades(w, ret)
    t["year"] = pd.DatetimeIndex(t.start).year
    rows = []
    for y in range(2019, 2026):
        s = (pnl.index.year == y)
        eq = (1 + pnl[s].fillna(0)).cumprod()
        ty = t[t.year == y]
        rows.append({"year": y if y < 2025 else "2025 (to Sep)",
                     "bot": eq.iloc[-1] - 1, "bot_maxdd": (eq / eq.cummax() - 1).min(),
                     "BTC": (1 + ret["BTCUSDT"][s].fillna(0)).prod() - 1,
                     "15 coins held": (1 + alts[s].fillna(0)).prod() - 1,
                     "invested": w[s].sum(axis=1).mean(),
                     "trend_on": float((gate[s] & m[s]).sum().sum() / m[s].sum().sum()),
                     "trades": len(ty), "won": (ty.ret > 0).mean(),
                     "avg_win": ty.ret[ty.ret > 0].mean(), "avg_loss": ty.ret[ty.ret <= 0].mean(),
                     "best_trade": ty.ret.max()})
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    fmt = {k: "{:+.0%}".format for k in ("bot", "bot_maxdd", "BTC", "15 coins held", "avg_win", "avg_loss", "best_trade")}
    fmt.update({k: "{:.0%}".format for k in ("invested", "trend_on", "won")})
    print(d.to_string(index=False, formatters=fmt))


if __name__ == "__main__":
    main()
