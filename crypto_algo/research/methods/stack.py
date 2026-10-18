"""Stack the method families that improved the bot on train, then open the lockbox once.

Components (each frozen by its own family on 2018-11..2022 and adversarially reviewed):
  R  regime : trade only while BTC's last completed weekly close is above its weekly EMA20 (research/methods/regime.py)
  X  exits  : Donchian exits on the 5- and 10-bar low instead of 10/20, daily-gate exit kept (research/methods/exits.py)
  Z  sizing : inverse-volatility slots, weight * min(1, ivol / coin's 30-day vol), top 3 (research/methods/sizing.py)

Rules declared before running anything in this file:
  * PRIMARY (train only): start from the full stack R+X+Z(0.7). Drop a component if the stack without it has a
    median Sharpe more than 0.05 higher AND beats the full stack in >= 6 of 9 variants. Components that failed
    their review are left out from the start.
  * Return-leaning options (reported, never used to pick PRIMARY): R+X without sizing, and the full stack with
    ivol 1.0 and with k=4 (added to the lockbox list after the train run, before the lockbox was opened).
  * Lockbox (2023-01..2026-10, opened once): adopt PRIMARY only if its median Sharpe beats the current bot's AND it
    beats the current bot in >= 6 of 9 variants. Otherwise keep the current bot. Nothing is re-chosen afterwards.

    python -m research.methods.stack train
    LOCKBOX=open python -m research.methods.stack lockbox
"""
import json
import sys

import pandas as pd

from research.harness import METHODS, Lab, lockbox_open, show, summary
from research.methods import exits, regime, sizing

FAMILY = "stack"
REVIEW_FAILED: set[str] = set()   # filled in from the reviews before running

STACKS = {
    "R+X+Z0.7": dict(R=True, X=True, ivol=0.7, k=3),
    "R+X": dict(R=True, X=True, ivol=None, k=3),
    "R+Z0.7": dict(R=True, X=False, ivol=0.7, k=3),
    "X+Z0.7": dict(R=False, X=True, ivol=0.7, k=3),
    "R+X+Z1.0": dict(R=True, X=True, ivol=1.0, k=3),
    "R+X+Z1.0 k4": dict(R=True, X=True, ivol=1.0, k=4),
    "R+X+Z0.7 k4": dict(R=True, X=True, ivol=0.7, k=4),
}
SINGLES = {"R": dict(R=True, X=False, ivol=None, k=3), "X": dict(R=False, X=True, ivol=None, k=3),
           "Z0.7": dict(R=False, X=False, ivol=0.7, k=3)}


def weights(lab: Lab, n: int, calendar: str, R: bool, X: bool, ivol: float | None, k: int) -> pd.DataFrame:
    return lab.build(n, calendar, k=k, slots=6,
                     regime=regime.regime(lab, calendar, **regime.FROZEN) if R else None,
                     signal=exits.signal(lab, **exits.FROZEN) if X else None,
                     sizing=sizing.make_sizing(ivol=ivol) if ivol else None)


def primary(results: dict[str, pd.DataFrame]) -> str:
    full = results["R+X+Z0.7"]
    keep = {"R", "X", "Z0.7"} - REVIEW_FAILED
    for comp, without in (("R", "X+Z0.7"), ("X", "R+Z0.7"), ("Z0.7", "R+X")):
        d = results[without]
        better = d.set_index(["n", "calendar"]).train_sharpe > full.set_index(["n", "calendar"]).train_sharpe
        if d.train_sharpe.median() > full.train_sharpe.median() + 0.05 and better.mean() >= 6 / 9:
            keep.discard(comp)
    return "+".join(c for c in ("R", "X", "Z0.7") if c in keep)


def train() -> None:
    lab = Lab.load()
    base = lab.baseline_results
    print("baseline   :", show(base), flush=True)
    results, rows = {}, []
    for name, p in STACKS.items():
        d = lab.evaluate(weights, FAMILY, p)
        results[name] = d
        print(f"{name:12s}:", show(d, base), flush=True)
        rows.append({"stack": name, **summary(d, base).to_dict()})
    pd.DataFrame(rows).to_csv(METHODS / "stack_train.csv", index=False)
    choice = primary(results)
    print("\nPRIMARY by the declared rule:", choice)
    (METHODS / "stack.json").write_text(json.dumps({"primary": choice, "review_failed": sorted(REVIEW_FAILED)}))


def spec(name: str) -> dict:
    """'R+X+Z0.7' -> weights() params."""
    parts = set(name.split(" ")[0].split("+"))
    z = [p for p in parts if p.startswith("Z")]
    return dict(R="R" in parts, X="X" in parts, ivol=float(z[0][1:]) if z else None, k=4 if name.endswith("k4") else 3)


def lockbox() -> None:
    assert lockbox_open(), "run with LOCKBOX=open"
    lab = Lab.load()
    choice = json.loads((METHODS / "stack.json").read_text())["primary"]
    names = ["baseline", *SINGLES, choice, "R+X", "R+X+Z1.0", "R+X+Z1.0 k4"]
    base = lab.baseline_results
    rows, per_variant = [], []
    for name in dict.fromkeys(names):
        d = base if name == "baseline" else lab.evaluate(weights, FAMILY + "-lockbox", SINGLES.get(name) or spec(name))
        s = summary(d)
        b = base.set_index(["n", "calendar"])
        dv = d.set_index(["n", "calendar"])
        s["oos_beats_base"] = (dv.oos_sharpe > b.oos_sharpe).mean()
        s["train_beats_base"] = (dv.train_sharpe > b.train_sharpe).mean()
        rows.append({"model": name, **s.to_dict()})
        per_variant.append(d.assign(model=name))
        print(name, flush=True)
    t = pd.DataFrame(rows).set_index("model")
    pd.concat(per_variant).to_csv(METHODS / "lockbox_variants.csv", index=False)
    t.to_csv(METHODS / "lockbox.csv")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string())
    p = t.loc[choice]
    adopt = p.oos_sharpe > t.loc["baseline", "oos_sharpe"] and p.oos_beats_base >= 6 / 9
    print(f"\nPRIMARY {choice}: OOS Sharpe {p.oos_sharpe:.2f} vs baseline {t.loc['baseline', 'oos_sharpe']:.2f}, "
          f"beats baseline in {p.oos_beats_base:.0%} of variants -> {'ADOPT' if adopt else 'KEEP CURRENT BOT'}")


if __name__ == "__main__":
    {"train": train, "lockbox": lockbox}[sys.argv[1]]()
