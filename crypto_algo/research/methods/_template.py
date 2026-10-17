"""TEMPLATE - copy to research/methods/<family>.py.

Idea: <one paragraph: what the method changes in the pipeline and why it should help>.
Layer: <universe | signal | regime | rank | sizing | smoothing>.
"""
import json
import sys

import pandas as pd

from research.harness import METHODS, Lab, show, summary

FAMILY = "template"
# Declare every trial before running anything. Keep it small (<= 20) and economically motivated.
GRID = [{"example": 1}, {"example": 2}]
FROZEN = None  # set to the chosen params after running GRID on train; None if nothing beat the baseline


def layer(lab: Lab, example: int = 1) -> pd.DataFrame:
    """Return the layer object lab.build expects (e.g. a signal DataFrame in [0, 1]).
    Only data up to and including each bar may be used (rolling/ewm/shift(+k) are fine; shift(-k), centred
    windows and full-sample statistics are not). Higher-timeframe values via lab.gate or algo.htf.align."""
    return lab.baseline_signal()


def weights(lab: Lab, n: int, calendar: str, **params) -> pd.DataFrame:
    return lab.build(n, calendar, signal=lab.memo((FAMILY, json.dumps(params, sort_keys=True)), lambda: layer(lab, **params)))


def main() -> None:
    lab = Lab.load()
    base = lab.baseline_results
    print("baseline :", show(base))
    rows = []
    for params in GRID:
        d = lab.evaluate(weights, FAMILY, params)
        print(params, ":", show(d, base), flush=True)
        rows.append({"params": params, **summary(d, base).to_dict()})
    pd.DataFrame(rows).to_csv(METHODS / f"{FAMILY}_train.csv", index=False)


if __name__ == "__main__":
    main()
