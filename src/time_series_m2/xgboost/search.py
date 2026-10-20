"""Hyperparameters, chosen on the validation year only.

Early stopping reads 2024 to decide how many trees to grow, so the validation
score is mildly optimistic. That is the price of picking on validation; the test
split is never read here.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..utils.metrics import score
from ..utils.splits import mask
from .model import fit, forecast

# Depth and leaf size are the knobs that move the result; the learning rate is
# fixed and the number of trees is set by early stopping.
GRID: List[Dict] = [
    dict(max_depth=d, min_child_weight=w) for d in (4, 6, 8) for w in (5, 20)
]

MAX_TREES = 6000
LEARNING_RATE = 0.05


def grid_search(
    frame: pd.DataFrame,
    columns,
    *,
    base: Optional[str] = None,
    grid: Iterable[Dict] = tuple(GRID),
    feature_set: Optional[str] = None,
    progress: bool = False,
) -> pd.DataFrame:
    """One row per configuration, sorted by validation MAE.

    ``feature_set`` labels the rows so that tables for several feature sets can be
    concatenated and still be told apart by ``best_params``.
    """
    train = frame[mask(frame.index, "train")]
    valid = frame[mask(frame.index, "val")]
    rows = []
    for params in grid:
        booster = fit(
            train,
            columns,
            base=base,
            valid=valid,
            params=dict(n_estimators=MAX_TREES, learning_rate=LEARNING_RATE, **params),
        )
        result = score(valid["load_mw"], forecast(booster, valid))
        row = {
            **({"feature_set": feature_set} if feature_set is not None else {}),
            **params,
            "n_trees": int(booster.model.best_iteration) + 1,
            "MAE": result["MAE"],
            "RMSE": result["RMSE"],
            "MAPE": result["MAPE"],
            "bias": result["bias"],
        }
        rows.append(row)
        if progress:
            print(row, flush=True)
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


# A model refitted on train + val sees about 20% more data than the one early
# stopping sized on train alone.
TREES_SCALE_UP = 1.2

# Everything a row carries that is a result or a label rather than a knob. Any
# other column is a hyperparameter and is carried into the refit, so widening
# GRID cannot silently produce a model that differs from the winner it claims.
_NOT_A_KNOB = frozenset({"feature_set", "n_trees", "MAE", "RMSE", "MAPE", "bias", "hours"})


def best_params(grid: pd.DataFrame, feature_set: Optional[str] = None) -> Dict:
    """Parameters of the validation winner, ready for ``fit``.

    ``feature_set`` selects among rows produced for several feature sets; omit it
    when the table holds only one.
    """
    table = grid
    if feature_set is not None:
        if "feature_set" not in table.columns:
            raise KeyError(
                "this grid carries no 'feature_set' column: build it with "
                "grid_search(..., feature_set=...), or call best_params without a name"
            )
        table = table[table["feature_set"] == feature_set]
        if table.empty:
            known = sorted(grid["feature_set"].dropna().unique())
            raise ValueError(f"no rows for feature set {feature_set!r}; known sets: {known}")

    row = table.sort_values("MAE").iloc[0]
    params = {
        column: _as_number(row[column])
        for column in table.columns
        if column not in _NOT_A_KNOB
    }
    params["n_estimators"] = int(round(row["n_trees"] * TREES_SCALE_UP))
    params["learning_rate"] = LEARNING_RATE
    return params


def _as_number(value):
    """Undo the float cast a round-trip through CSV inflicts on integer knobs."""
    if isinstance(value, str):
        return value
    number = float(value)
    return int(number) if number.is_integer() else number
