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
    progress: bool = False,
) -> pd.DataFrame:
    """One row per configuration, sorted by validation MAE."""
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


def best_params(grid: pd.DataFrame, feature_set: str) -> Dict:
    """Parameters of the validation winner for one feature set, ready for ``fit``."""
    row = grid[grid["feature_set"] == feature_set].sort_values("MAE").iloc[0]
    return dict(
        max_depth=int(row["max_depth"]),
        min_child_weight=int(row["min_child_weight"]),
        n_estimators=int(round(row["n_trees"] * TREES_SCALE_UP)),
        learning_rate=LEARNING_RATE,
    )
