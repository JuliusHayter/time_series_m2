"""Regenerates the XGBoost results that notebook 04 reads.

    uv run python scripts/run_xgboost.py grid   # hyperparameters on 2024, a few minutes
    uv run python scripts/run_xgboost.py test   # test forecasts, static and walk-forward

The test run reads the grid winner per feature set, so run ``grid`` first.
"""

import sys

import pandas as pd

from time_series_m2.utils.data import load_national, project_root
from time_series_m2.utils.splits import mask
from time_series_m2.xgboost import build_dataset, fit, forecast, walk_forward
from time_series_m2.xgboost.features import FEATURE_SETS
from time_series_m2.xgboost.search import LEARNING_RATE, grid_search

OUT = project_root() / "results" / "xgboost"

# A model refitted on train + val sees about 20% more data than the one early
# stopping sized on train alone.
TREES_SCALE_UP = 1.2


def run_grid() -> None:
    df = load_national()
    tables = []
    for feature_set in FEATURE_SETS:
        frame, _ = build_dataset(df, feature_set)
        table = grid_search(frame, frame.attrs["feature_columns"], progress=True)
        tables.append(table.assign(feature_set=feature_set))
    pd.concat(tables).to_csv(OUT / "grid_val.csv", index=False)


def _best_params(grid: pd.DataFrame, feature_set: str) -> dict:
    row = grid[grid["feature_set"] == feature_set].sort_values("MAE").iloc[0]
    return dict(
        max_depth=int(row["max_depth"]),
        min_child_weight=int(row["min_child_weight"]),
        n_estimators=int(round(row["n_trees"] * TREES_SCALE_UP)),
        learning_rate=LEARNING_RATE,
    )


def run_test() -> None:
    df = load_national()
    grid = pd.read_csv(OUT / "grid_val.csv")
    predictions, importance = {}, {}
    for feature_set in FEATURE_SETS:
        frame, _ = build_dataset(df, feature_set)
        columns = frame.attrs["feature_columns"]
        params = _best_params(grid, feature_set)

        fitted = frame[mask(frame.index, "train") | mask(frame.index, "val")]
        booster = fit(fitted, columns, params=params)
        test = frame[mask(frame.index, "test")]
        predictions[f"{feature_set}_static"] = forecast(booster, test)
        importance[feature_set] = pd.Series(
            booster.model.feature_importances_, index=columns
        ).sort_values(ascending=False)

        days = test.index.normalize().unique()
        predictions[f"{feature_set}_walk_forward"] = walk_forward(
            frame, columns, days, params=params
        )
        print(feature_set, params, flush=True)

    pd.DataFrame(predictions).to_csv(OUT / "test_forecasts.csv")
    pd.DataFrame(importance).to_csv(OUT / "feature_importance.csv")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"grid": run_grid, "test": run_test}[sys.argv[1]]()
