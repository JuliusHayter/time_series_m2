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
from time_series_m2.xgboost.search import best_params, grid_search

OUT = project_root() / "results" / "xgboost"


def run_grid() -> None:
    df = load_national()
    tables = []
    for feature_set in FEATURE_SETS:
        frame, _ = build_dataset(df, feature_set)
        tables.append(
            grid_search(
                frame,
                frame.attrs["feature_columns"],
                feature_set=feature_set,
                progress=True,
            )
        )
    pd.concat(tables).to_csv(OUT / "grid_val.csv", index=False)


def run_test() -> None:
    df = load_national()
    grid = pd.read_csv(OUT / "grid_val.csv")
    predictions, importance = {}, {}
    for feature_set in FEATURE_SETS:
        frame, _ = build_dataset(df, feature_set)
        columns = frame.attrs["feature_columns"]
        params = best_params(grid, feature_set)

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
