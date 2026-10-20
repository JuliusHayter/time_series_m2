"""Regenerates the SARIMAX results that notebook 03 reads.

    uv run python scripts/run_sarima.py grid   # order sweep on 2024, ~10 min
    uv run python scripts/run_sarima.py test   # day-ahead forecasts on the test split, ~30 min

BLAS threads are pinned to 1 before numpy loads: with several worker processes,
a multithreaded BLAS oversubscribes the cores and runs slower than sequential.
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from time_series_m2.sarima import search
from time_series_m2.sarima.model import forecast_days
from time_series_m2.utils.data import load_national, project_root
from time_series_m2.utils.splits import mask

OUT = project_root() / "results" / "sarima"
MAX_WORKERS = 2


def _evaluate(candidate):
    order, seasonal_order = candidate
    df = load_national()
    days = search.split_days(df.index, "val", search.SWEEP_EVERY)
    return search.evaluate(
        df, days, order, seasonal_order, refit_every=search.SWEEP_REFIT_EVERY
    )


def _forecast(days):
    return forecast_days(load_national(), days)


def run_grid() -> None:
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        rows = list(pool.map(_evaluate, search.CANDIDATES))
    table = pd.DataFrame(rows).set_index(["order", "seasonal_order"]).sort_values("MAE")
    table.to_csv(OUT / "grid_val.csv")
    print(table[["MAE", "RMSE", "MAPE", "bias"]].round(1))


def run_test() -> None:
    df = load_national()
    days = df.index[mask(df.index, "test")].normalize().unique()
    # Chunks of whole weeks keep the weekly refit schedule of a single run.
    size = -(-len(days) // MAX_WORKERS // 7) * 7
    chunks = [days[i : i + size] for i in range(0, len(days), size)]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        pred = pd.concat(list(pool.map(_forecast, chunks)))
    pred.to_csv(OUT / "sarimax_test.csv")
    print(f"{len(pred)} hours written")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"grid": run_grid, "test": run_test}[sys.argv[1]]()
