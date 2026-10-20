"""Regenerates the results that notebook 05 reads. Needs results/xgboost/grid_val.csv.

    uv run python scripts/run_uncertainty.py val        # out-of-sample forecasts on 2024, to seed the bands
    uv run python scripts/run_uncertainty.py quantile   # 5% and 95% quantile boosters, walk-forward on the test
    uv run python scripts/run_uncertainty.py weather    # the same model replayed on noisy temperature
"""

import sys

import pandas as pd

from time_series_m2.utils.data import load_national, project_root
from time_series_m2.utils.features import perturb_temperature
from time_series_m2.utils.metrics import peak_mask, score
from time_series_m2.utils.splits import mask
from time_series_m2.xgboost import build_dataset, fit, forecast, walk_forward
from time_series_m2.xgboost.features import FEATURE_SETS
from time_series_m2.xgboost.search import MAX_TREES, TREES_SCALE_UP, best_params

OUT = project_root() / "results" / "uncertainty"
GRID = project_root() / "results" / "xgboost" / "grid_val.csv"

TAILS = {"lower": 0.05, "upper": 0.95}
SIGMAS = (1.0, 2.0)
SEEDS = range(5)


def _grid() -> pd.DataFrame:
    return pd.read_csv(GRID)


def run_val() -> None:
    """A model fitted on train only, scored on 2024: calibration history for the bands."""
    df, grid = load_national(), _grid()
    columns_out = {}
    for feature_set in FEATURE_SETS:
        frame, _ = build_dataset(df, feature_set)
        params = {**best_params(grid, feature_set), "n_estimators": MAX_TREES}
        booster = fit(
            frame[mask(frame.index, "train")],
            frame.attrs["feature_columns"],
            valid=frame[mask(frame.index, "val")],
            params=params,
        )
        columns_out[feature_set] = forecast(booster, frame[mask(frame.index, "val")])
    pd.DataFrame(columns_out).to_csv(OUT / "val_forecasts.csv")


def run_quantile() -> None:
    df, grid = load_national(), _grid()
    frame, _ = build_dataset(df, "with_rte")
    columns = frame.attrs["feature_columns"]
    days = frame.index[mask(frame.index, "test")].normalize().unique()
    base = best_params(grid, "with_rte")

    bands = {}
    for name, alpha in TAILS.items():
        quantile = {**base, "objective": "reg:quantileerror", "quantile_alpha": alpha}
        # Early stopping on 2024 sizes the booster; the walk-forward then runs on a fixed count.
        sized = fit(
            frame[mask(frame.index, "train")],
            columns,
            valid=frame[mask(frame.index, "val")],
            params={**quantile, "n_estimators": MAX_TREES},
        )
        quantile["n_estimators"] = int(round((sized.model.best_iteration + 1) * TREES_SCALE_UP))
        bands[name] = walk_forward(frame, columns, days, params=quantile)
        print(name, quantile["n_estimators"], flush=True)

    out = pd.DataFrame(bands)
    # Independently fitted quantiles can cross; order them rather than report an inverted band.
    out = pd.DataFrame(
        {"lower": out.min(axis=1), "upper": out.max(axis=1)}, index=out.index
    )
    out.to_csv(OUT / "quantile_bands.csv")


def run_weather() -> None:
    df, grid = load_national(), _grid()
    rows = []
    for feature_set in FEATURE_SETS:
        frame, _ = build_dataset(df, feature_set)
        columns = frame.attrs["feature_columns"]
        days = frame.index[mask(frame.index, "test")].normalize().unique()

        versions = {("clean", 0.0, 0): frame}
        for sigma in SIGMAS:
            for seed in SEEDS:
                noisy, _ = build_dataset(perturb_temperature(df, sigma, seed), feature_set)
                versions[(f"noisy", sigma, seed)] = noisy
        keyed = {f"{kind}|{sigma}|{seed}": v for (kind, sigma, seed), v in versions.items()}

        preds = walk_forward(
            frame, columns, days, params=best_params(grid, feature_set), score_frames=keyed
        )
        actual = df["load_mw"]
        scored = frame["rte_debiased"].reindex(preds.index).notna()
        for name in preds:
            _, sigma, seed = name.split("|")
            p = preds[name][scored]
            result = score(actual, p)
            peaks = peak_mask(p.index)
            rows.append(
                {
                    "feature_set": feature_set,
                    "sigma_c": float(sigma),
                    "seed": int(seed),
                    "MAE": result["MAE"],
                    "RMSE": result["RMSE"],
                    "bias": result["bias"],
                    "peak_MAE": score(actual[p.index][peaks], p[peaks])["MAE"],
                    "hours": result["hours"],
                }
            )
        print(feature_set, "done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "weather_robustness.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"val": run_val, "quantile": run_quantile, "weather": run_weather}[sys.argv[1]]()
