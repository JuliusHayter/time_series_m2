"""Scoring.

Headline numbers hide where a forecast fails, so every metric is also available
sliced by hour, by season and on the hours that actually cost money.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

# Winter evening peaks: the hours where imbalance is most expensive.
PEAK_HOURS = range(18, 22)
WINTER_MONTHS = (12, 1, 2)


def score(actual: pd.Series, pred: pd.Series) -> Dict[str, float]:
    """MAE, RMSE, MAPE and bias on the hours where both series are defined."""
    both = pd.concat([actual.rename("a"), pred.rename("p")], axis=1).dropna()
    err = both["a"] - both["p"]
    return {
        "MAE": err.abs().mean(),
        "RMSE": float(np.sqrt((err**2).mean())),
        "MAPE": (err.abs() / both["a"]).mean() * 100,
        "bias": err.mean(),
        "hours": len(both),
    }


def score_table(
    actual: pd.Series, predictions: Dict[str, pd.Series], reference: Optional[str] = None
) -> pd.DataFrame:
    """One row per forecast. ``reference`` adds a gain column against that row."""
    table = pd.DataFrame({name: score(actual, pred) for name, pred in predictions.items()}).T
    table["hours"] = table["hours"].astype(int)
    if reference is not None:
        table["gain_vs_" + reference] = (
            (table.loc[reference, "MAE"] - table["MAE"]) / table.loc[reference, "MAE"] * 100
        )
    return table


def peak_mask(index: pd.DatetimeIndex) -> np.ndarray:
    return np.isin(index.hour, list(PEAK_HOURS)) & np.isin(index.month, list(WINTER_MONTHS))


def by_hour(actual: pd.Series, pred: pd.Series) -> pd.Series:
    err = (actual - pred).abs()
    return err.groupby(err.index.hour).mean().rename("MAE")


def by_month(actual: pd.Series, pred: pd.Series) -> pd.Series:
    err = (actual - pred).abs()
    return err.groupby(err.index.month).mean().rename("MAE")


def on_peaks(actual: pd.Series, pred: pd.Series) -> Dict[str, float]:
    m = peak_mask(actual.index)
    return score(actual[m], pred[m])


def interval_coverage(
    actual: pd.Series, lower: pd.Series, upper: pd.Series, nominal: float = 0.9
) -> Dict[str, float]:
    """Does a claimed 90% band actually contain 90% of outcomes?"""
    both = pd.concat(
        [actual.rename("a"), lower.rename("lo"), upper.rename("hi")], axis=1
    ).dropna()
    inside = (both["a"] >= both["lo"]) & (both["a"] <= both["hi"])
    return {
        "coverage": inside.mean(),
        "nominal": nominal,
        "mean_width": (both["hi"] - both["lo"]).mean(),
        "hours": len(both),
    }
