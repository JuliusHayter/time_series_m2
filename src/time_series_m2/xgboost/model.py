"""XGBoost regressor, fitted once and applied to every hour of a window.

The target can be the load itself or its gap to a reference known at decision
time (``base``). Trees cannot extrapolate, and consumption sits at a lower level
in 2025 than in the training years; predicting the gap to a recent reference
removes most of that level shift.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_PARAMS: Dict = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=4,
    random_state=0,
)


class Booster:
    """A fitted model plus what it needs to predict: columns and target base."""

    def __init__(self, model: xgb.XGBRegressor, columns, base: Optional[str]):
        self.model = model
        self.columns = list(columns)
        self.base = base

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        raw = self.model.predict(frame[self.columns])
        if self.base is not None:
            raw = raw + frame[self.base].to_numpy()
        return pd.Series(raw, index=frame.index, name="xgboost")


def _usable(frame: pd.DataFrame, columns, base: Optional[str]) -> pd.DataFrame:
    needed = list(columns) + ["load_mw"] + ([base] if base else [])
    return frame.dropna(subset=needed)


def fit(
    train: pd.DataFrame,
    columns,
    *,
    base: Optional[str] = None,
    valid: Optional[pd.DataFrame] = None,
    params: Optional[Dict] = None,
    early_stopping_rounds: int = 100,
) -> Booster:
    """Fit on ``train``. If ``valid`` is given, it drives early stopping only."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    train = _usable(train, columns, base)

    def target(frame):
        return frame["load_mw"] - (frame[base] if base else 0.0)

    kwargs = {}
    if valid is not None:
        valid = _usable(valid, columns, base)
        kwargs["eval_set"] = [(valid[list(columns)], target(valid))]
        p["early_stopping_rounds"] = early_stopping_rounds
    else:
        p.pop("early_stopping_rounds", None)

    model = xgb.XGBRegressor(**p)
    model.fit(train[list(columns)], target(train), verbose=False, **kwargs)
    return Booster(model, columns, base)


def forecast(booster: Booster, frame: pd.DataFrame) -> pd.Series:
    """Predictions on the rows of ``frame`` whose features are all defined."""
    needed = booster.columns + ([booster.base] if booster.base else [])
    return booster.predict(frame.dropna(subset=needed))


def walk_forward(
    frame: pd.DataFrame,
    columns,
    days: pd.DatetimeIndex,
    *,
    params: Dict,
    base: Optional[str] = None,
    refit_every_days: int = 90,
) -> pd.Series:
    """Day-ahead forecasts with the model refitted as time advances.

    Before each block, the model is fitted on every day whose load is fully
    realized at the first auction of the block: the auction for day S closes on
    S-1, so the last usable day is S-2. Fixed tree count, no early stopping, so
    nothing here reads the block being scored.
    """
    days = pd.DatetimeIndex(days)
    day_of_row = frame.index.normalize()
    pieces = []
    for start in range(0, len(days), refit_every_days):
        block = days[start : start + refit_every_days]
        last_usable = block[0] - pd.Timedelta(days=2)
        booster = fit(frame[day_of_row <= last_usable], columns, base=base, params=params)
        rows = frame[day_of_row.isin(block)]
        pieces.append(forecast(booster, rows))
    return pd.concat(pieces)
