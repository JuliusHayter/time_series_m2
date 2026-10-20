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
    score_frames: Optional[Dict[str, pd.DataFrame]] = None,
):
    """Day-ahead forecasts with the model refitted as time advances.

    Before each block, the model is fitted on every day whose load is fully
    realized at the first auction of the block: the auction for day S closes on
    S-1, so the last usable day is S-2. Fixed tree count, no early stopping, so
    nothing here reads the block being scored.

    ``score_frames`` maps a name to another version of ``frame`` (same index, other
    feature values, e.g. a noisy weather forecast). The model is still fitted on
    ``frame``; each block is scored on every version, and a DataFrame with one
    column per name is returned instead of a Series.
    """
    days = pd.DatetimeIndex(days)
    day_of_row = frame.index.normalize()
    # An empty mapping means "nothing to score", which is a silent no-op rather
    # than a choice anyone makes on purpose.
    versions = score_frames or {"forecast": frame}
    pieces = {name: [] for name in versions}
    for start in range(0, len(days), refit_every_days):
        block = days[start : start + refit_every_days]
        last_usable = block[0] - pd.Timedelta(days=2)
        booster = fit(frame[day_of_row <= last_usable], columns, base=base, params=params)
        for name, version in versions.items():
            # Each version carries its own index: masking it with a boolean array
            # built from ``frame`` would score the wrong rows whenever the two
            # disagree, and reassembling on ``version``'s index would hide it.
            rows = version[version.index.normalize().isin(block)]
            pieces[name].append(forecast(booster, rows))

    out = {name: pd.concat(parts) for name, parts in pieces.items()}
    if not score_frames:
        return out["forecast"]
    return pd.DataFrame(out)
