"""Baselines the models must beat.

Three rungs, each answering a different question:

* persistence  - is the problem hard at all?
* RTE raw      - what we already get for free
* RTE debiased - what we get for free once we spend five minutes looking at it

RTE's published forecast carries a systematic bias that grows from roughly zero
in 2021 to about 2% of load from 2023 on. Correcting it needs no model, so a
model that only matches the corrected baseline has added nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bias is estimated on days that are fully observed at decision time. For a target
# on day T the auction closes on T-1 at noon, so the last complete day is T-2.
_USABLE_DAY_OFFSET = pd.Timedelta(days=2)


def persistence(df: pd.DataFrame, days: int = 7) -> pd.Series:
    """Load observed the same hour, ``days`` days earlier."""
    if days < 2:
        raise ValueError("lags shorter than 2 days are not available at auction time")
    return df["load_mw"].shift(24 * days).rename(f"persistence_{days}d")


def rte_raw(df: pd.DataFrame) -> pd.Series:
    return df["rte_forecast_mw"].rename("rte_raw")


def rte_bias(df: pd.DataFrame, window_days: int = 15, per_hour: bool = True) -> pd.Series:
    """Rolling estimate of RTE's bias, using only fully observed past days."""
    idx = df.index
    err = df["load_mw"] - df["rte_forecast_mw"]
    target_day = pd.DatetimeIndex(idx.date)
    usable_day = target_day - _USABLE_DAY_OFFSET
    min_periods = max(3, window_days // 3)

    if per_hour:
        wide = pd.DataFrame(
            {"day": target_day, "hour": idx.hour, "err": err.to_numpy()}
        ).pivot_table(index="day", columns="hour", values="err")
        roll = wide.rolling(window_days, min_periods=min_periods).mean()
        picked = roll.reindex(usable_day).to_numpy()[np.arange(len(idx)), idx.hour]
    else:
        daily = err.groupby(target_day.to_numpy()).mean()
        roll = daily.rolling(window_days, min_periods=min_periods).mean()
        picked = roll.reindex(usable_day.to_numpy()).to_numpy()

    return pd.Series(picked, index=idx, name="rte_bias")


def rte_debiased(df: pd.DataFrame, window_days: int = 15, per_hour: bool = True) -> pd.Series:
    suffix = "hourly" if per_hour else "scalar"
    name = f"rte_debiased_{suffix}_{window_days}d"
    return (df["rte_forecast_mw"] + rte_bias(df, window_days, per_hour)).rename(name)


def all_baselines(df: pd.DataFrame, window_days: int = 15) -> dict:
    return {
        "persistence_7d": persistence(df, 7),
        "rte_raw": rte_raw(df),
        f"rte_debiased_scalar_{window_days}d": rte_debiased(df, window_days, per_hour=False),
        f"rte_debiased_hourly_{window_days}d": rte_debiased(df, window_days, per_hour=True),
    }
