"""Feature construction under the day-ahead information constraint.

Every load-derived feature declares the timestamp of the most recent observation
it uses. ``check_leakage`` asserts that this timestamp never exceeds the
information cutoff, so a leaking feature fails loudly instead of quietly
inflating the score.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .data import DATA_LAG_HOURS, DECISION_HOUR, LOCAL_TZ, information_cutoff

# Same hour, N days back. 24h is deliberately absent: for a target at 23:00 it
# would point 12 hours past the cutoff. 48h is the shortest uniformly safe lag.
SAME_HOUR_LAGS: Tuple[int, ...] = (48, 72, 96, 120, 144, 168, 192)

# Comfort threshold for heating/cooling degree hours.
BASE_TEMP_C = 15.0

Sources = Dict[str, pd.DatetimeIndex]


def _localize(days: pd.DatetimeIndex, hour: int) -> pd.DatetimeIndex:
    return (days + pd.Timedelta(hours=hour)).tz_localize(LOCAL_TZ)


def add_load_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Sources]:
    """Lags and recent aggregates of realized load, all known before the auction."""
    out = df.copy()
    idx = out.index
    load = out["load_mw"]
    sources: Sources = {}

    for lag in SAME_HOUR_LAGS:
        name = f"load_lag_{lag}h"
        out[name] = load.shift(lag).to_numpy()
        sources[name] = idx - pd.Timedelta(hours=lag)

    lag_cols = [f"load_lag_{lag}h" for lag in SAME_HOUR_LAGS]
    out["load_same_hour_mean_7d"] = out[lag_cols].mean(axis=1)
    sources["load_same_hour_mean_7d"] = idx - pd.Timedelta(hours=min(SAME_HOUR_LAGS))

    # Freshest usable observation: the load at the cutoff itself, shared by all
    # 24 hours of the target day.
    cutoff = information_cutoff(idx)
    out["load_at_cutoff"] = load.reindex(cutoff).to_numpy()
    sources["load_at_cutoff"] = cutoff

    target_day = pd.DatetimeIndex(idx.date)
    decision_day = target_day - pd.Timedelta(days=1)
    last_full_day = target_day - pd.Timedelta(days=2)

    by_day = load.groupby(target_day.to_numpy())
    for stat in ("mean", "max", "min"):
        name = f"load_last_full_day_{stat}"
        out[name] = by_day.agg(stat).reindex(last_full_day.to_numpy()).to_numpy()
        sources[name] = _localize(last_full_day, 23)

    # The decision day is only observed up to the cutoff, so aggregate that part only.
    last_known_hour = DECISION_HOUR - DATA_LAG_HOURS
    partial = load.where(idx.hour <= last_known_hour)
    out["load_decision_day_partial_mean"] = (
        partial.groupby(target_day.to_numpy()).mean().reindex(decision_day.to_numpy()).to_numpy()
    )
    sources["load_decision_day_partial_mean"] = cutoff

    # Week-on-week momentum, built only from safe lags.
    out["load_trend_7d"] = out["load_lag_48h"] - out["load_lag_192h"]
    sources["load_trend_7d"] = idx - pd.Timedelta(hours=48)

    return out, sources


def add_temperature_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Sources]:
    """Thermal drivers.

    These read the temperature *at the target hour*, which assumes a perfect
    weather forecast. Real desks only hold a forecast, so results built on these
    are an optimistic bound. ``perturb_temperature`` exists to quantify that gap.
    """
    out = df.copy()
    temp = out["temperature_c"]

    out["hdd"] = (BASE_TEMP_C - temp).clip(lower=0)
    out["cdd"] = (temp - BASE_TEMP_C).clip(lower=0)
    out["temp_sq"] = temp**2

    # Thermal inertia: buildings respond to the last day or two, not the spot value.
    out["temp_mean_24h"] = temp.rolling(24, min_periods=1).mean()
    out["temp_min_24h"] = temp.rolling(24, min_periods=1).min()

    names = ["temperature_c", "hdd", "cdd", "temp_sq", "temp_mean_24h", "temp_min_24h"]
    return out, {name: out.index for name in names}


def perturb_temperature(df: pd.DataFrame, sigma_c: float, seed: int = 0) -> pd.DataFrame:
    """Replace temperature by a simulated day-ahead weather forecast.

    Adds autocorrelated noise rather than white noise: real forecast errors
    persist over a day, they do not flip hour to hour.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n_days = len(out.index.normalize().unique())
    daily_error = pd.Series(
        rng.normal(0.0, sigma_c, n_days), index=out.index.normalize().unique()
    )
    out["temperature_c"] = out["temperature_c"] + daily_error.reindex(
        out.index.normalize()
    ).to_numpy()
    return out


def check_leakage(index: pd.DatetimeIndex, sources: Sources, *, exempt=()) -> pd.DataFrame:
    """Assert that no feature reads an observation past the information cutoff.

    Weather-derived features are exempt by construction: they encode the declared
    perfect-forecast assumption, not an accident.
    """
    cutoff = information_cutoff(index)
    rows = []
    for name, source in sources.items():
        if name in exempt:
            rows.append({"feature": name, "status": "exempt (weather assumption)", "worst_gap_h": np.nan})
            continue
        gap = (pd.DatetimeIndex(source) - cutoff).total_seconds() / 3600.0
        worst = float(np.nanmax(gap))
        rows.append(
            {
                "feature": name,
                "status": "ok" if worst <= 0 else "LEAK",
                "worst_gap_h": worst,
            }
        )
    report = pd.DataFrame(rows).sort_values("worst_gap_h", ascending=False)
    leaks = report[report["status"] == "LEAK"]
    if not leaks.empty:
        raise AssertionError(f"Features reading past the cutoff:\n{leaks.to_string(index=False)}")
    return report
