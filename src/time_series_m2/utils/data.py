"""Loading and cleaning of the French electricity demand data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

LOCAL_TZ = "Europe/Paris"

# The day-ahead auction closes at 12:00 local time on day D, for every hour of day D+1.
DECISION_HOUR = 12

# Realized load is not published instantly. We assume the most recent fully known
# hour at decision time is one hour old. Raising this makes every feature more
# conservative; it never makes one less safe.
DATA_LAG_HOURS = 1


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_national(path: Optional[Path] = None) -> pd.DataFrame:
    """Hourly national series, indexed by Europe/Paris local time.

    The seven missing load values (all at the October DST fall-back hour) are
    interpolated so that lag features stay defined, and flagged in
    ``load_is_imputed``. The RTE forecast is deliberately left NaN at those hours:
    scoring must not credit or blame a forecast that was never published.
    """
    path = path or project_root() / "data" / "france_electricity_hourly.csv"
    df = pd.read_csv(path)

    index = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
    out = pd.DataFrame(
        {
            "load_mw": pd.to_numeric(df["load_mw"], errors="coerce").to_numpy(),
            "rte_forecast_mw": pd.to_numeric(df["rte_forecast_d1_mw"], errors="coerce").to_numpy(),
            "temperature_c": pd.to_numeric(df["temperature_c"], errors="coerce").to_numpy(),
        },
        index=pd.DatetimeIndex(index, name="datetime"),
    ).sort_index()

    out["load_is_imputed"] = out["load_mw"].isna()
    out["load_mw"] = out["load_mw"].interpolate(limit_direction="both")
    return out


def load_regional(path: Optional[Path] = None) -> pd.DataFrame:
    """Regional series in long format. Only covers 2023 onwards."""
    path = path or project_root() / "data" / "france_regional_electricity_hourly.csv"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
    for col in ("load_mw", "temperature_c"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["datetime", "region", "load_mw", "temperature_c"]].sort_values(
        ["datetime", "region"]
    ).reset_index(drop=True)


def decision_time(target: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """When the forecast for ``target`` must be submitted.

    Every hour of day D+1 is auctioned on day D at 12:00 local. Built from naive
    calendar dates and localized afterwards: adding a Timedelta to a tz-aware
    midnight would drift by an hour on DST transition days.
    """
    days = pd.DatetimeIndex(target.date) - pd.Timedelta(days=1)
    return (days + pd.Timedelta(hours=DECISION_HOUR)).tz_localize(LOCAL_TZ)


def information_cutoff(target: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Latest timestamp whose realized load is known when forecasting ``target``.

    Any feature built from an observation after this instant is leakage.
    """
    return decision_time(target) - pd.Timedelta(hours=DATA_LAG_HOURS)
