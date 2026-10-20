"""Feature matrix for the day-ahead booster.

Every column is checked against the information cutoff before a model sees it.
Two feature sets are built so their difference is explicit:

* ``independent`` - realized load, calendar and weather only
* ``with_rte``    - the same, plus RTE's published forecast and its recent bias

RTE's forecast for day D+1 is published before the auction closes, so using it
is legitimate. It is a design choice that makes the model depend on RTE staying
available, not a free gain, and it is reported as such.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from ..baselines.naive import rte_bias
from ..utils.calendar_fr import add_calendar_features
from ..utils.data import information_cutoff
from ..utils.features import add_load_features, add_temperature_features, check_leakage

CALENDAR_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "is_holiday",
    "is_day_after_holiday",
    "is_day_before_holiday",
    "is_bridge_day",
    "is_dst_change_day",
]

FEATURE_SETS = ("independent", "with_rte")


def build_dataset(df: pd.DataFrame, feature_set: str = "independent") -> Tuple[pd.DataFrame, Dict]:
    """Returns the frame (features + ``load_mw`` + ``rte_debiased``) and the leakage report.

    ``check_leakage`` runs here, so a leaking column cannot reach a fit.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")

    frame = add_calendar_features(df)
    frame, load_sources = add_load_features(frame)
    frame, temp_sources = add_temperature_features(frame)

    sources = dict(load_sources)
    sources.update(temp_sources)

    # RTE's forecast and the bias estimated from fully observed past days.
    frame["rte_bias"] = rte_bias(df)
    frame["rte_debiased"] = frame["rte_forecast_mw"] + frame["rte_bias"]
    if feature_set == "with_rte":
        # Published before the auction closes: its availability instant is the cutoff.
        sources["rte_forecast_mw"] = information_cutoff(frame.index)
        sources["rte_bias"] = pd.DatetimeIndex(frame.index.date).tz_localize(frame.index.tz) - pd.Timedelta(
            days=2
        ) + pd.Timedelta(hours=23)

    report = check_leakage(frame.index, sources, exempt=set(temp_sources))

    columns = CALENDAR_COLUMNS + [c for c in sources if c != "temperature_c"] + ["temperature_c"]
    frame.attrs["feature_columns"] = list(dict.fromkeys(columns))
    return frame, report
