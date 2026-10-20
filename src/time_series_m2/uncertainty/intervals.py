"""Prediction intervals from past forecast errors.

A band is only worth the coverage it delivers, so every interval here is meant to
be scored with ``metrics.interval_coverage``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The auction for day D closes on D-1, so the last fully realized day is D-2.
USABLE_DAY_OFFSET = pd.Timedelta(days=2)


def rolling_conformal(
    actual: pd.Series,
    pred: pd.Series,
    *,
    nominal: float = 0.9,
    window_days: int = 60,
    min_days: int = 20,
) -> pd.DataFrame:
    """Hour-of-day bands from the model's own recent errors.

    For each target day, take the signed errors (actual - pred) of the last
    ``window_days`` days that are fully observed at the auction, per hour of day,
    and add their empirical quantiles to the new point forecast. Distribution-free
    and adaptive: when the model drifts, the band widens as soon as the errors do.

    Returns columns ``lower`` and ``upper`` on the index of ``pred``.
    """
    both = pd.concat([actual.rename("a"), pred.rename("p")], axis=1).dropna()
    if both.empty:
        raise ValueError("actual and pred never overlap: there is no error history to calibrate on")

    err = both["a"] - both["p"]
    days = pd.DatetimeIndex(err.index.date)
    wide = pd.DataFrame({"day": days, "hour": err.index.hour, "err": err.to_numpy()}).pivot_table(
        index="day", columns="hour", values="err", aggfunc="mean"
    )
    # Hours absent from the history would otherwise shift every later column left,
    # so a band asked for hour 19 would quietly be hour 20's. Columns are addressed
    # positionally below; this makes position and hour the same number again.
    wide = wide.reindex(columns=range(24))

    # Reindex to every calendar day so that "the last 60 rows" means 60 days.
    last_day = pred.index.max().date()
    if last_day < wide.index.min().date():
        raise ValueError(
            "pred ends before the first observed error day: no calibration history applies"
        )
    wide = wide.reindex(pd.date_range(wide.index.min(), last_day))

    tail = (1 - nominal) / 2
    roll = wide.rolling(window_days, min_periods=min_days)
    # Order statistics rather than interpolated quantiles: with ~60 errors per hour,
    # interpolating inside the tails undercovers by about three points. Taking the
    # outer neighbour on each side is the finite-sample correction of conformal
    # prediction, and brings coverage back to nominal under exchangeable errors.
    q_lo = roll.quantile(tail, interpolation="lower")
    q_hi = roll.quantile(1 - tail, interpolation="higher")

    target_day = pd.DatetimeIndex(pred.index.date)
    usable_day = target_day - USABLE_DAY_OFFSET
    rows = np.arange(len(pred))
    hours = pred.index.hour

    def pick(table: pd.DataFrame) -> np.ndarray:
        return table.reindex(usable_day).to_numpy()[rows, hours]

    return pd.DataFrame(
        {"lower": pred.to_numpy() + pick(q_lo), "upper": pred.to_numpy() + pick(q_hi)},
        index=pred.index,
    )
