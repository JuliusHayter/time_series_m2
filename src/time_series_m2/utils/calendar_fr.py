"""French calendar effects.

Calendar features are known arbitrarily far in advance, so none of them can leak.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def french_holidays(years) -> set:
    return set(holidays.France(years=list(years)).keys())


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour, weekday, season and holiday effects, derived from the local clock."""
    out = df.copy()
    idx = out.index

    out["hour"] = idx.hour
    out["day_of_week"] = idx.dayofweek
    out["month"] = idx.month
    out["day_of_year"] = idx.dayofyear
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # Smooth yearly and daily cycles: avoids the artificial jump between 23h and 0h,
    # and between 31 Dec and 1 Jan, that integer encodings create.
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    dates = pd.Series(idx.date, index=idx)
    hol = french_holidays(range(idx.year.min(), idx.year.max() + 1))
    out["is_holiday"] = dates.isin(hol).astype(int).to_numpy()

    one_day = pd.Timedelta(days=1)
    prev_holiday = dates.map(lambda d: (d - one_day) in hol)
    next_holiday = dates.map(lambda d: (d + one_day) in hol)
    out["is_day_after_holiday"] = prev_holiday.astype(int).to_numpy()
    out["is_day_before_holiday"] = next_holiday.astype(int).to_numpy()

    # "Pont": a working day wedged between a holiday and the weekend.
    is_workday = (out["is_weekend"] == 0) & (out["is_holiday"] == 0)
    friday_bridge = is_workday & (out["day_of_week"] == 4) & (out["is_day_before_holiday"] == 0) & (
        prev_holiday.to_numpy()
    )
    monday_bridge = is_workday & (out["day_of_week"] == 0) & next_holiday.to_numpy()
    out["is_bridge_day"] = (friday_bridge | monday_bridge).astype(int)

    # On DST transition days the local daily profile is shifted by one hour.
    offsets = pd.Series(idx.map(lambda t: t.utcoffset()), index=idx)
    changed = offsets.ne(offsets.shift())
    changed.iloc[0] = False
    dst_days = set(pd.Series(idx.date, index=idx)[changed.to_numpy()])
    out["is_dst_change_day"] = dates.isin(dst_days).astype(int).to_numpy()

    return out
