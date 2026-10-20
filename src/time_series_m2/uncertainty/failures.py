"""Where a forecast fails: error sliced by the conditions that drive it."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from ..utils.calendar_fr import add_calendar_features
from ..utils.metrics import peak_mask


def conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean and categorical descriptors of each hour, known before the auction."""
    cal = add_calendar_features(df)
    out = pd.DataFrame(index=df.index)
    out["temp_decile"] = pd.qcut(df["temperature_c"], 10, labels=False, duplicates="drop") + 1
    out["load_decile"] = pd.qcut(df["load_mw"], 10, labels=False, duplicates="drop") + 1
    out["winter_peak"] = peak_mask(df.index)
    out["holiday"] = cal["is_holiday"].astype(bool)
    out["bridge_day"] = cal["is_bridge_day"].astype(bool)
    out["dst_change_day"] = cal["is_dst_change_day"].astype(bool)
    out["weekend"] = cal["is_weekend"].astype(bool)
    return out


def slice_table(
    actual: pd.Series, preds: Dict[str, pd.Series], by: pd.Series, min_hours: int = 24
) -> pd.DataFrame:
    """MAE and bias of each forecast within each value of ``by``."""
    frame = pd.concat({"actual": actual, **preds}, axis=1).dropna()
    group = by.reindex(frame.index)
    rows = {}
    for key, part in frame.groupby(group):
        if len(part) < min_hours:
            continue
        row = {"hours": len(part)}
        for name in preds:
            err = part["actual"] - part[name]
            row[f"MAE_{name}"] = err.abs().mean()
            row[f"bias_{name}"] = err.mean()
        rows[key] = row
    return pd.DataFrame(rows).T
