"""Temporal splits.

Fixed, declared once, and never chosen by looking at the test window. RTE's own
accuracy degrades over the period, so a model must be judged on the same window
as the baseline it is compared against.
"""

from __future__ import annotations

from typing import Dict, Iterator, Tuple

import pandas as pd

TRAIN = ("2019-01-01", "2023-12-31")
VAL = ("2024-01-01", "2024-12-31")
TEST = ("2025-01-01", "2026-06-30")

SPLITS = {"train": TRAIN, "val": VAL, "test": TEST}


def mask(index: pd.DatetimeIndex, split: str) -> pd.Series:
    start, end = SPLITS[split]
    tz = index.tz
    lo = pd.Timestamp(start, tz=tz)
    hi = pd.Timestamp(end, tz=tz) + pd.Timedelta(days=1)
    return (index >= lo) & (index < hi)


def split_frame(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {name: df.loc[mask(df.index, name)] for name in SPLITS}


def describe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, part in split_frame(df).items():
        rows.append(
            {
                "split": name,
                "from": part.index.min(),
                "to": part.index.max(),
                "hours": len(part),
                "years": round(len(part) / 8766, 2),
            }
        )
    return pd.DataFrame(rows)


def walk_forward_folds(
    index: pd.DatetimeIndex, n_folds: int = 4, horizon_days: int = 90
) -> Iterator[Tuple[pd.Series, pd.Series]]:
    """Expanding-window folds, for models that must be refitted as time advances.

    Each fold trains on everything before a cut and scores the next
    ``horizon_days``. Folds never overlap and never look forward.
    """
    end = index.max()
    for k in range(n_folds, 0, -1):
        cut = end - pd.Timedelta(days=horizon_days * k)
        stop = cut + pd.Timedelta(days=horizon_days)
        yield (index < cut), (index >= cut) & (index < stop)
