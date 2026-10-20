"""Order selection, on the validation year only.

The ACF/PACF of notebook 03 narrows the field but does not settle it: those plots
describe one-step-ahead structure, and this model is judged 13 to 36 hours ahead.
A candidate that fits the in-sample autocorrelation best can forecast the auction
horizon worst. So the orders are ranked by out-of-sample MAE on 2024, and the test
years are never touched here.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ..utils.metrics import score
from ..utils.splits import mask
from .model import DEFAULT_REFIT_EVERY, DEFAULT_WINDOW_DAYS, forecast_days

Order = Tuple[int, int, int]
SeasonalOrder = Tuple[int, int, int, int]

# Around the ACF/PACF reading, plus the variants that question it: no regular
# differencing, a longer AR, a mixed ARMA, and a seasonal AR term. Deliberately
# short — each entry costs a minute of fitting, and the point is to test the
# identification, not to dredge the order space.
CANDIDATES: List[Tuple[Order, SeasonalOrder]] = [
    ((2, 1, 0), (0, 1, 1, 24)),  # what notebook 03 read in the ACF/PACF
    ((2, 0, 0), (0, 1, 1, 24)),  # same, without the regular differencing
    ((3, 0, 0), (0, 1, 1, 24)),
    ((1, 0, 1), (0, 1, 1, 24)),
    ((2, 0, 0), (1, 1, 1, 24)),
]

# The sweep runs on one day in seven of the validation year: 52 days spread over
# every season, enough to rank the candidates without a full year of fitting.
SWEEP_EVERY = 7
SWEEP_REFIT_EVERY = 4

def split_days(index: pd.DatetimeIndex, split: str, every: int = 1) -> pd.DatetimeIndex:
    """Days of a split, optionally thinned to one in ``every`` for a cheaper sweep."""
    days = index[mask(index, split)].normalize().unique()
    return days[::every]


def evaluate(
    df: pd.DataFrame,
    days: pd.DatetimeIndex,
    order: Order,
    seasonal_order: SeasonalOrder,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    refit_every: int = DEFAULT_REFIT_EVERY,
    exog_cols: Optional[Sequence[str]] = None,
) -> dict:
    pred = forecast_days(
        df,
        days,
        order=order,
        seasonal_order=seasonal_order,
        window_days=window_days,
        refit_every=refit_every,
        exog_cols=exog_cols,
    )
    out = score(df["load_mw"], pred)
    out["order"] = str(order)
    out["seasonal_order"] = str(seasonal_order)
    return out


def grid_search(
    df: pd.DataFrame,
    days: pd.DatetimeIndex,
    candidates: Iterable[Tuple[Order, SeasonalOrder]] = tuple(CANDIDATES),
    *,
    progress: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """One row per candidate, sorted by MAE. Expensive: a fit is seconds, not milliseconds."""
    rows = []
    for order, seasonal_order in candidates:
        row = evaluate(df, days, order, seasonal_order, **kwargs)
        rows.append(row)
        if progress:
            print(f"{order}{seasonal_order}  MAE={row['MAE']:.1f}", flush=True)

    table = pd.DataFrame(rows).set_index(["order", "seasonal_order"])
    return table.sort_values("MAE")
