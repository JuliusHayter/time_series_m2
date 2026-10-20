"""Day-ahead SARIMAX, fitted and forecast under the auction constraint.

The model never produces a one-step-ahead forecast. At 11:00 on day D it holds
the realized load up to that hour, and must deliver the 24 hours of day D+1,
i.e. horizons of 13 to 36 hours. Every number this module returns is produced
that way, so it is directly comparable to the baselines of notebook 02.

The weekly cycle survives seasonal differencing at s=24 (notebook 03), and a
seasonal term of period 168 is not computable in reasonable time. Day of week,
holidays and temperature therefore enter as exogenous regressors.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..utils.calendar_fr import add_calendar_features
from ..utils.data import information_cutoff
from ..utils.features import BASE_TEMP_C

# Picked by sarima.search on 52 days of 2024 (MAE 3008 MW). The (2, 1, 0)(0, 1, 1)
# order read off the ACF/PACF scored 7471 MW there: regular differencing is
# over-differencing at a 13 to 36 hour horizon.
DEFAULT_ORDER: Tuple[int, int, int] = (2, 0, 0)
DEFAULT_SEASONAL_ORDER: Tuple[int, int, int, int] = (1, 1, 1, 24)

# Rolling window used for estimation. Long enough for the seasonal terms to be
# identified, short enough to track the post-2022 shift in consumption levels.
DEFAULT_WINDOW_DAYS = 60

# Re-estimating the parameters every day costs ~15s per day and buys little: an
# operational desk recalibrates weekly and refilters daily. In between refits the
# same parameters are applied to the updated history, which is a Kalman pass only.
DEFAULT_REFIT_EVERY = 7


def build_exog(df: pd.DataFrame) -> pd.DataFrame:
    """Regressors known before the auction closes.

    Calendar terms are known arbitrarily far ahead. Temperature is not: using the
    target-hour value assumes a perfect weather forecast, the same optimistic
    assumption as ``utils.features.add_temperature_features``, and the same one
    ``utils.features.perturb_temperature`` exists to stress.
    """
    cal = add_calendar_features(df)
    exog = pd.get_dummies(cal["day_of_week"], prefix="dow", drop_first=True).astype(float)
    exog["is_holiday"] = cal["is_holiday"].astype(float)
    exog["is_bridge_day"] = cal["is_bridge_day"].astype(float)
    exog["hdd"] = (BASE_TEMP_C - cal["temperature_c"]).clip(lower=0)
    exog["cdd"] = (cal["temperature_c"] - BASE_TEMP_C).clip(lower=0)
    return exog


def _specify(
    endog: np.ndarray,
    exog: Optional[np.ndarray],
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> SARIMAX:
    # simple_differencing=True is four times faster but silently corrupts
    # multi-step forecasts with exogenous regressors (MAE in the tens of
    # thousands of MW). The slow path is the only correct one here.
    return SARIMAX(
        endog,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        simple_differencing=False,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )


def _target_hours(index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.DatetimeIndex:
    return index[index.normalize() == day]


def forecast_days(
    df: pd.DataFrame,
    days: Iterable[pd.Timestamp],
    *,
    order: Tuple[int, int, int] = DEFAULT_ORDER,
    seasonal_order: Tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
    window_days: int = DEFAULT_WINDOW_DAYS,
    refit_every: int = DEFAULT_REFIT_EVERY,
    exog_cols: Optional[Sequence[str]] = None,
    progress: bool = False,
) -> pd.Series:
    """Day-ahead forecasts for every day in ``days``.

    One fit per ``refit_every`` days; the other days reuse those parameters on an
    updated history. Returns an hourly series aligned on ``df.index``.
    """
    index = df.index
    load = df["load_mw"].to_numpy()
    exog_frame = build_exog(df)
    if exog_cols is not None:
        exog_frame = exog_frame[list(exog_cols)]
    exog = exog_frame.to_numpy()

    pieces = []
    fitted = None
    start_params = None
    for n, day in enumerate(pd.DatetimeIndex(days)):
        target = _target_hours(index, day)
        if len(target) == 0:
            continue

        # All 24 hours of the day share one cutoff: the 11:00 before the auction.
        cutoff = information_cutoff(target)[0]
        if cutoff not in index:
            continue
        i_cut = index.get_loc(cutoff)
        i_last = index.get_loc(target[-1])
        i_start = i_cut - 24 * window_days
        if i_start < 0:
            continue

        hist = slice(i_start, i_cut + 1)
        future = index[i_cut + 1 : i_last + 1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter on every refit
            if fitted is None or n % refit_every == 0:
                # Warm start from the previous refit: same model, two months later,
                # so the optimizer lands close to where it left off. Halves the fit
                # time and reads nothing but past estimates.
                fitted = _specify(load[hist], exog[hist], order, seasonal_order).fit(
                    disp=False, maxiter=100, start_params=start_params
                )
                start_params = fitted.params
                state = fitted
            else:
                state = fitted.apply(load[hist], exog=exog[hist], refit=False)
            mean = state.get_forecast(
                steps=i_last - i_cut, exog=exog_frame.loc[future].to_numpy()
            ).predicted_mean

        pieces.append(pd.Series(np.asarray(mean), index=future).loc[target])
        if progress and n % 30 == 0:
            print(f"{day.date()} ({n + 1} days)", flush=True)

    if not pieces:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz=index.tz))
    return pd.concat(pieces).rename("sarimax")


def forecast_split(df: pd.DataFrame, split: str, **kwargs) -> pd.Series:
    """Day-ahead forecasts for every day of a named split."""
    from ..utils.splits import mask

    days = df.index[mask(df.index, split)].normalize().unique()
    return forecast_days(df, days, **kwargs)
