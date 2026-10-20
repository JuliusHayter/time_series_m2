"""The SARIMAX wrapper must forecast the auction horizon and nothing else."""

import numpy as np
import pandas as pd
import pytest

from time_series_m2.sarima.model import build_exog, forecast_days
from time_series_m2.utils.data import information_cutoff, load_national

# Small window: these tests check the information constraint, not accuracy.
FAST = dict(window_days=20, refit_every=1)


@pytest.fixture(scope="module")
def df():
    return load_national()


def day(ts):
    return pd.Timestamp(ts, tz="Europe/Paris")


def test_forecast_covers_exactly_the_auctioned_day(df):
    target = day("2024-03-13")
    out = forecast_days(df, [target], **FAST)

    assert len(out) == 24
    assert (out.index.normalize() == target).all()
    assert out.notna().all()


def test_forecast_starts_thirteen_hours_after_the_cutoff(df):
    target = day("2024-03-13")
    out = forecast_days(df, [target], **FAST)
    cutoff = information_cutoff(out.index)[0]

    ahead = (out.index - cutoff) / pd.Timedelta(hours=1)
    assert ahead.min() == 13
    assert ahead.max() == 36


def test_load_after_the_cutoff_cannot_change_the_forecast(df):
    """The sharpest leakage test available: poison the future and look away.

    If any realized load after the cutoff reached the fit or the filter, the two
    forecasts would differ.
    """
    target = day("2024-03-13")
    honest = forecast_days(df, [target], **FAST)

    cutoff = information_cutoff(honest.index)[0]
    poisoned = df.copy()
    poisoned.loc[poisoned.index > cutoff, "load_mw"] = 0.0
    tampered = forecast_days(poisoned, [target], **FAST)

    np.testing.assert_allclose(honest.to_numpy(), tampered.to_numpy())


def test_dst_days_keep_their_real_length(df):
    """October has a 25-hour day and March a 23-hour one. Neither is padded."""
    spring = forecast_days(df, [day("2024-03-31")], **FAST)
    autumn = forecast_days(df, [day("2024-10-27")], **FAST)

    assert len(spring) == 23
    assert len(autumn) == 25


def test_exog_carries_no_realized_load(df):
    exog = build_exog(df.loc["2024"])

    assert not any("load" in col for col in exog.columns)
    assert exog.notna().all().all()
