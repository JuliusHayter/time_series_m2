"""Bands must be causal and honest about their coverage."""

import numpy as np
import pandas as pd
import pytest

from time_series_m2.uncertainty import pinball, rolling_conformal
from time_series_m2.utils.metrics import interval_coverage

TZ = "Europe/Paris"


@pytest.fixture(scope="module")
def synthetic():
    """Hourly series with iid noise: a 90% band should cover about 90%."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=24 * 300, freq="h", tz=TZ)
    pred = pd.Series(50_000 + 5_000 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24), index=idx)
    actual = pred + rng.normal(0, 800, len(idx))
    return actual, pred


def test_band_covers_close_to_nominal_on_iid_errors(synthetic):
    actual, pred = synthetic
    band = rolling_conformal(actual, pred, nominal=0.9)
    cov = interval_coverage(actual, band["lower"], band["upper"])

    assert cov["coverage"] == pytest.approx(0.9, abs=0.03)


def test_band_for_a_day_ignores_actuals_from_the_day_before(synthetic):
    """The auction for day D closes on D-1: errors of D-1 and later are unknown."""
    actual, pred = synthetic
    target = pd.Timestamp("2024-09-15", tz=TZ)
    honest = rolling_conformal(actual, pred)

    poisoned = actual.copy()
    poisoned.loc[poisoned.index >= target - pd.Timedelta(days=1)] += 50_000
    tampered = rolling_conformal(poisoned, pred)

    day = honest.index.normalize() == target
    np.testing.assert_allclose(honest[day].to_numpy(), tampered[day].to_numpy())


def test_band_is_undefined_before_enough_history(synthetic):
    actual, pred = synthetic
    band = rolling_conformal(actual, pred, min_days=20)

    assert band.loc[: pd.Timestamp("2024-01-10", tz=TZ)].isna().all().all()


def test_pinball_at_one_half_is_half_the_mae():
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz=TZ)
    actual = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx)
    pred = pd.Series([12.0, 18.0, 33.0, 40.0], index=idx)

    assert pinball(actual, pred, 0.5) == pytest.approx(np.abs(actual - pred).mean() / 2)


def test_pinball_penalizes_the_short_side_when_tau_is_high():
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz=TZ)
    actual = pd.Series([100.0, 100.0], index=idx)
    under = pd.Series([90.0, 90.0], index=idx)   # short by 10
    over = pd.Series([110.0, 110.0], index=idx)  # long by 10

    assert pinball(actual, under, 0.8) > pinball(actual, over, 0.8)
