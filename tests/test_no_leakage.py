"""The information constraint is the one property the whole project rests on.

Every number we report is meaningless if a feature reads past the moment the
day-ahead auction closes, so it is tested rather than trusted.
"""

import pandas as pd
import pytest

from time_series_m2.baselines import naive
from time_series_m2.utils.data import (DECISION_HOUR, information_cutoff,
                                       decision_time, load_national)
from time_series_m2.utils.features import (add_load_features,
                                           add_temperature_features,
                                           check_leakage)


@pytest.fixture(scope="module")
def df():
    return load_national()


def test_decision_time_is_noon_the_day_before(df):
    idx = df.index[:500]
    dec = decision_time(idx)
    assert (dec.hour == DECISION_HOUR).all()
    assert ((pd.DatetimeIndex(idx.date) - pd.DatetimeIndex(dec.date)).days == 1).all()


def test_all_24_hours_share_one_cutoff(df):
    day = df.loc["2025-03-15"]
    assert information_cutoff(day.index).nunique() == 1


def test_cutoff_survives_dst_days(df):
    for day, expected_hours in (("2025-03-30", 23), ("2025-10-26", 25)):
        idx = df.loc[day].index
        assert len(idx) == expected_hours
        assert (information_cutoff(idx).hour == DECISION_HOUR - 1).all()


def test_load_features_never_read_past_the_cutoff(df):
    featured, sources = add_load_features(df)
    report = check_leakage(featured.index, sources)
    assert (report["status"] == "ok").all()


def test_lag_24h_is_rejected(df):
    """The obvious lag is unusable: at 23:00 it points 12h past the cutoff."""
    sources = {"load_lag_24h": df.index - pd.Timedelta(hours=24)}
    with pytest.raises(AssertionError, match="past the cutoff"):
        check_leakage(df.index, sources)


def test_temperature_features_are_flagged_not_hidden(df):
    _, sources = add_temperature_features(df)
    report = check_leakage(df.index, sources, exempt=set(sources))
    assert report["status"].str.startswith("exempt").all()


def test_persistence_refuses_unavailable_lags(df):
    with pytest.raises(ValueError):
        naive.persistence(df, days=1)


def test_bias_correction_uses_only_fully_observed_days(df):
    idx = df.index
    last_used = (
        pd.DatetimeIndex(idx.date) - pd.Timedelta(days=2) + pd.Timedelta(hours=23)
    ).tz_localize("Europe/Paris")
    assert (last_used <= information_cutoff(idx)).all()


def test_debiasing_removes_the_bias_out_of_sample(df):
    from time_series_m2.utils import metrics
    from time_series_m2.utils.splits import split_frame

    test = split_frame(df)["test"]
    raw = metrics.score(test["load_mw"], naive.rte_raw(df).loc[test.index])
    fixed = metrics.score(test["load_mw"], naive.rte_debiased(df, 15).loc[test.index])
    assert raw["bias"] > 500
    assert abs(fixed["bias"]) < 100
    assert fixed["MAE"] < raw["MAE"]
