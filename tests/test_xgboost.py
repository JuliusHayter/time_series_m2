"""The booster must see the same inputs whatever happens after the cutoff."""

import pandas as pd
import pytest

from time_series_m2.utils.data import information_cutoff, load_national
from time_series_m2.xgboost import build_dataset, fit, forecast, walk_forward

TARGET = pd.Timestamp("2024-03-13", tz="Europe/Paris")


@pytest.fixture(scope="module")
def df():
    return load_national()


@pytest.mark.parametrize("feature_set", ["independent", "with_rte"])
def test_features_ignore_load_after_the_cutoff(df, feature_set):
    """Poison the future of the target day: its feature rows must not move."""
    honest, _ = build_dataset(df, feature_set)
    rows = honest.index[honest.index.normalize() == TARGET]
    cutoff = information_cutoff(rows)[0]

    poisoned = df.copy()
    poisoned.loc[poisoned.index > cutoff, "load_mw"] = 0.0
    tampered, _ = build_dataset(poisoned, feature_set)

    columns = honest.attrs["feature_columns"]
    pd.testing.assert_frame_equal(honest.loc[rows, columns], tampered.loc[rows, columns])


def test_independent_set_carries_no_rte_column(df):
    frame, _ = build_dataset(df, "independent")
    assert not any("rte" in c for c in frame.attrs["feature_columns"])


def test_rte_features_are_checked_when_used(df):
    _, report = build_dataset(df, "with_rte")
    checked = set(report["feature"])

    assert {"rte_forecast_mw", "rte_bias"} <= checked
    assert "LEAK" not in set(report["status"])


def test_unknown_feature_set_is_rejected(df):
    with pytest.raises(ValueError):
        build_dataset(df, "everything")


def test_walk_forward_never_trains_on_the_block_it_scores(df):
    """Poisoning the load after the first cutoff must leave the first day unchanged."""
    frame, _ = build_dataset(df, "independent")
    columns = frame.attrs["feature_columns"]
    days = pd.date_range("2024-03-11", periods=3, tz="Europe/Paris")
    params = dict(n_estimators=20, max_depth=3)

    honest = walk_forward(frame, columns, days, params=params, refit_every_days=3)

    poisoned_df = df.copy()
    first_cutoff = information_cutoff(pd.DatetimeIndex([days[0]]))[0]
    poisoned_df.loc[poisoned_df.index > first_cutoff, "load_mw"] = 0.0
    poisoned, _ = build_dataset(poisoned_df, "independent")
    tampered = walk_forward(poisoned, columns, days, params=params, refit_every_days=3)

    # Later days of the block legitimately read the poisoned hours through their
    # own, later cutoffs; only the first target day is comparable.
    first = days[0]
    a = honest[honest.index.normalize() == first]
    b = tampered[tampered.index.normalize() == first]
    pd.testing.assert_series_equal(a, b)
