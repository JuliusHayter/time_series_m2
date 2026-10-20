"""Asymmetric cost of forecast errors.

The data holds no imbalance prices, so no euro figure is claimed. Under-forecasting
(actual above forecast) leaves the desk short, over-forecasting leaves it long;
if a shortfall costs ``tau / (1 - tau)`` times a surplus, the right loss is the
pinball loss at level ``tau``. ``tau = 0.5`` is half the MAE.
"""

from __future__ import annotations

import pandas as pd


def pinball(actual: pd.Series, pred: pd.Series, tau: float = 0.5) -> float:
    """Mean pinball loss in MW. Short positions (actual > pred) weigh ``tau``."""
    both = pd.concat([actual.rename("a"), pred.rename("p")], axis=1).dropna()
    err = both["a"] - both["p"]
    return float((tau * err.clip(lower=0) + (1 - tau) * (-err).clip(lower=0)).mean())
