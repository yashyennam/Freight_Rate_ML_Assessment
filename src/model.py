"""Model definition, target transform and evaluation metrics.

The model predicts `log(posted_rate / distance)` -- log rate-per-mile -- rather
than the dollar rate directly. Two reasons:

*   Rate is close to multiplicative in distance (corr(distance, rate) = 0.91),
    so dividing it out leaves the model to learn the part that is actually
    uncertain instead of re-learning the length of the haul.
*   Errors are proportional rather than absolute: a $200 miss on a 3,000-mile
    load is not the same mistake as a $200 miss on a 300-mile load. Working in
    logs makes the loss match that.

Predictions are converted back with a smearing correction so the exponential
of a mean-unbiased log prediction is not biased low in dollars.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


def to_log_rpm(rate: pd.Series, distance: pd.Series) -> pd.Series:
    return np.log(rate / distance)


def to_rate(log_rpm: np.ndarray, distance: pd.Series, smearing: float = 1.0) -> np.ndarray:
    return np.exp(log_rpm) * np.asarray(distance, dtype=float) * smearing


def smearing_factor(residuals: np.ndarray) -> float:
    """Duan's smearing estimate: E[exp(residual)] on the log scale."""
    return float(np.mean(np.exp(residuals)))


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE": float(np.mean(np.abs(error / actual)) * 100),
        "R2": 1 - ss_res / ss_tot,
        "bias": float(np.mean(error)),
    }


LGB_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 0,
    "verbosity": -1,
    "seed": config.RANDOM_STATE,
}


@dataclass
class FittedModel:
    booster: object
    features: list[str]
    smearing: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        log_rpm = self.booster.predict(frame[self.features])
        return to_rate(log_rpm, frame["distance"], self.smearing)


# Columns the linear backbone owns. They are withheld from the booster so it
# cannot convert a smooth, extrapolatable relationship into day-specific steps.
LEVEL_FEATURES = ["market_index", "trend_days"]

RESIDUAL_EXCLUDE = set(LEVEL_FEATURES)


class HybridModel:
    """Linear level backbone + gradient-boosted cross-sectional residual.

    The daily rate level is driven by two things that a tree cannot represent
    beyond the training window: the market index, and a steady upward drift of
    roughly +0.64% per month. Regressing daily level on those two recovers
    R2 = 0.83, and because the fit is linear it keeps extrapolating into
    November and December instead of flattening at the last day it saw.

    Everything else -- lane, geography, equipment, weight, position within the
    month -- is cross-sectional and stationary, so it is handed to LightGBM,
    fitted on what the backbone leaves behind.

    `trend_damping` shrinks the extrapolated slope beyond the training window.
    A value of 1.0 trusts the trend fully; lower values hedge against it
    flattening. The value used is chosen by rolling-origin validation, not
    assumed -- see `src/evaluate.py`, experiment E5.
    """

    def __init__(self, features: list[str], trend_damping: float = 1.0, rounds: int = 900) -> None:
        self.features = features
        self.residual_features = [f for f in features if f not in RESIDUAL_EXCLUDE]
        self.trend_damping = trend_damping
        self.rounds = rounds
        self.coefficients_: np.ndarray | None = None
        self.booster_ = None
        self.smearing_ = 1.0
        self.train_end_: float = 0.0

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        """Backbone design matrix: market level, trend, size and equipment."""
        trend = frame["trend_days"].to_numpy(dtype=float)
        # Beyond the fitted window the slope is damped rather than trusted flat
        # out; inside the window this is the identity.
        overshoot = np.clip(trend - self.train_end_, 0, None)
        trend = np.minimum(trend, self.train_end_) + self.trend_damping * overshoot

        equipment = pd.get_dummies(
            pd.Categorical(frame["equipment"], categories=["Dry Van", "Reefer", "Flatbed"])
        ).to_numpy(dtype=float)[:, 1:]

        return np.column_stack(
            [
                frame["market_index"].to_numpy(dtype=float),
                trend / 100.0,
                frame["log_distance"].to_numpy(dtype=float),
                equipment,
                np.ones(len(frame)),
            ]
        )

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "HybridModel":
        self.train_end_ = float(frame["trend_days"].max())
        design = self._design(frame)
        self.coefficients_, *_ = np.linalg.lstsq(design, target.to_numpy(dtype=float), rcond=None)
        residual = target.to_numpy(dtype=float) - design @ self.coefficients_

        self.booster_ = train_lightgbm(
            frame, self.residual_features, pd.Series(residual, index=frame.index),
            num_boost_round=self.rounds,
        )
        final_residual = residual - self.booster_.predict(frame[self.residual_features])
        self.smearing_ = smearing_factor(final_residual)
        return self

    def predict_log_rpm(self, frame: pd.DataFrame) -> np.ndarray:
        return self._design(frame) @ self.coefficients_ + self.booster_.predict(
            frame[self.residual_features]
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return to_rate(self.predict_log_rpm(frame), frame["distance"], self.smearing_)


def train_lightgbm(
    train: pd.DataFrame,
    features: list[str],
    target: pd.Series,
    num_boost_round: int = config.N_BOOST_ROUNDS,
) -> object:
    """Fit a booster for a fixed number of rounds.

    The round count is fixed rather than early-stopped: an early-stopping split
    would have to come from the training window, and every such split here is
    either random (which the quote_signal analysis shows is misleading) or eats
    the most recent month, which is the most valuable training data for a
    forward prediction.
    """
    import lightgbm as lgb

    dataset = lgb.Dataset(
        train[features],
        label=target,
        categorical_feature=[f for f in ("equipment_code",) if f in features],
        free_raw_data=False,
    )
    return lgb.train(
        LGB_PARAMS, dataset, num_boost_round=num_boost_round, callbacks=[lgb.log_evaluation(period=0)]
    )
