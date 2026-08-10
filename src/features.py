"""Feature engineering.

Two constraints shape this module:

1.  Every feature must be computable for `december_chart_inputs.csv`, which
    only carries pickup, delivery, distance, equipment, weight and date. That
    rules out anything derived from per-load columns.
2.  Nothing may require extrapolation beyond the training window. The training
    data covers January-October 2025 while predictions are needed for November
    and December, so absolute-time features (month, day-of-year, a trend index)
    would force every prediction into the last bucket a tree ever saw. The
    market level is supplied instead by `market_index`, whose November/December
    values (0.919 / 0.935) sit comfortably inside the training range
    (0.89 - 1.30). Only *within*-month and *within*-week calendar position is
    used, and both are fully in-distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import haversine_miles

EQUIPMENT_ORDER = ["Dry Van", "Reefer", "Flatbed"]

# Days since the first day of the training window. Only the linear backbone in
# `model.HybridModel` is allowed to see this; it is withheld from the booster,
# which would otherwise turn a smooth drift into steps that stop at Oct 31.
TREND_EPOCH = pd.Timestamp("2025-01-01")

BASE_FEATURES = [
    "trend_days",
    "log_distance",
    "distance",
    "weight",
    "weight_missing",
    "weight_per_mile",
    "equipment_code",
    "market_index",
    "market_index_missing",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "mid_lat",
    "mid_lon",
    "haversine",
    "circuity",
    "bearing_sin",
    "bearing_cos",
    "day_of_week",
    "is_weekend",
    "day_of_month",
    "month_progress",
    "days_to_month_end",
]

ENCODED_FEATURES = ["lane_te", "pickup_te", "delivery_te", "lane_count"]

FEATURES = BASE_FEATURES + ENCODED_FEATURES

# Feature set for a plain (non-hybrid) booster. `trend_days` is removed because
# a tree given a raw time index simply memorises it and then predicts the last
# training bucket forever; it is only meaningful inside the linear backbone.
GBM_FEATURES = [f for f in FEATURES if f != "trend_days"]

CATEGORICAL = ["equipment_code"]


def attach_coordinates(frame: pd.DataFrame, coordinates: pd.DataFrame) -> pd.DataFrame:
    """Fill pickup/delivery lat-lon from the city table when absent.

    The December inputs carry city names only; the lookup makes them equivalent
    to a normal row.
    """
    result = frame.copy()
    lookup = coordinates.set_index("city")
    for side in ("pickup", "delivery"):
        for axis in ("lat", "lon"):
            column = f"{side}_{axis}"
            mapped = result[side].map(lookup[axis])
            if column in result.columns:
                result[column] = result[column].fillna(mapped)
            else:
                result[column] = mapped
    return result


def bearing(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial great-circle bearing, in radians, origin -> destination."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


def build_base(frame: pd.DataFrame, coordinates: pd.DataFrame) -> pd.DataFrame:
    """Compute every feature that depends only on a single row."""
    result = attach_coordinates(frame, coordinates)

    if "weight_missing" not in result.columns:
        result["weight_missing"] = result["weight"].isna().astype("int8")
    if "market_index_missing" not in result.columns:
        result["market_index_missing"] = result.get(
            "market_index", pd.Series(np.nan, index=result.index)
        ).isna().astype("int8")

    result["log_distance"] = np.log(result["distance"])
    result["weight_per_mile"] = result["weight"] / result["distance"]

    equipment = pd.Categorical(result["equipment"], categories=EQUIPMENT_ORDER)
    result["equipment_code"] = equipment.codes.astype("int16")

    result["haversine"] = haversine_miles(
        result["pickup_lat"], result["pickup_lon"], result["delivery_lat"], result["delivery_lon"]
    )
    # Road distance runs ~1.18x the great-circle distance; the ratio flags
    # lanes that are more indirect than their straight-line distance suggests.
    result["circuity"] = result["distance"] / result["haversine"].replace(0, np.nan)
    result["circuity"] = result["circuity"].fillna(result["circuity"].median())

    angle = bearing(
        result["pickup_lat"], result["pickup_lon"], result["delivery_lat"], result["delivery_lon"]
    )
    result["bearing_sin"] = np.sin(angle)
    result["bearing_cos"] = np.cos(angle)
    result["mid_lat"] = (result["pickup_lat"] + result["delivery_lat"]) / 2
    result["mid_lon"] = (result["pickup_lon"] + result["delivery_lon"]) / 2

    dates = result[config.DATE_COL]
    result["trend_days"] = (dates - TREND_EPOCH).dt.days.astype(float)
    result["day_of_week"] = dates.dt.dayofweek.astype("int16")
    result["is_weekend"] = (result["day_of_week"] >= 5).astype("int8")
    result["day_of_month"] = dates.dt.day.astype("int16")
    days_in_month = dates.dt.days_in_month
    # Rates ramp through the month (-1.1% on the 1st to +1.5% on the 29th once
    # market level, equipment and distance are controlled for), so position
    # within the month matters more than the raw day number.
    result["month_progress"] = (result["day_of_month"] - 1) / (days_in_month - 1)
    result["days_to_month_end"] = (days_in_month - result["day_of_month"]).astype("int16")

    result["lane"] = result["pickup"].astype(str) + " -> " + result["delivery"].astype(str)
    return result


class TargetEncoder:
    """Smoothed mean-target encoding for lane, origin and destination.

    Encodes log rate-per-mile rather than the raw rate so the statistic is not
    dominated by lane length. Smoothing pulls thin categories toward the global
    prior, which is what keeps the 8 cities that only ever appear in the
    validation set from producing a wild estimate -- they fall back to the
    prior exactly.
    """

    def __init__(self, smoothing: float = 25.0) -> None:
        self.smoothing = smoothing
        self.prior_: float = 0.0
        self.maps_: dict[str, pd.Series] = {}
        self.counts_: pd.Series | None = None

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "TargetEncoder":
        self.prior_ = float(target.mean())
        for name, key in (("lane_te", "lane"), ("pickup_te", "pickup"), ("delivery_te", "delivery")):
            stats = target.groupby(frame[key]).agg(["mean", "count"])
            weight = stats["count"] / (stats["count"] + self.smoothing)
            self.maps_[name] = weight * stats["mean"] + (1 - weight) * self.prior_
        self.counts_ = frame["lane"].value_counts()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for name, key in (("lane_te", "lane"), ("pickup_te", "pickup"), ("delivery_te", "delivery")):
            result[name] = result[key].map(self.maps_[name]).fillna(self.prior_)
        result["lane_count"] = result["lane"].map(self.counts_).fillna(0).astype(float)
        return result

    def fit_transform(self, frame: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        return self.fit(frame, target).transform(frame)


def out_of_fold_encoding(
    frame: pd.DataFrame, target: pd.Series, n_splits: int = 5, smoothing: float = 25.0
) -> pd.DataFrame:
    """Encode the training rows out-of-fold so the encoder cannot leak labels."""
    from sklearn.model_selection import KFold

    result = frame.copy()
    for column in ENCODED_FEATURES:
        result[column] = np.nan

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    for fit_idx, apply_idx in splitter.split(result):
        encoder = TargetEncoder(smoothing=smoothing)
        encoder.fit(result.iloc[fit_idx], target.iloc[fit_idx])
        encoded = encoder.transform(result.iloc[apply_idx])
        for column in ENCODED_FEATURES:
            result.iloc[apply_idx, result.columns.get_loc(column)] = encoded[column].to_numpy()
    return result
