"""Loading and cleaning of the freight rate datasets.

The cleaning steps here are deliberately conservative: everything that is
removed or imputed is counted and returned in a `QualityReport` so the choices
can be justified in the write-up rather than buried in the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@dataclass
class QualityReport:
    """Counts of every issue found and what was done about it."""

    rows_in: int = 0
    rows_out: int = 0
    duplicate_load_ids: int = 0
    duplicate_rows: int = 0
    missing_weight: int = 0
    missing_market_index: int = 0
    nonpositive_rate: int = 0
    rpm_outliers: int = 0
    unseen_cities: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "| Check | Count | Action |",
            "| --- | --- | --- |",
            f"| Rows read | {self.rows_in:,} | - |",
            f"| Duplicate `load_id` | {self.duplicate_load_ids:,} | dropped |",
            f"| Fully duplicated rows | {self.duplicate_rows:,} | dropped |",
            f"| Missing `weight` | {self.missing_weight:,} | median impute + missing flag |",
            f"| Missing `market_index` | {self.missing_market_index:,} | daily mean impute + missing flag |",
            f"| Non-positive `posted_rate` | {self.nonpositive_rate:,} | dropped |",
            f"| Rate-per-mile outside "
            f"[{config.RPM_LOWER}, {config.RPM_UPPER}] $/mi | {self.rpm_outliers:,} | dropped from training |",
            f"| Rows used for training | {self.rows_out:,} | - |",
        ]
        return "\n".join(lines)


def load_raw(path: Path) -> pd.DataFrame:
    """Read a dataset and coerce the columns to their intended types."""
    frame = pd.read_csv(path, parse_dates=[config.DATE_COL])
    numeric = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "weight",
        "market_index",
        "quote_signal",
        config.TARGET,
    ]
    for column in numeric:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def impute(frame: pd.DataFrame, weight_median: float | None = None) -> tuple[pd.DataFrame, float]:
    """Fill the two columns that carry missing values, flagging both.

    `market_index` is a market-wide daily signal (between-day variance is 6.6x
    the within-day variance), so a same-day mean is a far better estimate than
    a global constant. `weight` has no such structure and gets a median fill.
    """
    result = frame.copy()

    result["weight_missing"] = result["weight"].isna().astype("int8")
    if weight_median is None:
        weight_median = float(result["weight"].median())
    result["weight"] = result["weight"].fillna(weight_median)

    result["market_index_missing"] = result["market_index"].isna().astype("int8")
    daily_mean = result.groupby(config.DATE_COL)["market_index"].transform("mean")
    result["market_index"] = result["market_index"].fillna(daily_mean)
    result["market_index"] = result["market_index"].fillna(result["market_index"].median())

    return result, weight_median


def clean_training(frame: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    """Apply training-only cleaning (label-dependent filters live here)."""
    report = QualityReport(rows_in=len(frame))
    result = frame.copy()

    report.duplicate_load_ids = int(result["load_id"].duplicated().sum())
    result = result.drop_duplicates(subset="load_id", keep="first")

    feature_cols = [c for c in result.columns if c != "load_id"]
    report.duplicate_rows = int(result.duplicated(subset=feature_cols).sum())
    result = result.drop_duplicates(subset=feature_cols, keep="first")

    report.missing_weight = int(result["weight"].isna().sum())
    report.missing_market_index = int(result["market_index"].isna().sum())

    report.nonpositive_rate = int((result[config.TARGET] <= 0).sum())
    result = result[result[config.TARGET] > 0]

    rpm = result[config.TARGET] / result["distance"]
    keep = rpm.between(config.RPM_LOWER, config.RPM_UPPER)
    report.rpm_outliers = int((~keep).sum())
    result = result[keep]

    report.rows_out = len(result)
    return result.reset_index(drop=True), report


def rate_per_mile(frame: pd.DataFrame) -> pd.Series:
    return frame[config.TARGET] / frame["distance"]


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """Build one coordinate table from the pickup and delivery sides.

    Coordinates are internally consistent (each of the 72 cities has exactly one
    lat/lon, identical whether it appears as an origin or a destination), so the
    two sides can be stacked safely.
    """
    parts = []
    for frame in frames:
        parts.append(
            frame[["pickup", "pickup_lat", "pickup_lon"]].rename(
                columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
            )
        )
        parts.append(
            frame[["delivery", "delivery_lat", "delivery_lon"]].rename(
                columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
            )
        )
    stacked = pd.concat(parts, ignore_index=True).dropna(subset=["city"])
    return stacked.groupby("city", as_index=False)[["lat", "lon"]].first()


def load_all() -> dict[str, object]:
    """Load train/validation/December inputs and clean the training frame."""
    train_raw = load_raw(config.TRAIN_PATH)
    validation = load_raw(config.VALIDATION_PATH)
    december = pd.read_csv(config.DECEMBER_PATH, parse_dates=[config.DATE_COL])

    train, report = clean_training(train_raw)

    train_cities = set(train["pickup"]) | set(train["delivery"])
    validation_cities = set(validation["pickup"]) | set(validation["delivery"])
    report.unseen_cities = sorted(validation_cities - train_cities)

    return {
        "train_raw": train_raw,
        "train": train,
        "validation": validation,
        "december": december,
        "report": report,
        "coordinates": city_coordinates(train_raw, validation),
    }


def december_market_index(validation: pd.DataFrame) -> pd.DataFrame:
    """Recover the daily `market_index` for December from the validation set.

    `december_chart_inputs.csv` ships without `market_index`, but every one of
    its 31 dates is present in `validation.csv`, and the column is a market-wide
    daily signal rather than a per-load one. The same-day mean across the ~200
    validation loads is therefore the correct value for the fixed scenario, and
    it uses no information the model would not have on those dates.
    """
    december = validation[validation[config.DATE_COL].dt.month == 12]
    daily = december.groupby(config.DATE_COL, as_index=False)["market_index"].mean()
    return daily.rename(columns={"market_index": "market_index"})


def haversine_miles(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    radius = 3958.7613
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    inner = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(inner, 0, 1)))
