"""Paths, column groups and modelling constants shared across the pipeline."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR = ROOT / "artifacts"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

PREDICTIONS_PATH = ROOT / "validation_predictions.csv"

TARGET = "posted_rate"
DATE_COL = "date"

# `quote_signal` is deliberately excluded from the feature set. See
# `notes/quote_signal.md` and reports/data_quality.md: the column is a genuine
# rate-per-mile quote in 5 of the 10 training months, a sign-inverted copy in
# 4 months, and pure noise in August -- which is the same regime as the entire
# November/December prediction window. Using it trains on a signal that does
# not exist at prediction time.
DROPPED_COLUMNS = ["quote_signal"]

# Rate-per-mile bounds used to drop physically implausible training labels.
# Chosen from the empirical distribution: the 0.1% tails are 0.44 and 10.1 $/mi
# against a median of 2.15 $/mi.
RPM_LOWER = 0.75
RPM_UPPER = 6.0

RANDOM_STATE = 42

# Boosting rounds for the residual model, and how much of the fitted upward
# drift to project past the end of the training window. Both are fixed by the
# rolling-origin sweep in `src/evaluate.py` (experiments E3 and E5); damping of
# 0.0 -- hold the level at the last observed day -- won that sweep, and is also
# the conservative reading of a drift that flattens over August-October.
N_BOOST_ROUNDS = 900
TREND_DAMPING = 0.0

# Fixed December scenario, mirrored from score.py so predictions stay in sync.
DECEMBER_FIXED = {
    "pickup": "Lexington",
    "delivery": "Fort Wayne",
    "distance": 360.0,
    "equipment": "Dry Van",
    "weight": 32_000.0,
}
