"""Validation experiments.

Run with `python -m src.evaluate`. Everything printed here is evidence for the
write-up, so each experiment states what it is testing and what the result
implies for the final model.

The experiments are:

  E1  Why the split matters -- the same feature set scored under a random
      split and under a forward-in-time split.
  E2  Baselines, so the gradient-boosted model has something to beat.
  E3  Rolling-origin validation -- train on everything up to month m, predict
      month m+1, for m = 4..9.
  E4  Feature ablations -- what each block of features is actually worth.
  E5  How far to trust the extrapolated trend, swept over the same folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data, features, model


def prepare(frames: dict) -> pd.DataFrame:
    """Featurised frame with **no** outlier filter and **no** encodings yet.

    Both are deliberately deferred to `make_fold`. Filtering outliers here would
    also strip them out of every test fold, which flatters the metrics; encoding
    here would let a fold's training rows carry statistics computed from months
    that fold is supposed to be predicting.
    """
    raw = data.load_raw(config.TRAIN_PATH)
    raw, _ = data.impute(raw)
    raw = features.build_base(raw, frames["coordinates"])
    raw["log_rpm"] = model.to_log_rpm(raw[config.TARGET], raw["distance"])
    return raw


def make_fold(
    frame: pd.DataFrame, train_mask: pd.Series, test_mask: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one fold: training rows cleaned and encoded, test rows untouched.

    The outlier filter is a *training* decision, so the test half keeps its
    extreme loads and the reported error includes them. Encodings are computed
    out-of-fold within this fold's training window only.
    """
    train = frame[train_mask].copy()
    rate_per_mile = train[config.TARGET] / train["distance"]
    train = train[rate_per_mile.between(config.RPM_LOWER, config.RPM_UPPER)].reset_index(drop=True)
    train = features.out_of_fold_encoding(train, train["log_rpm"])
    return train, frame[test_mask].copy()


def fit_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_list: list[str],
    rounds: int = config.N_BOOST_ROUNDS,
    hybrid: bool = False,
    trend_damping: float = config.TREND_DAMPING,
) -> dict[str, float]:
    """Fit on `train`, score on `test`, with encodings refit on train only."""
    encoder = features.TargetEncoder()
    encoder.fit(train, train["log_rpm"])
    test_encoded = encoder.transform(test)

    if hybrid:
        fitted = model.HybridModel(feature_list, trend_damping=trend_damping, rounds=rounds)
        fitted.fit(train, train["log_rpm"])
    else:
        booster = model.train_lightgbm(
            train, feature_list, train["log_rpm"], num_boost_round=rounds
        )
        residuals = train["log_rpm"].to_numpy() - booster.predict(train[feature_list])
        fitted = model.FittedModel(booster, feature_list, model.smearing_factor(residuals))

    predicted = fitted.predict(test_encoded)
    return model.metrics(test_encoded[config.TARGET], predicted)


def experiment_split_design(frame: pd.DataFrame) -> str:
    """E1: the `quote_signal` trap, shown under two split designs."""
    lines = [
        "## E1  Split design: why a random split is misleading here",
        "",
        "`quote_signal` equals rate-per-mile almost exactly for ~50% of training",
        "rows. Under a random split those rows appear in both halves and the",
        "feature looks decisive. Under a forward split it collapses, because the",
        "regime that generated it changes month to month -- and November and",
        "December (the rows we must actually predict) are in the regime where the",
        "column is pure noise.",
        "",
    ]
    with_qs = features.GBM_FEATURES + ["quote_signal"]
    without_qs = features.GBM_FEATURES

    # -- random split -------------------------------------------------------
    shuffled = frame.sample(frac=1.0, random_state=config.RANDOM_STATE).reset_index(drop=True)
    is_train = pd.Series(shuffled.index < int(len(shuffled) * 0.8), index=shuffled.index)
    rnd_train, rnd_test = make_fold(shuffled, is_train, ~is_train)

    rows = []
    rows.append(("random 80/20", "with quote_signal", fit_and_score(rnd_train, rnd_test, with_qs)))
    rows.append(("random 80/20", "without quote_signal", fit_and_score(rnd_train, rnd_test, without_qs)))

    # -- forward split: train <= September, test October ---------------------
    month = frame[config.DATE_COL].dt.month
    fwd_train, fwd_test = make_fold(frame, month <= 9, month == 10)
    rows.append(("forward (<=Sep -> Oct)", "with quote_signal", fit_and_score(fwd_train, fwd_test, with_qs)))
    rows.append(("forward (<=Sep -> Oct)", "without quote_signal", fit_and_score(fwd_train, fwd_test, without_qs)))

    # -- forward split into August, the month whose regime matches Nov/Dec ---
    aug_train, aug_test = make_fold(frame, month <= 7, month == 8)
    rows.append(("forward (<=Jul -> Aug)", "with quote_signal", fit_and_score(aug_train, aug_test, with_qs)))
    rows.append(("forward (<=Jul -> Aug)", "without quote_signal", fit_and_score(aug_train, aug_test, without_qs)))

    lines += ["| Split | Feature set | MAE | RMSE | MAPE | R2 |", "| --- | --- | --- | --- | --- | --- |"]
    for split, variant, m in rows:
        lines.append(
            f"| {split} | {variant} | ${m['MAE']:.2f} | ${m['RMSE']:.2f} | "
            f"{m['MAPE']:.2f}% | {m['R2']:.4f} |"
        )
    lines += [
        "",
        "August is the informative row: it is the one *labelled* month sharing the",
        "November/December regime, and there `quote_signal` actively hurts.",
        "It is dropped from the final feature set.",
        "",
    ]
    return "\n".join(lines)


def experiment_baselines(frame: pd.DataFrame) -> str:
    """E2: simple baselines on the forward October holdout."""
    month = frame[config.DATE_COL].dt.month
    train, test = make_fold(frame, month <= 9, month == 10)
    actual = test[config.TARGET].to_numpy()

    results: list[tuple[str, dict[str, float]]] = []

    flat = float(train["log_rpm"].mean())
    results.append(
        ("Global mean $/mi x distance", model.metrics(actual, np.exp(flat) * test["distance"]))
    )

    lane_mean = train.groupby("lane")["log_rpm"].mean()
    mapped = test["lane"].map(lane_mean).fillna(flat)
    results.append(("Lane mean $/mi x distance", model.metrics(actual, np.exp(mapped) * test["distance"])))

    equip_mean = train.groupby(["equipment", pd.cut(train["distance"], [0, 500, 1000, 2000, 4000])],
                              observed=True)["log_rpm"].mean()
    key = list(zip(test["equipment"], pd.cut(test["distance"], [0, 500, 1000, 2000, 4000])))
    mapped = pd.Series([equip_mean.get(k, flat) for k in key], index=test.index)
    results.append(
        ("Equipment x distance band", model.metrics(actual, np.exp(mapped) * test["distance"]))
    )

    from sklearn.linear_model import Ridge

    # `test` arrives unencoded from make_fold; the Ridge design needs the same
    # encoder the model stage uses, fitted on this fold's training rows only.
    test_encoded = features.TargetEncoder().fit(train, train["log_rpm"]).transform(test)

    numeric = [f for f in features.FEATURES if f != "equipment_code"]
    design_train = pd.get_dummies(train[numeric + ["equipment"]], columns=["equipment"]).astype(float)
    design_test = pd.get_dummies(test_encoded[numeric + ["equipment"]], columns=["equipment"]).astype(float)
    design_test = design_test.reindex(columns=design_train.columns, fill_value=0.0)
    ridge = Ridge(alpha=1.0).fit(design_train, train["log_rpm"])
    predicted = np.exp(ridge.predict(design_test)) * test["distance"]
    results.append(("Ridge on log $/mi", model.metrics(actual, predicted)))

    results.append(("LightGBM on log $/mi", fit_and_score(train, test, features.GBM_FEATURES)))
    results.append(
        ("Hybrid: linear level + LightGBM", fit_and_score(train, test, features.FEATURES, hybrid=True))
    )

    lines = [
        "## E2  Baselines (train <= September, test October)",
        "",
        "| Model | MAE | RMSE | MAPE | R2 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, m in results:
        lines.append(f"| {name} | ${m['MAE']:.2f} | ${m['RMSE']:.2f} | {m['MAPE']:.2f}% | {m['R2']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def experiment_rolling_origin(frame: pd.DataFrame) -> str:
    """E3: rolling-origin validation, the design used to pick the final model."""
    month = frame[config.DATE_COL].dt.month
    lines = [
        "## E3  Rolling-origin validation",
        "",
        "Train on every load up to and including month *m*, predict month *m+1*.",
        "This mirrors the real task -- fit on the past, predict a month that has",
        "not happened yet -- and it is the score the final configuration was",
        "chosen on.",
        "",
        "| Train through | Test month | Rows | Plain LightGBM MAE | Hybrid MAE | Hybrid MAPE | Hybrid R2 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    plain_all, hybrid_all = [], []
    names = {5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October"}
    for cutoff in range(4, 10):
        train, test = make_fold(frame, month <= cutoff, month == cutoff + 1)
        if test.empty:
            continue
        plain = fit_and_score(train, test, features.GBM_FEATURES)
        hybrid = fit_and_score(train, test, features.FEATURES, hybrid=True)
        plain_all.append(plain)
        hybrid_all.append(hybrid)
        lines.append(
            f"| month {cutoff} | {names[cutoff + 1]} | {len(test):,} | ${plain['MAE']:.2f} | "
            f"**${hybrid['MAE']:.2f}** | {hybrid['MAPE']:.2f}% | {hybrid['R2']:.4f} |"
        )
    mean_plain = float(np.mean([m["MAE"] for m in plain_all]))
    mean_hybrid = {k: float(np.mean([m[k] for m in hybrid_all])) for k in hybrid_all[0]}
    lines.append(
        f"| **mean** | | | **${mean_plain:.2f}** | **${mean_hybrid['MAE']:.2f}** | "
        f"**{mean_hybrid['MAPE']:.2f}%** | **{mean_hybrid['R2']:.4f}** |"
    )
    lines.append("")
    return "\n".join(lines)


def experiment_trend_damping(frame: pd.DataFrame) -> str:
    """E5: how much of the extrapolated trend to trust, chosen by validation.

    The backbone projects a +0.64%/month drift into a window with no labels. If
    the drift flattens, a full projection over-predicts. Rather than guess, the
    damping factor is swept over the same rolling-origin folds; the last two
    folds are weighted most since they extrapolate furthest.
    """
    month = frame[config.DATE_COL].dt.month
    dampings = [0.0, 0.25, 0.5, 0.75, 1.0]
    lines = [
        "## E5  How much of the trend to extrapolate",
        "",
        "`trend_damping` = 0 freezes the level at the last training day;",
        "1.0 projects the fitted drift forward in full.",
        "",
        "| Damping | " + " | ".join(f"MAE {n}" for n in ["Jun", "Jul", "Aug", "Sep", "Oct"]) + " | mean MAE |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    best, best_mae = 1.0, float("inf")
    for damping in dampings:
        maes = []
        for cutoff in range(5, 10):
            train, test = make_fold(frame, month <= cutoff, month == cutoff + 1)
            m = fit_and_score(train, test, features.FEATURES, hybrid=True, trend_damping=damping)
            maes.append(m["MAE"])
        mean_mae = float(np.mean(maes))
        if mean_mae < best_mae:
            best, best_mae = damping, mean_mae
        lines.append(
            f"| {damping:.2f} | " + " | ".join(f"${v:.2f}" for v in maes) + f" | **${mean_mae:.2f}** |"
        )
    lines += [
        "",
        f"Selected `trend_damping = {best:.2f}` (mean MAE ${best_mae:.2f}).",
        "",
        "Note this is a one-month-ahead sweep, while December is two months past",
        "the end of training. The chosen value is applied to both horizons.",
        "",
    ]
    return "\n".join(lines)


def experiment_ablation(frame: pd.DataFrame) -> str:
    """E4: what each block of features contributes on the October holdout."""
    month = frame[config.DATE_COL].dt.month
    train, test = make_fold(frame, month <= 9, month == 10)

    blocks = {
        "All features": features.GBM_FEATURES,
        "- market_index": [f for f in features.GBM_FEATURES if not f.startswith("market_index")],
        "- calendar": [
            f
            for f in features.GBM_FEATURES
            if f not in {"day_of_week", "is_weekend", "day_of_month", "month_progress", "days_to_month_end"}
        ],
        "- geography": [
            f
            for f in features.GBM_FEATURES
            if f
            not in {
                "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
                "mid_lat", "mid_lon", "haversine", "circuity", "bearing_sin", "bearing_cos",
            }
        ],
        "- target encodings": [f for f in features.GBM_FEATURES if f not in features.ENCODED_FEATURES],
        "- weight": [f for f in features.GBM_FEATURES if f not in {"weight", "weight_missing", "weight_per_mile"}],
    }

    lines = [
        "## E4  Feature ablation (train <= September, test October)",
        "",
        "| Feature set | MAE | RMSE | MAPE | R2 | dMAE vs all |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    reference = None
    for name, feature_list in blocks.items():
        m = fit_and_score(train, test, feature_list)
        if reference is None:
            reference = m["MAE"]
        delta = m["MAE"] - reference
        lines.append(
            f"| {name} | ${m['MAE']:.2f} | ${m['RMSE']:.2f} | {m['MAPE']:.2f}% | "
            f"{m['R2']:.4f} | {delta:+.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    frames = data.load_all()
    print(frames["report"].to_markdown())
    print()

    frame = prepare(frames)

    sections = [
        "# Validation results",
        "",
        "Generated by `python -m src.evaluate`.",
        "",
        "## Data quality",
        "",
        frames["report"].to_markdown(),
        "",
        f"Cities appearing only in the validation set: "
        f"{', '.join(frames['report'].unseen_cities)} "
        f"({len(frames['report'].unseen_cities)} of "
        f"{frames['coordinates'].shape[0]}). Handled by geographic features plus "
        "prior-smoothed encodings rather than city identity.",
        "",
        experiment_split_design(frame),
        experiment_baselines(frame),
        experiment_rolling_origin(frame),
        experiment_trend_damping(frame),
        experiment_ablation(frame),
    ]
    report = "\n".join(sections)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = config.REPORTS_DIR / "validation_results.md"
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
