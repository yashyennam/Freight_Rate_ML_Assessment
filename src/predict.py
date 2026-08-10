"""Fit the final model on all labelled data and write both prediction files.

Run with `python -m src.predict`. Produces:

  validation_predictions.csv        12,000 rows, load_id + predicted_rate
  data/december_chart_inputs.csv    the fixed scenario, predicted_rate filled
  reports/feature_importance.md     what the residual model leaned on
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data, features, model


def build_training_matrix(frames: dict) -> pd.DataFrame:
    train, weight_median = data.impute(frames["train"])
    train = features.build_base(train, frames["coordinates"])
    train["log_rpm"] = model.to_log_rpm(train[config.TARGET], train["distance"])
    train.attrs["weight_median"] = weight_median
    return features.out_of_fold_encoding(train, train["log_rpm"])


def build_scoring_matrix(
    frame: pd.DataFrame, frames: dict, encoder: features.TargetEncoder, weight_median: float
) -> pd.DataFrame:
    scored, _ = data.impute(frame, weight_median=weight_median)
    scored = features.build_base(scored, frames["coordinates"])
    return encoder.transform(scored)


def december_frame(frames: dict) -> pd.DataFrame:
    """Attach the recovered daily `market_index` to the fixed December inputs."""
    december = frames["december"].copy()
    daily = data.december_market_index(frames["validation"])
    merged = december.merge(daily, on=config.DATE_COL, how="left")
    if merged["market_index"].isna().any():
        missing = merged.loc[merged["market_index"].isna(), config.DATE_COL].dt.date.tolist()
        raise SystemExit(f"ERROR: no market_index recoverable for {missing}")
    return merged


def main() -> None:
    frames = data.load_all()
    report = frames["report"]
    print(report.to_markdown())
    print()

    train = build_training_matrix(frames)
    weight_median = train.attrs["weight_median"]

    encoder = features.TargetEncoder().fit(train, train["log_rpm"])
    fitted = model.HybridModel(
        features.FEATURES,
        trend_damping=config.TREND_DAMPING,
        rounds=config.N_BOOST_ROUNDS,
    ).fit(train, train["log_rpm"])

    names = ["market_index", "trend_days/100", "log_distance", "Reefer", "Flatbed", "intercept"]
    print("Linear backbone coefficients (log $/mi):")
    for name, value in zip(names, fitted.coefficients_):
        print(f"  {name:<16} {value:+.5f}")
    print(f"  smearing factor  {fitted.smearing_:.5f}")
    print()

    # -- validation set -----------------------------------------------------
    validation = build_scoring_matrix(frames["validation"], frames, encoder, weight_median)
    validation["predicted_rate"] = fitted.predict(validation)

    template = pd.read_csv(config.TEMPLATE_PATH)
    predictions = template[["load_id"]].merge(
        validation[["load_id", "predicted_rate"]], on="load_id", how="left"
    )
    if predictions["predicted_rate"].isna().any():
        raise SystemExit("ERROR: some template load_ids received no prediction")
    predictions["predicted_rate"] = predictions["predicted_rate"].round(2)
    predictions.to_csv(config.PREDICTIONS_PATH, index=False)
    print(f"Wrote {config.PREDICTIONS_PATH} ({len(predictions):,} rows)")
    print(
        f"  predicted rate  min ${predictions.predicted_rate.min():,.2f} | "
        f"median ${predictions.predicted_rate.median():,.2f} | "
        f"max ${predictions.predicted_rate.max():,.2f}"
    )

    # -- fixed December scenario -------------------------------------------
    december = december_frame(frames)
    scored = build_scoring_matrix(december, frames, encoder, weight_median)
    december_out = frames["december"].copy()
    december_out["predicted_rate"] = np.round(fitted.predict(scored), 2)
    december_out[config.DATE_COL] = december_out[config.DATE_COL].dt.strftime("%Y-%m-%d")
    december_out.to_csv(config.DECEMBER_PATH, index=False)
    print(f"Wrote {config.DECEMBER_PATH} (31 rows)")
    print(
        f"  December rate   min ${december_out.predicted_rate.min():,.2f} | "
        f"mean ${december_out.predicted_rate.mean():,.2f} | "
        f"max ${december_out.predicted_rate.max():,.2f} | "
        f"spread {(december_out.predicted_rate.max() / december_out.predicted_rate.min() - 1) * 100:.1f}%"
    )

    # -- feature importance -------------------------------------------------
    gains = pd.DataFrame(
        {
            "feature": fitted.residual_features,
            "gain": fitted.booster_.feature_importance("gain"),
        }
    ).sort_values("gain", ascending=False)
    gains["share"] = gains["gain"] / gains["gain"].sum() * 100

    lines = [
        "# Residual model feature importance",
        "",
        "Gain of the LightGBM stage only. The market level and the time trend are",
        "handled by the linear backbone and are deliberately not available here.",
        "",
        "| Feature | Gain share |",
        "| --- | --- |",
    ]
    lines += [f"| `{row.feature}` | {row.share:.1f}% |" for row in gains.itertuples()]
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "feature_importance.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {config.REPORTS_DIR / 'feature_importance.md'}")
    print(gains.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
