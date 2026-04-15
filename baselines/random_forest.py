from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from th229_bench.baselines import WeightedHarmonicRegression  # noqa: E402


DEFAULT_PERIODS_DAYS = np.array([7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 240, 365], dtype=float)


def _load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load benchmark examples and the 55-row metadata used to build periodogram features."""
    pkl_path = PROJECT_ROOT / "data" / "benchmark_dataset.pkl"
    if pkl_path.exists():
        with pkl_path.open("rb") as handle:
            payload = pickle.load(handle)
        catalog = pd.DataFrame(
            {
                "split": np.asarray(payload["split"]),
                "label": np.asarray(payload["y"], dtype=int),
            }
        )
        residuals = np.asarray(payload["X"], dtype=float)
    else:
        catalog = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "benchmark_catalog.csv")
        arrays = np.load(PROJECT_ROOT / "data" / "interim" / "benchmark_arrays.npz")
        residuals = np.asarray(arrays["observed_residual_hz"], dtype=float)

    metadata = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "primary_peak_b.csv")
    if len(catalog) != residuals.shape[0]:
        raise ValueError(f"catalog rows ({len(catalog)}) do not match residual rows ({residuals.shape[0]})")
    if residuals.shape[1] != len(metadata):
        raise ValueError(f"residual length ({residuals.shape[1]}) does not match metadata rows ({len(metadata)})")
    return catalog, residuals, metadata


def _feature_names(periods_days: np.ndarray) -> list[str]:
    period_names = [f"wls_power_{int(period)}d" for period in periods_days]
    return period_names + [
        "mean_hz",
        "variance_hz2",
        "min_hz",
        "max_hz",
        "range_hz",
        "autocorr_lag1",
        "peak_to_peak_hz",
    ]


def _extract_features(
    residuals: np.ndarray,
    metadata: pd.DataFrame,
    periods_days: np.ndarray = DEFAULT_PERIODS_DAYS,
) -> tuple[np.ndarray, list[str]]:
    """Extract periodogram and summary-statistic features from length-55 residual vectors."""
    frequencies_hz = 1.0 / (periods_days * 86400.0)
    periodogram = WeightedHarmonicRegression.from_frame(metadata, frequencies_hz)

    features = np.zeros((residuals.shape[0], len(periods_days) + 7), dtype=float)
    for idx, y in enumerate(residuals):
        powers, _ = periodogram.periodogram(y)
        features[idx, : len(periods_days)] = powers
        centered = y - np.mean(y)
        denom = np.sqrt(np.sum(centered[:-1] ** 2) * np.sum(centered[1:] ** 2))
        lag1 = 0.0 if denom <= 0 else float(np.sum(centered[:-1] * centered[1:]) / denom)
        min_y = float(np.min(y))
        max_y = float(np.max(y))
        features[idx, len(periods_days) :] = [
            float(np.mean(y)),
            float(np.var(y)),
            min_y,
            max_y,
            max_y - min_y,
            lag1,
            float(np.ptp(y)),
        ]
    return features, _feature_names(periods_days)


def run_random_forest(output_path: Path) -> dict[str, object]:
    catalog, residuals, metadata = _load_data()
    features, names = _extract_features(residuals, metadata)

    train_mask = catalog["split"].eq("train").to_numpy()
    validation_mask = catalog["split"].eq("validation").to_numpy()
    test_mask = catalog["split"].eq("test").to_numpy()
    labels = catalog["label"].to_numpy(dtype=int)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(features[train_mask], labels[train_mask])

    test_scores = model.predict_proba(features[test_mask])[:, 1]
    validation_scores = model.predict_proba(features[validation_mask])[:, 1]
    threshold = float(np.quantile(validation_scores[labels[validation_mask] == 0], 0.95))

    predictions = pd.DataFrame(
        {
            "instance_id": catalog.loc[test_mask, "instance_id"].to_numpy()
            if "instance_id" in catalog.columns
            else np.arange(int(test_mask.sum())),
            "split": "test",
            "label": labels[test_mask],
            "score": test_scores,
        }
    )
    pred_path = output_path.with_name("rf_baseline_predictions.csv")
    predictions.to_csv(pred_path, index=False)

    importances = pd.DataFrame(
        {
            "feature": names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_path = output_path.with_name("rf_feature_importances.csv")
    importances.to_csv(importance_path, index=False)

    result = {
        "method": "random_forest_periodogram_features",
        "training_protocol": "train split only; validation split used only to calibrate a 5% null threshold",
        "n_estimators": 100,
        "random_state": 42,
        "n_train": int(train_mask.sum()),
        "n_validation": int(validation_mask.sum()),
        "n_test": int(test_mask.sum()),
        "test_auroc": float(roc_auc_score(labels[test_mask], test_scores)),
        "test_average_precision": float(average_precision_score(labels[test_mask], test_scores)),
        "validation_fpr5_threshold": threshold,
        "test_null_false_positive_rate": float(np.mean(test_scores[labels[test_mask] == 0] >= threshold)),
        "test_signal_true_positive_rate": float(np.mean(test_scores[labels[test_mask] == 1] >= threshold)),
        "features": names,
        "feature_importances_path": str(importance_path.relative_to(PROJECT_ROOT)),
        "predictions_path": str(pred_path.relative_to(PROJECT_ROOT)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the random-forest baseline for Th229-ScanBench.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "rf_baseline_results.json",
        help="Path for the JSON metrics output.",
    )
    args = parser.parse_args()
    result = run_random_forest(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
