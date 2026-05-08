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
DEFAULT_AMPLITUDES_HZ = np.array([100, 200, 350, 500, 750, 1000, 1500, 2200, 3200, 4500, 6500], dtype=float)
GLOBAL_SEED = 229026


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


def _interpolate_a95(amplitudes_hz: np.ndarray, rates: list[float]) -> float:
    for idx, rate in enumerate(rates):
        if rate >= 0.95:
            if idx == 0:
                return float(amplitudes_hz[idx])
            x0, x1 = amplitudes_hz[idx - 1], amplitudes_hz[idx]
            y0, y1 = rates[idx - 1], rate
            return float(x0 + (0.95 - y0) * (x1 - x0) / max(y1 - y0, 1e-12))
    return float("nan")


def _a95_with_uncertainty(
    scores: np.ndarray,
    catalog: pd.DataFrame,
    periods_days: np.ndarray = DEFAULT_PERIODS_DAYS,
    amplitudes_hz: np.ndarray = DEFAULT_AMPLITUDES_HZ,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    """Bootstrap RF A95 with the same validation-threshold and test-phase protocol as core baselines."""
    required = {"split", "label", "period_days", "frequency_hz", "amplitude_hz"}
    missing = sorted(required.difference(catalog.columns))
    if missing:
        raise ValueError(f"catalog is missing columns required for A95: {missing}")

    baseline = "random_forest_periodogram_features"
    rng = np.random.default_rng(GLOBAL_SEED + 900_000 + sum(ord(ch) for ch in baseline))
    validation_null = catalog["split"].eq("validation").to_numpy() & catalog["label"].eq(0).to_numpy()
    validation_scores = scores[validation_null]
    point_threshold = float(np.quantile(validation_scores, 0.95))
    test_signal = catalog["split"].eq("test").to_numpy() & catalog["label"].eq(1).to_numpy()
    rows = []
    for period_days in periods_days:
        frequency_hz = float(1.0 / (period_days * 86400.0))
        point_rates = []
        cell_scores: list[np.ndarray] = []
        for amplitude_hz in amplitudes_hz:
            mask = (
                test_signal
                & np.isclose(catalog["period_days"].to_numpy(dtype=float), period_days)
                & np.isclose(catalog["amplitude_hz"].to_numpy(dtype=float), amplitude_hz)
            )
            values = scores[mask]
            cell_scores.append(values)
            point_rates.append(float(np.mean(values >= point_threshold)))
        point = _interpolate_a95(amplitudes_hz, point_rates)

        draws = []
        for _ in range(n_bootstrap):
            threshold = float(np.quantile(rng.choice(validation_scores, size=len(validation_scores), replace=True), 0.95))
            rates = []
            for values in cell_scores:
                resampled = rng.choice(values, size=len(values), replace=True)
                rates.append(float(np.mean(resampled >= threshold)))
            draws.append(_interpolate_a95(amplitudes_hz, rates))
        draw_array = np.asarray(draws, dtype=float)
        finite = draw_array[np.isfinite(draw_array)]
        rows.append(
            {
                "baseline": baseline,
                "period_days": float(period_days),
                "frequency_hz": frequency_hz,
                "a95_hz": point,
                "a95_bootstrap_median_hz": float(np.median(finite)) if len(finite) else float("nan"),
                "a95_lower_hz": float(np.percentile(finite, 16)) if len(finite) else float("nan"),
                "a95_upper_hz": float(np.percentile(finite, 84)) if len(finite) else float("nan"),
                "a95_std_hz": float(np.std(finite)) if len(finite) else float("nan"),
                "bootstrap_replicates": n_bootstrap,
                "finite_bootstrap_replicates": int(len(finite)),
                "n_test_phases_per_cell": 5,
                "uncertainty_source": "validation-null threshold resampling plus test-phase resampling within each amplitude cell",
            }
        )
    return pd.DataFrame(rows)


def _representative_frequency_rows(
    scores: np.ndarray,
    catalog: pd.DataFrame,
    threshold: float,
    a95_uncertainty: pd.DataFrame,
    periods_days: tuple[float, float] = (30.0, 180.0),
) -> pd.DataFrame:
    rows = []
    test_null = catalog["split"].eq("test").to_numpy() & catalog["label"].eq(0).to_numpy()
    test_signal = catalog["split"].eq("test").to_numpy() & catalog["label"].eq(1).to_numpy()
    for period in periods_days:
        period_signal = test_signal & np.isclose(catalog["period_days"].to_numpy(dtype=float), period)
        y = np.concatenate([np.zeros(int(test_null.sum()), dtype=int), np.ones(int(period_signal.sum()), dtype=int)])
        s = np.concatenate([scores[test_null], scores[period_signal]])
        a95_row = a95_uncertainty.loc[np.isclose(a95_uncertainty["period_days"].to_numpy(dtype=float), period)]
        rows.append(
            {
                "baseline": "random_forest_periodogram_features",
                "period_days": period,
                "frequency_hz": float(1.0 / (period * 86400.0)),
                "roc_auc_at_period": float(roc_auc_score(y, s)),
                "average_precision_at_period": float(average_precision_score(y, s)),
                "validation_fpr5_threshold": threshold,
                "tpr_at_validation_fpr_5pct": float(np.mean(scores[period_signal] >= threshold)),
                "a95_hz": float(a95_row["a95_hz"].iloc[0]) if not a95_row.empty else float("nan"),
                "a95_lower_hz": float(a95_row["a95_lower_hz"].iloc[0]) if not a95_row.empty else float("nan"),
                "a95_upper_hz": float(a95_row["a95_upper_hz"].iloc[0]) if not a95_row.empty else float("nan"),
                "n_test_signal": int(period_signal.sum()),
                "n_test_null": int(test_null.sum()),
            }
        )
    return pd.DataFrame(rows)


def run_random_forest(output_path: Path) -> dict[str, object]:
    catalog, residuals, metadata = _load_data()
    features, names = _extract_features(residuals, metadata)

    train_mask = catalog["split"].eq("train").to_numpy()
    validation_mask = catalog["split"].eq("validation").to_numpy()
    test_mask = catalog["split"].eq("test").to_numpy()
    labels = catalog["label"].to_numpy(dtype=int)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(features[train_mask], labels[train_mask])
    model_path = PROJECT_ROOT / "models" / "random_forest_periodogram_features_v1.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump({"model": model, "feature_names": names, "periods_days": DEFAULT_PERIODS_DAYS}, handle)

    all_scores = model.predict_proba(features)[:, 1]
    test_scores = all_scores[test_mask]
    validation_scores = all_scores[validation_mask]
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

    all_predictions = pd.DataFrame(
        {
            "instance_id": catalog["instance_id"].to_numpy()
            if "instance_id" in catalog.columns
            else np.arange(len(catalog)),
            "split": catalog["split"].to_numpy(),
            "label": labels,
            "score": all_scores,
        }
    )
    for column in ("family", "period_days", "frequency_hz", "amplitude_hz", "phase_index"):
        if column in catalog.columns:
            all_predictions[column] = catalog[column].to_numpy()
    all_pred_path = output_path.with_name("rf_baseline_all_predictions.csv")
    all_predictions.to_csv(all_pred_path, index=False)

    a95_uncertainty = _a95_with_uncertainty(all_scores, catalog)
    a95_path = PROJECT_ROOT / "results" / "tables" / "rf_a95_vs_frequency_with_uncertainty.csv"
    a95_path.parent.mkdir(parents=True, exist_ok=True)
    a95_uncertainty.to_csv(a95_path, index=False)

    representative_rf = _representative_frequency_rows(all_scores, catalog, threshold, a95_uncertainty)
    core_representative_path = PROJECT_ROOT / "results" / "tables" / "final_representative_frequency_baseline_table.csv"
    representative_with_rf_path = PROJECT_ROOT / "results" / "tables" / "final_representative_frequency_baseline_table_with_rf.csv"
    if core_representative_path.exists():
        core_representative = pd.read_csv(core_representative_path)
    elif representative_with_rf_path.exists():
        core_representative = pd.read_csv(representative_with_rf_path)
        core_representative = core_representative.loc[
            ~core_representative["baseline"].eq("random_forest_periodogram_features")
        ].copy()
    else:
        core_representative = pd.DataFrame()
    if not core_representative.empty:
        representative_with_rf = pd.concat([core_representative, representative_rf], ignore_index=True)
    else:
        representative_with_rf = representative_rf
    representative_with_rf.to_csv(representative_with_rf_path, index=False)

    comparison_with_rf_path = PROJECT_ROOT / "results" / "tables" / "baseline_comparison_with_rf.csv"
    core_comparison_path = PROJECT_ROOT / "results" / "tables" / "baseline_comparison.csv"
    rf_comparison = pd.DataFrame(
        [
            {
                "baseline": "random_forest_periodogram_features",
                "roc_auc": float(roc_auc_score(labels[test_mask], test_scores)),
                "average_precision": float(average_precision_score(labels[test_mask], test_scores)),
                "validation_fpr5_threshold": threshold,
                "tpr_at_validation_fpr_5pct": float(np.mean(test_scores[labels[test_mask] == 1] >= threshold)),
                "test_null_false_positive_rate": float(np.mean(test_scores[labels[test_mask] == 0] >= threshold)),
                "runtime_sec": np.nan,
            }
        ]
    )
    if core_comparison_path.exists():
        core_comparison = pd.read_csv(core_comparison_path)
    elif comparison_with_rf_path.exists():
        core_comparison = pd.read_csv(comparison_with_rf_path)
        core_comparison = core_comparison.loc[
            ~core_comparison["baseline"].eq("random_forest_periodogram_features")
        ].copy()
    else:
        core_comparison = pd.DataFrame()
    if not core_comparison.empty:
        comparison_with_rf = pd.concat([core_comparison, rf_comparison], ignore_index=True)
    else:
        comparison_with_rf = rf_comparison
    comparison_with_rf.to_csv(comparison_with_rf_path, index=False)

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
        "a95_path": str(a95_path.relative_to(PROJECT_ROOT)),
        "representative_table_with_rf_path": str(representative_with_rf_path.relative_to(PROJECT_ROOT)),
        "baseline_comparison_with_rf_path": str(comparison_with_rf_path.relative_to(PROJECT_ROOT)),
        "features": names,
        "feature_importances_path": str(importance_path.relative_to(PROJECT_ROOT)),
        "checkpoint_path": str(model_path.relative_to(PROJECT_ROOT)),
        "predictions_path": str(pred_path.relative_to(PROJECT_ROOT)),
        "all_predictions_path": str(all_pred_path.relative_to(PROJECT_ROOT)),
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
