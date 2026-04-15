from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .baselines import GeneralizedLombScargle, WeightedHarmonicRegression, metric_definitions
from .hierarchical_model import HierarchicalSinusoidModel
from .null_models import FittedNullModel, fit_null_models, null_model_comparison_frame
from .paths import FIGURES_DIR, INTERIM_DIR, TABLES_DIR
from .synthetic import SECONDS_PER_DAY, injection_family_definitions, signal_from_row, slow_linear_drift
from .utils import write_json, write_text


@dataclass(frozen=True)
class BenchmarkConfig:
    periods_days: tuple[float, ...] = (7.0, 10.0, 14.0, 21.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 240.0, 365.0)
    amplitudes_hz: tuple[float, ...] = (100.0, 200.0, 350.0, 500.0, 750.0, 1000.0, 1500.0, 2200.0, 3200.0, 4500.0, 6500.0)
    n_phases_per_cell: int = 24
    train_phase_count: int = 14
    val_phase_count: int = 5
    test_phase_count: int = 5
    global_seed: int = 229026
    detection_grid_size: int = 96
    default_null_model_name: str | None = None

    @property
    def frequencies_hz(self) -> np.ndarray:
        return 1.0 / (np.asarray(self.periods_days, dtype=np.float64) * SECONDS_PER_DAY)

    @property
    def detection_grid_hz(self) -> np.ndarray:
        frequencies = self.frequencies_hz
        return np.geomspace(frequencies.min(), frequencies.max(), self.detection_grid_size)


OFFGRID_GENERALIZATION_PERIODS_DAYS = (8.5, 17.0, 36.0, 75.0, 150.0, 300.0)
DRIFT_SLOPES_HZ_PER_DAY = (0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 45.0, 65.0)


def _phase_split(phase_index: int, config: BenchmarkConfig) -> str:
    if phase_index < config.train_phase_count:
        return "train"
    if phase_index < config.train_phase_count + config.val_phase_count:
        return "validation"
    return "test"


def write_injection_docs(config: BenchmarkConfig) -> None:
    definitions = injection_family_definitions()
    lines = ["# Injection Family Definitions", ""]
    for name, info in definitions.items():
        lines.append(f"## {name}")
        lines.append(f"- Equation: `{info['equation']}`")
        lines.append(f"- Role: {info['role']}")
        lines.append(f"- Physics note: {info['physics_note']}")
        lines.append("")
    write_text("\n".join(lines), TABLES_DIR / "injection_family_definitions.md")

    rows = []
    for period_days, frequency_hz in zip(config.periods_days, config.frequencies_hz):
        for amplitude_hz in config.amplitudes_hz:
            rows.append(
                {
                    "family": "pure_sinusoid",
                    "period_days": period_days,
                    "frequency_hz": frequency_hz,
                    "amplitude_hz": amplitude_hz,
                    "is_primary_grid": True,
                }
            )
    for period_days, frequency_hz in zip((30.0, 90.0, 180.0), 1.0 / (np.asarray((30.0, 90.0, 180.0)) * SECONDS_PER_DAY)):
        rows.append(
            {
                "family": "finite_coherence_sinusoid",
                "period_days": period_days,
                "frequency_hz": frequency_hz,
                "amplitude_hz": 2200.0,
                "coherence_days": 90.0,
                "is_primary_grid": False,
            }
        )
    for slope in (2.0, 5.0, 10.0):
        rows.append({"family": "slow_linear_drift", "slope_hz_per_day": slope, "is_primary_grid": False})
    for width in (7.0, 21.0, 45.0):
        rows.append({"family": "gaussian_transient", "amplitude_hz": 3200.0, "center_day": 240.0, "width_days": width, "is_primary_grid": False})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "injection_grid.csv", index=False)


def _catalog_rows(config: BenchmarkConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for freq_index, (period_days, frequency_hz) in enumerate(zip(config.periods_days, config.frequencies_hz)):
        for amp_index, amplitude_hz in enumerate(config.amplitudes_hz):
            for phase_index in range(config.n_phases_per_cell):
                phase_rad = 2.0 * np.pi * (phase_index + 0.5) / config.n_phases_per_cell
                split = _phase_split(phase_index, config)
                for label in (0, 1):
                    rows.append(
                        {
                            "instance_id": f"pure_sinusoid_f{freq_index:02d}_a{amp_index:02d}_p{phase_index:02d}_y{label}",
                            "family": "pure_sinusoid",
                            "split": split,
                            "label": label,
                            "frequency_hz": frequency_hz,
                            "period_days": period_days,
                            "amplitude_hz": amplitude_hz if label == 1 else 0.0,
                            "phase_rad": phase_rad,
                            "coherence_days": np.nan,
                            "slope_hz_per_day": np.nan,
                            "center_day": np.nan,
                            "width_days": np.nan,
                            "frequency_index": freq_index,
                            "amplitude_index": amp_index,
                            "phase_index": phase_index,
                        }
                    )
    return rows


def build_benchmark_dataset(
    primary_df: pd.DataFrame,
    null_model: FittedNullModel,
    config: BenchmarkConfig | None = None,
    prefix: str = "primary_peak_b",
    write_outputs: bool = True,
) -> dict[str, Any]:
    config = config or BenchmarkConfig(default_null_model_name=null_model.name)
    df = primary_df.reset_index(drop=True).copy()
    times_sec = df["seconds_since_first_observation"].to_numpy(dtype=np.float64)
    catalog = pd.DataFrame(_catalog_rows(config))

    observed_residuals = np.zeros((len(catalog), len(df)), dtype=np.float64)
    signals = np.zeros_like(observed_residuals)
    noises = np.zeros_like(observed_residuals)
    for idx, row in catalog.iterrows():
        seed = config.global_seed + 1_000_000 * int(row["label"]) + 10_000 * int(row["frequency_index"]) + 100 * int(row["amplitude_index"]) + int(row["phase_index"])
        rng = np.random.default_rng(seed)
        noise = null_model.sample(df, rng)
        signal = signal_from_row(times_sec, row, rng) if int(row["label"]) == 1 else np.zeros(len(df), dtype=np.float64)
        noises[idx] = noise
        signals[idx] = signal
        observed_residuals[idx] = noise + signal
        catalog.loc[idx, "noise_seed"] = seed
    catalog["noise_seed"] = catalog["noise_seed"].astype(int)

    arrays = {
        "times_sec": times_sec,
        "observed_residual_hz": observed_residuals,
        "signal_hz": signals,
        "noise_hz": noises,
        "formal_sigma_hz": np.tile(df["freq_unc_hz"].to_numpy(dtype=np.float64), (len(catalog), 1)),
        "effective_sigma_hz": np.tile(null_model.effective_sigma(df), (len(catalog), 1)),
    }

    if write_outputs:
        catalog_path = INTERIM_DIR / f"{prefix}_benchmark_catalog.csv"
        arrays_path = INTERIM_DIR / f"{prefix}_benchmark_arrays.npz"
        catalog.to_csv(catalog_path, index=False)
        np.savez_compressed(arrays_path, **arrays)
        # Authoritative current files for backward compatibility.
        if prefix == "primary_peak_b":
            catalog.to_csv(INTERIM_DIR / "benchmark_catalog.csv", index=False)
            np.savez_compressed(INTERIM_DIR / "benchmark_arrays.npz", **arrays)
        write_injection_docs(config)
    return {"catalog": catalog, "arrays": arrays, "config": config}


def _baseline_scorers(df: pd.DataFrame, config: BenchmarkConfig, null_model: FittedNullModel) -> dict[str, Any]:
    return {
        "weighted_harmonic_regression": WeightedHarmonicRegression.from_frame(df, config.detection_grid_hz),
        "generalized_lomb_scargle": GeneralizedLombScargle.from_frame(df, config.detection_grid_hz),
        "hierarchical_sinusoid_jitter": HierarchicalSinusoidModel.from_frame(df, config.detection_grid_hz, null_model),
    }


def _score_all(scorer: Any, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = np.zeros(residuals.shape[0], dtype=np.float64)
    pred_freq = np.zeros_like(scores)
    pred_amp = np.zeros_like(scores)
    periodograms = []
    for idx, sample in enumerate(residuals):
        result = scorer.score(sample)
        scores[idx] = result.score
        pred_freq[idx] = result.pred_frequency_hz
        pred_amp[idx] = result.pred_amplitude_hz
        periodograms.append(result.periodogram)
    return scores, pred_freq, pred_amp, np.vstack(periodograms)


def _threshold(scores: np.ndarray, catalog: pd.DataFrame, split: str = "validation") -> float:
    mask = catalog["split"].eq(split).to_numpy() & catalog["label"].eq(0).to_numpy()
    return float(np.quantile(scores[mask], 0.95))


def _comparison_row(name: str, scores: np.ndarray, threshold: float, catalog: pd.DataFrame, runtime_sec: float) -> dict[str, Any]:
    test_mask = catalog["split"].eq("test").to_numpy()
    y = catalog.loc[test_mask, "label"].to_numpy(dtype=int)
    s = scores[test_mask]
    signal_mask = y == 1
    return {
        "baseline": name,
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "validation_fpr5_threshold": threshold,
        "tpr_at_validation_fpr_5pct": float(np.mean(s[signal_mask] >= threshold)),
        "test_null_false_positive_rate": float(np.mean(s[~signal_mask] >= threshold)),
        "runtime_sec": runtime_sec,
    }


def _interpolate_a95(amplitudes_hz: tuple[float, ...], rates: list[float]) -> float:
    for idx, rate in enumerate(rates):
        if rate >= 0.95:
            if idx == 0:
                return float(amplitudes_hz[idx])
            x0, x1 = amplitudes_hz[idx - 1], amplitudes_hz[idx]
            y0, y1 = rates[idx - 1], rate
            return float(x0 + (0.95 - y0) * (x1 - x0) / max(y1 - y0, 1e-12))
    return float("nan")


def _a95(scores: np.ndarray, threshold: float, catalog: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    rows = []
    test_signal = catalog["split"].eq("test").to_numpy() & catalog["label"].eq(1).to_numpy()
    for period_days, frequency_hz in zip(config.periods_days, config.frequencies_hz):
        rates = []
        for amplitude_hz in config.amplitudes_hz:
            mask = test_signal & np.isclose(catalog["frequency_hz"].to_numpy(), frequency_hz) & np.isclose(catalog["amplitude_hz"].to_numpy(), amplitude_hz)
            rates.append(float(np.mean(scores[mask] >= threshold)))
        a95 = _interpolate_a95(config.amplitudes_hz, rates)
        rows.append({"period_days": period_days, "frequency_hz": frequency_hz, "a95_hz": a95})
    return pd.DataFrame(rows)


def _a95_with_uncertainty(
    scores: np.ndarray,
    catalog: pd.DataFrame,
    config: BenchmarkConfig,
    baseline: str,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    """Bootstrap A95 over validation-null threshold and test phase realizations."""
    rng = np.random.default_rng(config.global_seed + 900_000 + sum(ord(ch) for ch in baseline))
    validation_null = catalog["split"].eq("validation").to_numpy() & catalog["label"].eq(0).to_numpy()
    validation_scores = scores[validation_null]
    test_signal = catalog["split"].eq("test").to_numpy() & catalog["label"].eq(1).to_numpy()
    rows = []
    for period_days, frequency_hz in zip(config.periods_days, config.frequencies_hz):
        point_rates = []
        cell_scores: list[np.ndarray] = []
        for amplitude_hz in config.amplitudes_hz:
            mask = (
                test_signal
                & np.isclose(catalog["frequency_hz"].to_numpy(), frequency_hz)
                & np.isclose(catalog["amplitude_hz"].to_numpy(), amplitude_hz)
            )
            values = scores[mask]
            cell_scores.append(values)
            point_rates.append(float(np.mean(values >= np.quantile(validation_scores, 0.95))))
        point = _interpolate_a95(config.amplitudes_hz, point_rates)
        draws = []
        for _ in range(n_bootstrap):
            threshold = float(np.quantile(rng.choice(validation_scores, size=len(validation_scores), replace=True), 0.95))
            rates = []
            for values in cell_scores:
                resampled = rng.choice(values, size=len(values), replace=True)
                rates.append(float(np.mean(resampled >= threshold)))
            draws.append(_interpolate_a95(config.amplitudes_hz, rates))
        draw_array = np.asarray(draws, dtype=np.float64)
        finite = draw_array[np.isfinite(draw_array)]
        median = float(np.median(finite)) if len(finite) else float("nan")
        lower = float(np.percentile(finite, 16)) if len(finite) else float("nan")
        upper = float(np.percentile(finite, 84)) if len(finite) else float("nan")
        std = float(np.std(finite)) if len(finite) else float("nan")
        rows.append(
            {
                "baseline": baseline,
                "period_days": period_days,
                "frequency_hz": frequency_hz,
                "a95_hz": point,
                "a95_bootstrap_median_hz": median,
                "a95_lower_hz": lower,
                "a95_upper_hz": upper,
                "a95_std_hz": std,
                "bootstrap_replicates": n_bootstrap,
                "finite_bootstrap_replicates": int(len(finite)),
                "n_test_phases_per_cell": int(config.test_phase_count),
                "uncertainty_source": "validation-null threshold resampling plus test-phase resampling within each amplitude cell",
            }
        )
    return pd.DataFrame(rows)


def _heatmap(scores: np.ndarray, threshold: float, catalog: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    rows = []
    test_signal = catalog["split"].eq("test").to_numpy() & catalog["label"].eq(1).to_numpy()
    for period_days, frequency_hz in zip(config.periods_days, config.frequencies_hz):
        for amplitude_hz in config.amplitudes_hz:
            mask = test_signal & np.isclose(catalog["frequency_hz"].to_numpy(), frequency_hz) & np.isclose(catalog["amplitude_hz"].to_numpy(), amplitude_hz)
            rows.append(
                {
                    "period_days": period_days,
                    "frequency_hz": frequency_hz,
                    "amplitude_hz": amplitude_hz,
                    "detection_rate": float(np.mean(scores[mask] >= threshold)),
                }
            )
    return pd.DataFrame(rows)


def _auc_heatmap_table(scores: np.ndarray, threshold: float, catalog: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    rows = []
    test_null = catalog["split"].eq("test").to_numpy() & catalog["label"].eq(0).to_numpy()
    null_scores = scores[test_null]
    test_signal = catalog["split"].eq("test").to_numpy() & catalog["label"].eq(1).to_numpy()
    for period_days, frequency_hz in zip(config.periods_days, config.frequencies_hz):
        for amplitude_hz in config.amplitudes_hz:
            signal_mask = (
                test_signal
                & np.isclose(catalog["frequency_hz"].to_numpy(), frequency_hz)
                & np.isclose(catalog["amplitude_hz"].to_numpy(), amplitude_hz)
            )
            signal_scores = scores[signal_mask]
            y = np.concatenate([np.zeros(len(null_scores), dtype=int), np.ones(len(signal_scores), dtype=int)])
            s = np.concatenate([null_scores, signal_scores])
            rows.append(
                {
                    "period_days": period_days,
                    "frequency_hz": frequency_hz,
                    "amplitude_hz": amplitude_hz,
                    "roc_auc": float(roc_auc_score(y, s)),
                    "detection_rate_at_validation_fpr5": float(np.mean(signal_scores >= threshold)),
                    "n_test_signal": int(len(signal_scores)),
                    "n_test_null": int(len(null_scores)),
                }
            )
    return pd.DataFrame(rows)


def _representative_frequency_table(
    primary_result: dict[str, Any],
    a95_uncertainty_df: pd.DataFrame,
    config: BenchmarkConfig,
    periods_days: tuple[float, float] = (30.0, 180.0),
) -> pd.DataFrame:
    catalog = primary_result["catalog"]
    rows = []
    test_null = catalog["split"].eq("test").to_numpy() & catalog["label"].eq(0).to_numpy()
    test_signal = catalog["split"].eq("test").to_numpy() & catalog["label"].eq(1).to_numpy()
    for baseline, values in primary_result["scores"].items():
        scores = values["scores"]
        threshold = values["threshold"]
        for period in periods_days:
            period_signal = test_signal & np.isclose(catalog["period_days"].to_numpy(dtype=np.float64), period)
            y = np.concatenate([np.zeros(int(test_null.sum()), dtype=int), np.ones(int(period_signal.sum()), dtype=int)])
            s = np.concatenate([scores[test_null], scores[period_signal]])
            a95_row = a95_uncertainty_df.loc[
                a95_uncertainty_df["baseline"].eq(baseline)
                & np.isclose(a95_uncertainty_df["period_days"].to_numpy(dtype=np.float64), period)
            ]
            rows.append(
                {
                    "baseline": baseline,
                    "period_days": period,
                    "frequency_hz": float(1.0 / (period * SECONDS_PER_DAY)),
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


def _observed_calibration(df: pd.DataFrame, config: BenchmarkConfig, null_model: FittedNullModel) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rows = []
    curves = []
    for scope, subset in {
        "all_peak_b": df,
        "peak_b_c10_c13": df.loc[df["target"].isin(["C10", "C13"])].copy(),
        "peak_b_x2": df.loc[df["target"].eq("X2")].copy(),
    }.items():
        if len(subset) < 5:
            continue
        scorer = HierarchicalSinusoidModel.from_frame(subset.reset_index(drop=True), config.detection_grid_hz, null_model)
        observed = subset["residual_hz"].to_numpy(dtype=np.float64)
        result = scorer.score(observed)
        null_scores = []
        for draw in range(512):
            rng = np.random.default_rng(config.global_seed + 700_000 + draw + len(subset))
            null_scores.append(scorer.score(null_model.sample(subset.reset_index(drop=True), rng)).score)
        null_scores = np.asarray(null_scores)
        rows.append(
            {
                "scope": scope,
                "null_model": null_model.name,
                "observed_score": result.score,
                "best_period_days": 1.0 / (result.pred_frequency_hz * SECONDS_PER_DAY),
                "null_score_p95": float(np.quantile(null_scores, 0.95)),
                "observed_empirical_p_value": float((1 + np.sum(null_scores >= result.score)) / (len(null_scores) + 1)),
            }
        )
        curves.append(
            pd.DataFrame(
                {
                    "scope": scope,
                    "period_days": 1.0 / (scorer.frequency_grid_hz * SECONDS_PER_DAY),
                    "frequency_hz": scorer.frequency_grid_hz,
                    "score": result.periodogram,
                }
            )
        )
    calibration = pd.DataFrame(rows)
    periodogram = pd.concat(curves, ignore_index=True)
    x2_row = calibration.loc[calibration["scope"].eq("peak_b_x2")]
    c_rows = calibration.loc[calibration["scope"].eq("peak_b_c10_c13")]
    note = "No observed-series detection claim is made."
    if not x2_row.empty and not c_rows.empty and x2_row["observed_score"].iloc[0] > c_rows["observed_score"].iloc[0]:
        note = "The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal."
    return calibration, periodogram, note


def _plot_outputs(
    comparison: pd.DataFrame,
    heatmap_df: pd.DataFrame,
    a95_uncertainty_df: pd.DataFrame,
    calibration_periodogram: pd.DataFrame,
    ablation: pd.DataFrame,
    catalog: pd.DataFrame,
    scores_main: np.ndarray,
) -> dict[str, str]:
    figures: dict[str, str] = {}
    test_mask = catalog["split"].eq("test").to_numpy()
    y = catalog.loc[test_mask, "label"].to_numpy(dtype=int)

    roc_path = FIGURES_DIR / "final_roc_curve_main_baseline.png"
    fpr, tpr, _ = roc_curve(y, scores_main[test_mask])
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.plot(fpr, tpr, lw=2.0, color="#0b4f6c", label="Weighted harmonic regression")
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Held-out ROC for primary sinusoid task")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)
    figures["final_roc_curve_main_baseline"] = str(roc_path)

    heatmap_path = FIGURES_DIR / "final_sensitivity_heatmap_main_baseline.png"
    pivot = heatmap_df.pivot(index="amplitude_hz", columns="period_days", values="detection_rate")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    im = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.0f}" for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{x:.0f}" for x in pivot.index])
    ax.set_xlabel("Injected period (days)")
    ax.set_ylabel("Injected amplitude (Hz)")
    ax.set_title("Detection rate at validation-calibrated 5% FPR")
    fig.colorbar(im, ax=ax, label="Detection rate")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)
    figures["final_sensitivity_heatmap_main_baseline"] = str(heatmap_path)

    a95_path = FIGURES_DIR / "final_a95_vs_frequency_with_uncertainty.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for baseline, group in a95_uncertainty_df.groupby("baseline"):
        group = group.sort_values("period_days")
        x = group["period_days"].to_numpy(dtype=np.float64)
        y = group["a95_hz"].to_numpy(dtype=np.float64)
        low = group["a95_lower_hz"].to_numpy(dtype=np.float64)
        high = group["a95_upper_hz"].to_numpy(dtype=np.float64)
        ax.plot(x, y, marker="o", lw=2, label=baseline)
        ax.fill_between(x, low, high, alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlabel("Injected period (days)")
    ax.set_ylabel("A95 amplitude (Hz)")
    ax.set_title("A95 sensitivity with bootstrap intervals")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(a95_path, dpi=200)
    plt.close(fig)
    figures["final_a95_vs_frequency_with_uncertainty"] = str(a95_path)

    calib_path = FIGURES_DIR / "final_observed_series_periodogram_calibration.png"
    fig, ax = plt.subplots(figsize=(8.8, 5))
    for scope, group in calibration_periodogram.groupby("scope"):
        ax.plot(group["period_days"], group["score"], lw=2, label=scope)
    ax.set_xscale("log")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Hierarchical score")
    ax.set_title("Observed JILA residual periodogram by subset")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(calib_path, dpi=200)
    plt.close(fig)
    figures["final_observed_series_periodogram_calibration"] = str(calib_path)

    ablation_path = FIGURES_DIR / "final_ablation_remove_x2.png"
    fig, ax = plt.subplots(figsize=(8, 4.8))
    subset = ablation.loc[
        ablation["baseline"].eq("weighted_harmonic_regression")
        & ablation["dataset_scope"].isin(["primary_peak_b_all", "official_clean_c10_c13_peak_b"])
    ]
    ax.bar(subset["dataset_scope"], subset["roc_auc"], color=["#0b4f6c", "#5f9ea0", "#b08968"][: len(subset)])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Held-out AUROC")
    ax.set_title("Ablation: effect of dataset scope")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(ablation_path, dpi=200)
    plt.close(fig)
    figures["final_ablation_remove_x2"] = str(ablation_path)

    return figures


def _plot_auc_heatmap_with_a95_contour(auc_df: pd.DataFrame, baseline: str) -> str:
    figure_path = FIGURES_DIR / "final_auc_heatmap_with_a95_contour.png"
    auc_pivot = auc_df.pivot(index="amplitude_hz", columns="period_days", values="roc_auc").sort_index()
    det_pivot = auc_df.pivot(index="amplitude_hz", columns="period_days", values="detection_rate_at_validation_fpr5").sort_index()
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    image = ax.imshow(auc_pivot.to_numpy(), origin="lower", aspect="auto", cmap="magma", vmin=0.5, vmax=1.0)
    ax.set_xticks(np.arange(len(auc_pivot.columns)))
    ax.set_xticklabels([f"{value:.0f}" for value in auc_pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(auc_pivot.index)))
    ax.set_yticklabels([f"{value:.0f}" for value in auc_pivot.index])
    ax.set_xlabel("Injected period (days)")
    ax.set_ylabel("Injected amplitude (Hz)")
    ax.set_title(f"AUC heatmap with 95% detection contour: {baseline}")
    det_values = det_pivot.to_numpy(dtype=np.float64)
    if np.nanmin(det_values) <= 0.95 <= np.nanmax(det_values):
        contour = ax.contour(
            np.arange(len(det_pivot.columns)),
            np.arange(len(det_pivot.index)),
            det_values,
            levels=[0.95],
            colors="white",
            linewidths=2.0,
        )
        ax.clabel(contour, fmt={0.95: "95% detection"}, inline=True, fontsize=8)
    fig.colorbar(image, ax=ax, label="ROC AUC")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return str(figure_path)


def evaluate_dataset(
    df: pd.DataFrame,
    null_model: FittedNullModel,
    config: BenchmarkConfig,
    prefix: str,
    write_outputs: bool = False,
) -> dict[str, Any]:
    dataset = build_benchmark_dataset(df, null_model, config=config, prefix=prefix, write_outputs=write_outputs)
    catalog = dataset["catalog"]
    residuals = dataset["arrays"]["observed_residual_hz"]
    scorers = _baseline_scorers(df.reset_index(drop=True), config, null_model)
    score_tables = {}
    comparison_rows = []
    runtime_rows = []
    for name, scorer in scorers.items():
        start = time.perf_counter()
        scores, pred_freq, pred_amp, periodograms = _score_all(scorer, residuals)
        runtime = time.perf_counter() - start
        threshold = _threshold(scores, catalog)
        comparison_rows.append(_comparison_row(name, scores, threshold, catalog, runtime))
        runtime_rows.append({"baseline": name, "runtime_sec": runtime, "dataset_scope": prefix})
        score_tables[name] = {
            "scores": scores,
            "pred_frequency_hz": pred_freq,
            "pred_amplitude_hz": pred_amp,
            "periodograms": periodograms,
            "threshold": threshold,
        }
    return {
        "catalog": catalog,
        "arrays": dataset["arrays"],
        "comparison": pd.DataFrame(comparison_rows),
        "runtime": pd.DataFrame(runtime_rows),
        "scores": score_tables,
    }


def _run_heldout_period_generalization(
    df: pd.DataFrame,
    null_model: FittedNullModel,
    source_config: BenchmarkConfig,
    source_result: dict[str, Any],
) -> dict[str, Any]:
    """Secondary out-of-grid-style test on period values absent from the main injection grid."""
    target_config = BenchmarkConfig(
        periods_days=OFFGRID_GENERALIZATION_PERIODS_DAYS,
        amplitudes_hz=source_config.amplitudes_hz,
        n_phases_per_cell=source_config.n_phases_per_cell,
        train_phase_count=source_config.train_phase_count,
        val_phase_count=source_config.val_phase_count,
        test_phase_count=source_config.test_phase_count,
        global_seed=source_config.global_seed + 17,
        detection_grid_size=source_config.detection_grid_size,
        default_null_model_name=source_config.default_null_model_name,
    )
    target_dataset = build_benchmark_dataset(
        df,
        null_model,
        config=target_config,
        prefix="heldout_period_generalization",
        write_outputs=False,
    )
    target_catalog = target_dataset["catalog"]
    target_arrays = target_dataset["arrays"]
    target_catalog.to_csv(INTERIM_DIR / "generalization_heldout_periods_catalog.csv", index=False)
    np.savez_compressed(INTERIM_DIR / "generalization_heldout_periods_arrays.npz", **target_arrays)

    scorers = _baseline_scorers(df.reset_index(drop=True), source_config, null_model)
    results = []
    for baseline, scorer in scorers.items():
        start = time.perf_counter()
        scores, pred_freq, pred_amp, _ = _score_all(scorer, target_arrays["observed_residual_hz"])
        runtime = time.perf_counter() - start
        source_scores = source_result["scores"][baseline]["scores"]
        source_catalog = source_result["catalog"]
        source_threshold = _threshold(source_scores, source_catalog, split="validation")
        test_mask = target_catalog["split"].eq("test").to_numpy()
        y = target_catalog.loc[test_mask, "label"].to_numpy(dtype=int)
        s = scores[test_mask]
        signal_mask = y == 1
        results.append(
            {
                "baseline": baseline,
                "generalization_task": "off_grid_heldout_periods",
                "source_periods_days": "|".join(str(value) for value in source_config.periods_days),
                "heldout_periods_days": "|".join(str(value) for value in target_config.periods_days),
                "roc_auc": float(roc_auc_score(y, s)),
                "average_precision": float(average_precision_score(y, s)),
                "source_validation_fpr5_threshold": source_threshold,
                "tpr_at_source_validation_fpr_5pct": float(np.mean(s[signal_mask] >= source_threshold)),
                "target_test_null_false_positive_rate": float(np.mean(s[~signal_mask] >= source_threshold)),
                "target_test_examples": int(len(s)),
                "runtime_sec": runtime,
            }
        )
        target_catalog[f"score_{baseline}"] = scores
        target_catalog[f"pred_frequency_hz_{baseline}"] = pred_freq
        target_catalog[f"pred_amplitude_hz_{baseline}"] = pred_amp
    target_catalog.to_csv(INTERIM_DIR / "generalization_heldout_periods_catalog.csv", index=False)
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "generalization_results.csv", index=False)
    protocol = {
        "task": "off_grid_heldout_periods",
        "role": "secondary evaluation, not a replacement for the main phase-index split",
        "source_periods_days": list(source_config.periods_days),
        "heldout_periods_days": list(target_config.periods_days),
        "threshold_calibration": "source-grid validation null examples only",
        "evaluation": "held-out-period test examples only",
        "no_leakage_rule": "no held-out-period examples are used to choose thresholds or model settings",
        "global_seed": target_config.global_seed,
        "target_catalog": str(INTERIM_DIR / "generalization_heldout_periods_catalog.csv"),
    }
    write_json(protocol, TABLES_DIR / "generalization_protocol.json")
    notes = [
        "# Generalization Notes",
        "",
        "- This is a secondary evaluation; it does not replace the main phase-index benchmark.",
        "- The held-out periods are off the main period grid: "
        + ", ".join(f"{value:g} days" for value in target_config.periods_days)
        + ".",
        "- Thresholds are copied from the main source-grid validation-null scores.",
        "- Test metrics are computed only on held-out-period test examples.",
        "- This quantifies whether score calibration transfers across period values, not whether the model can extrapolate beyond the total observation span.",
    ]
    write_text("\n".join(notes), TABLES_DIR / "generalization_notes.md")

    figure_path = FIGURES_DIR / "final_generalization_summary.png"
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(results_df["baseline"], results_df["roc_auc"], color="#4c6f7b")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC on held-out periods")
    ax.set_title("Secondary generalization test: off-grid held-out periods")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return {"results": results_df, "protocol": protocol, "figure": str(figure_path), "target_catalog": target_catalog}


def _drift_catalog_rows(config: BenchmarkConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slope_index, slope in enumerate(DRIFT_SLOPES_HZ_PER_DAY):
        for repeat_index in range(config.n_phases_per_cell):
            split = _phase_split(repeat_index, config)
            sign = -1.0 if repeat_index % 2 else 1.0
            for label in (0, 1):
                rows.append(
                    {
                        "instance_id": f"linear_drift_s{slope_index:02d}_r{repeat_index:02d}_y{label}",
                        "family": "slow_linear_drift",
                        "split": split,
                        "label": label,
                        "slope_abs_hz_per_day": slope,
                        "slope_hz_per_day": sign * slope if label == 1 else 0.0,
                        "repeat_index": repeat_index,
                    }
                )
    return rows


def _weighted_rss(y: np.ndarray, design: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    lhs = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * y)
    beta = np.linalg.solve(lhs, rhs)
    residual = y - design @ beta
    return float(np.sum(weights * residual**2)), beta


def _drift_score_samples(df: pd.DataFrame, null_model: FittedNullModel, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    targets = pd.get_dummies(df["target"].astype(str), dtype=float).to_numpy()
    centered_days = (
        df["seconds_since_first_observation"].to_numpy(dtype=np.float64)
        - df["seconds_since_first_observation"].mean()
    ) / SECONDS_PER_DAY
    full_design = np.column_stack([targets, centered_days])
    weights = 1.0 / np.maximum(null_model.effective_sigma(df), 1.0) ** 2
    scores = np.zeros(residuals.shape[0], dtype=np.float64)
    slopes = np.zeros(residuals.shape[0], dtype=np.float64)
    for idx, sample in enumerate(residuals):
        rss_null, _ = _weighted_rss(sample, targets, weights)
        rss_full, beta = _weighted_rss(sample, full_design, weights)
        scores[idx] = max(0.0, rss_null - rss_full)
        slopes[idx] = float(beta[-1])
    return scores, slopes


def _run_drift_sanity_task(df: pd.DataFrame, null_model: FittedNullModel, config: BenchmarkConfig) -> dict[str, Any]:
    drift_config = BenchmarkConfig(
        periods_days=config.periods_days,
        amplitudes_hz=config.amplitudes_hz,
        n_phases_per_cell=config.n_phases_per_cell,
        train_phase_count=config.train_phase_count,
        val_phase_count=config.val_phase_count,
        test_phase_count=config.test_phase_count,
        global_seed=config.global_seed + 31,
        detection_grid_size=config.detection_grid_size,
        default_null_model_name=config.default_null_model_name,
    )
    frame = df.reset_index(drop=True).copy()
    times_sec = frame["seconds_since_first_observation"].to_numpy(dtype=np.float64)
    catalog = pd.DataFrame(_drift_catalog_rows(drift_config))
    residuals = np.zeros((len(catalog), len(frame)), dtype=np.float64)
    signal = np.zeros_like(residuals)
    noise = np.zeros_like(residuals)
    for idx, row in catalog.iterrows():
        seed = drift_config.global_seed + 10_000 * int(row["slope_abs_hz_per_day"] * 10) + int(row["repeat_index"]) + 1_000_000 * int(row["label"])
        rng = np.random.default_rng(seed)
        noise_i = null_model.sample(frame, rng)
        signal_i = slow_linear_drift(times_sec, float(row["slope_hz_per_day"])) if int(row["label"]) == 1 else np.zeros(len(frame), dtype=np.float64)
        noise[idx] = noise_i
        signal[idx] = signal_i
        residuals[idx] = noise_i + signal_i
        catalog.loc[idx, "noise_seed"] = seed
    catalog["noise_seed"] = catalog["noise_seed"].astype(int)
    scores, slopes = _drift_score_samples(frame, null_model, residuals)
    catalog["score_linear_drift"] = scores
    catalog["pred_slope_hz_per_day"] = slopes
    threshold = _threshold(scores, catalog, split="validation")
    test_mask = catalog["split"].eq("test").to_numpy()
    y = catalog.loc[test_mask, "label"].to_numpy(dtype=int)
    s = scores[test_mask]
    signal_mask = y == 1
    summary = {
        "task": "linear_drift_sanity",
        "baseline": "weighted_linear_drift_with_crystal_offsets",
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "validation_fpr5_threshold": threshold,
        "tpr_at_validation_fpr_5pct": float(np.mean(s[signal_mask] >= threshold)),
        "test_null_false_positive_rate": float(np.mean(s[~signal_mask] >= threshold)),
        "test_examples": int(len(s)),
    }
    rows = []
    test_signal = catalog["split"].eq("test").to_numpy() & catalog["label"].eq(1).to_numpy()
    for slope in DRIFT_SLOPES_HZ_PER_DAY:
        mask = test_signal & np.isclose(catalog["slope_abs_hz_per_day"].to_numpy(dtype=np.float64), slope)
        rows.append(
            {
                **summary,
                "slope_abs_hz_per_day": slope,
                "detection_rate_at_validation_fpr5": float(np.mean(scores[mask] >= threshold)),
                "n_test_signal": int(mask.sum()),
            }
        )
    result = pd.DataFrame(rows)
    rates = result["detection_rate_at_validation_fpr5"].tolist()
    min_slope = _interpolate_a95(DRIFT_SLOPES_HZ_PER_DAY, rates)
    result["minimum_detectable_abs_slope_hz_per_day"] = min_slope
    result.to_csv(TABLES_DIR / "drift_sanity_results.csv", index=False)
    catalog.to_csv(INTERIM_DIR / "drift_sanity_catalog.csv", index=False)
    np.savez_compressed(INTERIM_DIR / "drift_sanity_arrays.npz", observed_residual_hz=residuals, signal_hz=signal, noise_hz=noise)
    protocol = {
        "task": "linear_drift_sanity",
        "role": "secondary stress test, not a replacement for the sinusoidal main task",
        "slope_abs_hz_per_day_grid": list(DRIFT_SLOPES_HZ_PER_DAY),
        "threshold_calibration": "validation null examples only",
        "minimum_detectable_abs_slope_hz_per_day": min_slope,
        "global_seed": drift_config.global_seed,
        "catalog": str(INTERIM_DIR / "drift_sanity_catalog.csv"),
    }
    write_json(protocol, TABLES_DIR / "drift_sanity_protocol.json")
    notes = [
        "# Drift Sanity Task Notes",
        "",
        "- This task injects a linear trend in Hz/day on the same 55 peak-b timestamps.",
        "- It is a secondary low-frequency stress test inspired by the draft, not the main benchmark.",
        "- The score is the weighted RSS improvement from adding one linear slope term to crystal offsets.",
        "- The detection threshold is set from validation null examples only.",
        f"- Minimum detectable absolute drift rate at 95% detection, interpolated over the tested grid: `{min_slope:.6g}` Hz/day.",
    ]
    write_text("\n".join(notes), TABLES_DIR / "drift_sanity_notes.md")
    figure_path = FIGURES_DIR / "final_drift_sanity_curve.png"
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(result["slope_abs_hz_per_day"], result["detection_rate_at_validation_fpr5"], marker="o", lw=2)
    ax.axhline(0.95, color="black", ls="--", lw=1)
    if np.isfinite(min_slope):
        ax.axvline(min_slope, color="#b08968", ls=":", lw=2, label=f"95% at {min_slope:.2g} Hz/day")
        ax.legend(frameon=False)
    ax.set_xscale("log")
    ax.set_xlabel("Injected |drift| (Hz/day)")
    ax.set_ylabel("Detection rate at validation 5% FPR")
    ax.set_title("Secondary drift sanity task")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return {"results": result, "protocol": protocol, "figure": str(figure_path), "catalog": catalog}


def run_benchmark_suite(primary_df: pd.DataFrame, secondary_bc_df: pd.DataFrame | None = None) -> dict[str, Any]:
    null_fit_df, null_models, default_name = fit_null_models(primary_df)
    default_model = null_models[default_name]
    config = BenchmarkConfig(default_null_model_name=default_name)

    null_comparison = null_model_comparison_frame(null_models, default_name)
    null_comparison["parameters_json"] = null_comparison["parameters_json"].apply(json.dumps)
    null_comparison.to_csv(TABLES_DIR / "null_model_comparison.csv", index=False)

    notes = [
        "# Null Model Notes",
        "",
        "- `formal_gaussian_scaled`: Gaussian residuals with formal frequency uncertainties multiplied by one global scale factor fitted on the observation-level training split.",
        "- `crystal_gaussian_jitter`: Gaussian residuals with per-crystal extra variance terms fitted on training residuals.",
        "- `crystal_student_t_jitter`: Student-t residuals with fixed 4 degrees of freedom and per-crystal jitter terms fitted on training residuals.",
        "- `crystal_student_t_fitted_df`: Student-t residuals with target-specific jitter and target-specific degrees of freedom fitted on training residuals.",
        "- `crystal_gaussian_x2_mixture`: Gaussian residuals with crystal jitter plus an X2-only Gaussian outlier component fitted on X2 training residuals.",
        f"- Default selected by held-out residual negative log likelihood: `{default_name}`.",
        "- X2 is allowed to have separate scatter and optional tail/outlier behavior; model comparison is still based on held-out residual likelihood, not on tuning to the observed periodogram.",
    ]
    write_text("\n".join(notes), TABLES_DIR / "null_model_notes.md")

    primary = evaluate_dataset(primary_df, default_model, config, prefix="primary_peak_b", write_outputs=True)
    comparison = primary["comparison"].copy()
    runtime = primary["runtime"].copy()

    all_a95 = []
    all_a95_uncertainty = []
    for baseline in ("weighted_harmonic_regression", "generalized_lomb_scargle", "hierarchical_sinusoid_jitter"):
        a95 = _a95(primary["scores"][baseline]["scores"], primary["scores"][baseline]["threshold"], primary["catalog"], config)
        a95["baseline"] = baseline
        all_a95.append(a95)
        all_a95_uncertainty.append(
            _a95_with_uncertainty(primary["scores"][baseline]["scores"], primary["catalog"], config, baseline)
        )
    a95_df = pd.concat(all_a95, ignore_index=True)
    a95_uncertainty_df = pd.concat(all_a95_uncertainty, ignore_index=True)
    heatmap_df = _heatmap(
        primary["scores"]["weighted_harmonic_regression"]["scores"],
        primary["scores"]["weighted_harmonic_regression"]["threshold"],
        primary["catalog"],
        config,
    )

    ablations = []
    for scope, frame in {
        "primary_peak_b_all": primary_df,
        "official_clean_c10_c13_peak_b": primary_df.loc[primary_df["target"].isin(["C10", "C13"])].copy(),
    }.items():
        result = evaluate_dataset(frame, default_model, config, prefix=scope, write_outputs=False)
        ab = result["comparison"].copy()
        ab["dataset_scope"] = scope
        ablations.append(ab)
    if secondary_bc_df is not None:
        exploratory = evaluate_dataset(secondary_bc_df, default_model, config, prefix="exploratory_peaks_bc", write_outputs=False)
        ab = exploratory["comparison"].copy()
        ab["dataset_scope"] = "exploratory_peaks_bc_not_headline"
        ablations.append(ab)
    ablation_df = pd.concat(ablations, ignore_index=True)

    calibration, periodogram, calibration_note = _observed_calibration(primary_df, config, default_model)
    calibration.to_csv(TABLES_DIR / "observed_series_false_alarm_behavior.csv", index=False)
    periodogram.to_csv(TABLES_DIR / "observed_series_periodogram.csv", index=False)

    x2_study_rows = []
    observed_rows = []
    for name, model in null_models.items():
        cal, _, note_for_model = _observed_calibration(primary_df, config, model)
        for _, row in cal.iterrows():
            x2_study_rows.append(
                {
                    "null_model": name,
                    "distribution": model.distribution,
                    "is_default": name == default_name,
                    "scope": row["scope"],
                    "observed_score": row["observed_score"],
                    "best_period_days": row["best_period_days"],
                    "null_score_p95": row["null_score_p95"],
                    "observed_empirical_p_value": row["observed_empirical_p_value"],
                    "model_note": note_for_model,
                }
            )
        all_row = cal.loc[cal["scope"].eq("all_peak_b")].iloc[0]
        observed_rows.append(
            {
                "null_model": name,
                "observed_empirical_p_value": all_row["observed_empirical_p_value"],
                "observed_score": all_row["observed_score"],
                "observed_null_p95": all_row["null_score_p95"],
            }
        )
    x2_study = pd.DataFrame(x2_study_rows)
    x2_study.to_csv(TABLES_DIR / "x2_calibration_study.csv", index=False)
    selected_x2 = x2_study.loc[
        x2_study["is_default"].astype(bool) & x2_study["scope"].eq("peak_b_x2")
    ].iloc[0]
    best_x2 = x2_study.loc[x2_study["scope"].eq("peak_b_x2")].sort_values(
        "observed_empirical_p_value", ascending=False
    ).iloc[0]
    clean_decision = (
        "The selected null does not fully calibrate the X2-only observed periodogram, "
        "so C10+C13 peak-b is promoted to an official clean secondary condition."
        if float(selected_x2["observed_empirical_p_value"]) < 0.05
        else "The selected null gives an acceptable X2-only calibration diagnostic; C10+C13 remains a clean secondary condition."
    )
    decision_lines = [
        "# X2 Handling Decision",
        "",
        "- X2 is retained in the default 55-row peak-b benchmark because it is part of the published peak-b record.",
        "- Additional candidate nulls were fitted on training residuals only: target-specific fitted-df Student-t and X2 Gaussian mixture.",
        f"- Default null selected by held-out residual likelihood: `{default_name}`.",
        f"- Default-null X2-only empirical p-value: `{float(selected_x2['observed_empirical_p_value']):.6g}`.",
        f"- Best X2-only empirical p-value among tested nulls: `{float(best_x2['observed_empirical_p_value']):.6g}` from `{best_x2['null_model']}`.",
        f"- Decision: {clean_decision}",
        "- The X2-only observed diagnostic is not a detection claim and is not a headline result.",
        "- The official clean condition means rerunning the same peak-b task on C10+C13 only, where the observed calibration diagnostic is not X2 dominated.",
    ]
    write_text("\n".join(decision_lines), TABLES_DIR / "x2_handling_decision.md")

    observed_notes = notes + [
        "",
        "## Observed-Series Calibration",
        f"- {calibration_note}",
    ]
    for _, row in calibration.iterrows():
        observed_notes.append(
            f"- `{row['scope']}`: score `{row['observed_score']:.6g}`, "
            f"null p95 `{row['null_score_p95']:.6g}`, empirical p-value `{row['observed_empirical_p_value']:.6g}`."
        )
    observed_notes.append(
        "- Low p-values in the X2-only diagnostic are treated as evidence of residual scatter/model mismatch, not as a detection claim.",
    )
    observed_notes.extend(["", "## X2 Decision", f"- {clean_decision}"])
    write_text("\n".join(observed_notes), TABLES_DIR / "null_model_notes.md")

    observed_false_alarm = pd.DataFrame(observed_rows)
    null_comparison = null_comparison.merge(observed_false_alarm, on="null_model", how="left")
    null_comparison.to_csv(TABLES_DIR / "null_model_comparison.csv", index=False)

    comparison.to_csv(TABLES_DIR / "baseline_comparison.csv", index=False)
    runtime.to_csv(TABLES_DIR / "runtime_summary.csv", index=False)
    runtime.to_csv(TABLES_DIR / "baseline_runtime_summary.csv", index=False)
    a95_df.to_csv(TABLES_DIR / "a95_vs_frequency.csv", index=False)
    a95_uncertainty_df.to_csv(TABLES_DIR / "a95_vs_frequency_with_uncertainty.csv", index=False)
    heatmap_df.to_csv(TABLES_DIR / "benchmark_sensitivity_heatmap.csv", index=False)
    ablation_df.to_csv(TABLES_DIR / "ablation_results.csv", index=False)
    a95_notes = [
        "# A95 Uncertainty Notes",
        "",
        "- The point estimate `a95_hz` is the interpolated amplitude where the test-set detection rate reaches 95% at a validation-null 5% false-positive threshold.",
        "- The interval columns resample the validation-null scores used to set the threshold and the five test phase realizations in each period-amplitude cell.",
        "- These intervals quantify finite synthetic-grid and phase-sampling uncertainty.",
        "- They do not quantify uncertainty in the measured JILA scan frequencies, the temperature correction, or the choice of null model.",
    ]
    write_text("\n".join(a95_notes), TABLES_DIR / "a95_uncertainty_notes.md")

    generalization = _run_heldout_period_generalization(primary_df, default_model, config, primary)
    drift_sanity = _run_drift_sanity_task(primary_df, default_model, config)

    predictions = primary["catalog"].copy()
    for baseline, values in primary["scores"].items():
        predictions[f"score_{baseline}"] = values["scores"]
        predictions[f"pred_frequency_hz_{baseline}"] = values["pred_frequency_hz"]
        predictions[f"pred_amplitude_hz_{baseline}"] = values["pred_amplitude_hz"]
    predictions.to_csv(TABLES_DIR / "all_predictions_and_scores.csv", index=False)
    predictions.loc[predictions["split"].eq("test")].to_csv(TABLES_DIR / "test_predictions_and_scores.csv", index=False)

    main_heatmap_baseline = "hierarchical_sinusoid_jitter"
    auc_heatmap_df = _auc_heatmap_table(
        primary["scores"][main_heatmap_baseline]["scores"],
        primary["scores"][main_heatmap_baseline]["threshold"],
        primary["catalog"],
        config,
    )
    auc_heatmap_df["baseline"] = main_heatmap_baseline
    auc_heatmap_df.to_csv(TABLES_DIR / "auc_heatmap_with_a95_contour.csv", index=False)
    representative_table = _representative_frequency_table(primary, a95_uncertainty_df, config)
    representative_table.to_csv(TABLES_DIR / "final_representative_frequency_baseline_table.csv", index=False)

    protocol = {
        "primary_dataset": "primary_peak_b",
        "primary_rows": int(len(primary_df)),
        "default_null_model": default_name,
        "periods_days": list(config.periods_days),
        "frequencies_hz": config.frequencies_hz.tolist(),
        "amplitudes_hz": list(config.amplitudes_hz),
        "n_phases_per_cell": config.n_phases_per_cell,
        "official_clean_secondary_condition": "C10+C13 peak-b only; same task and split rule, used to quarantine X2 calibration concerns",
        "secondary_generalization_task": "off-grid held-out periods; thresholds from source-grid validation nulls",
        "phase_split_rule": {
            "train_phase_indices": f"0-{config.train_phase_count - 1}",
            "validation_phase_indices": f"{config.train_phase_count}-{config.train_phase_count + config.val_phase_count - 1}",
            "test_phase_indices": f"{config.train_phase_count + config.val_phase_count}-{config.n_phases_per_cell - 1}",
        },
        "global_seed": config.global_seed,
    }
    split_counts = primary["catalog"].groupby(["split", "label"]).size().rename("count").reset_index()
    protocol["split_counts"] = split_counts.to_dict(orient="records")
    write_json(protocol, TABLES_DIR / "benchmark_protocol_fixed.json")
    write_json(protocol, TABLES_DIR / "benchmark_protocol.json")
    split_counts.to_csv(TABLES_DIR / "split_counts_fixed.csv", index=False)

    leakage = "\n".join(
        [
            "# Leakage Check",
            "",
            "- One authoritative split is used: phase-index splits within every frequency-amplitude cell.",
            "- Validation thresholds are calibrated only on validation null examples.",
            "- Test examples are not used for model selection, threshold calibration, or null-model fitting.",
            "- Null-model parameters are fitted on a target-stratified chronological training subset of the observed peak-b residuals.",
            "- The held-out-period generalization test calibrates thresholds on source-grid validation nulls and evaluates only off-grid held-out-period test examples.",
            "- The previous logistic-periodogram baseline was removed from headline outputs to avoid overemphasizing tiny-data ML.",
        ]
    )
    write_text(leakage, TABLES_DIR / "leakage_check.md")

    metrics_md = ["# Metric Definitions", ""]
    for key, value in metric_definitions().items():
        metrics_md.append(f"- `{key}`: {value}")
    write_text("\n".join(metrics_md), TABLES_DIR / "metric_definitions.md")

    figures = _plot_outputs(
        comparison,
        heatmap_df,
        a95_uncertainty_df,
        periodogram,
        ablation_df,
        primary["catalog"],
        primary["scores"]["weighted_harmonic_regression"]["scores"],
    )
    figures["final_generalization_summary"] = generalization["figure"]
    figures["final_drift_sanity_curve"] = drift_sanity["figure"]
    figures["final_auc_heatmap_with_a95_contour"] = _plot_auc_heatmap_with_a95_contour(
        auc_heatmap_df, main_heatmap_baseline
    )

    # Null calibration figure.
    calib_path = FIGURES_DIR / "final_null_model_calibration_comparison.png"
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_df = null_comparison.sort_values("holdout_nll_per_point")
    ax.bar(plot_df["null_model"], plot_df["holdout_nll_per_point"], color="#5f9ea0")
    ax.set_ylabel("Held-out NLL per residual")
    ax.set_title("Null-model calibration comparison")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(calib_path, dpi=200)
    plt.close(fig)
    figures["final_null_model_calibration_comparison"] = str(calib_path)

    figure_captions = {
        "final_peak_b_residuals_over_time_by_crystal": "Peak-b temperature-corrected residuals on the observed JILA cadence, separated by crystal.",
        "final_residual_scatter_uncertainty_by_crystal": "Residual scatter versus formal uncertainty and crystal identity; X2 is visibly broader.",
        "final_null_model_calibration_comparison": "Held-out negative log likelihood for formal, crystal-aware Gaussian, fitted-tail Student-t, and X2-mixture null models.",
        "final_sensitivity_heatmap_main_baseline": "Detection rate for the weighted harmonic-regression baseline across the primary sinusoid grid at validation-calibrated 5% FPR.",
        "final_auc_heatmap_with_a95_contour": "Cell-wise ROC AUC for the hierarchical sinusoid baseline, with a 95% detection-rate contour from the validation-calibrated threshold.",
        "final_a95_vs_frequency_with_uncertainty": "A95 sensitivity curves for all safe baselines with bootstrap intervals from validation-threshold and test-phase resampling.",
        "final_observed_series_periodogram_calibration": "Observed peak-b residual periodogram by subset; interpreted conservatively as a calibration diagnostic, not a signal claim.",
        "final_generalization_summary": "Secondary off-grid held-out-period evaluation using thresholds calibrated on the main period grid.",
        "final_drift_sanity_curve": "Secondary linear-drift sanity task showing detection rate versus injected absolute drift.",
        "final_ablation_remove_x2": "Comparison of the default peak-b condition and the official clean C10+C13-only condition.",
        "final_roc_curve_main_baseline": "Held-out ROC curve for the weighted harmonic-regression baseline.",
    }
    lines = ["# Figure Captions Draft", ""]
    for key, caption in figure_captions.items():
        lines.append(f"- `{key}`: {caption}")
    write_text("\n".join(lines), TABLES_DIR / "figure_captions_draft.md")
    write_text("\n".join(["# Figure Caption Notes", "", *lines[2:]]), TABLES_DIR / "figure_caption_notes.md")
    manifest_lines = [
        "# Figures Manifest",
        "",
        "This manifest is for the final technical package only. It is not a publication figure list.",
        "",
    ]
    for key, caption in figure_captions.items():
        path = figures.get(key, str(FIGURES_DIR / f"{key}.png"))
        manifest_lines.append(f"- `{key}`: `{path}`. {caption}")
    write_text("\n".join(manifest_lines), TABLES_DIR / "figures_manifest.md")

    return {
        "config": config,
        "null_fit_df": null_fit_df,
        "null_models": null_models,
        "default_null_model": default_name,
        "comparison": comparison,
        "runtime": runtime,
        "a95": a95_df,
        "a95_uncertainty": a95_uncertainty_df,
        "heatmap": heatmap_df,
        "ablation": ablation_df,
        "generalization": generalization,
        "drift_sanity": drift_sanity,
        "auc_heatmap": auc_heatmap_df,
        "representative_frequency_table": representative_table,
        "x2_calibration_study": x2_study,
        "calibration": calibration,
        "calibration_note": calibration_note,
        "figures": figures,
        "split_counts": split_counts,
        "protocol": protocol,
    }
