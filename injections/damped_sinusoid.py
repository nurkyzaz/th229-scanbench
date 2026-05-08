from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
BASELINES_DIR = PROJECT_ROOT / "baselines"
for path in (SRC_DIR, BASELINES_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baselines.random_forest import _extract_features  # noqa: E402
from baselines.sbi_npe import DEFAULT_CHECKPOINT as SBI_CHECKPOINT, score_residuals  # noqa: E402
from neural_cnn import NeuralCNN, build_sequence_features  # noqa: E402
from neural_transformer import NeuralTransformer  # noqa: E402
from th229_bench.benchmarking import (  # noqa: E402
    BenchmarkConfig,
    _a95_with_uncertainty,
    _baseline_scorers,
    _comparison_row,
    _score_all,
    _threshold,
)
from th229_bench.null_models import fit_null_models  # noqa: E402
from th229_bench.paths import INTERIM_DIR, TABLES_DIR  # noqa: E402
from th229_bench.synthetic import SECONDS_PER_DAY  # noqa: E402

TAUS_DAYS = (30.0, 90.0, 180.0)
STAT_BASELINES = ("weighted_harmonic_regression", "generalized_lomb_scargle", "hierarchical_sinusoid_jitter")
ML_BASELINES = ("random_forest_periodogram_features", "sbi_npe", "neural_cnn", "neural_transformer")
ALL_BASELINES = (*STAT_BASELINES, *ML_BASELINES)


def damped_sinusoid(times_sec: np.ndarray, amplitude_hz: float, frequency_hz: float, phase_rad: float, tau_days: float) -> np.ndarray:
    elapsed_days = (times_sec - float(np.min(times_sec))) / SECONDS_PER_DAY
    return amplitude_hz * np.exp(-elapsed_days / tau_days) * np.cos(2.0 * np.pi * frequency_hz * times_sec + phase_rad)


def _catalog_rows(config: BenchmarkConfig, tau_days: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for freq_index, (period_days, frequency_hz) in enumerate(zip(config.periods_days, config.frequencies_hz)):
        for amp_index, amplitude_hz in enumerate(config.amplitudes_hz):
            for phase_index in range(config.n_phases_per_cell):
                if phase_index < config.train_phase_count:
                    split = "train"
                elif phase_index < config.train_phase_count + config.val_phase_count:
                    split = "validation"
                else:
                    split = "test"
                phase_rad = 2.0 * np.pi * (phase_index + 0.5) / config.n_phases_per_cell
                for label in (0, 1):
                    rows.append(
                        {
                            "instance_id": f"damped_tau{int(tau_days)}_f{freq_index:02d}_a{amp_index:02d}_p{phase_index:02d}_y{label}",
                            "family": "damped_sinusoid",
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


def load_primary_and_null() -> tuple[pd.DataFrame, Any, BenchmarkConfig]:
    primary = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "primary_peak_b.csv").reset_index(drop=True)
    _, models, default_name = fit_null_models(primary)
    config = BenchmarkConfig(default_null_model_name=default_name)
    return primary, models[default_name], config


def generate_stress_dataset(tau_days: float, write_outputs: bool = True) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    primary, null_model, config = load_primary_and_null()
    times_sec = primary["seconds_since_first_observation"].to_numpy(dtype=np.float64)
    catalog = pd.DataFrame(_catalog_rows(config, tau_days))
    observed = np.zeros((len(catalog), len(primary)), dtype=np.float64)
    signal = np.zeros_like(observed)
    noise = np.zeros_like(observed)
    for idx, row in catalog.iterrows():
        seed = config.global_seed + 1_000_000 * int(row["label"]) + 10_000 * int(row["frequency_index"]) + 100 * int(row["amplitude_index"]) + int(row["phase_index"])
        rng = np.random.default_rng(seed)
        draw = null_model.sample(primary, rng)
        injected = (
            damped_sinusoid(times_sec, row["amplitude_hz"], row["frequency_hz"], row["phase_rad"], tau_days)
            if int(row["label"]) == 1
            else np.zeros(len(primary), dtype=np.float64)
        )
        noise[idx] = draw
        signal[idx] = injected
        observed[idx] = draw + injected
        catalog.loc[idx, "noise_seed"] = seed
    catalog["noise_seed"] = catalog["noise_seed"].astype(int)
    arrays = {
        "times_sec": times_sec,
        "observed_residual_hz": observed,
        "signal_hz": signal,
        "noise_hz": noise,
        "formal_sigma_hz": np.tile(primary["freq_unc_hz"].to_numpy(dtype=np.float64), (len(catalog), 1)),
        "effective_sigma_hz": np.tile(null_model.effective_sigma(primary), (len(catalog), 1)),
    }
    if write_outputs:
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        catalog.to_csv(INTERIM_DIR / f"stress_damped_tau{int(tau_days)}_catalog.csv", index=False)
        np.savez_compressed(INTERIM_DIR / f"stress_damped_tau{int(tau_days)}_arrays.npz", **arrays)
    return catalog, arrays


def _test_metrics(name: str, scores: np.ndarray, catalog: pd.DataFrame, threshold: float, runtime: float) -> dict[str, Any]:
    row = _comparison_row(name, scores, threshold, catalog, runtime)
    return {
        "baseline": name,
        "test_auroc": row["roc_auc"],
        "test_ap": row["average_precision"],
        "validation_fpr5_threshold": row["validation_fpr5_threshold"],
        "test_null_fpr": row["test_null_false_positive_rate"],
        "runtime_sec": row["runtime_sec"],
    }


def _load_rf_scores(residuals: np.ndarray, primary: pd.DataFrame) -> np.ndarray:
    path = PROJECT_ROOT / "models" / "random_forest_periodogram_features_v1.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing RF checkpoint {path}; run `make reproduce` before stress evaluation.")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    features, _ = _extract_features(residuals, primary)
    return payload["model"].predict_proba(features)[:, 1]


def _load_neural_scores(name: str, catalog: pd.DataFrame, residuals: np.ndarray, primary: pd.DataFrame) -> np.ndarray:
    checkpoint = PROJECT_ROOT / "models" / f"{name}_v1.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing neural checkpoint {checkpoint}; run `make reproduce` before stress evaluation.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = NeuralCNN() if name == "neural_cnn" else NeuralTransformer()
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    features, _ = build_sequence_features(catalog, residuals, primary, stats=payload["feature_stats"])
    scores = []
    with torch.no_grad():
        for start in range(0, len(features), 512):
            x = torch.as_tensor(features[start : start + 512], dtype=torch.float32)
            scores.append(torch.sigmoid(model(x)).numpy())
    return np.concatenate(scores).astype(np.float64)


def _baseline_scores(primary: pd.DataFrame, null_model: Any, config: BenchmarkConfig, residuals: np.ndarray) -> dict[str, tuple[np.ndarray, float]]:
    scores: dict[str, tuple[np.ndarray, float]] = {}
    for name, scorer in _baseline_scorers(primary, config, null_model).items():
        start = time.perf_counter()
        values = _score_all(scorer, residuals)[0]
        scores[name] = (values, time.perf_counter() - start)
    start = time.perf_counter()
    scores["random_forest_periodogram_features"] = (_load_rf_scores(residuals, primary), time.perf_counter() - start)
    start = time.perf_counter()
    scores["sbi_npe"] = (score_residuals(residuals.astype(np.float32), checkpoint_path=SBI_CHECKPOINT, num_posterior_samples=256), time.perf_counter() - start)
    for name in ("neural_cnn", "neural_transformer"):
        start = time.perf_counter()
        scores[name] = (_load_neural_scores(name, pd.DataFrame(), residuals, primary), time.perf_counter() - start)
    return scores


def _main_auc_lookup() -> dict[str, float]:
    frame = pd.read_csv(TABLES_DIR / "baseline_comparison_with_ml.csv")
    lookup = dict(zip(frame["baseline"], frame["roc_auc"]))
    appendix = TABLES_DIR / "neural_baselines_appendix.csv"
    if appendix.exists():
        neural = pd.read_csv(appendix)
        lookup.update(dict(zip(neural["baseline"], neural["test_auroc"])))
    return lookup


def evaluate_tau(tau_days: float, n_bootstrap: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary, null_model, config = load_primary_and_null()
    catalog_path = INTERIM_DIR / f"stress_damped_tau{int(tau_days)}_catalog.csv"
    arrays_path = INTERIM_DIR / f"stress_damped_tau{int(tau_days)}_arrays.npz"
    if not catalog_path.exists() or not arrays_path.exists():
        catalog, arrays = generate_stress_dataset(tau_days, write_outputs=True)
    else:
        catalog = pd.read_csv(catalog_path)
        arrays = dict(np.load(arrays_path))
    residuals = np.asarray(arrays["observed_residual_hz"], dtype=np.float64)
    main_auc = _main_auc_lookup()
    summary_rows = []
    a95_rows = []
    for name, (scores, runtime) in _baseline_scores(primary, null_model, config, residuals).items():
        threshold = _threshold(scores, catalog)
        metrics = _test_metrics(name, scores, catalog, threshold, runtime)
        metrics["tau_days"] = tau_days
        metrics["test_auroc_main_benchmark"] = float(main_auc.get(name, np.nan))
        metrics["auroc_delta"] = float(metrics["test_auroc"] - metrics["test_auroc_main_benchmark"])
        summary_rows.append(metrics)
        a95 = _a95_with_uncertainty(scores, catalog, config, name, n_bootstrap=n_bootstrap)
        a95["tau_days"] = tau_days
        a95_rows.append(a95)
    return pd.DataFrame(summary_rows), pd.concat(a95_rows, ignore_index=True)


def _plot_summary(summary: pd.DataFrame) -> Path:
    fig_path = PROJECT_ROOT / "results" / "figures" / "stress_damped_auroc_degradation.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    baselines = list(ALL_BASELINES)
    x = np.arange(len(baselines), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    colors = {30.0: "#4c78a8", 90.0: "#f58518", 180.0: "#54a24b"}
    display = {
        "weighted_harmonic_regression": "Weighted harmonic",
        "generalized_lomb_scargle": "GLS",
        "hierarchical_sinusoid_jitter": "Hierarchical",
        "random_forest_periodogram_features": "Random forest",
        "sbi_npe": "SBI NPE v2",
        "neural_cnn": "Neural CNN",
        "neural_transformer": "Neural Transformer",
    }
    for offset, tau in enumerate(TAUS_DAYS):
        group = summary.loc[np.isclose(summary["tau_days"], tau)].set_index("baseline")
        values = [group.loc[name, "test_auroc"] if name in group.index else np.nan for name in baselines]
        ax.bar(x + (offset - 1) * width, values, width=width, label=f"tau={int(tau)} d", color=colors[tau])
    main = summary.drop_duplicates("baseline").set_index("baseline")
    for idx, name in enumerate(baselines):
        y = float(main.loc[name, "test_auroc_main_benchmark"]) if name in main.index else np.nan
        if np.isfinite(y):
            ax.hlines(y, idx - 0.42, idx + 0.42, colors="black", linestyles="--", lw=1)
    ax.axhline(0.5, color="0.65", lw=1, linestyle="-", zorder=0)
    ax.set_ylim(0.45, 0.9)
    ax.set_ylabel("Test AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels([display.get(name, name) for name in baselines], rotation=25, ha="right")
    ax.set_title("Zero-shot damped-sinusoid stress test")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    return fig_path


def run_stress_damped(n_bootstrap: int = 500) -> dict[str, Any]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    all_summary = []
    output_a95 = []
    for tau in TAUS_DAYS:
        generate_stress_dataset(tau, write_outputs=True)
        summary, a95 = evaluate_tau(tau, n_bootstrap=n_bootstrap)
        summary.to_csv(TABLES_DIR / f"stress_damped_tau{int(tau)}_summary.csv", index=False)
        a95_path = TABLES_DIR / f"stress_damped_tau{int(tau)}_a95_vs_frequency.csv"
        a95.to_csv(a95_path, index=False)
        all_summary.append(summary)
        output_a95.append(str(a95_path.relative_to(PROJECT_ROOT)))
    summary = pd.concat(all_summary, ignore_index=True)
    summary_path = TABLES_DIR / "stress_damped_summary.csv"
    summary.to_csv(summary_path, index=False)
    fig_path = _plot_summary(summary)
    result = {
        "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
        "a95_paths": output_a95,
        "figure_path": str(fig_path.relative_to(PROJECT_ROOT)),
        "taus_days": list(TAUS_DAYS),
        "baselines": list(ALL_BASELINES),
        "n_bootstrap": n_bootstrap,
    }
    out = PROJECT_ROOT / "results" / "stress_damped_results.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and evaluate damped-sinusoid stress tests.")
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(run_stress_damped(n_bootstrap=args.bootstrap), indent=2))


if __name__ == "__main__":
    main()
