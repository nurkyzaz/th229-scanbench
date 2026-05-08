from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
BASELINES_DIR = PROJECT_ROOT / "baselines"
for path in (SRC_DIR, BASELINES_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baselines.random_forest import _extract_features  # noqa: E402
from baselines.sbi_npe import DEFAULT_CHECKPOINT, score_residuals  # noqa: E402
from null_models.gmm import save_gmm_model  # noqa: E402
from null_models.normalizing_flow import save_flow_model  # noqa: E402
from th229_bench.benchmarking import BenchmarkConfig, _baseline_scorers, _score_all  # noqa: E402
from th229_bench.hierarchical_model import HierarchicalSinusoidModel  # noqa: E402
from th229_bench.null_models import FittedNullModel, assign_null_fit_split, fit_null_models  # noqa: E402


def _load_primary() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data" / "processed" / "primary_peak_b.csv").reset_index(drop=True)


def _null_models(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    split, models, _ = fit_null_models(primary, model_names=["gmm", "normalizing_flow"])
    models = {
        "crystal_gaussian_x2_mixture": models["crystal_gaussian_x2_mixture"],
        "crystal_gmm_3comp": models["crystal_gmm_3comp"],
        "conditional_spline_flow": models["conditional_spline_flow"],
    }
    gmm = models["crystal_gmm_3comp"]
    flow = models["conditional_spline_flow"]
    save_gmm_model(gmm)
    save_flow_model(flow)
    return split, models


def _observed_pvalues(primary: pd.DataFrame, models: dict[str, Any], config: BenchmarkConfig, n_resamples: int = 512) -> pd.DataFrame:
    rows = []
    subsets = {
        "all_peak_b": primary,
        "peak_b_c10_c13": primary.loc[primary["target"].isin(["C10", "C13"])].copy(),
        "peak_b_x2": primary.loc[primary["target"].eq("X2")].copy(),
    }
    for model_name, model in models.items():
        for subset_name, subset in subsets.items():
            subset = subset.reset_index(drop=True)
            scorer = HierarchicalSinusoidModel.from_frame(subset, config.detection_grid_hz, model)
            observed = subset["residual_hz"].to_numpy(dtype=np.float64)
            score = scorer.score(observed).score
            null_scores = []
            for draw in range(n_resamples):
                rng = np.random.default_rng(config.global_seed + 810_000 + draw + len(subset) + sum(ord(c) for c in model_name))
                null_scores.append(scorer.score(model.sample(subset, rng)).score)
            null_scores = np.asarray(null_scores, dtype=np.float64)
            rows.append(
                {
                    "null_model": model_name,
                    "subset": subset_name,
                    "n_observations": int(len(subset)),
                    "score": float(score),
                    "empirical_pvalue": float((1 + np.sum(null_scores >= score)) / (len(null_scores) + 1)),
                    "n_resamples": int(n_resamples),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(PROJECT_ROOT / "results" / "tables" / "observed_series_pvalues_by_null.csv", index=False)
    return out


def _score_stat_baselines(primary: pd.DataFrame, residuals: np.ndarray, null_model: Any, config: BenchmarkConfig) -> dict[str, np.ndarray]:
    scorers = _baseline_scorers(primary, config, null_model)
    return {name: _score_all(scorer, residuals)[0] for name, scorer in scorers.items()}


def _rf_scores(train_catalog: pd.DataFrame, train_residuals: np.ndarray, metadata: pd.DataFrame, residuals: np.ndarray) -> np.ndarray:
    features_train, _ = _extract_features(train_residuals, metadata)
    labels = train_catalog["label"].to_numpy(dtype=int)
    train_mask = train_catalog["split"].eq("train").to_numpy()
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(features_train[train_mask], labels[train_mask])
    features, _ = _extract_features(residuals, metadata)
    return model.predict_proba(features)[:, 1]


def _test_null_fpr(primary: pd.DataFrame, models: dict[str, Any], config: BenchmarkConfig) -> pd.DataFrame:
    catalog = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "benchmark_catalog.csv")
    arrays = np.load(PROJECT_ROOT / "data" / "interim" / "benchmark_arrays.npz")
    train_residuals = arrays["observed_residual_hz"]
    val_count = int((catalog["split"].eq("validation") & catalog["label"].eq(0)).sum())
    test_count = int((catalog["split"].eq("test") & catalog["label"].eq(0)).sum())
    rows = []
    for model_name, model in models.items():
        rng = np.random.default_rng(config.global_seed + 830_000 + sum(ord(c) for c in model_name))
        val_null = np.vstack([model.sample(primary, rng) for _ in range(val_count)])
        test_null = np.vstack([model.sample(primary, rng) for _ in range(test_count)])
        residuals = np.vstack([val_null, test_null])
        stat_scores = _score_stat_baselines(primary, residuals, model, config)
        rf = _rf_scores(catalog, train_residuals, primary, residuals)
        sbi = score_residuals(residuals.astype(np.float32), checkpoint_path=DEFAULT_CHECKPOINT, num_posterior_samples=128)
        all_scores = {**stat_scores, "random_forest_periodogram_features": rf, "sbi_npe": sbi}
        for baseline, scores in all_scores.items():
            threshold = float(np.quantile(scores[:val_count], 0.95))
            rows.append(
                {
                    "null_model": model_name,
                    "baseline": baseline,
                    "validation_fpr5_threshold": threshold,
                    "test_null_false_positive_rate": float(np.mean(scores[val_count:] >= threshold)),
                    "n_validation_null": val_count,
                    "n_test_null": test_count,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(PROJECT_ROOT / "results" / "tables" / "test_null_fpr_by_null_model.csv", index=False)
    return out


def _heldout_loglik(split: pd.DataFrame, models: dict[str, Any], n_bootstrap: int = 1000) -> pd.DataFrame:
    holdout = split.loc[split["null_fit_split"].eq("holdout")].reset_index(drop=True)
    rng = np.random.default_rng(22902683)
    rows = []
    for model_name, model in models.items():
        logp = model.log_prob(holdout["residual_hz"].to_numpy(dtype=np.float64), holdout)
        draws = [float(np.mean(logp[rng.integers(0, len(logp), size=len(logp))])) for _ in range(n_bootstrap)]
        rows.append(
            {
                "null_model": model_name,
                "n_holdout": int(len(holdout)),
                "mean_loglik": float(np.mean(logp)),
                "mean_loglik_bootstrap_lower": float(np.percentile(draws, 16)),
                "mean_loglik_bootstrap_upper": float(np.percentile(draws, 84)),
                "bootstrap_replicates": n_bootstrap,
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_loglik", ascending=False)
    out.to_csv(PROJECT_ROOT / "results" / "tables" / "null_model_heldout_loglik.csv", index=False)
    return out


def _model_cdf_samples(model: Any, rows: pd.DataFrame, rng: np.random.Generator, n: int = 8192) -> np.ndarray:
    pieces = []
    for _ in range(max(1, n // max(len(rows), 1))):
        pieces.append(model.sample(rows.reset_index(drop=True), rng))
    return np.concatenate(pieces)


def _calibration_plot(split: pd.DataFrame, models: dict[str, Any]) -> Path:
    train = split.loc[split["null_fit_split"].eq("train")].reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    rng = np.random.default_rng(22902684)
    for ax, target in zip(axes, ["C10", "C13", "X2"]):
        group = train.loc[train["target"].eq(target)].reset_index(drop=True)
        empirical = np.sort(group["residual_hz"].to_numpy(dtype=np.float64))
        y = np.arange(1, len(empirical) + 1) / max(len(empirical), 1)
        ax.step(empirical, y, where="post", label="empirical train", color="black", lw=2)
        grid = np.linspace(float(empirical.min() - 2000), float(empirical.max() + 2000), 200)
        for model_name, model in models.items():
            samples = np.sort(_model_cdf_samples(model, group, rng))
            cdf = np.searchsorted(samples, grid, side="right") / len(samples)
            ax.plot(grid, cdf, lw=1.8, label=model_name)
        ax.axvline(0.0, color="0.65", lw=1, linestyle=":")
        ax.set_title(target)
        ax.set_xlabel("Residual (Hz)")
    axes[0].set_ylabel("CDF")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    path = PROJECT_ROOT / "results" / "figures" / "null_model_calibration_by_crystal.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    primary = _load_primary()
    config = BenchmarkConfig()
    split, models = _null_models(primary)
    pvalues = _observed_pvalues(primary, models, config)
    fpr = _test_null_fpr(primary, models, config)
    loglik = _heldout_loglik(split, models)
    fig = _calibration_plot(split, models)
    summary = {
        "observed_series_pvalues": str((PROJECT_ROOT / "results" / "tables" / "observed_series_pvalues_by_null.csv").relative_to(PROJECT_ROOT)),
        "test_null_fpr": str((PROJECT_ROOT / "results" / "tables" / "test_null_fpr_by_null_model.csv").relative_to(PROJECT_ROOT)),
        "heldout_loglik": str((PROJECT_ROOT / "results" / "tables" / "null_model_heldout_loglik.csv").relative_to(PROJECT_ROOT)),
        "calibration_plot": str(fig.relative_to(PROJECT_ROOT)),
        "x2_pvalues": pvalues.loc[pvalues["subset"].eq("peak_b_x2"), ["null_model", "empirical_pvalue"]].to_dict(orient="records"),
        "best_heldout_loglik": loglik.iloc[0].to_dict(),
        "fpr_rows": int(len(fpr)),
    }
    out = PROJECT_ROOT / "results" / "learned_null_results.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
