from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from th229_bench.benchmarking import BenchmarkConfig  # noqa: E402
from th229_bench.synthetic import SECONDS_PER_DAY  # noqa: E402

GLOBAL_SEED = 22902691
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_EPOCHS = 200
DEFAULT_PATIENCE = 20
DEFAULT_BOOTSTRAP = 500
TARGETS = ("C10", "C13", "X2")


@dataclass(frozen=True)
class NeuralConfig:
    baseline_name: str
    checkpoint_path: Path
    result_path: Path
    seed: int = GLOBAL_SEED
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    batch_size: int = DEFAULT_BATCH_SIZE
    n_bootstrap: int = DEFAULT_BOOTSTRAP
    learning_rates: tuple[float, ...] = (1e-3, 3e-4, 1e-4)
    weight_decays: tuple[float, ...] = (0.0, 1e-4)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(out_channels)
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        y = self.conv(x)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return self.activation(y + residual)


class NeuralCNN(nn.Module):
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(input_dim, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(x.transpose(1, 2))
        pooled = y.mean(dim=2)
        return self.head(pooled).squeeze(-1)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    catalog = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "benchmark_catalog.csv")
    arrays = np.load(PROJECT_ROOT / "data" / "interim" / "benchmark_arrays.npz")
    residuals = np.asarray(arrays["observed_residual_hz"], dtype=np.float32)
    metadata = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "primary_peak_b.csv")
    if residuals.shape[1] != len(metadata):
        raise ValueError(f"expected length-{len(metadata)} residuals, got {residuals.shape}")
    return catalog, residuals, metadata


def _feature_stats(catalog: pd.DataFrame, residuals: np.ndarray, metadata: pd.DataFrame) -> dict[str, float]:
    train_mask = catalog["split"].eq("train").to_numpy()
    train_residual = residuals[train_mask].reshape(-1)
    sigma = metadata["freq_unc_hz"].to_numpy(dtype=np.float32)
    temp = metadata["temp_k"].to_numpy(dtype=np.float32)
    return {
        "residual_mean": float(np.mean(train_residual)),
        "residual_std": float(max(np.std(train_residual), 1.0)),
        "unc_mean": float(np.mean(sigma)),
        "unc_std": float(max(np.std(sigma), 1.0)),
        "temp_mean": float(np.mean(temp)),
        "temp_std": float(max(np.std(temp), 1e-6)),
    }


def build_sequence_features(
    catalog: pd.DataFrame,
    residuals: np.ndarray,
    metadata: pd.DataFrame,
    stats: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build length-55, 8-channel sequences using train-split normalization only."""
    if stats is None:
        stats = _feature_stats(catalog, residuals, metadata)
    n, length = residuals.shape
    features = np.zeros((n, length, 8), dtype=np.float32)
    features[:, :, 0] = (residuals - stats["residual_mean"]) / stats["residual_std"]
    sigma = metadata["freq_unc_hz"].to_numpy(dtype=np.float32)
    temp = metadata["temp_k"].to_numpy(dtype=np.float32)
    features[:, :, 1] = (sigma[None, :] - stats["unc_mean"]) / stats["unc_std"]
    features[:, :, 2] = (temp[None, :] - stats["temp_mean"]) / stats["temp_std"]
    target = metadata["target"].astype(str)
    for offset, name in enumerate(TARGETS):
        features[:, :, 3 + offset] = (target == name).to_numpy(dtype=np.float32)[None, :]
    seconds = metadata["seconds_since_first_observation"].to_numpy(dtype=np.float32)
    span = float(max(seconds.max() - seconds.min(), 1.0))
    features[:, :, 6] = ((seconds - seconds.min()) / span)[None, :]
    features[:, :, 7] = 1.0
    return features, stats


def _make_model(baseline_name: str) -> nn.Module:
    if baseline_name == "neural_cnn":
        return NeuralCNN()
    if baseline_name == "neural_transformer":
        from neural_transformer import NeuralTransformer

        return NeuralTransformer()
    raise ValueError(f"Unknown neural baseline: {baseline_name}")


def _predict_scores(model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.as_tensor(features, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (x,) in loader:
            logits = model(x.to(device))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def _train_one(
    baseline_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    lr: float,
    weight_decay: float,
    config: NeuralConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    _set_seed(config.seed + int(lr * 1e6) + int(weight_decay * 1e8))
    model = _make_model(baseline_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    train_ds = TensorDataset(
        torch.as_tensor(features[train_mask], dtype=torch.float32),
        torch.as_tensor(labels[train_mask], dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, generator=generator)

    best_state = None
    best_val_auc = -np.inf
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(config.max_epochs):
        model.train()
        losses = []
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device))
            loss = loss_fn(logits, y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_scores = _predict_scores(model, features[validation_mask], device, config.batch_size)
        val_auc = float(roc_auc_score(labels[validation_mask], val_scores))
        history.append({"epoch": epoch + 1, "train_bce": float(np.mean(losses)), "validation_auroc": val_auc})
        if val_auc > best_val_auc + 1e-5:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "best_validation_auroc": best_val_auc,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "history": history,
    }


def _threshold(scores: np.ndarray, catalog: pd.DataFrame) -> float:
    mask = catalog["split"].eq("validation").to_numpy() & catalog["label"].eq(0).to_numpy()
    return float(np.quantile(scores[mask], 0.95))


def _comparison_row(baseline_name: str, scores: np.ndarray, threshold: float, catalog: pd.DataFrame, runtime_sec: float) -> dict[str, Any]:
    test_mask = catalog["split"].eq("test").to_numpy()
    y = catalog.loc[test_mask, "label"].to_numpy(dtype=int)
    s = scores[test_mask]
    signal = y == 1
    return {
        "baseline": baseline_name,
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "validation_fpr5_threshold": threshold,
        "tpr_at_validation_fpr_5pct": float(np.mean(s[signal] >= threshold)),
        "test_null_false_positive_rate": float(np.mean(s[~signal] >= threshold)),
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


def _a95_with_uncertainty(
    baseline_name: str,
    scores: np.ndarray,
    catalog: pd.DataFrame,
    benchmark_config: BenchmarkConfig,
    n_bootstrap: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(benchmark_config.global_seed + 910_000 + sum(ord(ch) for ch in baseline_name))
    validation_null = catalog["split"].eq("validation").to_numpy() & catalog["label"].eq(0).to_numpy()
    validation_scores = scores[validation_null]
    test_signal = catalog["split"].eq("test").to_numpy() & catalog["label"].eq(1).to_numpy()
    rows = []
    for period_days, frequency_hz in zip(benchmark_config.periods_days, benchmark_config.frequencies_hz):
        point_threshold = float(np.quantile(validation_scores, 0.95))
        point_rates = []
        cell_scores: list[np.ndarray] = []
        for amplitude_hz in benchmark_config.amplitudes_hz:
            mask = (
                test_signal
                & np.isclose(catalog["frequency_hz"].to_numpy(dtype=np.float64), frequency_hz)
                & np.isclose(catalog["amplitude_hz"].to_numpy(dtype=np.float64), amplitude_hz)
            )
            values = scores[mask]
            cell_scores.append(values)
            point_rates.append(float(np.mean(values >= point_threshold)))
        point = _interpolate_a95(benchmark_config.amplitudes_hz, point_rates)
        draws = []
        for _ in range(n_bootstrap):
            threshold = float(np.quantile(rng.choice(validation_scores, size=len(validation_scores), replace=True), 0.95))
            rates = []
            for values in cell_scores:
                rates.append(float(np.mean(rng.choice(values, size=len(values), replace=True) >= threshold)))
            draws.append(_interpolate_a95(benchmark_config.amplitudes_hz, rates))
        finite = np.asarray([value for value in draws if np.isfinite(value)], dtype=np.float64)
        rows.append(
            {
                "baseline": baseline_name,
                "period_days": period_days,
                "frequency_hz": frequency_hz,
                "a95_hz": point,
                "a95_bootstrap_median_hz": float(np.median(finite)) if len(finite) else float("nan"),
                "a95_lower_hz": float(np.percentile(finite, 16)) if len(finite) else float("nan"),
                "a95_upper_hz": float(np.percentile(finite, 84)) if len(finite) else float("nan"),
                "a95_std_hz": float(np.std(finite)) if len(finite) else float("nan"),
                "bootstrap_replicates": n_bootstrap,
                "finite_bootstrap_replicates": int(len(finite)),
                "n_test_phases_per_cell": int(benchmark_config.test_phase_count),
                "uncertainty_source": "validation-null threshold resampling plus test-phase resampling within each amplitude cell",
            }
        )
    return pd.DataFrame(rows)


def _representative_rows(baseline_name: str, scores: np.ndarray, catalog: pd.DataFrame, threshold: float, a95: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test_null = catalog["split"].eq("test").to_numpy() & catalog["label"].eq(0).to_numpy()
    test_signal = catalog["split"].eq("test").to_numpy() & catalog["label"].eq(1).to_numpy()
    for period in (30.0, 180.0):
        period_signal = test_signal & np.isclose(catalog["period_days"].to_numpy(dtype=np.float64), period)
        y = np.concatenate([np.zeros(int(test_null.sum()), dtype=int), np.ones(int(period_signal.sum()), dtype=int)])
        s = np.concatenate([scores[test_null], scores[period_signal]])
        a95_row = a95.loc[np.isclose(a95["period_days"].to_numpy(dtype=np.float64), period)]
        rows.append(
            {
                "baseline": baseline_name,
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


def _plot_a95_with_ml(headline_a95: pd.DataFrame) -> Path:
    pieces = []
    tables_dir = PROJECT_ROOT / "results" / "tables"
    for name in (
        "a95_vs_frequency_with_uncertainty.csv",
        "rf_a95_vs_frequency_with_uncertainty.csv",
        "sbi_npe_a95_vs_frequency_with_uncertainty.csv",
        "sbi_npe_v2_a95_vs_frequency_with_uncertainty.csv",
        "neural_cnn_a95_vs_frequency_with_uncertainty.csv",
        "neural_transformer_a95_vs_frequency_with_uncertainty.csv",
    ):
        path = tables_dir / name
        if path.exists():
            pieces.append(pd.read_csv(path))
    pieces.append(headline_a95)
    frame = pd.concat(pieces, ignore_index=True)
    figure_path = PROJECT_ROOT / "results" / "figures" / "final_a95_vs_frequency_with_uncertainty.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.drop_duplicates(["baseline", "period_days"], keep="last")
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    styles = {
        "weighted_harmonic_regression": ("Weighted harmonic", "--", "o"),
        "generalized_lomb_scargle": ("GLS", "--", "s"),
        "hierarchical_sinusoid_jitter": ("Hierarchical", "--", "^"),
        "random_forest_periodogram_features": ("Random forest", ":", "D"),
        "sbi_npe": ("SBI NPE v2", "-", "P"),
        "neural_cnn": ("Neural CNN", "-", "X"),
        "neural_transformer": ("Neural Transformer", "-", "v"),
    }
    for baseline, group in frame.groupby("baseline"):
        group = group.sort_values("period_days")
        x = group["period_days"].to_numpy(dtype=np.float64)
        y = group["a95_hz"].to_numpy(dtype=np.float64)
        low = group["a95_lower_hz"].to_numpy(dtype=np.float64)
        high = group["a95_upper_hz"].to_numpy(dtype=np.float64)
        label, linestyle, marker = styles.get(baseline, (baseline, "-", "o"))
        line = ax.plot(x, y, marker=marker, linestyle=linestyle, lw=2, label=label)[0]
        ax.fill_between(x, low, high, color=line.get_color(), alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Injected period (days)")
    ax.set_ylabel("A95 amplitude (Hz)")
    ax.set_title("A95 sensitivity with bootstrap intervals")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def _write_appendix_and_headline() -> None:
    rows = []
    for name in ("neural_cnn", "neural_transformer"):
        path = PROJECT_ROOT / "results" / f"{name}_results.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "baseline": name,
                    "selected_learning_rate": payload["selected_hyperparameters"]["learning_rate"],
                    "selected_weight_decay": payload["selected_hyperparameters"]["weight_decay"],
                    "best_validation_auroc": payload["best_validation_auroc"],
                    "test_auroc": payload["roc_auc"],
                    "test_average_precision": payload["average_precision"],
                    "validation_fpr5_threshold": payload["validation_fpr5_threshold"],
                    "test_null_false_positive_rate": payload["test_null_false_positive_rate"],
                    "checkpoint_path": payload["checkpoint_path"],
                }
            )
    if not rows:
        return
    appendix = pd.DataFrame(rows).sort_values("best_validation_auroc", ascending=False)
    appendix_path = PROJECT_ROOT / "results" / "tables" / "neural_baselines_appendix.csv"
    appendix_path.parent.mkdir(parents=True, exist_ok=True)
    appendix.to_csv(appendix_path, index=False)

    headline = str(appendix.iloc[0]["baseline"])
    headline_comparison = pd.read_csv(PROJECT_ROOT / "results" / "tables" / f"{headline}_baseline_comparison.csv")
    headline_representative = pd.read_csv(PROJECT_ROOT / "results" / "tables" / f"{headline}_representative_frequency_table.csv")
    headline_a95 = pd.read_csv(PROJECT_ROOT / "results" / "tables" / f"{headline}_a95_vs_frequency_with_uncertainty.csv")

    tables_dir = PROJECT_ROOT / "results" / "tables"
    comparison_ml_path = tables_dir / "baseline_comparison_with_ml.csv"
    representative_ml_path = tables_dir / "final_representative_frequency_baseline_table_with_ml.csv"
    if not comparison_ml_path.exists() and (tables_dir / "baseline_comparison_with_rf.csv").exists():
        shutil.copyfile(tables_dir / "baseline_comparison_with_rf.csv", comparison_ml_path)
    if not representative_ml_path.exists() and (tables_dir / "final_representative_frequency_baseline_table_with_rf.csv").exists():
        shutil.copyfile(tables_dir / "final_representative_frequency_baseline_table_with_rf.csv", representative_ml_path)
    comparison_base = pd.read_csv(comparison_ml_path) if comparison_ml_path.exists() else pd.DataFrame()
    representative_base = pd.read_csv(representative_ml_path) if representative_ml_path.exists() else pd.DataFrame()
    comparison_base = comparison_base.loc[~comparison_base["baseline"].isin(["neural_cnn", "neural_transformer"])].copy()
    representative_base = representative_base.loc[~representative_base["baseline"].isin(["neural_cnn", "neural_transformer"])].copy()
    pd.concat([comparison_base, headline_comparison], ignore_index=True).to_csv(comparison_ml_path, index=False)
    pd.concat([representative_base, headline_representative], ignore_index=True).to_csv(representative_ml_path, index=False)
    _plot_a95_with_ml(headline_a95)


def _score_after_freeze(model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    scores = _predict_scores(model, features, device, batch_size)
    return scores, time.perf_counter() - start


def run_neural_baseline(config: NeuralConfig) -> dict[str, Any]:
    _set_seed(config.seed)
    catalog, residuals, metadata = _load_data()
    labels = catalog["label"].to_numpy(dtype=int)
    train_mask = catalog["split"].eq("train").to_numpy()
    validation_mask = catalog["split"].eq("validation").to_numpy()
    test_mask = catalog["split"].eq("test").to_numpy()
    features, feature_stats = build_sequence_features(catalog, residuals, metadata)
    device = _device()

    started = time.perf_counter()
    candidates = []
    best_model = None
    best_info: dict[str, Any] | None = None
    for lr in config.learning_rates:
        for weight_decay in config.weight_decays:
            model, info = _train_one(
                config.baseline_name,
                features,
                labels,
                train_mask,
                validation_mask,
                lr,
                weight_decay,
                config,
                device,
            )
            info["num_parameters"] = int(sum(p.numel() for p in model.parameters()))
            candidates.append(info)
            if best_info is None or info["best_validation_auroc"] > best_info["best_validation_auroc"]:
                best_model = model
                best_info = info
    if best_model is None or best_info is None:
        raise RuntimeError("No neural model was trained")
    training_runtime = time.perf_counter() - started

    scores, scoring_runtime = _score_after_freeze(best_model, features, device, config.batch_size)
    threshold = _threshold(scores, catalog)
    comparison = pd.DataFrame([_comparison_row(config.baseline_name, scores, threshold, catalog, scoring_runtime)])
    benchmark_config = BenchmarkConfig()
    a95 = _a95_with_uncertainty(config.baseline_name, scores, catalog, benchmark_config, config.n_bootstrap)
    representative = _representative_rows(config.baseline_name, scores, catalog, threshold, a95)

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_model.cpu().state_dict(),
            "baseline_name": config.baseline_name,
            "feature_stats": feature_stats,
            "metadata": {
                "seed": config.seed,
                "selected_hyperparameters": {k: best_info[k] for k in ("learning_rate", "weight_decay")},
                "best_validation_auroc": best_info["best_validation_auroc"],
                "best_epoch": best_info["best_epoch"],
                "num_parameters": best_info["num_parameters"],
                "train_examples": int(train_mask.sum()),
                "validation_examples": int(validation_mask.sum()),
                "test_examples": int(test_mask.sum()),
                "input_channels": 8,
                "sequence_length": int(features.shape[1]),
            },
        },
        config.checkpoint_path,
    )

    tables_dir = PROJECT_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions = catalog.copy()
    predictions[f"score_{config.baseline_name}"] = scores
    predictions.to_csv(tables_dir / f"{config.baseline_name}_predictions.csv", index=False)
    comparison.to_csv(tables_dir / f"{config.baseline_name}_baseline_comparison.csv", index=False)
    a95.to_csv(tables_dir / f"{config.baseline_name}_a95_vs_frequency_with_uncertainty.csv", index=False)
    representative.to_csv(tables_dir / f"{config.baseline_name}_representative_frequency_table.csv", index=False)

    candidate_summary = [
        {
            key: value
            for key, value in candidate.items()
            if key in {"learning_rate", "weight_decay", "best_validation_auroc", "best_epoch", "epochs_ran", "num_parameters"}
        }
        for candidate in candidates
    ]

    result = {
        **comparison.iloc[0].to_dict(),
        "baseline": config.baseline_name,
        "training_protocol": "train split only; validation split used for early stopping, hyperparameter selection, and 5% null threshold calibration",
        "best_validation_auroc": best_info["best_validation_auroc"],
        "selected_hyperparameters": {k: best_info[k] for k in ("learning_rate", "weight_decay")},
        "best_epoch": best_info["best_epoch"],
        "epochs_ran_selected": best_info["epochs_ran"],
        "candidate_grid": candidate_summary,
        "feature_stats": feature_stats,
        "device": str(device),
        "training_runtime_sec": training_runtime,
        "checkpoint_path": str(config.checkpoint_path.relative_to(PROJECT_ROOT)),
        "predictions_path": f"results/tables/{config.baseline_name}_predictions.csv",
        "a95_path": f"results/tables/{config.baseline_name}_a95_vs_frequency_with_uncertainty.csv",
        "representative_table_path": f"results/tables/{config.baseline_name}_representative_frequency_table.csv",
        "n_bootstrap": config.n_bootstrap,
    }
    config.result_path.parent.mkdir(parents=True, exist_ok=True)
    config.result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_appendix_and_headline()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the supervised CNN sequence baseline.")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    args = parser.parse_args()
    config = NeuralConfig(
        baseline_name="neural_cnn",
        checkpoint_path=PROJECT_ROOT / "models" / "neural_cnn_v1.pt",
        result_path=PROJECT_ROOT / "results" / "neural_cnn_results.json",
        max_epochs=args.max_epochs,
        patience=args.patience,
        n_bootstrap=args.bootstrap,
    )
    result = run_neural_baseline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
