from __future__ import annotations

import argparse
import shutil
import json
import os
import sys
import time
import types
import warnings
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sbi_npe_simulator import SBISimulatorConfig, Th229SBISimulator  # noqa: E402
from th229_bench.benchmarking import BenchmarkConfig  # noqa: E402
from th229_bench.synthetic import SECONDS_PER_DAY  # noqa: E402

BASELINE_NAME = "sbi_npe"
V1_CHECKPOINT = PROJECT_ROOT / "models" / "sbi_npe_v1.pt"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "sbi_npe_v2.pt"
SMOKE_CHECKPOINT = PROJECT_ROOT / "models" / "sbi_npe_v2_smoke.pt"
DEFAULT_TRAINING_SEED = 22902651
DEFAULT_SCORING_SEED = 22902652
U_BOUND = 10.0
LOG_A_MIN = float(np.log(1.0))
LOG_A_MAX = float(np.log(6500.0))


def _install_tensorboard_stub() -> None:
    """Avoid importing a broken local TensorFlow through torch.utils.tensorboard."""
    if "torch.utils.tensorboard.writer" in sys.modules:
        return
    package = types.ModuleType("torch.utils.tensorboard")
    package.__path__ = []
    writer = types.ModuleType("torch.utils.tensorboard.writer")

    class SummaryWriter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __getattr__(self, name: str) -> Any:
            return lambda *args, **kwargs: None

        def close(self) -> None:
            pass

    writer.SummaryWriter = SummaryWriter
    package.SummaryWriter = SummaryWriter
    sys.modules["torch.utils.tensorboard"] = package
    sys.modules["torch.utils.tensorboard.writer"] = writer


def _import_sbi() -> tuple[Any, Any, Any, Any]:
    _install_tensorboard_stub()
    import torch
    from sbi.inference import SNPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform

    return torch, SNPE, posterior_nn, BoxUniform


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _prior_bounds(config: SBISimulatorConfig) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray([-U_BOUND, -U_BOUND, -U_BOUND], dtype=np.float32)
    high = np.asarray([U_BOUND, U_BOUND, U_BOUND], dtype=np.float32)
    return low, high


def _logit(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, 1.0 / (1.0 + np.exp(U_BOUND)), 1.0 / (1.0 + np.exp(-U_BOUND)))
    return np.log(q / (1.0 - q))


def _sigmoid(u: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(u, dtype=np.float64)))


def _amp_to_u(amplitude_hz: np.ndarray) -> np.ndarray:
    log_amp = np.log(np.clip(amplitude_hz, np.exp(LOG_A_MIN), np.exp(LOG_A_MAX)))
    return _logit((log_amp - LOG_A_MIN) / (LOG_A_MAX - LOG_A_MIN))


def _u_to_amp(u: np.ndarray) -> np.ndarray:
    return np.exp(LOG_A_MIN + _sigmoid(u) * (LOG_A_MAX - LOG_A_MIN))


def _period_to_u(log_period_days: np.ndarray, config: SBISimulatorConfig) -> np.ndarray:
    return _logit((log_period_days - config.log_period_min) / (config.log_period_max - config.log_period_min))


def _u_to_log_period(u: np.ndarray, prior: dict[str, Any]) -> np.ndarray:
    return float(prior["log_period_min"]) + _sigmoid(u) * (float(prior["log_period_max"]) - float(prior["log_period_min"]))


def _phi_to_u(phi_rad: np.ndarray) -> np.ndarray:
    return _logit(np.asarray(phi_rad, dtype=np.float64) / (2.0 * np.pi))


def _u_to_phi(u: np.ndarray) -> np.ndarray:
    return _sigmoid(u) * (2.0 * np.pi)


def _amp_threshold_u(threshold_hz: float = 50.0) -> float:
    return float(_amp_to_u(np.asarray([threshold_hz], dtype=np.float64))[0])


def _theta_to_training_space(theta: np.ndarray) -> np.ndarray:
    transformed = np.asarray(theta, dtype=np.float32).copy()
    config = SBISimulatorConfig(a_min_hz=1.0)
    transformed[:, 0] = _amp_to_u(np.maximum(transformed[:, 0], 1.0))
    transformed[:, 1] = _period_to_u(transformed[:, 1], config)
    transformed[:, 2] = _phi_to_u(transformed[:, 2])
    return transformed


def _sample_continuous_theta(simulator: Th229SBISimulator, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cfg = simulator.config
    theta = np.zeros((n, 3), dtype=np.float32)
    theta[:, 0] = np.exp(rng.uniform(np.log(1.0), np.log(cfg.a_max_hz), size=n))
    theta[:, 1] = rng.uniform(cfg.log_period_min, cfg.log_period_max, size=n)
    theta[:, 2] = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return theta, simulator.simulate(theta, rng)


def train_sbi_npe(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    num_simulations: int = 100_000,
    seed: int = DEFAULT_TRAINING_SEED,
    max_num_epochs: int = 12,
    training_batch_size: int = 512,
    hidden_features: int = 64,
    num_transforms: int = 4,
    force: bool = False,
    simulator_null_model: Any | None = None,
    simulator_null_model_name: str | None = None,
    version_label: str = "v2",
) -> dict[str, Any]:
    if checkpoint_path.exists() and not force:
        return load_checkpoint(checkpoint_path)["metadata"]

    torch, SNPE, posterior_nn, BoxUniform = _import_sbi()
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu"

    simulator = Th229SBISimulator(
        null_model=simulator_null_model,
        null_model_name=simulator_null_model_name,
        config=SBISimulatorConfig(a_min_hz=1.0, pi_null=0.0, seed=seed),
    )
    theta_np, x_np = _sample_continuous_theta(simulator, num_simulations, seed)
    theta = torch.as_tensor(_theta_to_training_space(theta_np), dtype=torch.float32, device=device)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    low, high = _prior_bounds(simulator.config)
    prior = BoxUniform(
        low=torch.as_tensor(low, dtype=torch.float32, device=device),
        high=torch.as_tensor(high, dtype=torch.float32, device=device),
        device=device,
    )
    density_builder = posterior_nn(
        "nsf",
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        num_bins=8,
        z_score_theta="independent",
        z_score_x="independent",
    )
    inference = SNPE(
        prior=prior,
        density_estimator=density_builder,
        device=device,
        show_progress_bars=False,
        summary_writer=None,
    )

    start = time.perf_counter()
    density_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=training_batch_size,
        validation_fraction=0.1,
        stop_after_epochs=5,
        max_num_epochs=max_num_epochs,
        show_train_summary=False,
    )
    posterior = inference.build_posterior(density_estimator, prior=prior, sample_with="direct")
    runtime_sec = time.perf_counter() - start

    metadata = {
        "baseline": BASELINE_NAME,
        "sbi_version_label": version_label,
        "method": "SNPE-C/NPE neural spline flow via sbi; continuous log-uniform amplitude prior with bounded reparameterization",
        "num_simulations": int(num_simulations),
        "training_seed": int(seed),
        "device": device,
        "training_runtime_sec": runtime_sec,
        "sbi_version": _package_version("sbi"),
        "torch_version": _package_version("torch"),
        "nflows_version": _package_version("nflows"),
        "density_estimator": {
            "model": "nsf",
            "hidden_features": hidden_features,
            "num_transforms": num_transforms,
            "num_bins": 8,
            "theta_training_space": ["u_log_amp_bounded", "u_log_period_bounded", "u_phi_bounded"],
            "u_bound": U_BOUND,
            "amplitude_transform": "u=logit((log(A_hz)-log(1))/(log(6500)-log(1)))",
            "period_transform": "u=logit((log_period_days-log(7))/(log(365)-log(7)))",
            "phi_transform": "u=logit(phi_rad/(2*pi))",
            "detection_threshold_training_space": _amp_threshold_u(50.0),
        },
        "simulator": simulator.metadata(),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "posterior": posterior,
            "metadata": metadata,
        },
        checkpoint_path,
    )
    return metadata


def run_continuous_prior_diagnostic(
    output_path: Path = PROJECT_ROOT / "results" / "sbi_npe_continuous_prior_diagnostic.json",
    checkpoint_path: Path = PROJECT_ROOT / "models" / "sbi_npe_continuous_prior_smoke.pt",
    num_simulations: int = 5_000,
    seed: int = DEFAULT_TRAINING_SEED + 303,
    max_num_epochs: int = 3,
    num_posterior_samples: int = 64,
    force_train: bool = False,
) -> dict[str, Any]:
    torch, SNPE, posterior_nn, BoxUniform = _import_sbi()
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu"
    if checkpoint_path.exists() and not force_train:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    else:
        simulator = Th229SBISimulator(config=SBISimulatorConfig(seed=seed, pi_null=0.5))
        rng = np.random.default_rng(seed)
        class_label = (rng.random(num_simulations) >= 0.5).astype(np.float32)
        log_amp = rng.uniform(np.log(1.0), np.log(simulator.config.a_max_hz), size=num_simulations).astype(np.float32)
        theta_physical = np.column_stack(
            [
                np.exp(log_amp) * class_label,
                rng.uniform(simulator.config.log_period_min, simulator.config.log_period_max, size=num_simulations),
                rng.uniform(0.0, 2.0 * np.pi, size=num_simulations),
            ]
        ).astype(np.float32)
        x_np = simulator.simulate(theta_physical, rng)
        theta_np = np.column_stack([log_amp, theta_physical[:, 1], theta_physical[:, 2], class_label]).astype(np.float32)
        prior = BoxUniform(
            low=torch.tensor([0.0, simulator.config.log_period_min, 0.0, 0.0], dtype=torch.float32, device=device),
            high=torch.tensor([np.log(simulator.config.a_max_hz), simulator.config.log_period_max, 2.0 * np.pi, 1.0], dtype=torch.float32, device=device),
            device=device,
        )
        density_builder = posterior_nn("nsf", hidden_features=64, num_transforms=4, num_bins=8)
        inference = SNPE(prior=prior, density_estimator=density_builder, device=device, show_progress_bars=False, summary_writer=None)
        density_estimator = inference.append_simulations(
            torch.as_tensor(theta_np, dtype=torch.float32, device=device),
            torch.as_tensor(x_np, dtype=torch.float32, device=device),
        ).train(
            training_batch_size=512,
            validation_fraction=0.1,
            stop_after_epochs=3,
            max_num_epochs=max_num_epochs,
            show_train_summary=False,
        )
        posterior = inference.build_posterior(density_estimator, prior=prior, sample_with="direct")
        checkpoint = {
            "posterior": posterior,
            "metadata": {
                "baseline": "sbi_npe_continuous_prior_diagnostic",
                "num_simulations": int(num_simulations),
                "training_seed": int(seed),
                "theta_training_space": ["log_A_slab_hz", "log_period_days", "phi_rad", "class_label"],
                "score": "P(class_label > 0.5 and A_hz > 50 | x)",
                "sbi_version": _package_version("sbi"),
                "torch_version": _package_version("torch"),
                "simulator": simulator.metadata(),
            },
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

    catalog = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "benchmark_catalog.csv")
    residuals = np.load(PROJECT_ROOT / "data" / "interim" / "benchmark_arrays.npz")["observed_residual_hz"].astype(np.float32)
    posterior = checkpoint["posterior"]
    torch.manual_seed(DEFAULT_SCORING_SEED + 303)
    scores = np.zeros(residuals.shape[0], dtype=np.float64)
    for idx, row in enumerate(residuals):
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*reject_outside_prior=False.*")
            samples = posterior.sample(
                (num_posterior_samples,),
                x=torch.as_tensor(row, dtype=torch.float32),
                show_progress_bars=False,
                reject_outside_prior=False,
            )
        scores[idx] = float(torch.mean(((samples[:, 3] > 0.5) & (samples[:, 0] > np.log(50.0))).to(torch.float32)).cpu().item())
    test_mask = catalog["split"].eq("test").to_numpy()
    y = catalog.loc[test_mask, "label"].to_numpy(dtype=int)
    result = {
        "diagnostic": "continuous_amplitude_prior_with_class_label",
        "checkpoint_path": _display_path(checkpoint_path),
        "num_simulations": int(num_simulations),
        "num_posterior_samples": int(num_posterior_samples),
        "test_auroc": float(roc_auc_score(y, scores[test_mask])),
        "test_average_precision": float(average_precision_score(y, scores[test_mask])),
        "score": "P(class_label > 0.5 and A_hz > 50 | x)",
        "metadata": checkpoint["metadata"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_checkpoint(checkpoint_path: Path = DEFAULT_CHECKPOINT) -> dict[str, Any]:
    torch, _, _, _ = _import_sbi()
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _sample_posterior(
    posterior: Any,
    row: np.ndarray,
    num_posterior_samples: int,
) -> Any:
    torch, _, _, _ = _import_sbi()
    x = torch.as_tensor(row, dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*reject_outside_prior=False.*")
        return posterior.sample(
            (num_posterior_samples,),
            x=x,
            show_progress_bars=False,
            reject_outside_prior=False,
        )


def score_residuals(
    residuals: np.ndarray,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    num_posterior_samples: int = 256,
    seed: int = DEFAULT_SCORING_SEED,
) -> np.ndarray:
    torch, _, _, _ = _import_sbi()
    checkpoint = load_checkpoint(checkpoint_path)
    posterior = checkpoint["posterior"]
    torch.manual_seed(seed)
    metadata = checkpoint["metadata"]
    threshold = float(metadata.get("density_estimator", {}).get("detection_threshold_training_space", np.log1p(50.0)))
    scores = np.zeros(residuals.shape[0], dtype=np.float64)
    with torch.no_grad():
        for idx, row in enumerate(np.asarray(residuals, dtype=np.float32)):
            samples = _sample_posterior(posterior, row, num_posterior_samples)
            scores[idx] = float(torch.mean((samples[:, 0] > threshold).to(torch.float32)).cpu().item())
    return scores


def out_of_prior_diagnostics(
    residuals: np.ndarray,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    num_posterior_samples: int = 1000,
    seed: int = DEFAULT_SCORING_SEED + 101,
) -> dict[str, Any]:
    torch, _, _, _ = _import_sbi()
    checkpoint = load_checkpoint(checkpoint_path)
    posterior = checkpoint["posterior"]
    prior = checkpoint["metadata"]["simulator"]["prior"]
    version = checkpoint["metadata"].get("sbi_version_label", "v1")
    torch.manual_seed(seed)
    fractions = {"A_hz": [], "log_period_days": [], "phi_rad": [], "any_axis": []}
    for row in np.asarray(residuals, dtype=np.float32):
        with torch.no_grad():
            samples = _sample_posterior(posterior, row, num_posterior_samples).cpu().numpy()
        if version == "v2" or version == "v3":
            a_hz = _u_to_amp(samples[:, 0])
            log_period = _u_to_log_period(samples[:, 1], prior)
            phi = _u_to_phi(samples[:, 2])
        else:
            a_hz = np.expm1(samples[:, 0])
            log_period = samples[:, 1]
            phi = samples[:, 2]
        bad_a = (a_hz < float(prior.get("a_min_hz", 0.0))) | (a_hz > float(prior["a_max_hz"]))
        bad_period = (log_period < float(prior["log_period_min"])) | (log_period > float(prior["log_period_max"]))
        bad_phi = (phi < 0.0) | (phi > 2.0 * np.pi)
        fractions["A_hz"].append(float(np.mean(bad_a)))
        fractions["log_period_days"].append(float(np.mean(bad_period)))
        fractions["phi_rad"].append(float(np.mean(bad_phi)))
        fractions["any_axis"].append(float(np.mean(bad_a | bad_period | bad_phi)))
    stats = {}
    for axis, values in fractions.items():
        arr = np.asarray(values, dtype=np.float64)
        stats[axis] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
        }
    return {
        "baseline": BASELINE_NAME,
        "checkpoint_path": _display_path(checkpoint_path),
        "num_posterior_samples": int(num_posterior_samples),
        "n_examples": int(len(residuals)),
        "posterior_sampling": "direct with reject_outside_prior=False",
        "stats": stats,
    }


def _threshold(scores: np.ndarray, catalog: pd.DataFrame, split: str = "validation") -> float:
    mask = catalog["split"].eq(split).to_numpy() & catalog["label"].eq(0).to_numpy()
    return float(np.quantile(scores[mask], 0.95))


def _comparison_row(scores: np.ndarray, threshold: float, catalog: pd.DataFrame, runtime_sec: float) -> dict[str, Any]:
    test_mask = catalog["split"].eq("test").to_numpy()
    y = catalog.loc[test_mask, "label"].to_numpy(dtype=int)
    s = scores[test_mask]
    signal_mask = y == 1
    return {
        "baseline": BASELINE_NAME,
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


def _a95_with_uncertainty(
    scores: np.ndarray,
    catalog: pd.DataFrame,
    config: BenchmarkConfig,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.global_seed + 900_000 + sum(ord(ch) for ch in BASELINE_NAME))
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
                & np.isclose(catalog["frequency_hz"].to_numpy(dtype=np.float64), frequency_hz)
                & np.isclose(catalog["amplitude_hz"].to_numpy(dtype=np.float64), amplitude_hz)
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
                rates.append(float(np.mean(rng.choice(values, size=len(values), replace=True) >= threshold)))
            draws.append(_interpolate_a95(config.amplitudes_hz, rates))
        draw_array = np.asarray(draws, dtype=np.float64)
        finite = draw_array[np.isfinite(draw_array)]
        rows.append(
            {
                "baseline": BASELINE_NAME,
                "period_days": period_days,
                "frequency_hz": frequency_hz,
                "a95_hz": point,
                "a95_bootstrap_median_hz": float(np.median(finite)) if len(finite) else float("nan"),
                "a95_lower_hz": float(np.percentile(finite, 16)) if len(finite) else float("nan"),
                "a95_upper_hz": float(np.percentile(finite, 84)) if len(finite) else float("nan"),
                "a95_std_hz": float(np.std(finite)) if len(finite) else float("nan"),
                "bootstrap_replicates": n_bootstrap,
                "finite_bootstrap_replicates": int(len(finite)),
                "n_test_phases_per_cell": int(config.test_phase_count),
                "uncertainty_source": "validation-null threshold resampling plus test-phase resampling within each amplitude cell",
            }
        )
    return pd.DataFrame(rows)


def _representative_rows(scores: np.ndarray, catalog: pd.DataFrame, threshold: float, a95: pd.DataFrame) -> pd.DataFrame:
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
                "baseline": BASELINE_NAME,
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


def _merge_append(path: Path, row_df: pd.DataFrame, key_column: str = "baseline") -> pd.DataFrame:
    if path.exists():
        base = pd.read_csv(path)
        base = base.loc[~base[key_column].eq(BASELINE_NAME)].copy()
        merged = pd.concat([base, row_df], ignore_index=True)
    else:
        merged = row_df
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged


def _plot_a95_with_sbi(a95_sbi: pd.DataFrame) -> Path:
    figure_path = PROJECT_ROOT / "results" / "figures" / "final_a95_vs_frequency_with_uncertainty.png"
    pieces = []
    core_path = PROJECT_ROOT / "results" / "tables" / "a95_vs_frequency_with_uncertainty.csv"
    rf_path = PROJECT_ROOT / "results" / "tables" / "rf_a95_vs_frequency_with_uncertainty.csv"
    if core_path.exists():
        pieces.append(pd.read_csv(core_path))
    if rf_path.exists():
        pieces.append(pd.read_csv(rf_path))
    pieces.append(a95_sbi)
    frame = pd.concat(pieces, ignore_index=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
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


def run_sbi_npe(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    num_simulations: int = 100_000,
    force_train: bool = False,
    max_num_epochs: int = 12,
    num_posterior_samples: int = 256,
    n_bootstrap: int = 500,
    seed: int = DEFAULT_TRAINING_SEED,
    output_tag: str = BASELINE_NAME,
    simulator_null_model: Any | None = None,
    simulator_null_model_name: str | None = None,
    version_label: str = "v2",
) -> dict[str, Any]:
    train_metadata = train_sbi_npe(
        checkpoint_path=checkpoint_path,
        num_simulations=num_simulations,
        seed=seed,
        max_num_epochs=max_num_epochs,
        force=force_train,
        simulator_null_model=simulator_null_model,
        simulator_null_model_name=simulator_null_model_name,
        version_label=version_label,
    )
    catalog = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "benchmark_catalog.csv")
    arrays = np.load(PROJECT_ROOT / "data" / "interim" / "benchmark_arrays.npz")
    residuals = np.asarray(arrays["observed_residual_hz"], dtype=np.float32)

    start = time.perf_counter()
    scores = score_residuals(
        residuals,
        checkpoint_path=checkpoint_path,
        num_posterior_samples=num_posterior_samples,
        seed=DEFAULT_SCORING_SEED,
    )
    scoring_runtime = time.perf_counter() - start
    threshold = _threshold(scores, catalog)
    comparison = pd.DataFrame([_comparison_row(scores, threshold, catalog, scoring_runtime)])
    config = BenchmarkConfig()
    a95 = _a95_with_uncertainty(scores, catalog, config, n_bootstrap=n_bootstrap)
    representative = _representative_rows(scores, catalog, threshold, a95)
    test_mask = catalog["split"].eq("test").to_numpy()
    oop = out_of_prior_diagnostics(
        residuals[test_mask],
        checkpoint_path=checkpoint_path,
        num_posterior_samples=max(1000, num_posterior_samples),
    )

    results_dir = PROJECT_ROOT / "results"
    tables_dir = results_dir / "tables"
    scores_path = tables_dir / f"{output_tag}_predictions.csv"
    output_scores = catalog.copy()
    output_scores["score_sbi_npe"] = scores
    output_scores.to_csv(scores_path, index=False)
    comparison_path = tables_dir / f"{output_tag}_baseline_comparison.csv"
    a95_path = tables_dir / f"{output_tag}_a95_vs_frequency_with_uncertainty.csv"
    representative_path = tables_dir / f"{output_tag}_representative_frequency_table.csv"
    oop_path = results_dir / f"{output_tag}_oop_diagnostics.json"
    comparison.to_csv(comparison_path, index=False)
    a95.to_csv(a95_path, index=False)
    representative.to_csv(representative_path, index=False)
    oop_path.write_text(json.dumps(oop, indent=2), encoding="utf-8")

    comparison_ml_path = tables_dir / "baseline_comparison_with_ml.csv"
    representative_ml_path = tables_dir / "final_representative_frequency_baseline_table_with_ml.csv"
    if output_tag == BASELINE_NAME:
        comparison_source = tables_dir / "baseline_comparison_with_rf.csv"
        representative_source = tables_dir / "final_representative_frequency_baseline_table_with_rf.csv"
        if comparison_source.exists():
            shutil.copyfile(comparison_source, comparison_ml_path)
        if representative_source.exists():
            shutil.copyfile(representative_source, representative_ml_path)
        comparison_ml = _merge_append(comparison_ml_path, comparison)
        representative_ml = _merge_append(representative_ml_path, representative)
        _plot_a95_with_sbi(a95)
    else:
        comparison_ml = pd.DataFrame()
        representative_ml = pd.DataFrame()

    result = {
        **comparison.iloc[0].to_dict(),
        "checkpoint_path": _display_path(checkpoint_path),
        "training_metadata": train_metadata,
        "num_posterior_samples": int(num_posterior_samples),
        "posterior_sampling": "direct with reject_outside_prior=False to avoid low-acceptance stalls; score uses amplitude marginal only",
        "n_bootstrap": int(n_bootstrap),
        "predictions_path": _display_path(scores_path),
        "comparison_path": _display_path(comparison_path),
        "a95_path": _display_path(a95_path),
        "representative_table_path": _display_path(representative_path),
        "out_of_prior_diagnostics_path": _display_path(oop_path),
        "out_of_prior_median_any_axis": oop["stats"]["any_axis"]["median"],
        "comparison_with_ml_path": _display_path(comparison_ml_path),
        "representative_with_ml_path": _display_path(representative_ml_path),
        "comparison_with_ml_rows": int(len(comparison_ml)),
        "representative_with_ml_rows": int(len(representative_ml)),
    }
    result_path = results_dir / f"{output_tag}_results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def predict(inputs: np.ndarray, checkpoint_path: Path = DEFAULT_CHECKPOINT, num_posterior_samples: int = 256) -> np.ndarray:
    arr = np.asarray(inputs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 55:
        raise ValueError(f"SBI NPE expects length-55 residual vectors, got shape {arr.shape}")
    return score_residuals(arr, checkpoint_path=checkpoint_path, num_posterior_samples=num_posterior_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the Th229-ScanBench SBI NPE baseline.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-simulations", type=int, default=int(os.environ.get("SBI_NUM_SIMULATIONS", "100000")))
    parser.add_argument("--max-epochs", type=int, default=int(os.environ.get("SBI_MAX_EPOCHS", "12")))
    parser.add_argument("--posterior-samples", type=int, default=int(os.environ.get("SBI_POSTERIOR_SAMPLES", "256")))
    parser.add_argument("--bootstrap", type=int, default=int(os.environ.get("SBI_BOOTSTRAP", "500")))
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--output-tag", default=BASELINE_NAME)
    parser.add_argument("--simulator-null", choices=["parametric", "flow"], default="parametric")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--continuous-prior-diagnostic", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    simulator_null_model = None
    simulator_null_model_name = None
    version_label = "v2"
    if args.simulator_null == "flow":
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from null_models.normalizing_flow import fit_flow_null
        from sbi_npe_simulator import load_primary_peak_b

        simulator_null_model = fit_flow_null(load_primary_peak_b())
        simulator_null_model_name = simulator_null_model.name
        version_label = "v3"
    if args.continuous_prior_diagnostic:
        result = run_continuous_prior_diagnostic(force_train=args.force_train)
    else:
        result = run_sbi_npe(
            checkpoint_path=checkpoint,
            num_simulations=args.num_simulations,
            force_train=args.force_train,
            max_num_epochs=args.max_epochs,
            num_posterior_samples=args.posterior_samples,
            n_bootstrap=args.bootstrap,
        seed=args.seed,
        output_tag=args.output_tag,
        simulator_null_model=simulator_null_model,
        simulator_null_model_name=simulator_null_model_name,
        version_label=version_label,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
