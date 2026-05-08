from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from th229_bench.benchmarking import BenchmarkConfig  # noqa: E402
from th229_bench.null_models import FittedNullModel, fit_null_models  # noqa: E402
from th229_bench.synthetic import SECONDS_PER_DAY  # noqa: E402


@dataclass(frozen=True)
class SBISimulatorConfig:
    a_min_hz: float = 50.0
    a_max_hz: float = 6500.0
    pi_null: float = 0.5
    min_period_days: float = 7.0
    max_period_days: float = 365.0
    seed: int = 22902651

    @property
    def log_period_min(self) -> float:
        return float(np.log(self.min_period_days))

    @property
    def log_period_max(self) -> float:
        return float(np.log(self.max_period_days))


def load_primary_peak_b() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "primary_peak_b.csv"
    frame = pd.read_csv(path)
    return frame.sort_values("seconds_since_first_observation").reset_index(drop=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def simulator_config_hash(metadata: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(metadata), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class Th229SBISimulator:
    """Forward simulator for SBI training; it never reads benchmark split examples."""

    def __init__(
        self,
        primary_df: pd.DataFrame | None = None,
        null_model: FittedNullModel | None = None,
        null_model_name: str | None = None,
        config: SBISimulatorConfig | None = None,
    ) -> None:
        self.config = config or SBISimulatorConfig()
        self.primary_df = (primary_df.copy() if primary_df is not None else load_primary_peak_b()).reset_index(drop=True)
        if len(self.primary_df) != 55:
            raise ValueError(f"SBI simulator requires the immutable 55 peak-b rows, got {len(self.primary_df)}")
        if null_model is None or null_model_name is None:
            _, null_models, default_name = fit_null_models(self.primary_df)
            null_model = null_models[default_name]
            null_model_name = default_name
        self.null_model = null_model
        self.null_model_name = null_model_name
        self.times_sec = self.primary_df["seconds_since_first_observation"].to_numpy(dtype=np.float64)
        self.formal_sigma_hz = self.primary_df["freq_unc_hz"].to_numpy(dtype=np.float64)
        self.targets = self.primary_df["target"].astype(str).to_numpy()

    def metadata(self) -> dict[str, Any]:
        benchmark_config = BenchmarkConfig()
        payload = {
            "simulator": "th229_sbi_npe_v1",
            "primary_rows": int(len(self.primary_df)),
            "covariate_columns": [
                "seconds_since_first_observation",
                "target",
                "freq_unc_hz",
                "temp_k",
            ],
            "periods_days_for_evaluation": list(benchmark_config.periods_days),
            "theta": ["A_hz", "log_period_days", "phi_rad"],
            "prior": asdict(self.config)
            | {
                "log_period_min": self.config.log_period_min,
                "log_period_max": self.config.log_period_max,
            },
            "null_model_name": self.null_model_name,
            "null_model_distribution": self.null_model.distribution,
            "null_model_parameters": self.null_model.parameters,
            "temperature_correction": "assumed already applied in primary_peak_b residual_hz",
        }
        payload["simulator_config_hash"] = simulator_config_hash(payload)
        return _jsonable(payload)

    def sample_theta(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be positive")
        cfg = self.config
        theta = np.zeros((n, 3), dtype=np.float32)
        is_signal = rng.random(n) >= cfg.pi_null
        theta[:, 0] = 0.0
        theta[is_signal, 0] = np.exp(
            rng.uniform(np.log(cfg.a_min_hz), np.log(cfg.a_max_hz), size=int(is_signal.sum()))
        )
        theta[:, 1] = rng.uniform(cfg.log_period_min, cfg.log_period_max, size=n)
        theta[:, 2] = rng.uniform(0.0, 2.0 * np.pi, size=n)
        return theta

    def _sample_noise(self, n: int, rng: np.random.Generator) -> np.ndarray:
        scale = self.null_model.effective_sigma(self.primary_df)
        if self.null_model.distribution == "gaussian":
            return rng.normal(loc=0.0, scale=scale, size=(n, len(scale))).astype(np.float32)
        if self.null_model.distribution == "student_t":
            degrees = float(self.null_model.parameters["student_t_df"])
            return student_t.rvs(degrees, loc=0.0, scale=scale, size=(n, len(scale)), random_state=rng).astype(np.float32)
        if self.null_model.distribution == "x2_gaussian_mixture":
            jitters = self.null_model.parameters["crystal_jitter_hz"]
            jitter = np.asarray([float(jitters.get(target, 0.0)) for target in self.targets], dtype=np.float64)
            core_scale = np.sqrt(self.formal_sigma_hz**2 + jitter**2)
            samples = rng.normal(loc=0.0, scale=core_scale, size=(n, len(core_scale)))
            is_x2 = self.targets == "X2"
            probability = float(self.null_model.parameters["x2_outlier_probability"])
            multiplier = float(self.null_model.parameters["x2_outlier_scale_multiplier"])
            replace = (rng.random((n, len(core_scale))) < probability) & is_x2[None, :]
            if np.any(replace):
                samples[replace] = rng.normal(loc=0.0, scale=np.broadcast_to(core_scale * multiplier, samples.shape)[replace])
            return samples.astype(np.float32)
        return np.vstack([self.null_model.sample(self.primary_df, rng) for _ in range(n)]).astype(np.float32)

    def simulate(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta[None, :]
        if theta.shape[1] != 3:
            raise ValueError(f"theta must have shape (n, 3), got {theta.shape}")
        amplitude = theta[:, 0]
        period_days = np.exp(theta[:, 1])
        frequency_hz = 1.0 / (period_days * SECONDS_PER_DAY)
        phase = theta[:, 2]
        signal = amplitude[:, None] * np.cos(2.0 * np.pi * frequency_hz[:, None] * self.times_sec[None, :] + phase[:, None])
        noise = self._sample_noise(theta.shape[0], rng)
        return (signal + noise).astype(np.float32)

    def simulate_training_batch(self, n: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        theta = self.sample_theta(n, rng)
        x = self.simulate(theta, rng)
        return theta, x


def build_default_simulator(seed: int = 22902651) -> Th229SBISimulator:
    # TODO(future): allow replacing the parametric null with null_models/normalizing_flow.py
    # after learned-null SBI is promoted as a separate reported variant.
    return Th229SBISimulator(config=SBISimulatorConfig(seed=seed))
