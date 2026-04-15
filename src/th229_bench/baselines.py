from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PeriodogramResult:
    score: float
    pred_frequency_hz: float
    pred_amplitude_hz: float
    periodogram: np.ndarray


class WeightedLeastSquaresPeriodogram:
    """Weighted sinusoid search with a configurable offset design."""

    def __init__(self, times_sec: np.ndarray, sigma_hz: np.ndarray, frequency_grid_hz: np.ndarray, offset_design: np.ndarray):
        self.times_sec = np.asarray(times_sec, dtype=np.float64)
        self.sigma_hz = np.asarray(sigma_hz, dtype=np.float64)
        self.frequency_grid_hz = np.asarray(frequency_grid_hz, dtype=np.float64)
        self.offset_design = np.asarray(offset_design, dtype=np.float64)
        self.weights = 1.0 / np.maximum(self.sigma_hz, 1.0) ** 2

    @staticmethod
    def target_offset_design(targets: np.ndarray) -> np.ndarray:
        return pd.get_dummies(pd.Series(targets).astype(str), dtype=float).to_numpy()

    @staticmethod
    def global_offset_design(n_rows: int) -> np.ndarray:
        return np.ones((n_rows, 1), dtype=np.float64)

    def _solve(self, y: np.ndarray, design: np.ndarray) -> tuple[float, np.ndarray]:
        lhs = design.T @ (self.weights[:, None] * design)
        rhs = design.T @ (self.weights * y)
        beta = np.linalg.solve(lhs, rhs)
        residual = y - design @ beta
        rss = float(np.sum(self.weights * residual**2))
        return rss, beta

    def periodogram(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=np.float64)
        rss_null, _ = self._solve(y, self.offset_design)
        powers = np.zeros(len(self.frequency_grid_hz), dtype=np.float64)
        amplitudes = np.zeros(len(self.frequency_grid_hz), dtype=np.float64)
        for idx, frequency_hz in enumerate(self.frequency_grid_hz):
            phase = 2.0 * np.pi * frequency_hz * self.times_sec
            design = np.column_stack([self.offset_design, np.sin(phase), np.cos(phase)])
            rss_full, beta = self._solve(y, design)
            powers[idx] = max(0.0, rss_null - rss_full)
            amplitudes[idx] = float(np.sqrt(beta[-2] ** 2 + beta[-1] ** 2))
        return powers, amplitudes

    def score(self, y: np.ndarray) -> PeriodogramResult:
        powers, amplitudes = self.periodogram(y)
        best = int(np.argmax(powers))
        return PeriodogramResult(
            score=float(powers[best]),
            pred_frequency_hz=float(self.frequency_grid_hz[best]),
            pred_amplitude_hz=float(amplitudes[best]),
            periodogram=powers,
        )


class WeightedHarmonicRegression(WeightedLeastSquaresPeriodogram):
    @classmethod
    def from_frame(cls, df: pd.DataFrame, frequency_grid_hz: np.ndarray) -> "WeightedHarmonicRegression":
        return cls(
            df["seconds_since_first_observation"].to_numpy(dtype=np.float64),
            df["freq_unc_hz"].to_numpy(dtype=np.float64),
            frequency_grid_hz,
            cls.target_offset_design(df["target"].to_numpy()),
        )


class GeneralizedLombScargle(WeightedLeastSquaresPeriodogram):
    @classmethod
    def from_frame(cls, df: pd.DataFrame, frequency_grid_hz: np.ndarray) -> "GeneralizedLombScargle":
        return cls(
            df["seconds_since_first_observation"].to_numpy(dtype=np.float64),
            df["freq_unc_hz"].to_numpy(dtype=np.float64),
            frequency_grid_hz,
            cls.global_offset_design(len(df)),
        )


def metric_definitions() -> dict[str, str]:
    return {
        "roc_auc": "Area under the ROC curve on the held-out synthetic test set.",
        "average_precision": "Area under the precision-recall curve on the held-out synthetic test set.",
        "validation_fpr5_threshold": "Score threshold set to the 95th percentile of validation null scores.",
        "tpr_at_validation_fpr_5pct": "Fraction of held-out signal examples above a threshold calibrated to the 95th percentile of validation null scores.",
        "test_null_false_positive_rate": "Fraction of held-out null examples above the validation-calibrated threshold.",
        "a95_hz": "Smallest injected amplitude with interpolated detection rate at least 95% at the validation-calibrated 5% false-positive threshold.",
        "a95_lower_hz": "16th percentile bootstrap estimate for A95 from validation-threshold and test-phase resampling.",
        "a95_upper_hz": "84th percentile bootstrap estimate for A95 from validation-threshold and test-phase resampling.",
        "observed_empirical_p_value": "Bootstrap null tail probability for the observed JILA residual series under the specified null model.",
        "source_validation_fpr5_threshold": "Generalization-test threshold copied from source-grid validation null scores.",
        "target_test_null_false_positive_rate": "Generalization-test null false-positive rate on off-grid held-out-period test examples.",
    }
