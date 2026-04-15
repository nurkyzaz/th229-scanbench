from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import PeriodogramResult, WeightedLeastSquaresPeriodogram
from .null_models import FittedNullModel


class HierarchicalSinusoidModel(WeightedLeastSquaresPeriodogram):
    """Shared sinusoid with crystal-specific offsets and null-model jitter."""

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        frequency_grid_hz: np.ndarray,
        null_model: FittedNullModel,
    ) -> "HierarchicalSinusoidModel":
        sigma = null_model.effective_sigma(df)
        return cls(
            df["seconds_since_first_observation"].to_numpy(dtype=np.float64),
            sigma,
            frequency_grid_hz,
            cls.target_offset_design(df["target"].to_numpy()),
        )

    def score(self, y: np.ndarray) -> PeriodogramResult:
        result = super().score(y)
        # Convert weighted RSS improvement into a log-likelihood-ratio-like score.
        return PeriodogramResult(
            score=float(0.5 * result.score),
            pred_frequency_hz=result.pred_frequency_hz,
            pred_amplitude_hz=result.pred_amplitude_hz,
            periodogram=0.5 * result.periodogram,
        )

