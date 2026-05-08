from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from th229_bench.null_models import assign_null_fit_split  # noqa: E402


@dataclass
class CrystalGMMNullModel:
    name: str
    gmms: dict[str, GaussianMixture]
    fallback_sigma: dict[str, float]
    metadata: dict[str, Any]

    distribution: str = "crystal_gmm"

    @property
    def parameters(self) -> dict[str, Any]:
        return self.metadata

    def effective_sigma(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray([self.fallback_sigma.get(str(t), 1.0) for t in df["target"]], dtype=np.float64)

    def sample(self, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        out = np.zeros(len(df), dtype=np.float64)
        for target, index in df.groupby("target").groups.items():
            gmm = self.gmms[str(target)]
            sampled, _ = gmm.sample(len(index))
            # sklearn uses its internal RNG; shuffle through provided RNG for deterministic row order.
            values = sampled[:, 0]
            rng.shuffle(values)
            out[df.index.get_indexer(index)] = values
        return out

    def log_prob(self, residual_hz: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        residual = np.asarray(residual_hz, dtype=np.float64)
        out = np.zeros(len(df), dtype=np.float64)
        for target, index in df.groupby("target").groups.items():
            gmm = self.gmms[str(target)]
            loc = df.index.get_indexer(index)
            out[loc] = gmm.score_samples(residual[loc, None])
        return out

    def nll(self, df: pd.DataFrame) -> float:
        return float(-np.sum(self.log_prob(df["residual_hz"].to_numpy(dtype=np.float64), df)))

    def cdf(self, values: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        out = np.zeros(len(values), dtype=np.float64)
        target = str(rows["target"].iloc[0])
        gmm = self.gmms[target]
        weights = gmm.weights_
        means = gmm.means_[:, 0]
        std = np.sqrt(gmm.covariances_.reshape(-1))
        for idx, value in enumerate(values):
            out[idx] = float(np.sum(weights * norm.cdf(value, loc=means, scale=std)))
        return out


def fit_gmm_null(primary_peak_b: pd.DataFrame, n_components: int = 3, seed: int = 22902671) -> CrystalGMMNullModel:
    split = assign_null_fit_split(primary_peak_b)
    train = split.loc[split["null_fit_split"].eq("train")].copy()
    gmms: dict[str, GaussianMixture] = {}
    fallback_sigma: dict[str, float] = {}
    metadata = {"model": "crystal_gmm", "n_components": n_components, "seed": seed, "trained_rows": int(len(train))}
    for offset, (target, group) in enumerate(train.groupby("target")):
        values = group["residual_hz"].to_numpy(dtype=np.float64)[:, None]
        components = min(n_components, len(values))
        gmm = GaussianMixture(
            n_components=components,
            covariance_type="full",
            reg_covar=1e-3,
            random_state=seed + offset,
            max_iter=500,
        )
        gmm.fit(values)
        gmms[str(target)] = gmm
        fallback_sigma[str(target)] = float(max(np.std(values[:, 0]), 1.0))
    return CrystalGMMNullModel("crystal_gmm_3comp", gmms, fallback_sigma, metadata)


def save_gmm_model(model: CrystalGMMNullModel, path: Path = PROJECT_ROOT / "models" / "null_gmm_v1.joblib") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    (path.with_suffix(".json")).write_text(json.dumps(model.metadata, indent=2), encoding="utf-8")
