from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zuko

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from th229_bench.null_models import assign_null_fit_split  # noqa: E402

TARGETS = ("C10", "C13", "X2")


def _context_frame(df: pd.DataFrame, stats: dict[str, float]) -> np.ndarray:
    target = df["target"].astype(str)
    one_hot = np.column_stack([(target == name).to_numpy(dtype=np.float32) for name in TARGETS])
    log_sigma = np.log(np.maximum(df["freq_unc_hz"].to_numpy(dtype=np.float64), 1.0))
    temp = df["temp_k"].to_numpy(dtype=np.float64)
    ctx = np.column_stack(
        [
            one_hot,
            (log_sigma - stats["log_sigma_mean"]) / max(stats["log_sigma_std"], 1e-6),
            (temp - stats["temp_mean"]) / max(stats["temp_std"], 1e-6),
        ]
    )
    return ctx.astype(np.float32)


@dataclass
class ConditionalFlowNullModel:
    name: str
    flow: torch.nn.Module
    stats: dict[str, float]
    metadata: dict[str, Any]
    distribution: str = "conditional_neural_spline_flow"

    @property
    def parameters(self) -> dict[str, Any]:
        return self.metadata

    def _context(self, df: pd.DataFrame) -> torch.Tensor:
        return torch.as_tensor(_context_frame(df, self.stats), dtype=torch.float32)

    def _standardize(self, residual: np.ndarray) -> np.ndarray:
        return ((np.asarray(residual, dtype=np.float64) - self.stats["residual_mean"]) / self.stats["residual_std"]).astype(np.float32)

    def _unstandardize(self, z: np.ndarray) -> np.ndarray:
        return z * self.stats["residual_std"] + self.stats["residual_mean"]

    def effective_sigma(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), float(self.stats["residual_std"]), dtype=np.float64)

    def log_prob(self, residual_hz: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        self.flow.eval()
        z = torch.as_tensor(self._standardize(residual_hz)[:, None], dtype=torch.float32)
        context = self._context(df)
        with torch.no_grad():
            logp = self.flow(context).log_prob(z).cpu().numpy()
        return logp - np.log(self.stats["residual_std"])

    def nll(self, df: pd.DataFrame) -> float:
        return float(-np.sum(self.log_prob(df["residual_hz"].to_numpy(dtype=np.float64), df)))

    def sample(self, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        self.flow.eval()
        context = self._context(df)
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
        with torch.no_grad():
            z = self.flow(context).sample().cpu().numpy()[:, 0]
        return self._unstandardize(z)

    def cdf(self, values: np.ndarray, rows: pd.DataFrame, n_mc: int = 4096) -> np.ndarray:
        rng = np.random.default_rng(22902677 + len(rows))
        repeated = pd.concat([rows.iloc[[0]].copy()] * n_mc, ignore_index=True)
        samples = self.sample(repeated, rng)
        return np.asarray([np.mean(samples <= value) for value in values], dtype=np.float64)


def fit_flow_null(
    primary_peak_b: pd.DataFrame,
    seed: int = 22902672,
    max_epochs: int = 600,
    hidden_width: int = 64,
    transforms: int = 3,
) -> ConditionalFlowNullModel:
    split = assign_null_fit_split(primary_peak_b)
    train = split.loc[split["null_fit_split"].eq("train")].copy()
    residual = train["residual_hz"].to_numpy(dtype=np.float64)
    log_sigma = np.log(np.maximum(train["freq_unc_hz"].to_numpy(dtype=np.float64), 1.0))
    temp = train["temp_k"].to_numpy(dtype=np.float64)
    stats = {
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(max(np.std(residual), 1.0)),
        "log_sigma_mean": float(np.mean(log_sigma)),
        "log_sigma_std": float(max(np.std(log_sigma), 1e-6)),
        "temp_mean": float(np.mean(temp)),
        "temp_std": float(max(np.std(temp), 1e-6)),
    }
    torch.manual_seed(seed)
    flow = zuko.flows.NSF(features=1, context=5, transforms=transforms, hidden_features=[hidden_width, hidden_width])
    x = torch.as_tensor(((residual - stats["residual_mean"]) / stats["residual_std"])[:, None], dtype=torch.float32)
    context = torch.as_tensor(_context_frame(train, stats), dtype=torch.float32)
    optimizer = torch.optim.Adam(flow.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state = None
    best_loss = float("inf")
    patience = 80
    stale = 0
    for _ in range(max_epochs):
        optimizer.zero_grad()
        loss = -flow(context).log_prob(x).mean()
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss - 1e-5:
            best_loss = value
            best_state = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        flow.load_state_dict(best_state)
    metadata = {
        "model": "conditional_neural_spline_flow",
        "seed": seed,
        "trained_rows": int(len(train)),
        "transforms": transforms,
        "hidden_width": hidden_width,
        "best_train_nll_standardized": best_loss,
        "context": ["target_one_hot_C10_C13_X2", "log_freq_unc_hz_z", "temp_k_z"],
    }
    return ConditionalFlowNullModel("conditional_spline_flow", flow.cpu(), stats, metadata)


def save_flow_model(model: ConditionalFlowNullModel, path: Path = PROJECT_ROOT / "models" / "null_flow_v1.pt") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.flow.state_dict(),
            "stats": model.stats,
            "metadata": model.metadata,
        },
        path,
    )
    path.with_suffix(".json").write_text(json.dumps(model.metadata | {"stats": model.stats}, indent=2), encoding="utf-8")
