from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(data: Any, path: Path) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def unix_seconds_from_timestamps(series: pd.Series) -> np.ndarray:
    if not isinstance(series.dtype, pd.DatetimeTZDtype):
        raise TypeError("Expected timezone-aware datetime series.")
    return series.astype("int64").to_numpy(dtype=np.float64) / 1e9


def mjd_from_timestamps(series: pd.Series) -> np.ndarray:
    return unix_seconds_from_timestamps(series) / 86400.0 + 40587.0


def weighted_quadratic_fit(
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, Any]:
    weights = 1.0 / np.maximum(sigma, 1.0) ** 2
    y_offset = float(np.average(y, weights=weights))
    y_centered = y - y_offset
    design = np.column_stack([x**2, x, np.ones_like(x)])
    lhs = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * y_centered)
    beta = np.linalg.solve(lhs, rhs)
    covariance = np.linalg.inv(lhs)
    fitted = design @ beta + y_offset
    residual = y - fitted
    a, b, c_centered = beta
    t0 = float(-b / (2.0 * a)) if abs(a) > 1e-20 else None
    return {
        "coefficients_hz": [float(a), float(b), float(c_centered + y_offset)],
        "covariance": covariance.tolist(),
        "weighted_rms_hz": float(np.sqrt(np.average(residual**2, weights=weights))),
        "turning_point_temperature_k": t0,
        "y_offset_hz": y_offset,
    }


def evaluate_quadratic(coefficients: list[float], x: np.ndarray) -> np.ndarray:
    a, b, c = coefficients
    return a * x**2 + b * x + c


def weighted_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, float]:
    weights = 1.0 / np.maximum(sigma, 1.0) ** 2
    design = np.column_stack([x, np.ones_like(x)])
    lhs = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * y)
    beta = np.linalg.solve(lhs, rhs)
    covariance = np.linalg.inv(lhs)
    fitted = design @ beta
    residual = y - fitted
    chi2 = float(np.sum(weights * residual**2))
    dof = max(len(x) - design.shape[1], 1)
    return {
        "slope": float(beta[0]),
        "intercept": float(beta[1]),
        "slope_stderr": float(np.sqrt(covariance[0, 0])),
        "intercept_stderr": float(np.sqrt(covariance[1, 1])),
        "weighted_rms": float(np.sqrt(np.average(residual**2, weights=weights))),
        "reduced_chi2": chi2 / dof,
    }


def empirical_quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))
