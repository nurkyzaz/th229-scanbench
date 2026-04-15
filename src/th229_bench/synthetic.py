from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SECONDS_PER_DAY = 86400.0


def pure_sinusoid(times_sec: np.ndarray, amplitude_hz: float, frequency_hz: float, phase_rad: float) -> np.ndarray:
    return amplitude_hz * np.cos(2.0 * np.pi * frequency_hz * times_sec + phase_rad)


def finite_coherence_sinusoid(
    times_sec: np.ndarray,
    amplitude_hz: float,
    frequency_hz: float,
    phase_rad: float,
    coherence_days: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sinusoid with independent random phase offsets in fixed coherence windows."""
    if coherence_days <= 0:
        raise ValueError("coherence_days must be positive.")
    elapsed_days = (times_sec - float(np.min(times_sec))) / SECONDS_PER_DAY
    windows = np.floor(elapsed_days / coherence_days).astype(int)
    unique_windows = np.unique(windows)
    phase_offsets = {window: rng.uniform(0.0, 2.0 * np.pi) for window in unique_windows}
    phases = np.asarray([phase_rad + phase_offsets[window] for window in windows], dtype=np.float64)
    return amplitude_hz * np.cos(2.0 * np.pi * frequency_hz * times_sec + phases)


def slow_linear_drift(times_sec: np.ndarray, slope_hz_per_day: float) -> np.ndarray:
    centered_days = (times_sec - np.mean(times_sec)) / SECONDS_PER_DAY
    return slope_hz_per_day * centered_days


def gaussian_transient(times_sec: np.ndarray, amplitude_hz: float, center_day: float, width_days: float) -> np.ndarray:
    elapsed_days = (times_sec - float(np.min(times_sec))) / SECONDS_PER_DAY
    width = max(width_days, 0.1)
    return amplitude_hz * np.exp(-0.5 * ((elapsed_days - center_day) / width) ** 2)


def signal_from_row(times_sec: np.ndarray, row: pd.Series, rng: np.random.Generator) -> np.ndarray:
    family = row["family"]
    if family == "pure_sinusoid":
        return pure_sinusoid(times_sec, row["amplitude_hz"], row["frequency_hz"], row["phase_rad"])
    if family == "finite_coherence_sinusoid":
        return finite_coherence_sinusoid(
            times_sec,
            row["amplitude_hz"],
            row["frequency_hz"],
            row["phase_rad"],
            row["coherence_days"],
            rng,
        )
    if family == "slow_linear_drift":
        return slow_linear_drift(times_sec, row["slope_hz_per_day"])
    if family == "gaussian_transient":
        return gaussian_transient(times_sec, row["amplitude_hz"], row["center_day"], row["width_days"])
    raise ValueError(f"Unknown injection family: {family}")


def injection_family_definitions() -> dict[str, dict[str, Any]]:
    return {
        "pure_sinusoid": {
            "equation": "nu(t) = A cos(omega t + phi)",
            "role": "primary benchmark task",
            "physics_note": "Motivated by time-varying fields that modulate the nuclear transition frequency.",
        },
        "finite_coherence_sinusoid": {
            "equation": "nu(t) = A cos(omega t + phi_k) for t in coherence window k",
            "role": "secondary stress test",
            "physics_note": "Approximates finite coherence by phase re-randomization in fixed windows on the observed cadence.",
        },
        "slow_linear_drift": {
            "equation": "nu(t) = slope * (t - mean(t))",
            "role": "stress test, not a headline BSM model",
            "physics_note": "Captures the slow-modulation limit where a sinusoid can look locally like a drift.",
        },
        "gaussian_transient": {
            "equation": "nu(t) = A exp[-0.5 ((t - t0)/sigma_t)^2]",
            "role": "controlled transient benchmark family",
            "physics_note": "Stylized localized event; not claimed as a unique physical prediction.",
        },
        "susy_note": {
            "equation": "not implemented as an injection family",
            "role": "scope guard",
            "physics_note": "Generic SUSY virtual corrections may motivate enhanced sensitivity, but do not by themselves define a time-domain morphology.",
        },
    }

