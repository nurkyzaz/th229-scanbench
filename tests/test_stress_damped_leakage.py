from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "baselines"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from injections import damped_sinusoid  # noqa: E402


def test_stress_damped_uses_same_55_timestamps() -> None:
    catalog, arrays = damped_sinusoid.generate_stress_dataset(30.0, write_outputs=False)
    main = np.load(PROJECT_ROOT / "data" / "interim" / "benchmark_arrays.npz")
    assert len(catalog) == 6336
    assert arrays["observed_residual_hz"].shape == (6336, 55)
    assert np.allclose(arrays["times_sec"], main["times_sec"])


def test_stress_damped_catalog_schema_matches_main_benchmark() -> None:
    catalog, _ = damped_sinusoid.generate_stress_dataset(90.0, write_outputs=False)
    main_catalog = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "benchmark_catalog.csv")
    assert list(catalog.columns) == list(main_catalog.columns)
    assert set(catalog["split"]) == {"train", "validation", "test"}
    assert catalog.groupby(["split", "label"]).size().to_dict() == main_catalog.groupby(["split", "label"]).size().to_dict()


def test_stress_damped_uses_main_parametric_null_seed_protocol() -> None:
    source = inspect.getsource(damped_sinusoid.generate_stress_dataset)
    assert "fit_null_models(primary)" in inspect.getsource(damped_sinusoid.load_primary_and_null)
    assert "models[default_name]" in inspect.getsource(damped_sinusoid.load_primary_and_null)
    assert "1_000_000 * int(row[\"label\"])" in source
    assert "10_000 * int(row[\"frequency_index\"])" in source
    assert "100 * int(row[\"amplitude_index\"])" in source


def test_stress_damped_loads_frozen_ml_checkpoints() -> None:
    source = inspect.getsource(damped_sinusoid._baseline_scores)
    assert "_load_rf_scores" in source
    assert "score_residuals" in source
    assert "_load_neural_scores" in source
    assert ".fit(" not in inspect.getsource(damped_sinusoid._load_rf_scores)
    assert ".fit(" not in inspect.getsource(damped_sinusoid._load_neural_scores)
    assert "torch.load" in inspect.getsource(damped_sinusoid._load_neural_scores)


def test_stress_damped_stat_baselines_use_existing_sinusoid_scorers() -> None:
    source = inspect.getsource(damped_sinusoid._baseline_scores)
    assert "_baseline_scorers" in source
    assert "_score_all" in source
    assert "damped" not in inspect.getsource(damped_sinusoid._baseline_scores).replace("_baseline_scores", "")
