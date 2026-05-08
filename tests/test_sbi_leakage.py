from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = PROJECT_ROOT / "baselines"
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from sbi_npe import _threshold, train_sbi_npe  # noqa: E402
from sbi_npe_simulator import SBISimulatorConfig, Th229SBISimulator, load_primary_peak_b  # noqa: E402
from th229_bench.null_models import fit_null_models  # noqa: E402


def test_sbi_simulator_uses_immutable_55_peak_b_covariates() -> None:
    simulator = Th229SBISimulator(config=SBISimulatorConfig(seed=123))
    assert len(simulator.primary_df) == 55
    assert simulator.primary_df["seconds_since_first_observation"].is_monotonic_increasing
    assert {"target", "freq_unc_hz", "temp_k"}.issubset(simulator.primary_df.columns)


def test_sbi_null_parameters_match_training_fit_default() -> None:
    primary = load_primary_peak_b()
    _, null_models, default_name = fit_null_models(primary)
    simulator = Th229SBISimulator(primary_df=primary)
    assert simulator.null_model_name == default_name
    assert simulator.null_model.parameters == null_models[default_name].parameters


def test_sbi_training_helper_does_not_read_benchmark_examples_or_labels() -> None:
    source = inspect.getsource(train_sbi_npe)
    forbidden = [
        "benchmark_catalog",
        "benchmark_arrays",
        "data/interim",
        '"split"',
        '"label"',
    ]
    for token in forbidden:
        assert token not in source


def test_sbi_prior_config_records_log_period_and_null_balance() -> None:
    config = SBISimulatorConfig()
    simulator = Th229SBISimulator(config=config)
    metadata = simulator.metadata()
    prior = metadata["prior"]
    assert prior["pi_null"] == 0.5
    assert np.isclose(prior["log_period_min"], np.log(7.0))
    assert np.isclose(prior["log_period_max"], np.log(365.0))
    assert metadata["theta"] == ["A_hz", "log_period_days", "phi_rad"]


def test_sbi_threshold_uses_validation_null_only() -> None:
    catalog = pd.DataFrame(
        {
            "split": ["train", "validation", "validation", "validation", "test"],
            "label": [0, 0, 0, 1, 0],
        }
    )
    scores = np.asarray([100.0, 1.0, 3.0, 1000.0, 2000.0])
    assert np.isclose(_threshold(scores, catalog), np.quantile([1.0, 3.0], 0.95))
