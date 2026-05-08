from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = PROJECT_ROOT / "baselines"
for path in (BASELINES_DIR, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from neural_cnn import _score_after_freeze, _threshold, _train_one, build_sequence_features, run_neural_baseline  # noqa: E402


def test_neural_feature_normalization_uses_train_split_only() -> None:
    catalog = pd.DataFrame({"split": ["train", "validation", "test"], "label": [0, 1, 0]})
    residuals = np.asarray([[1.0, 3.0], [100.0, 100.0], [-100.0, -100.0]], dtype=np.float32)
    metadata = pd.DataFrame(
        {
            "target": ["C10", "X2"],
            "freq_unc_hz": [10.0, 20.0],
            "temp_k": [1.0, 2.0],
            "seconds_since_first_observation": [0.0, 10.0],
        }
    )
    features, stats = build_sequence_features(catalog, residuals, metadata)
    assert np.isclose(stats["residual_mean"], 2.0)
    assert np.isclose(stats["residual_std"], 1.0)
    assert features.shape == (3, 2, 8)


def test_neural_training_helper_uses_train_and_validation_only() -> None:
    source = inspect.getsource(_train_one)
    assert "features[train_mask]" in source
    assert "labels[train_mask]" in source
    assert "features[validation_mask]" in source
    assert "test_mask" not in source
    assert "benchmark_catalog" not in source
    assert "benchmark_arrays" not in source


def test_neural_threshold_uses_validation_null_only() -> None:
    catalog = pd.DataFrame({"split": ["train", "validation", "validation", "validation", "test"], "label": [0, 0, 0, 1, 0]})
    scores = np.asarray([100.0, 0.1, 0.5, 0.99, 1000.0])
    assert np.isclose(_threshold(scores, catalog), np.quantile([0.1, 0.5], 0.95))


def test_neural_test_scores_are_after_freeze() -> None:
    source = inspect.getsource(run_neural_baseline)
    assert source.index("best_model is None") < source.index("_score_after_freeze")
    frozen_source = inspect.getsource(_score_after_freeze)
    assert "_predict_scores" in frozen_source
    assert "optimizer" not in frozen_source
    assert "loss.backward" not in frozen_source
