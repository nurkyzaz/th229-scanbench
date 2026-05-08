from __future__ import annotations

import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from null_models.gmm import fit_gmm_null  # noqa: E402
from null_models.normalizing_flow import fit_flow_null  # noqa: E402
from null_models.run_learned_nulls import _heldout_loglik  # noqa: E402
from th229_bench.null_models import assign_null_fit_split, fit_null_models  # noqa: E402


def test_learned_null_training_sources_are_train_split_only() -> None:
    for fn in (fit_gmm_null, fit_flow_null):
        source = inspect.getsource(fn)
        assert 'null_fit_split"].eq("train")' in source
        assert "benchmark_catalog" not in source
        assert "benchmark_arrays" not in source
        assert "data/interim" not in source


def test_learned_null_heldout_loglik_uses_holdout_only() -> None:
    source = inspect.getsource(_heldout_loglik)
    assert 'null_fit_split"].eq("holdout")' in source
    assert "benchmark_catalog" not in source
    assert "benchmark_arrays" not in source


def test_null_fit_split_has_train_and_holdout_alias_for_validation() -> None:
    import pandas as pd

    primary = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "primary_peak_b.csv")
    split = assign_null_fit_split(primary)
    assert set(split["null_fit_split"]) == {"train", "holdout"}
    assert len(split.loc[split["null_fit_split"].eq("train")]) > len(split.loc[split["null_fit_split"].eq("holdout")])


def test_learned_nulls_register_through_factory() -> None:
    import pandas as pd

    primary = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "primary_peak_b.csv")
    _, models, default_name = fit_null_models(primary, model_names=["gmm", "normalizing_flow"])
    assert "crystal_gaussian_x2_mixture" in models
    assert "crystal_gmm_3comp" in models
    assert "conditional_spline_flow" in models
    assert default_name in models
