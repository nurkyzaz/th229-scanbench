from __future__ import annotations

from th229_bench.benchmarking import BenchmarkConfig, build_benchmark_dataset
from th229_bench.null_models import fit_null_models
from th229_bench.preprocessing import prepare_preprocessed_data


def test_canonical_counts_match_file_backed_expectation() -> None:
    prepared = prepare_preprocessed_data()
    canonical = prepared["canonical"]
    assert len(canonical) == 73
    assert int((canonical["peak"] == "b").sum()) == 55
    assert int(canonical["peak"].isin(["b", "c"]).sum()) == 73


def test_timestamps_are_timezone_aware_utc() -> None:
    prepared = prepare_preprocessed_data()
    raw = prepared["raw"]
    assert str(raw["scan_time_utc"].dt.tz) == "UTC"
    assert raw["scan_time_utc"].isna().sum() == 0


def test_temperature_model_peak_b_matches_jila_zero_shift_band() -> None:
    prepared = prepare_preprocessed_data()
    model_b = prepared["temperature_models"]["b"]
    assert 191.0 <= model_b["turning_point_temperature_k"] <= 201.0


def test_benchmark_split_protocol_has_no_overlap() -> None:
    prepared = prepare_preprocessed_data()
    primary_df = prepared["views"]["primary_peak_b"]
    _, null_models, default_name = fit_null_models(primary_df)
    config = BenchmarkConfig(
        n_phases_per_cell=6,
        train_phase_count=3,
        val_phase_count=1,
        test_phase_count=2,
        detection_grid_size=16,
    )
    dataset = build_benchmark_dataset(
        primary_df,
        null_models[default_name],
        config=config,
        prefix="pytest_no_write",
        write_outputs=False,
    )
    catalog = dataset["catalog"]
    split_ids = {
        split: set(catalog.loc[catalog["split"].eq(split), "instance_id"].tolist())
        for split in ("train", "validation", "test")
    }
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
