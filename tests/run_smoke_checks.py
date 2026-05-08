from __future__ import annotations

from test_preprocessing import (
    test_benchmark_split_protocol_has_no_overlap,
    test_canonical_counts_match_file_backed_expectation,
    test_temperature_model_peak_b_matches_jila_zero_shift_band,
    test_timestamps_are_timezone_aware_utc,
)
from test_learned_null_leakage import (
    test_learned_null_heldout_loglik_uses_holdout_only,
    test_learned_null_training_sources_are_train_split_only,
    test_learned_nulls_register_through_factory,
    test_null_fit_split_has_train_and_holdout_alias_for_validation,
)
from test_sbi_leakage import (
    test_sbi_null_parameters_match_training_fit_default,
    test_sbi_prior_config_records_log_period_and_null_balance,
    test_sbi_simulator_uses_immutable_55_peak_b_covariates,
    test_sbi_threshold_uses_validation_null_only,
    test_sbi_training_helper_does_not_read_benchmark_examples_or_labels,
)
from test_neural_leakage import (
    test_neural_feature_normalization_uses_train_split_only,
    test_neural_test_scores_are_after_freeze,
    test_neural_threshold_uses_validation_null_only,
    test_neural_training_helper_uses_train_and_validation_only,
)
from test_stress_damped_leakage import (
    test_stress_damped_catalog_schema_matches_main_benchmark,
    test_stress_damped_loads_frozen_ml_checkpoints,
    test_stress_damped_stat_baselines_use_existing_sinusoid_scorers,
    test_stress_damped_uses_main_parametric_null_seed_protocol,
    test_stress_damped_uses_same_55_timestamps,
)


def main() -> None:
    checks = [
        test_canonical_counts_match_file_backed_expectation,
        test_timestamps_are_timezone_aware_utc,
        test_temperature_model_peak_b_matches_jila_zero_shift_band,
        test_benchmark_split_protocol_has_no_overlap,
        test_sbi_simulator_uses_immutable_55_peak_b_covariates,
        test_sbi_null_parameters_match_training_fit_default,
        test_sbi_training_helper_does_not_read_benchmark_examples_or_labels,
        test_sbi_prior_config_records_log_period_and_null_balance,
        test_sbi_threshold_uses_validation_null_only,
        test_learned_null_training_sources_are_train_split_only,
        test_learned_null_heldout_loglik_uses_holdout_only,
        test_null_fit_split_has_train_and_holdout_alias_for_validation,
        test_learned_nulls_register_through_factory,
        test_neural_feature_normalization_uses_train_split_only,
        test_neural_training_helper_uses_train_and_validation_only,
        test_neural_threshold_uses_validation_null_only,
        test_neural_test_scores_are_after_freeze,
        test_stress_damped_uses_same_55_timestamps,
        test_stress_damped_catalog_schema_matches_main_benchmark,
        test_stress_damped_uses_main_parametric_null_seed_protocol,
        test_stress_damped_loads_frozen_ml_checkpoints,
        test_stress_damped_stat_baselines_use_existing_sinusoid_scorers,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")


if __name__ == "__main__":
    main()
