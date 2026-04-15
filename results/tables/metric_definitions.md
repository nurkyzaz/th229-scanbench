# Metric Definitions

- `roc_auc`: Area under the ROC curve on the held-out synthetic test set.
- `average_precision`: Area under the precision-recall curve on the held-out synthetic test set.
- `validation_fpr5_threshold`: Score threshold set to the 95th percentile of validation null scores.
- `tpr_at_validation_fpr_5pct`: Fraction of held-out signal examples above a threshold calibrated to the 95th percentile of validation null scores.
- `test_null_false_positive_rate`: Fraction of held-out null examples above the validation-calibrated threshold.
- `a95_hz`: Smallest injected amplitude with interpolated detection rate at least 95% at the validation-calibrated 5% false-positive threshold.
- `a95_lower_hz`: 16th percentile bootstrap estimate for A95 from validation-threshold and test-phase resampling.
- `a95_upper_hz`: 84th percentile bootstrap estimate for A95 from validation-threshold and test-phase resampling.
- `observed_empirical_p_value`: Bootstrap null tail probability for the observed JILA residual series under the specified null model.
- `source_validation_fpr5_threshold`: Generalization-test threshold copied from source-grid validation null scores.
- `target_test_null_false_positive_rate`: Generalization-test null false-positive rate on off-grid held-out-period test examples.