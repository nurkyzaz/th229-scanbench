# Leakage Check

- One authoritative split is used: phase-index splits within every frequency-amplitude cell.
- Validation thresholds are calibrated only on validation null examples.
- Test examples are not used for model selection, threshold calibration, or null-model fitting.
- Null-model parameters are fitted on a target-stratified chronological training subset of the observed peak-b residuals.
- The held-out-period generalization test calibrates thresholds on source-grid validation nulls and evaluates only off-grid held-out-period test examples.
- The previous logistic-periodogram baseline was removed from headline outputs to avoid overemphasizing tiny-data ML.