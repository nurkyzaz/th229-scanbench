# Null Model Notes

- `formal_gaussian_scaled`: Gaussian residuals with formal frequency uncertainties multiplied by one global scale factor fitted on the observation-level training split.
- `crystal_gaussian_jitter`: Gaussian residuals with per-crystal extra variance terms fitted on training residuals.
- `crystal_student_t_jitter`: Student-t residuals with fixed 4 degrees of freedom and per-crystal jitter terms fitted on training residuals.
- `crystal_student_t_fitted_df`: Student-t residuals with target-specific jitter and target-specific degrees of freedom fitted on training residuals.
- `crystal_gaussian_x2_mixture`: Gaussian residuals with crystal jitter plus an X2-only Gaussian outlier component fitted on X2 training residuals.
- Default selected by held-out residual negative log likelihood: `crystal_gaussian_x2_mixture`.
- X2 is allowed to have separate scatter and optional tail/outlier behavior; model comparison is still based on held-out residual likelihood, not on tuning to the observed periodogram.

## Observed-Series Calibration
- The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
- `all_peak_b`: score `3.09526`, null p95 `6.80126`, empirical p-value `0.674464`.
- `peak_b_c10_c13`: score `1.86147`, null p95 `6.54203`, empirical p-value `0.947368`.
- `peak_b_x2`: score `9.30793`, null p95 `6.25618`, empirical p-value `0.00389864`.
- Low p-values in the X2-only diagnostic are treated as evidence of residual scatter/model mismatch, not as a detection claim.

## X2 Decision
- The selected null does not fully calibrate the X2-only observed periodogram, so C10+C13 peak-b is promoted to an official clean secondary condition.