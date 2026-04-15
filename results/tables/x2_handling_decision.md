# X2 Handling Decision

- X2 is retained in the default 55-row peak-b benchmark because it is part of the published peak-b record.
- Additional candidate nulls were fitted on training residuals only: target-specific fitted-df Student-t and X2 Gaussian mixture.
- Default null selected by held-out residual likelihood: `crystal_gaussian_x2_mixture`.
- Default-null X2-only empirical p-value: `0.00389864`.
- Best X2-only empirical p-value among tested nulls: `0.0896686` from `crystal_student_t_jitter`.
- Decision: The selected null does not fully calibrate the X2-only observed periodogram, so C10+C13 peak-b is promoted to an official clean secondary condition.
- The X2-only observed diagnostic is not a detection claim and is not a headline result.
- The official clean condition means rerunning the same peak-b task on C10+C13 only, where the observed calibration diagnostic is not X2 dominated.