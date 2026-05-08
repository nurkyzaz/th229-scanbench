# Figure Caption Notes

- `final_peak_b_residuals_over_time_by_crystal`: Peak-b temperature-corrected residuals on the observed JILA cadence, separated by crystal.
- `final_residual_scatter_uncertainty_by_crystal`: Residual scatter versus formal uncertainty and crystal identity; X2 is visibly broader.
- `final_null_model_calibration_comparison`: Held-out negative log likelihood for formal, crystal-aware Gaussian, fitted-tail Student-t, and X2-mixture null models.
- `final_sensitivity_heatmap_main_baseline`: Detection rate for the weighted harmonic-regression baseline across the primary sinusoid grid at validation-calibrated 5% FPR.
- `final_auc_heatmap_with_a95_contour`: Cell-wise ROC AUC for the hierarchical sinusoid baseline, with a 95% detection-rate contour from the validation-calibrated threshold.
- `final_a95_vs_frequency_with_uncertainty`: A95 sensitivity curves for all safe baselines with bootstrap intervals from validation-threshold and test-phase resampling.
- `final_observed_series_periodogram_calibration`: Observed peak-b residual periodogram by subset; interpreted conservatively as a calibration diagnostic, not a signal claim.
- `final_generalization_summary`: Secondary off-grid held-out-period evaluation using thresholds calibrated on the main period grid.
- `final_drift_sanity_curve`: Secondary linear-drift sanity task showing detection rate versus injected absolute drift.
- `final_ablation_remove_x2`: Comparison of the default peak-b condition and the official clean C10+C13-only condition.
- `final_roc_curve_main_baseline`: Held-out ROC curve for the weighted harmonic-regression baseline.