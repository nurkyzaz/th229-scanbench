# Figures Manifest

This manifest is for the final technical package only. It is not a publication figure list.

- `final_peak_b_residuals_over_time_by_crystal`: `results/figures/final_peak_b_residuals_over_time_by_crystal.png`. Peak-b temperature-corrected residuals on the observed JILA cadence, separated by crystal.
- `final_residual_scatter_uncertainty_by_crystal`: `results/figures/final_residual_scatter_uncertainty_by_crystal.png`. Residual scatter versus formal uncertainty and crystal identity; X2 is visibly broader.
- `final_null_model_calibration_comparison`: `results/figures/final_null_model_calibration_comparison.png`. Held-out negative log likelihood for formal, crystal-aware Gaussian, fitted-tail Student-t, and X2-mixture null models.
- `final_sensitivity_heatmap_main_baseline`: `results/figures/final_sensitivity_heatmap_main_baseline.png`. Detection rate for the weighted harmonic-regression baseline across the primary sinusoid grid at validation-calibrated 5% FPR.
- `final_auc_heatmap_with_a95_contour`: `results/figures/final_auc_heatmap_with_a95_contour.png`. Cell-wise ROC AUC for the hierarchical sinusoid baseline, with a 95% detection-rate contour from the validation-calibrated threshold.
- `final_a95_vs_frequency_with_uncertainty`: `results/figures/final_a95_vs_frequency_with_uncertainty.png`. A95 sensitivity curves for all safe baselines with bootstrap intervals from validation-threshold and test-phase resampling.
- `final_observed_series_periodogram_calibration`: `results/figures/final_observed_series_periodogram_calibration.png`. Observed peak-b residual periodogram by subset; interpreted conservatively as a calibration diagnostic, not a signal claim.
- `final_generalization_summary`: `results/figures/final_generalization_summary.png`. Secondary off-grid held-out-period evaluation using thresholds calibrated on the main period grid.
- `final_drift_sanity_curve`: `results/figures/final_drift_sanity_curve.png`. Secondary linear-drift sanity task showing detection rate versus injected absolute drift.
- `final_ablation_remove_x2`: `results/figures/final_ablation_remove_x2.png`. Comparison of the default peak-b condition and the official clean C10+C13-only condition.
- `final_roc_curve_main_baseline`: `results/figures/final_roc_curve_main_baseline.png`. Held-out ROC curve for the weighted harmonic-regression baseline.