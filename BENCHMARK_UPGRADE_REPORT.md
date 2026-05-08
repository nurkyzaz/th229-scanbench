# Thorium-229 Benchmark Final Technical Report

## Benchmark Definition
The package uses the published JILA scan-level frequency record. The main task uses the 55 peak-b measurements and asks whether an added time-domain frequency modulation is present on the observed JILA cadence.

Peak c is retained only as optional metadata and an exploratory table row. The pkl scan arrays remain a future-work branch through `data/interim/lineshape_manifest.csv`.

## What Changed In This Final Revision
- Canonical scan-level subset: `73` rows.
- Primary peak-b subset: `55` rows.
- Optional peak-b+c subset: `73` rows.
- Strict QC subset with `chisq_red <= 10`: `41` rows.
- Added `results/tables/subset_reconciliation_scan_keys.csv` so the 72-versus-73 issue is auditable scan by scan.
- Added fitted-tail and X2-mixture null candidates, then recorded the X2 decision in `results/tables/x2_handling_decision.md`.
- Added A95 bootstrap intervals in `results/tables/a95_vs_frequency_with_uncertainty.csv`.
- Added a secondary off-grid held-out-period test in `results/tables/generalization_results.csv`.
- Removed the point-only A95 figure from the final figure set and replaced it with `final_a95_vs_frequency_with_uncertainty.png`.
- After reviewing the example draft, added a secondary linear-drift sanity task, an AUC heatmap with an A95 contour, a representative-period baseline table, and a residual-over-time figure with a `+/-220` Hz reference band.

## Preprocessing
- Timestamp span verified: `487.727546` days.
- Processed CSV exports now drop the raw `Unnamed: 0` index column and fail if a stale `Unnamed` column would be saved.
- Peak-b temperature correction is fitted from canonical C10+C13 records and remains the default preprocessing path.
- Peak-c correction is treated as data-derived exploratory support only; it is not used by the main task.

## X2 Handling
- `formal_gaussian_scaled` fits one global uncertainty inflation factor on the observation-level training subset.
- `crystal_gaussian_jitter` fits a separate extra variance term for each crystal, allowing X2 to be wider than C10/C13.
- `crystal_student_t_jitter` fits crystal-specific jitter with Student-t tails using fixed 4 degrees of freedom.
- `crystal_student_t_fitted_df` fits target-specific Student-t degrees of freedom on training residuals.
- `crystal_gaussian_x2_mixture` fits an X2-only Gaussian outlier component on X2 training residuals.
- Default null model selected by held-out residual negative log likelihood: `crystal_gaussian_x2_mixture`.
- Default null held-out NLL per residual: `9.1162`.
- X2 remains in the default peak-b task because it is part of the published peak-b record.
- C10+C13 peak b is now an official clean secondary condition, meaning the same task is rerun without X2 to show how much the X2 scatter affects conclusions.
```
                 null_model        distribution  is_default          scope  observed_score  best_period_days  null_score_p95  observed_empirical_p_value                                                                                                    model_note
     formal_gaussian_scaled            gaussian       False     all_peak_b        7.905143         15.436079        7.011471                    0.029240 The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
     formal_gaussian_scaled            gaussian       False peak_b_c10_c13        1.169029          7.930951        6.583313                    0.996101 The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
     formal_gaussian_scaled            gaussian       False      peak_b_x2       14.679199         15.436079        5.933082                    0.001949 The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
    crystal_gaussian_jitter            gaussian       False     all_peak_b        3.233823         25.435825        6.913678                    0.639376 The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
    crystal_gaussian_jitter            gaussian       False peak_b_c10_c13        1.861473          7.930951        6.542035                    0.947368 The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
    crystal_gaussian_jitter            gaussian       False      peak_b_x2        9.866401         15.436079        6.463906                    0.001949 The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
   crystal_student_t_jitter           student_t       False     all_peak_b        4.469622         25.435825       16.049122                    0.787524 The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
   crystal_student_t_jitter           student_t       False peak_b_c10_c13        2.642348          7.930951       14.444471                    0.947368 The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
   crystal_student_t_jitter           student_t       False      peak_b_x2       12.470169         15.436079       14.636469                    0.089669 The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
crystal_student_t_fitted_df student_t_target_df       False     all_peak_b        3.467370         25.435825        9.840644                    0.808967 The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
crystal_student_t_fitted_df student_t_target_df       False peak_b_c10_c13        2.288099          7.930951        9.871826                    0.947368 The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
crystal_student_t_fitted_df student_t_target_df       False      peak_b_x2        9.986641         15.436079        6.770855                    0.005848 The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
crystal_gaussian_x2_mixture x2_gaussian_mixture        True     all_peak_b        3.095259         25.435825        6.801260                    0.674464 The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
crystal_gaussian_x2_mixture x2_gaussian_mixture        True peak_b_c10_c13        1.861473          7.930951        6.542035                    0.947368 The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
crystal_gaussian_x2_mixture x2_gaussian_mixture        True      peak_b_x2        9.307926         15.436079        6.256183                    0.003899 The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
```

## Baselines
- `weighted_harmonic_regression`: uncertainty-weighted sinusoid fit with crystal offsets; this remains the main simple baseline.
- `generalized_lomb_scargle`: global-offset weighted periodogram baseline on the same search grid.
- `hierarchical_sinusoid_jitter`: shared sinusoid with crystal-specific offsets and the selected crystal-aware null variance; this is the main physics-aware comparison.

## Evaluation Protocol
- One authoritative split is used everywhere: phase-index splits within each frequency-amplitude cell.
- Null-model parameters are fitted on a target-stratified chronological subset of the published observed peak-b residuals.
- Detection thresholds are calibrated on validation null injections only.
- Test examples are never used for threshold calibration or model selection.
- The off-grid held-out-period test calibrates thresholds on source-grid validation null examples and evaluates only held-out-period test examples.

## Observed-Series Calibration Diagnostic
- The strongest observed-series score is X2-dominated; this is treated as scatter/model mismatch, not a signal.
- The all-peak-b observed series is used as a calibration diagnostic, not as a detection claim.
- The X2-only subset can produce a low empirical p-value under the selected null; this is documented as residual scatter or model mismatch, not evidence for a physical signal.
```
         scope                  null_model  observed_score  best_period_days  null_score_p95  observed_empirical_p_value
    all_peak_b crystal_gaussian_x2_mixture        3.095259         25.435825        6.801260                    0.674464
peak_b_c10_c13 crystal_gaussian_x2_mixture        1.861473          7.930951        6.542035                    0.947368
     peak_b_x2 crystal_gaussian_x2_mixture        9.307926         15.436079        6.256183                    0.003899
```

## Split Counts
```
     split  label  count
      test      0    660
      test      1    660
     train      0   1848
     train      1   1848
validation      0    660
validation      1    660
```

## A95 Uncertainty
- A95 intervals resample validation-null scores for the threshold and the five test phase realizations in each period-amplitude cell.
- The intervals quantify finite synthetic-grid and phase-sampling uncertainty only.
- They do not include uncertainty in the measured frequencies, temperature correction, or null-model choice.
```
                    baseline  period_days  frequency_hz      a95_hz  a95_bootstrap_median_hz  a95_lower_hz  a95_upper_hz  a95_std_hz  bootstrap_replicates  finite_bootstrap_replicates  n_test_phases_per_cell                                                                         uncertainty_source
weighted_harmonic_regression          7.0  1.653439e-06 3116.666667              3116.666667   3075.000000   3137.500000  161.329856                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
weighted_harmonic_regression         10.0  1.157407e-06 4175.000000              3150.000000   3075.000000   4337.500000  802.836152                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
weighted_harmonic_regression         14.0  8.267196e-07 3075.000000              3075.000000   2950.000000   3116.666667  350.728372                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
weighted_harmonic_regression         21.0  5.511464e-07 3116.666667              3116.666667   3075.000000   3137.500000  270.094719                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
weighted_harmonic_regression         30.0  3.858025e-07 2950.000000              2950.000000   2025.000000   3075.000000  535.630757                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
weighted_harmonic_regression         45.0  2.572016e-07 3116.666667              3116.666667   3075.000000   3137.500000  122.332351                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
weighted_harmonic_regression         60.0  1.929012e-07 4175.000000              4175.000000   3075.000000   4337.500000  654.759127                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
weighted_harmonic_regression         90.0  1.286008e-07 3116.666667              3116.666667   3075.000000   3137.500000   76.608786                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
weighted_harmonic_regression        120.0  9.645062e-08 2950.000000              2950.000000   2156.250000   3075.000000  397.902859                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
weighted_harmonic_regression        180.0  6.430041e-08 4418.750000              4418.750000   4391.666667   4435.000000   29.263767                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
weighted_harmonic_regression        240.0  4.822531e-08 6333.333333              6250.000000   6000.000000   6333.333333  590.263507                   500                          360                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
weighted_harmonic_regression        365.0  3.170979e-08 4337.500000              4337.500000   3148.000000   4391.666667  682.661462                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
    generalized_lomb_scargle          7.0  1.653439e-06 3075.000000              3075.000000   2950.000000   3116.666667  247.700245                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
    generalized_lomb_scargle         10.0  1.157407e-06 4175.000000              3137.500000   2950.000000   4337.500000  823.633021                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
    generalized_lomb_scargle         14.0  8.267196e-07 3075.000000              3075.000000   2950.000000   3116.666667  378.122647                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
    generalized_lomb_scargle         21.0  5.511464e-07 3075.000000              3075.000000   2950.000000   3137.500000  509.907903                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
    generalized_lomb_scargle         30.0  3.858025e-07 2950.000000              2950.000000   2025.000000   3075.000000  520.220347                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
    generalized_lomb_scargle         45.0  2.572016e-07 3116.666667              3116.666667   3075.000000   3137.500000  358.789911                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
    generalized_lomb_scargle         60.0  1.929012e-07 4175.000000              4175.000000   2950.000000   4337.500000  775.974860                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
    generalized_lomb_scargle         90.0  1.286008e-07 2950.000000              2950.000000   2165.000000   3075.000000  399.388873                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
    generalized_lomb_scargle        120.0  9.645062e-08 2950.000000              2950.000000   2156.250000   3075.000000  444.541209                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
    generalized_lomb_scargle        180.0  6.430041e-08 3075.000000              3075.000000   2950.000000   3116.666667  293.523095                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
    generalized_lomb_scargle        240.0  4.822531e-08 2112.500000              2112.500000   2025.000000   2141.666667  187.999482                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
    generalized_lomb_scargle        365.0  3.170979e-08 2950.000000              2950.000000   2156.250000   3075.000000  400.601862                   500                          500                       5 validation-null threshold resampling plus test-phase resampling within each amplitude cell
```

## Held-Out-Period Generalization
- The secondary test uses off-grid periods that are absent from the main period grid.
- Thresholds come from main-grid validation null examples.
- Metrics are computed on off-grid held-out-period test examples.
```
                    baseline      generalization_task                                            source_periods_days           heldout_periods_days  roc_auc  average_precision  source_validation_fpr5_threshold  tpr_at_source_validation_fpr_5pct  target_test_null_false_positive_rate  target_test_examples  runtime_sec
weighted_harmonic_regression off_grid_heldout_periods 7.0|10.0|14.0|21.0|30.0|45.0|60.0|90.0|120.0|180.0|240.0|365.0 8.5|17.0|36.0|75.0|150.0|300.0 0.721543           0.769345                         95.556158                           0.357576                              0.039394                   660     4.533115
    generalized_lomb_scargle off_grid_heldout_periods 7.0|10.0|14.0|21.0|30.0|45.0|60.0|90.0|120.0|180.0|240.0|365.0 8.5|17.0|36.0|75.0|150.0|300.0 0.733205           0.780201                         94.815280                           0.381818                              0.045455                   660     4.397185
hierarchical_sinusoid_jitter off_grid_heldout_periods 7.0|10.0|14.0|21.0|30.0|45.0|60.0|90.0|120.0|180.0|240.0|365.0 8.5|17.0|36.0|75.0|150.0|300.0 0.807897           0.857721                          6.958917                           0.527273                              0.027273                   660     4.532575
```

## Draft-Review Additions
- The example draft was weaker than the current project overall because it contained count errors and proposed methods that are less well matched to the 55-point peak-b task.
- Useful draft ideas were adopted where they strengthened the current package without changing scope.
- The linear-drift sanity task injects slopes in Hz/day on the same peak-b cadence and calibrates its threshold on validation null examples.
- The representative-period table reports the safe baselines at 30 and 180 days.
- The AUC heatmap with A95 contour is generated for `hierarchical_sinusoid_jitter`, the current strongest safe baseline.
```
               task                                   baseline  roc_auc  average_precision  validation_fpr5_threshold  tpr_at_validation_fpr_5pct  test_null_false_positive_rate  test_examples  minimum_detectable_abs_slope_hz_per_day
linear_drift_sanity weighted_linear_drift_with_crystal_offsets 0.803056            0.86917                   7.590932                        0.55                            0.0            120                                   14.375
```

## Baseline Summary
```
                    baseline  roc_auc  average_precision  validation_fpr5_threshold  tpr_at_validation_fpr_5pct  test_null_false_positive_rate  runtime_sec
hierarchical_sinusoid_jitter 0.773804           0.826751                   6.958917                    0.490909                       0.042424     9.467628
    generalized_lomb_scargle 0.724281           0.776158                  94.815280                    0.378788                       0.050000     8.772626
weighted_harmonic_regression 0.699653           0.747932                  95.556158                    0.328788                       0.050000     9.142252
```

## Ablation Summary
```
                    baseline  roc_auc  average_precision  validation_fpr5_threshold  tpr_at_validation_fpr_5pct  test_null_false_positive_rate  runtime_sec                     dataset_scope
weighted_harmonic_regression 0.699653           0.747932                  95.556158                    0.328788                       0.050000     9.118865                primary_peak_b_all
    generalized_lomb_scargle 0.724281           0.776158                  94.815280                    0.378788                       0.050000     8.867059                primary_peak_b_all
hierarchical_sinusoid_jitter 0.773804           0.826751                   6.958917                    0.490909                       0.042424     9.087779                primary_peak_b_all
weighted_harmonic_regression 0.755386           0.807651                  27.620249                    0.457576                       0.065152     8.536244     official_clean_c10_c13_peak_b
    generalized_lomb_scargle 0.770533           0.824725                  28.573834                    0.504545                       0.056061     8.482989     official_clean_c10_c13_peak_b
hierarchical_sinusoid_jitter 0.782493           0.834913                   6.397248                    0.507576                       0.046970     8.427325     official_clean_c10_c13_peak_b
weighted_harmonic_regression 0.712489           0.767384                  96.779695                    0.315152                       0.018182     9.243949 exploratory_peaks_bc_not_headline
    generalized_lomb_scargle 0.725358           0.784056                  97.026872                    0.356061                       0.028788     9.055698 exploratory_peaks_bc_not_headline
hierarchical_sinusoid_jitter 0.781019           0.836175                   6.526715                    0.530303                       0.059091     9.324330 exploratory_peaks_bc_not_headline
```

## Lineshape Future Branch
- Manifest: `data/interim/lineshape_manifest.csv`.
- Audit: `results/tables/lineshape_audit.md`.
- The pkl files are usable for a future branch manifest and loader, but not promoted into the main benchmark in this upgrade.

## Safest Outputs For External Reporting
- `results/tables/preprocessing_audit.json`
- `results/tables/subset_reconciliation.md`
- `results/tables/subset_reconciliation_scan_keys.csv`
- `results/tables/null_model_comparison.csv`
- `results/tables/x2_calibration_study.csv`
- `results/tables/x2_handling_decision.md`
- `results/tables/baseline_comparison.csv`
- `results/tables/a95_vs_frequency_with_uncertainty.csv`
- `results/tables/a95_uncertainty_notes.md`
- `results/tables/generalization_protocol.json`
- `results/tables/generalization_results.csv`
- `results/tables/generalization_notes.md`
- `results/tables/drift_sanity_results.csv`
- `results/tables/drift_sanity_protocol.json`
- `results/tables/drift_sanity_notes.md`
- `results/tables/auc_heatmap_with_a95_contour.csv`
- `results/tables/final_representative_frequency_baseline_table.csv`
- `results/tables/benchmark_sensitivity_heatmap.csv`
- `results/tables/observed_series_false_alarm_behavior.csv`
- `results/tables/ablation_results.csv`
- `results/figures/final_peak_b_residuals_over_time_by_crystal.png`
- `results/figures/final_residual_scatter_uncertainty_by_crystal.png`
- `results/figures/final_null_model_calibration_comparison.png`
- `results/figures/final_sensitivity_heatmap_main_baseline.png`
- `results/figures/final_a95_vs_frequency_with_uncertainty.png`
- `results/figures/final_observed_series_periodogram_calibration.png`
- `results/figures/final_generalization_summary.png`
- `results/figures/final_auc_heatmap_with_a95_contour.png`
- `results/figures/final_drift_sanity_curve.png`
- `results/tables/figures_manifest.md`

## Caveats That Must Remain
- Verify the peak-b temperature coefficient treatment against the final intended citation/table wording.
- Decide whether the 2024-09-25 X2 peak-c fallback row should be described as canonical or optional in any eventual manuscript.
- State clearly that the X2-only observed calibration diagnostic is not a signal claim.
- State clearly that A95 intervals are synthetic-resampling intervals, not full experimental uncertainty intervals.
- Review all figure captions before external use; they are technical notes, not final publication text.