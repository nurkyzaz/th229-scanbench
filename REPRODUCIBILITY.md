# Reproducibility

Run all commands from the repository root.

## Environment

```bash
python -m pip install -r requirements.txt
```

## Full Benchmark Regeneration

```bash
PYTHONPATH=src python benchmark/run_all.py
PYTHONPATH=src python -m pytest -q
```

Run only the random-forest baseline:

```bash
python run_benchmark.py --baseline rf
```

## Fixed Seeds And Splits

- Benchmark global seed: `229026`
- Generalization catalog seed: `229043`
- Split rule: phase-index splits within every frequency-amplitude cell
- Main split counts: `train: 1848 null + 1848 signal`, `validation: 660 null + 660 signal`, `test: 660 null + 660 signal`

## Main Outputs

- `data/processed/canonical_published_subset.csv`
- `data/processed/primary_peak_b.csv`
- `data/processed/secondary_peaks_bc.csv`
- `data/processed/strict_qc_subset.csv`
- `data/interim/benchmark_catalog.csv`
- `data/interim/benchmark_arrays.npz`
- `results/tables/benchmark_protocol_fixed.json`
- `results/tables/split_counts_fixed.csv`
- `results/tables/baseline_comparison_with_rf.csv`
- `results/tables/final_representative_frequency_baseline_table_with_rf.csv`
- `results/tables/a95_vs_frequency_with_uncertainty.csv`
- `results/tables/observed_series_false_alarm_behavior.csv`
- `results/tables/subset_reconciliation_scan_keys.csv`
- `results/rf_baseline_results.json`
- `results/figures/final_peak_b_residuals_over_time_by_crystal.png`
- `results/figures/final_auc_heatmap_with_a95_contour.png`
- `results/figures/final_a95_vs_frequency_with_uncertainty.png`

## Notes

- The benchmark treats the published scan-level frequencies as already silicon-cavity-drift corrected.
- The default task uses peak b only.
- Peak c is not used as a headline benchmark dependency.
- The X2-only observed-series low empirical p-value is treated as scatter/model mismatch, not as a signal.
