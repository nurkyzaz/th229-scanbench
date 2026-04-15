# Th229-ScanBench

Th229-ScanBench is a dataset and benchmark package for detecting injected time-varying frequency signals on the published JILA Thorium-229 scan-level frequency record.

The main benchmark is intentionally narrow:

- Data source: published JILA scan-level frequency records.
- Main task: binary detection of an added sinusoidal frequency modulation.
- Main series: 55 temperature-corrected peak-b measurements.
- Default null model: crystal-specific Gaussian scatter with an X2 outlier component.
- Peak c: retained only as optional/exploratory metadata.
- Within-scan pkl files: inventoried for future work, not used in the headline benchmark.

## Repository Layout

```text
baselines/          Random-forest baseline entry point.
benchmark/          End-to-end benchmark runner.
data/raw/           Source files used as inputs.
data/processed/     Canonical scan-level subsets.
data/interim/       Generated benchmark catalogs and arrays.
results/figures/    Benchmark figures.
results/tables/     Metrics, audits, calibration tables, and split files.
src/th229_bench/    Preprocessing, synthetic injection, null models, and baselines.
tests/              Validation tests.
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Reproduce The Benchmark

Run the full benchmark and tests from the repository root:

```bash
PYTHONPATH=src python benchmark/run_all.py
PYTHONPATH=src python -m pytest -q
```

Run only the random-forest baseline:

```bash
python run_benchmark.py --baseline rf
```

## Source Inputs

The authoritative source files are kept under `data/raw/`:

- `JILA_Jan2026_paper.pdf`
- `Fuchs_2025.pdf`
- `2025-11-13_Th_record_db.csv`
- `read_me.docx`
- `Freq reproducibility data.zip`
- `Freq reproducibility data/`

## Authoritative Counts

- Raw CSV rows: `428`
- Unique Lorentzian scan keys: `73`
- Literal readme subset rows: `72`
- Canonical published subset rows: `73`
- Primary peak-b rows: `55`
- Optional peak-b+c rows: `73`
- Strict QC rows with `chisq_red <= 10`: `41`
- Main benchmark examples: `6336`
- Test-only prediction rows: `1320`

The 72-versus-73 reconciliation is documented in `results/tables/subset_reconciliation.md` and `results/tables/subset_reconciliation_scan_keys.csv`.

## Fixed Splits

```text
split       label  count
train       null   1848
train       signal 1848
validation  null   660
validation  signal 660
test        null   660
test        signal 660
```

Splits are fixed by phase index within every period-amplitude cell.

## Current Baselines

- Weighted harmonic regression.
- Generalized Lomb-Scargle.
- Hierarchical sinusoid with crystal-specific jitter.
- Random forest on periodogram and summary features.

The random-forest baseline is included to show that the benchmark is usable by generic ML methods, but the strongest current safe method remains the hierarchical sinusoid model.

## Key Outputs

- `REPRODUCIBILITY.md`
- `data/processed/canonical_published_subset.csv`
- `data/processed/primary_peak_b.csv`
- `data/processed/secondary_peaks_bc.csv`
- `data/processed/strict_qc_subset.csv`
- `data/interim/benchmark_catalog.csv`
- `data/interim/benchmark_arrays.npz`
- `results/tables/baseline_comparison_with_rf.csv`
- `results/tables/final_representative_frequency_baseline_table_with_rf.csv`
- `results/tables/a95_vs_frequency_with_uncertainty.csv`
- `results/tables/observed_series_false_alarm_behavior.csv`
- `results/tables/subset_reconciliation_scan_keys.csv`
- `results/rf_baseline_results.json`
- `results/figures/final_peak_b_residuals_over_time_by_crystal.png`
- `results/figures/final_auc_heatmap_with_a95_contour.png`
- `results/figures/final_a95_vs_frequency_with_uncertainty.png`

## Caveats

- The benchmark is an injection-and-recovery task, not an observed-signal claim.
- X2-only observed residuals remain imperfectly calibrated under the selected null model; this is treated as scatter/model mismatch.
- A95 intervals quantify validation-threshold and phase-resampling variation only, not full experimental uncertainty.
- Peak-c and within-scan lineshape data remain future-work branches.

## License

Code in this repository is released under the MIT license. Curated JILA-derived data follow the upstream Ooi et al. CC BY 4.0 terms.
