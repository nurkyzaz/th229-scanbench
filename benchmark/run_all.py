from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.th229_bench.audit import write_audit_artifacts
from th229_bench.benchmarking import run_benchmark_suite
from th229_bench.diagnostics import generate_diagnostics, save_processed_csvs
from th229_bench.lineshape_loader import build_lineshape_manifest
from th229_bench.paths import FIGURES_DIR, INTERIM_DIR, PROJECT_ROOT, TABLES_DIR
from th229_bench.preprocessing import prepare_preprocessed_data
from th229_bench.utils import write_json, write_text

def _path(name: str) -> str:
    return name


def _remove_stale_outputs() -> None:
    """Remove artifacts that are known to be stale or outside the upgrade scope."""
    for figure in FIGURES_DIR.glob("*.png"):
        figure.unlink()

    stale_tables = [
        "injection_amplitude_grid.csv",
        "injection_families.json",
        "injection_frequency_grid.csv",
        "real_data_calibration_note.json",
        "real_data_calibration_summary.csv",
        "real_data_periodogram.csv",
        "split_counts.csv",
    ]
    for name in stale_tables:
        path = TABLES_DIR / name
        if path.exists():
            path.unlink()


def _counts(prepared: dict[str, Any]) -> dict[str, int]:
    raw = prepared["raw"]
    return {
        "raw_csv_rows": int(len(raw)),
        "raw_unique_lorentzian_scan_keys": int(
            raw.loc[raw["fitting_function"].eq("lorentzian"), ["scan_time_utc", "target", "peak"]]
            .drop_duplicates()
            .shape[0]
        ),
        "literal_readme_subset_rows": int(len(prepared["variants"].literal_readme)),
        "canonical_published_subset_rows": int(len(prepared["canonical"])),
        "strict_qc_subset_rows": int(len(prepared["strict_qc"])),
        "primary_peak_b_rows": int(len(prepared["views"]["primary_peak_b"])),
        "secondary_peaks_bc_rows": int(len(prepared["views"]["secondary_peaks_bc"])),
    }


def _write_lineshape_audit(manifest: pd.DataFrame) -> str:
    path = TABLES_DIR / "lineshape_audit.md"
    corrected = int(manifest["has_corrected_data"].sum())
    intensity = int(manifest["has_intensity"].sum())
    usable = int(manifest["usable_future_branch"].sum())
    by_peak = manifest.groupby(["target", "peak"]).size().rename("count").reset_index()
    lines = [
        "# Lineshape Audit",
        "",
        f"- Scan folders inspected: `{len(manifest)}`.",
        f"- Folders with `*_data_corr.pkl`: `{corrected}`.",
        f"- Folders with explicit intensity pickle files: `{intensity}`.",
        f"- Folders passing the conservative future-branch manifest check: `{usable}`.",
        "- The pkl data are retained as a future-work branch only; the primary benchmark remains scan-level peak-b modulation detection.",
        "- Within-scan lineshape reconstruction is not a headline task in this upgrade because corrected arrays are not uniformly present across all scans.",
        "",
        "## Folder Counts By Target And Peak",
        "```",
        by_peak.to_string(index=False),
        "```",
    ]
    write_text("\n".join(lines), path)
    return str(path)


def _write_upgrade_docs(
    prepared: dict[str, Any],
    processed_paths: dict[str, str],
    diagnostics: dict[str, Any],
    benchmark: dict[str, Any],
    lineshape_manifest_path: str,
    lineshape_audit_path: str,
    audit_outputs: dict[str, Any],
) -> dict[str, str]:
    counts = _counts(prepared)
    split_counts = benchmark["split_counts"].copy()
    split_text = split_counts.to_string(index=False)
    baseline = benchmark["comparison"].sort_values("roc_auc", ascending=False)
    null_comparison = pd.read_csv(TABLES_DIR / "null_model_comparison.csv")
    null_row = null_comparison.loc[null_comparison["is_default"].astype(bool)].iloc[0]
    ablation = benchmark["ablation"].copy()
    calibration = benchmark["calibration"].copy()
    x2_study = benchmark["x2_calibration_study"].copy()
    generalization = benchmark["generalization"]["results"].copy()
    drift_sanity = benchmark["drift_sanity"]["results"].copy()
    a95_uncertainty = benchmark["a95_uncertainty"].copy()
    time_span_days = float(prepared["raw"]["days_since_first_observation"].max())

    safe_outputs = [
        _path("results/tables/preprocessing_audit.json"),
        _path("results/tables/subset_reconciliation.md"),
        _path("results/tables/subset_reconciliation_scan_keys.csv"),
        _path("results/tables/null_model_comparison.csv"),
        _path("results/tables/x2_calibration_study.csv"),
        _path("results/tables/x2_handling_decision.md"),
        _path("results/tables/baseline_comparison.csv"),
        _path("results/tables/a95_vs_frequency_with_uncertainty.csv"),
        _path("results/tables/a95_uncertainty_notes.md"),
        _path("results/tables/generalization_protocol.json"),
        _path("results/tables/generalization_results.csv"),
        _path("results/tables/generalization_notes.md"),
        _path("results/tables/drift_sanity_results.csv"),
        _path("results/tables/drift_sanity_protocol.json"),
        _path("results/tables/drift_sanity_notes.md"),
        _path("results/tables/auc_heatmap_with_a95_contour.csv"),
        _path("results/tables/final_representative_frequency_baseline_table.csv"),
        _path("results/tables/benchmark_sensitivity_heatmap.csv"),
        _path("results/tables/observed_series_false_alarm_behavior.csv"),
        _path("results/tables/ablation_results.csv"),
        _path("results/figures/final_peak_b_residuals_over_time_by_crystal.png"),
        _path("results/figures/final_residual_scatter_uncertainty_by_crystal.png"),
        _path("results/figures/final_null_model_calibration_comparison.png"),
        _path("results/figures/final_sensitivity_heatmap_main_baseline.png"),
        _path("results/figures/final_a95_vs_frequency_with_uncertainty.png"),
        _path("results/figures/final_observed_series_periodogram_calibration.png"),
        _path("results/figures/final_generalization_summary.png"),
        _path("results/figures/final_auc_heatmap_with_a95_contour.png"),
        _path("results/figures/final_drift_sanity_curve.png"),
        _path("results/tables/figures_manifest.md"),
    ]

    report = [
        "# Thorium-229 Benchmark Final Technical Report",
        "",
        "## Benchmark Definition",
        "The package uses the published JILA scan-level frequency record. The main task uses the 55 peak-b measurements and asks whether an added time-domain frequency modulation is present on the observed JILA cadence.",
        "",
        "Peak c is retained only as optional metadata and an exploratory table row. The pkl scan arrays remain a future-work branch through `data/interim/lineshape_manifest.csv`.",
        "",
        "## What Changed In This Final Revision",
        f"- Canonical scan-level subset: `{counts['canonical_published_subset_rows']}` rows.",
        f"- Primary peak-b subset: `{counts['primary_peak_b_rows']}` rows.",
        f"- Optional peak-b+c subset: `{counts['secondary_peaks_bc_rows']}` rows.",
        f"- Strict QC subset with `chisq_red <= 10`: `{counts['strict_qc_subset_rows']}` rows.",
        "- Added `results/tables/subset_reconciliation_scan_keys.csv` so the 72-versus-73 issue is auditable scan by scan.",
        "- Added fitted-tail and X2-mixture null candidates, then recorded the X2 decision in `results/tables/x2_handling_decision.md`.",
        "- Added A95 bootstrap intervals in `results/tables/a95_vs_frequency_with_uncertainty.csv`.",
        "- Added a secondary off-grid held-out-period test in `results/tables/generalization_results.csv`.",
        "- Removed the point-only A95 figure from the final figure set and replaced it with `final_a95_vs_frequency_with_uncertainty.png`.",
        "- After reviewing the example draft, added a secondary linear-drift sanity task, an AUC heatmap with an A95 contour, a representative-period baseline table, and a residual-over-time figure with a `+/-220` Hz reference band.",
        "",
        "## Preprocessing",
        f"- Timestamp span verified: `{time_span_days:.6f}` days.",
        "- Processed CSV exports now drop the raw `Unnamed: 0` index column and fail if a stale `Unnamed` column would be saved.",
        "- Peak-b temperature correction is fitted from canonical C10+C13 records and remains the default preprocessing path.",
        "- Peak-c correction is treated as data-derived exploratory support only; it is not used by the main task.",
        "",
        "## X2 Handling",
        "- `formal_gaussian_scaled` fits one global uncertainty inflation factor on the observation-level training subset.",
        "- `crystal_gaussian_jitter` fits a separate extra variance term for each crystal, allowing X2 to be wider than C10/C13.",
        "- `crystal_student_t_jitter` fits crystal-specific jitter with Student-t tails using fixed 4 degrees of freedom.",
        "- `crystal_student_t_fitted_df` fits target-specific Student-t degrees of freedom on training residuals.",
        "- `crystal_gaussian_x2_mixture` fits an X2-only Gaussian outlier component on X2 training residuals.",
        f"- Default null model selected by held-out residual negative log likelihood: `{benchmark['default_null_model']}`.",
        f"- Default null held-out NLL per residual: `{float(null_row['holdout_nll_per_point']):.6g}`.",
        "- X2 remains in the default peak-b task because it is part of the published peak-b record.",
        "- C10+C13 peak b is now an official clean secondary condition, meaning the same task is rerun without X2 to show how much the X2 scatter affects conclusions.",
        "```",
        x2_study.to_string(index=False),
        "```",
        "",
        "## Baselines",
        "- `weighted_harmonic_regression`: uncertainty-weighted sinusoid fit with crystal offsets; this remains the main simple baseline.",
        "- `generalized_lomb_scargle`: global-offset weighted periodogram baseline on the same search grid.",
        "- `hierarchical_sinusoid_jitter`: shared sinusoid with crystal-specific offsets and the selected crystal-aware null variance; this is the main physics-aware comparison.",
        "",
        "## Evaluation Protocol",
        "- One authoritative split is used everywhere: phase-index splits within each frequency-amplitude cell.",
        "- Null-model parameters are fitted on a target-stratified chronological subset of the published observed peak-b residuals.",
        "- Detection thresholds are calibrated on validation null injections only.",
        "- Test examples are never used for threshold calibration or model selection.",
        "- The off-grid held-out-period test calibrates thresholds on source-grid validation null examples and evaluates only held-out-period test examples.",
        "",
        "## Observed-Series Calibration Diagnostic",
        f"- {benchmark['calibration_note']}",
        "- The all-peak-b observed series is used as a calibration diagnostic, not as a detection claim.",
        "- The X2-only subset can produce a low empirical p-value under the selected null; this is documented as residual scatter or model mismatch, not evidence for a physical signal.",
        "```",
        calibration.to_string(index=False),
        "```",
        "",
        "## Split Counts",
        "```",
        split_text,
        "```",
        "",
        "## A95 Uncertainty",
        "- A95 intervals resample validation-null scores for the threshold and the five test phase realizations in each period-amplitude cell.",
        "- The intervals quantify finite synthetic-grid and phase-sampling uncertainty only.",
        "- They do not include uncertainty in the measured frequencies, temperature correction, or null-model choice.",
        "```",
        a95_uncertainty.head(24).to_string(index=False),
        "```",
        "",
        "## Held-Out-Period Generalization",
        "- The secondary test uses off-grid periods that are absent from the main period grid.",
        "- Thresholds come from main-grid validation null examples.",
        "- Metrics are computed on off-grid held-out-period test examples.",
        "```",
        generalization.to_string(index=False),
        "```",
        "",
        "## Draft-Review Additions",
        "- The example draft was weaker than the current project overall because it contained count errors and proposed methods that are less well matched to the 55-point peak-b task.",
        "- Useful draft ideas were adopted where they strengthened the current package without changing scope.",
        "- The linear-drift sanity task injects slopes in Hz/day on the same peak-b cadence and calibrates its threshold on validation null examples.",
        "- The representative-period table reports the safe baselines at 30 and 180 days.",
        "- The AUC heatmap with A95 contour is generated for `hierarchical_sinusoid_jitter`, the current strongest safe baseline.",
        "```",
        drift_sanity.head(1).drop(columns=["slope_abs_hz_per_day", "detection_rate_at_validation_fpr5", "n_test_signal"]).to_string(index=False),
        "```",
        "",
        "## Baseline Summary",
        "```",
        baseline.to_string(index=False),
        "```",
        "",
        "## Ablation Summary",
        "```",
        ablation.to_string(index=False),
        "```",
        "",
        "## Lineshape Future Branch",
        f"- Manifest: `{lineshape_manifest_path}`.",
        f"- Audit: `{lineshape_audit_path}`.",
        "- The pkl files are usable for a future branch manifest and loader, but not promoted into the main benchmark in this upgrade.",
        "",
        "## Safest Outputs For External Reporting",
        *[f"- `{path}`" for path in safe_outputs],
        "",
        "## Caveats That Must Remain",
        "- Verify the peak-b temperature coefficient treatment against the final intended citation/table wording.",
        "- Decide whether the 2024-09-25 X2 peak-c fallback row should be described as canonical or optional in any eventual manuscript.",
        "- State clearly that the X2-only observed calibration diagnostic is not a signal claim.",
        "- State clearly that A95 intervals are synthetic-resampling intervals, not full experimental uncertainty intervals.",
        "- Review all figure captions before external use; they are technical notes, not final publication text.",
    ]

    release = [
        "# Final Technical Release Summary",
        "",
        "This technical release keeps the main task fixed at peak-b scan-level modulation detection and adds the final calibration, uncertainty, reconciliation, and generalization artifacts needed for external review.",
        "",
        "After the example draft review, the release also includes the adopted draft improvements that were justified by the current package: a secondary linear-drift sanity task, an AUC heatmap with an A95 contour, a representative-period baseline table, and updated all-safe-baseline A95 curves.",
        "",
        "## Authoritative Counts",
        *[f"- `{key}`: `{value}`" for key, value in counts.items()],
        "",
        "## Authoritative Split Counts",
        "```",
        split_text,
        "```",
        "",
        "## Key Output Paths",
        *[f"- `{path}`" for path in safe_outputs],
        "",
        "## Not In Scope",
        "- No publication manuscript is generated by this public benchmark command.",
        "- No within-scan lineshape benchmark was promoted to a headline result.",
    ]

    reproducibility = [
        "# Final Technical Reproducibility Notes",
        "",
        "## Environment",
        "- Python environment is specified by `requirements.txt`.",
        "- Run from the project root.",
        "",
        "## Full Regeneration Command",
        "```bash",
        "PYTHONPATH=src python benchmark/run_all.py",
        "```",
        "",
        "## Validation Command",
        "```bash",
        "PYTHONPATH=src python -m pytest -q",
        "```",
        "",
        "## Fixed Seeds",
        f"- Benchmark global seed: `{benchmark['config'].global_seed}`.",
        "- Each injected example has a deterministic `noise_seed` saved in `data/interim/benchmark_catalog.csv`.",
        "- The held-out-period generalization catalog has deterministic seeds saved in `data/interim/generalization_heldout_periods_catalog.csv`.",
        "",
        "## Source Files Used",
        f"- `{_path('data/raw/JILA_Jan2026_paper.pdf')}`",
        f"- `{_path('data/raw/Fuchs_2025.pdf')}`",
        f"- `{_path('data/raw/2025-11-13_Th_record_db.csv')}`",
        f"- `{_path('data/raw/read_me.docx')}`",
        f"- `{_path('data/raw/Freq reproducibility data.zip')}`",
        "",
        "## Freshness Checks",
        "- Processed CSVs are regenerated by the full command.",
        "- Benchmark figure manifests are written under `results/tables/`; no paper artifacts are generated by the public benchmark run.",
        "- Debug/old figures in `results/figures` are removed before final figures are regenerated.",
    ]

    docs = {
        "benchmark_final_technical_report": str(PROJECT_ROOT / "BENCHMARK_FINAL_TECHNICAL_REPORT.md"),
        "release_summary_final_tech": str(PROJECT_ROOT / "RELEASE_SUMMARY_FINAL_TECH.md"),
        "reproducibility_final_tech": str(PROJECT_ROOT / "REPRODUCIBILITY_FINAL_TECH.md"),
        "benchmark_upgrade_report": str(PROJECT_ROOT / "BENCHMARK_UPGRADE_REPORT.md"),
        "release_summary_upgraded": str(PROJECT_ROOT / "RELEASE_SUMMARY_UPGRADED.md"),
        "reproducibility_upgraded": str(PROJECT_ROOT / "REPRODUCIBILITY_UPGRADED.md"),
    }
    for path in (PROJECT_ROOT / "BENCHMARK_FINAL_TECHNICAL_REPORT.md", PROJECT_ROOT / "BENCHMARK_UPGRADE_REPORT.md"):
        write_text("\n".join(report), path)
    for path in (PROJECT_ROOT / "RELEASE_SUMMARY_FINAL_TECH.md", PROJECT_ROOT / "RELEASE_SUMMARY_UPGRADED.md"):
        write_text("\n".join(release), path)
    for path in (PROJECT_ROOT / "REPRODUCIBILITY_FINAL_TECH.md", PROJECT_ROOT / "REPRODUCIBILITY_UPGRADED.md"):
        write_text("\n".join(reproducibility), path)

    summary = {
        "counts": counts,
        "time_span_days": time_span_days,
        "default_null_model": benchmark["default_null_model"],
        "split_counts": split_counts.to_dict(orient="records"),
        "x2_handling": str(TABLES_DIR / "x2_handling_decision.md"),
        "generalization": benchmark["generalization"]["results"].to_dict(orient="records"),
        "processed_paths": processed_paths,
        "diagnostic_tables": diagnostics["tables"],
        "diagnostic_figures": diagnostics["figures"],
        "benchmark_figures": benchmark["figures"],
        "lineshape_manifest": lineshape_manifest_path,
        "lineshape_audit": lineshape_audit_path,
        "audit_outputs": {key: value for key, value in audit_outputs.items() if key != "audit"},
        "safe_outputs_for_future_paper": safe_outputs,
    }
    write_json(summary, TABLES_DIR / "pipeline_summary.json")
    return docs


def main() -> None:
    _remove_stale_outputs()

    audit_outputs = write_audit_artifacts()
    prepared = prepare_preprocessed_data()
    processed_paths = save_processed_csvs(prepared)
    diagnostics = generate_diagnostics(prepared)

    benchmark = run_benchmark_suite(
        prepared["views"]["primary_peak_b"],
        secondary_bc_df=prepared["views"]["secondary_peaks_bc"],
    )

    lineshape_manifest = build_lineshape_manifest()
    lineshape_manifest_path = INTERIM_DIR / "lineshape_manifest.csv"
    lineshape_manifest.to_csv(lineshape_manifest_path, index=False)
    lineshape_audit_path = _write_lineshape_audit(lineshape_manifest)

    docs = _write_upgrade_docs(
        prepared,
        processed_paths,
        diagnostics,
        benchmark,
        str(lineshape_manifest_path),
        lineshape_audit_path,
        audit_outputs,
    )

    print(json.dumps({
        "counts": _counts(prepared),
        "default_null_model": benchmark["default_null_model"],
        "split_counts": benchmark["split_counts"].to_dict(orient="records"),
        "docs": docs,
    }, indent=2))


if __name__ == "__main__":
    main()
