from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import FIGURES_DIR, PROCESSED_DIR, TABLES_DIR
from .utils import weighted_linear_fit, write_text


EXPORT_DROP_COLUMNS = {"source_row_index"}


def _frame_as_code_block(frame: pd.DataFrame) -> str:
    return "```\n" + frame.to_string(index=False) + "\n```"


def _export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.drop(columns=[col for col in EXPORT_DROP_COLUMNS if col in frame.columns]).copy()
    if any(col.startswith("Unnamed") for col in out.columns):
        raise ValueError("Export frame contains an accidental Unnamed index column.")
    return out


def save_processed_csvs(prepared: dict[str, Any]) -> dict[str, str]:
    outputs = {}
    canonical_path = PROCESSED_DIR / "canonical_published_subset.csv"
    strict_path = PROCESSED_DIR / "strict_qc_subset.csv"
    default_path = PROCESSED_DIR / "processed_default.csv"
    peak_b_path = PROCESSED_DIR / "primary_peak_b.csv"
    peaks_bc_path = PROCESSED_DIR / "secondary_peaks_bc.csv"

    _export_frame(prepared["canonical"]).to_csv(canonical_path, index=False)
    _export_frame(prepared["strict_qc"]).to_csv(strict_path, index=False)
    _export_frame(prepared["views"]["processed_default"]).to_csv(default_path, index=False)
    _export_frame(prepared["views"]["primary_peak_b"]).to_csv(peak_b_path, index=False)
    _export_frame(prepared["views"]["secondary_peaks_bc"]).to_csv(peaks_bc_path, index=False)

    outputs.update(
        {
            "canonical_published_subset": str(canonical_path),
            "strict_qc_subset": str(strict_path),
            "processed_default": str(default_path),
            "primary_peak_b": str(peak_b_path),
            "secondary_peaks_bc": str(peaks_bc_path),
        }
    )
    return outputs


def _row_count_summary(prepared: dict[str, Any]) -> pd.DataFrame:
    variants = prepared["variants"]
    views = prepared["views"]
    rows = [
        ("raw_csv_rows", len(prepared["raw"])),
        (
            "raw_unique_lorentzian_scan_keys",
            prepared["raw"].loc[
                prepared["raw"]["fitting_function"].eq("lorentzian"), ["scan_time_utc", "target", "peak"]
            ].drop_duplicates().shape[0],
        ),
        ("literal_readme_subset_rows", len(variants.literal_readme)),
        ("canonical_published_subset_rows", len(prepared["canonical"])),
        ("strict_qc_subset_rows", len(prepared["strict_qc"])),
        ("primary_peak_b_rows", len(views["primary_peak_b"])),
        ("secondary_peaks_bc_rows", len(views["secondary_peaks_bc"])),
    ]
    return pd.DataFrame(rows, columns=["dataset", "row_count"])


def _per_crystal_counts(processed_df: pd.DataFrame) -> pd.DataFrame:
    return (
        processed_df.groupby(["target", "peak"])
        .size()
        .rename("row_count")
        .reset_index()
        .sort_values(["target", "peak"])
    )


def _uncertainty_summary(processed_df: pd.DataFrame) -> pd.DataFrame:
    grouped = processed_df.groupby(["target", "peak"])
    rows = []
    for (target, peak), group in grouped:
        rows.append(
            {
                "target": target,
                "peak": peak,
                "row_count": len(group),
                "median_freq_unc_hz": float(group["freq_unc_hz"].median()),
                "p90_freq_unc_hz": float(group["freq_unc_hz"].quantile(0.9)),
                "residual_std_hz": float(group["residual_hz"].std(ddof=1)),
                "median_abs_residual_hz": float(group["residual_hz"].abs().median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "peak"])


def _drift_checks(processed_df: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "all_processed_rows": processed_df.copy(),
        "peak_b_only": processed_df.loc[processed_df["peak"].eq("b")].copy(),
        "peak_b_c10_c13_only": processed_df.loc[
            processed_df["peak"].eq("b") & processed_df["target"].isin(["C10", "C13"])
        ].copy(),
        "peak_b_x2_only": processed_df.loc[
            processed_df["peak"].eq("b") & processed_df["target"].eq("X2")
        ].copy(),
    }
    rows = []
    for name, frame in checks.items():
        fit = weighted_linear_fit(
            x=frame["days_since_first_observation"].to_numpy(dtype=np.float64),
            y=frame["residual_hz"].to_numpy(dtype=np.float64),
            sigma=frame["freq_unc_hz"].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "subset": name,
                "n_rows": len(frame),
                "slope_hz_per_day": fit["slope"],
                "slope_stderr_hz_per_day": fit["slope_stderr"],
                "weighted_rms_hz": fit["weighted_rms"],
                "reduced_chi2": fit["reduced_chi2"],
            }
        )
    return pd.DataFrame(rows)


def _scan_key_set(frame: pd.DataFrame) -> set[tuple[pd.Timestamp, str, str]]:
    keys = frame[["scan_time_utc", "target", "peak"]].drop_duplicates()
    return set(keys.itertuples(index=False, name=None))


def _subset_reconciliation_scan_keys(prepared: dict[str, Any]) -> pd.DataFrame:
    raw = prepared["raw"]
    variants = prepared["variants"]
    canonical = prepared["canonical"]
    literal_keys = _scan_key_set(variants.literal_readme)
    canonical_keys = _scan_key_set(canonical)
    lorentzian = raw.loc[raw["fitting_function"].eq("lorentzian")].copy()
    rows = []
    for key, group in lorentzian.groupby(["scan_time_utc", "target", "peak"], sort=True):
        scan_time, target, peak = key
        corrections = sorted(group["correction"].astype(str).unique())
        has_mem_int = "mem int" in corrections
        has_mem = "mem" in corrections
        included_literal = key in literal_keys
        included_canonical = key in canonical_keys
        if included_literal and has_mem_int:
            reason = "included by literal readme rule because mem int Lorentzian row exists"
        elif included_literal and has_mem and target == "X2" and scan_time.year == 2024 and scan_time.month == 5:
            reason = "included by literal readme May 2024 X2 mem fallback"
        elif included_canonical and not included_literal and has_mem and not has_mem_int:
            reason = "included only by file-backed mem fallback to reconcile 73 Lorentzian scan keys"
        elif included_canonical:
            reason = "included by canonical fallback rule"
        else:
            reason = "excluded from both published-subset rules"
        rows.append(
            {
                "timestamp_utc": scan_time.isoformat(),
                "target": target,
                "peak": peak,
                "correction_rows_present": "|".join(corrections),
                "has_mem_int": has_mem_int,
                "has_mem": has_mem,
                "included_by_literal_readme_rule": included_literal,
                "included_by_canonical_fallback_rule": included_canonical,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _plot_residuals_over_time(primary_peak_b: pd.DataFrame) -> str:
    figure_path = FIGURES_DIR / "final_peak_b_residuals_over_time_by_crystal.png"
    colors = {"C10": "#1f77b4", "C13": "#2ca02c", "X2": "#d62728"}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axhspan(-220.0, 220.0, color="#8ecae6", alpha=0.18, label="C10/C13 220 Hz reference band")
    for target, group in primary_peak_b.groupby("target"):
        ax.errorbar(
            group["days_since_first_observation"],
            group["residual_hz"],
            yerr=group["freq_unc_hz"],
            fmt="o",
            ms=5,
            alpha=0.85,
            label=target,
            color=colors.get(target, None),
        )
    ax.axhline(0.0, color="black", lw=1.0, ls="--")
    ax.set_xlabel("Days Since First Observation")
    ax.set_ylabel("Temperature-Corrected Residual (Hz)")
    ax.set_title("Peak-b residuals over time by crystal")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return str(figure_path)


def _plot_scatter_uncertainty(primary_peak_b: pd.DataFrame) -> str:
    figure_path = FIGURES_DIR / "final_residual_scatter_uncertainty_by_crystal.png"
    colors = {"C10": "#1f77b4", "C13": "#2ca02c", "X2": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for target, group in primary_peak_b.groupby("target"):
        axes[0].scatter(
            group["freq_unc_hz"],
            group["residual_hz"].abs(),
            s=40,
            alpha=0.8,
            color=colors.get(target, None),
            label=target,
        )
        axes[1].scatter(
            np.full(len(group), target),
            group["residual_hz"],
            s=35,
            alpha=0.8,
            color=colors.get(target, None),
        )
    axes[0].set_xlabel("Reported Frequency Uncertainty (Hz)")
    axes[0].set_ylabel("|Residual| (Hz)")
    axes[0].set_title("Residual size versus reported uncertainty")
    axes[1].axhline(0.0, color="black", lw=1.0, ls="--")
    axes[1].set_xlabel("Crystal")
    axes[1].set_ylabel("Residual (Hz)")
    axes[1].set_title("Residual spread by crystal")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return str(figure_path)


def _plot_residuals_vs_temperature(processed_df: pd.DataFrame) -> str:
    figure_path = FIGURES_DIR / "final_residuals_vs_temperature_diagnostic.png"
    colors = {"b": "#1f77b4", "c": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    plotted_any = False
    for peak, group in processed_df.groupby("peak"):
        if group["temperature_model_available"].all():
            ax.scatter(
                group["temp_k"],
                group["residual_hz"],
                s=40,
                alpha=0.8,
                label=f"peak {peak}",
                color=colors.get(peak, None),
            )
            plotted_any = True
    if plotted_any:
        ax.axhline(0.0, color="black", lw=1.0, ls="--")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Residual After Temperature Correction (Hz)")
        ax.set_title("Residual-versus-temperature diagnostic")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return str(figure_path)


def generate_diagnostics(prepared: dict[str, Any]) -> dict[str, Any]:
    processed = prepared["views"]["processed_default"].copy()
    primary_peak_b = prepared["views"]["primary_peak_b"].copy()

    row_counts = _row_count_summary(prepared)
    per_crystal = _per_crystal_counts(processed)
    uncertainty = _uncertainty_summary(processed)
    drift = _drift_checks(processed)

    row_count_path = TABLES_DIR / "row_count_summary.csv"
    preprocessing_summary_path = TABLES_DIR / "preprocessing_summary.csv"
    per_crystal_path = TABLES_DIR / "per_crystal_peak_counts.csv"
    uncertainty_path = TABLES_DIR / "uncertainty_diagnostics.csv"
    drift_path = TABLES_DIR / "drift_checks.csv"
    dataset_summary_path = TABLES_DIR / "main_preprocessing_dataset_summary.csv"
    reconciliation_scan_keys_path = TABLES_DIR / "subset_reconciliation_scan_keys.csv"

    row_counts.to_csv(row_count_path, index=False)
    row_counts.to_csv(preprocessing_summary_path, index=False)
    per_crystal.to_csv(per_crystal_path, index=False)
    uncertainty.to_csv(uncertainty_path, index=False)
    drift.to_csv(drift_path, index=False)
    per_crystal.to_csv(dataset_summary_path, index=False)
    reconciliation_scan_keys = _subset_reconciliation_scan_keys(prepared)
    reconciliation_scan_keys.to_csv(reconciliation_scan_keys_path, index=False)

    summary_markdown = "\n".join(
        [
            "# Main Preprocessing Summary",
            "",
            "## Row Counts",
            _frame_as_code_block(row_counts),
            "",
            "## Per-Crystal / Per-Peak Counts",
            _frame_as_code_block(per_crystal),
            "",
            "## Drift Checks",
            _frame_as_code_block(drift),
            "",
            "## Uncertainty Diagnostics",
            _frame_as_code_block(uncertainty),
        ]
    )
    write_text(summary_markdown, TABLES_DIR / "main_preprocessing_dataset_summary.md")
    reconciliation = "\n".join(
        [
            "# Subset Reconciliation",
            "",
            "The raw CSV has 73 unique Lorentzian scan keys after grouping by UTC scan time, crystal target, and peak.",
            "A literal reading of `read_me.docx` selects Lorentzian `mem int` rows plus only the first May 2024 X2 `mem` fallback rows, which gives 72 rows.",
            "The file-backed canonical subset used here selects Lorentzian `mem int` when available and otherwise uses the Lorentzian `mem` row for scans without a `mem int` row.",
            "That rule yields 73 rows, 55 peak-b rows, and 73 peak-b+c rows.",
            "",
            "The machine-readable companion table is `results/tables/subset_reconciliation_scan_keys.csv`. It lists every Lorentzian scan key, the correction rows present, the literal-readme inclusion flag, the canonical-fallback inclusion flag, and the inclusion reason.",
            "",
            "The non-May reconciliation row is the 2024-09-25 X2 peak-c scan. It lacks a `mem int` row in the CSV and in its per-folder fits, so it is included only through the explicit `mem` fallback rule.",
            "",
            "Peak c remains optional metadata and an exploratory ablation. It is not a headline dependency of the upgraded benchmark.",
        ]
    )
    write_text(reconciliation, TABLES_DIR / "subset_reconciliation.md")

    figures = {
        "real_temperature_corrected_residuals_by_crystal": _plot_residuals_over_time(primary_peak_b),
        "residual_scatter_uncertainty_behavior": _plot_scatter_uncertainty(primary_peak_b),
    }

    return {
        "tables": {
            "row_count_summary": str(row_count_path),
            "preprocessing_summary": str(preprocessing_summary_path),
            "per_crystal_peak_counts": str(per_crystal_path),
            "uncertainty_diagnostics": str(uncertainty_path),
            "drift_checks": str(drift_path),
            "main_preprocessing_dataset_summary": str(dataset_summary_path),
            "subset_reconciliation_scan_keys": str(reconciliation_scan_keys_path),
        },
        "figures": figures,
        "row_counts": row_counts,
        "per_crystal": per_crystal,
        "uncertainty": uncertainty,
        "drift": drift,
    }
