from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import RAW_CSV_PATH, RAW_LINESHAPE_DIR
from .utils import (
    evaluate_quadratic,
    mjd_from_timestamps,
    unix_seconds_from_timestamps,
    weighted_quadratic_fit,
)


EXPECTED_COLUMNS = [
    "Unnamed: 0",
    "_time",
    "correction",
    "laser",
    "Fitting function",
    "Sweep range (kHz)",
    "Linear baseline subtraction",
    "peak",
    "target",
    "FWHM (kHz)",
    "FWHM unc (kHz)",
    "amp",
    "amp_unc",
    "baseline",
    "baseline_unc",
    "chisq_red",
    "Freq (Hz)",
    "Freq unc (Hz)",
    "Temp (K)",
]

SCAN_KEY = ["scan_time_utc", "target", "peak"]


@dataclass
class CanonicalVariants:
    literal_readme: pd.DataFrame
    default_selected: pd.DataFrame
    missing_mem_int_scans: pd.DataFrame


def load_raw_frequency_records(csv_path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "CSV schema mismatch.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Found: {list(df.columns)}"
        )
    timestamps = pd.to_datetime(df["_time"], utc=True, errors="coerce")
    if timestamps.isna().any():
        bad_rows = df.loc[timestamps.isna(), ["_time"]]
        raise ValueError(f"Failed to parse timestamps:\n{bad_rows.to_string(index=False)}")
    if timestamps.min().year < 2000 or timestamps.max().year > 2035:
        raise ValueError("Timestamp parse produced implausible years; possible unit bug.")

    clean = df.rename(
        columns={
            "Unnamed: 0": "source_row_index",
            "_time": "scan_time_raw",
            "Fitting function": "fitting_function",
            "Sweep range (kHz)": "sweep_range_khz",
            "Linear baseline subtraction": "linear_baseline_subtraction",
            "FWHM (kHz)": "fwhm_khz",
            "FWHM unc (kHz)": "fwhm_unc_khz",
            "Freq (Hz)": "freq_hz",
            "Freq unc (Hz)": "freq_unc_hz",
            "Temp (K)": "temp_k",
        }
    ).copy()
    clean["scan_time_utc"] = timestamps
    clean["time_unix_s"] = unix_seconds_from_timestamps(clean["scan_time_utc"])
    clean["mjd"] = mjd_from_timestamps(clean["scan_time_utc"])
    clean["scan_id"] = (
        clean["scan_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        + "__"
        + clean["target"]
        + "__"
        + clean["peak"]
    )
    t0 = clean["time_unix_s"].min()
    clean["days_since_first_observation"] = (clean["time_unix_s"] - t0) / 86400.0
    clean["seconds_since_first_observation"] = clean["time_unix_s"] - t0
    return clean


def load_lineshape_inventory(lineshape_dir: Path = RAW_LINESHAPE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder_records: list[dict[str, Any]] = []
    fits_records: list[pd.DataFrame] = []

    for folder in sorted(lineshape_dir.iterdir()):
        if not folder.is_dir():
            continue
        files = sorted([path.name for path in folder.iterdir() if path.is_file()])
        fits_path = next(folder.glob("*_fits.pkl"))
        fits_df = pd.read_pickle(fits_path).reset_index()
        fits_df["_time"] = pd.to_datetime(fits_df["_time"], utc=True)
        fits_df["folder_name"] = folder.name
        fits_records.append(fits_df)
        unique_scan_rows = fits_df[["_time", "target", "peak"]].drop_duplicates()
        if len(unique_scan_rows) != 1:
            raise ValueError(f"Folder {folder.name} maps to {len(unique_scan_rows)} scan keys.")
        unique_row = unique_scan_rows.iloc[0]
        folder_records.append(
            {
                "folder_name": folder.name,
                "scan_time_utc": unique_row["_time"],
                "target": unique_row["target"],
                "peak": unique_row["peak"],
                "has_data": any(name.endswith("_data.pkl") for name in files),
                "has_data_corr": any(name.endswith("_data_corr.pkl") for name in files),
                "has_fits": any(name.endswith("_fits.pkl") for name in files),
                "has_intensity": any(name.endswith("_intensity.pkl") for name in files),
                "n_files": len(files),
                "files": files,
            }
        )

    folder_inventory = pd.DataFrame(folder_records).sort_values("scan_time_utc").reset_index(drop=True)
    fits_inventory = pd.concat(fits_records, ignore_index=True)
    return folder_inventory, fits_inventory


def choose_canonical_variants(raw_df: pd.DataFrame) -> CanonicalVariants:
    lorentzian = raw_df.loc[raw_df["fitting_function"].eq("lorentzian")].copy()
    if lorentzian.empty:
        raise ValueError("No Lorentzian rows found in frequency record database.")

    literal_readme = pd.concat(
        [
            lorentzian.loc[lorentzian["correction"].eq("mem int")],
            lorentzian.loc[
                lorentzian["target"].eq("X2")
                & lorentzian["scan_time_utc"].dt.year.eq(2024)
                & lorentzian["scan_time_utc"].dt.month.eq(5)
                & lorentzian["correction"].eq("mem")
            ],
        ],
        ignore_index=False,
    )
    literal_readme = (
        literal_readme.sort_values("scan_time_utc")
        .drop_duplicates(subset=SCAN_KEY)
        .copy()
    )
    literal_readme["canonical_strategy"] = "literal_readme_rule"

    selected_groups: list[pd.DataFrame] = []
    missing_mem_int_records: list[dict[str, Any]] = []
    for _, group in lorentzian.groupby(SCAN_KEY, sort=True):
        if (group["correction"] == "mem int").any():
            selected = group.loc[group["correction"].eq("mem int")].head(1).copy()
            selected["selection_reason"] = "mem_int_available"
        elif (group["correction"] == "mem").any():
            selected = group.loc[group["correction"].eq("mem")].head(1).copy()
            selected["selection_reason"] = "mem_fallback_missing_mem_int"
            row = selected.iloc[0]
            missing_mem_int_records.append(
                {
                    "scan_time_utc": row["scan_time_utc"].isoformat(),
                    "target": row["target"],
                    "peak": row["peak"],
                    "folder_name": row.get("folder_name", ""),
                }
            )
        else:
            raise ValueError(f"Scan group {group[SCAN_KEY].iloc[0].to_dict()} lacks mem/mem int row.")
        selected_groups.append(selected)

    default_selected = pd.concat(selected_groups, ignore_index=False).sort_values("scan_time_utc").copy()
    default_selected["canonical_strategy"] = "mem_int_else_mem_fallback"
    missing_mem_int_scans = pd.DataFrame(missing_mem_int_records)
    return CanonicalVariants(
        literal_readme=literal_readme,
        default_selected=default_selected,
        missing_mem_int_scans=missing_mem_int_scans,
    )


def attach_folder_inventory(raw_df: pd.DataFrame, folder_inventory: pd.DataFrame) -> pd.DataFrame:
    merged = raw_df.merge(folder_inventory, on=["scan_time_utc", "target", "peak"], how="left")
    if merged["folder_name"].isna().any():
        missing = merged.loc[merged["folder_name"].isna(), ["scan_time_utc", "target", "peak"]].drop_duplicates()
        raise ValueError(f"Failed to match scan rows to folder inventory:\n{missing.to_string(index=False)}")
    return merged


def build_strict_qc_subset(canonical_df: pd.DataFrame, chisq_threshold: float = 10.0) -> pd.DataFrame:
    strict = canonical_df.loc[canonical_df["chisq_red"] <= chisq_threshold].copy()
    strict["strict_qc_threshold"] = chisq_threshold
    return strict


def fit_temperature_models(canonical_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for peak in sorted(canonical_df["peak"].unique()):
        subset = canonical_df.loc[
            canonical_df["peak"].eq(peak) & canonical_df["target"].isin(["C10", "C13"])
        ].copy()
        if len(subset) < 3:
            continue
        fit = weighted_quadratic_fit(
            subset["temp_k"].to_numpy(dtype=np.float64),
            subset["freq_hz"].to_numpy(dtype=np.float64),
            subset["freq_unc_hz"].to_numpy(dtype=np.float64),
        )
        fit.update(
            {
                "peak": peak,
                "n_points": int(len(subset)),
                "fit_targets": ["C10", "C13"],
                "temperature_range_k": [
                    float(subset["temp_k"].min()),
                    float(subset["temp_k"].max()),
                ],
                "source": "data_derived_from_canonical_C10_C13",
            }
        )
        turning_point = fit["turning_point_temperature_k"]
        temp_min, temp_max = fit["temperature_range_k"]
        fit["turning_point_within_range"] = bool(
            turning_point is not None and temp_min <= turning_point <= temp_max
        )
        models[peak] = fit
    return models


def apply_temperature_corrections(
    canonical_df: pd.DataFrame,
    models: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    processed = canonical_df.copy()
    processed["temperature_model_available"] = processed["peak"].map(lambda peak: peak in models)
    processed["temperature_model_source"] = processed["peak"].map(
        lambda peak: models[peak]["source"] if peak in models else "unavailable"
    )

    trend_values = np.full(len(processed), np.nan, dtype=np.float64)
    for peak, model in models.items():
        mask = processed["peak"].eq(peak)
        trend_values[mask.to_numpy()] = evaluate_quadratic(
            model["coefficients_hz"], processed.loc[mask, "temp_k"].to_numpy(dtype=np.float64)
        )
    processed["temperature_trend_hz"] = trend_values
    processed["temperature_residual_hz"] = processed["freq_hz"] - processed["temperature_trend_hz"]

    processed["target_peak_offset_hz"] = np.nan
    for (target, peak), group in processed.groupby(["target", "peak"]):
        if not group["temperature_model_available"].all():
            continue
        weights = 1.0 / np.maximum(group["freq_unc_hz"].to_numpy(dtype=np.float64), 1.0) ** 2
        offset = float(np.average(group["temperature_residual_hz"], weights=weights))
        processed.loc[group.index, "target_peak_offset_hz"] = offset

    processed["residual_hz"] = processed["temperature_residual_hz"] - processed["target_peak_offset_hz"]

    # A conservative, data-backed noise scale used by the synthetic generator.
    processed["residual_abs_z"] = np.abs(processed["residual_hz"]) / np.maximum(
        processed["freq_unc_hz"], 1.0
    )
    return processed


def build_processed_views(
    processed_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    peak_b = processed_df.loc[processed_df["peak"].eq("b")].copy()
    peaks_bc = processed_df.loc[processed_df["peak"].isin(["b", "c"])].copy()
    return {
        "processed_default": processed_df,
        "primary_peak_b": peak_b,
        "secondary_peaks_bc": peaks_bc,
    }


def prepare_preprocessed_data() -> dict[str, Any]:
    raw_df = load_raw_frequency_records()
    folder_inventory, fits_inventory = load_lineshape_inventory()
    raw_with_folders = attach_folder_inventory(raw_df, folder_inventory)
    variants = choose_canonical_variants(raw_with_folders)
    canonical = variants.default_selected.copy()
    strict = build_strict_qc_subset(canonical)
    models = fit_temperature_models(canonical)
    processed = apply_temperature_corrections(canonical, models)
    views = build_processed_views(processed)
    return {
        "raw": raw_with_folders,
        "raw_original_columns": EXPECTED_COLUMNS,
        "folder_inventory": folder_inventory,
        "fits_inventory": fits_inventory,
        "variants": variants,
        "canonical": canonical,
        "strict_qc": strict,
        "temperature_models": models,
        "processed": processed,
        "views": views,
    }
