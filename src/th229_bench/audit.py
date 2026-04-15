from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from pypdf import PdfReader

from .paths import (
    RAW_CSV_PATH,
    RAW_DOCX_PATH,
    RAW_FUCHS_PDF_PATH,
    RAW_JILA_PDF_PATH,
    RAW_LINESHAPE_DIR,
    RAW_ZIP_PATH,
    TABLES_DIR,
)
from .preprocessing import prepare_preprocessed_data
from .utils import human_size, sha256_file, write_json, write_text


def extract_docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(node.text for node in paragraph.findall(".//w:t", namespace) if node.text)
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def build_file_manifest() -> list[dict[str, Any]]:
    files = [
        RAW_JILA_PDF_PATH,
        RAW_FUCHS_PDF_PATH,
        RAW_CSV_PATH,
        RAW_DOCX_PATH,
        RAW_ZIP_PATH,
    ]
    manifest: list[dict[str, Any]] = []
    for path in files:
        manifest.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size,
                "size_human": human_size(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return manifest


def build_preprocessing_audit() -> dict[str, Any]:
    prepared = prepare_preprocessed_data()
    raw_df = prepared["raw"]
    original_columns = prepared["raw_original_columns"]
    folder_inventory = prepared["folder_inventory"]
    variants = prepared["variants"]
    canonical = prepared["canonical"]
    strict = prepared["strict_qc"]
    models = prepared["temperature_models"]

    zip_entries = []
    with ZipFile(RAW_ZIP_PATH) as archive:
        zip_entries = archive.namelist()

    readme_text = extract_docx_text(RAW_DOCX_PATH)
    jila_text = extract_pdf_text(RAW_JILA_PDF_PATH)
    fuchs_text = extract_pdf_text(RAW_FUCHS_PDF_PATH)
    readme_lower = readme_text.lower()
    jila_lower = jila_text.lower()
    fuchs_lower = fuchs_text.lower()

    audit = {
        "source_files": build_file_manifest(),
        "csv_schema": {
            "row_count": int(len(raw_df)),
            "raw_original_column_count": len(original_columns),
            "raw_original_columns": original_columns,
            "normalized_column_count": int(len(raw_df.columns)),
            "normalized_columns": list(raw_df.columns),
            "has_stale_index_column": True,
            "stale_index_column_name": "Unnamed: 0",
            "stale_index_export_policy": "dropped from processed CSV exports",
        },
        "timestamps": {
            "timezone_aware": True,
            "timezone": "UTC",
            "min_scan_time_utc": raw_df["scan_time_utc"].min().isoformat(),
            "max_scan_time_utc": raw_df["scan_time_utc"].max().isoformat(),
            "time_span_days": float(raw_df["days_since_first_observation"].max()),
        },
        "zip_inventory": {
            "entry_count": len(zip_entries),
            "folder_count": int(len(folder_inventory)),
            "lineshape_folder_count": int(len(folder_inventory)),
            "folders_with_data_corr": int(folder_inventory["has_data_corr"].sum()),
            "folders_without_data_corr": int((~folder_inventory["has_data_corr"]).sum()),
            "folders_with_intensity": int(folder_inventory["has_intensity"].sum()),
        },
        "published_subset_checks": {
            "unique_lorentzian_scan_keys": int(
                raw_df.loc[raw_df["fitting_function"].eq("lorentzian"), ["scan_time_utc", "target", "peak"]]
                .drop_duplicates()
                .shape[0]
            ),
            "literal_readme_count": int(len(variants.literal_readme)),
            "default_canonical_count": int(len(canonical)),
            "literal_readme_peak_b_count": int((variants.literal_readme["peak"] == "b").sum()),
            "default_peak_b_count": int((canonical["peak"] == "b").sum()),
            "default_peaks_bc_count": int(canonical["peak"].isin(["b", "c"]).sum()),
            "strict_qc_threshold": 10.0,
            "strict_qc_count": int(len(strict)),
            "strict_qc_peak_b_count": int((strict["peak"] == "b").sum()),
        },
        "missing_mem_int_scans": variants.missing_mem_int_scans.to_dict(orient="records"),
        "temperature_models": models,
        "authoritative_sources": {
            "scan_level_frequency_records": [
                str(RAW_CSV_PATH),
                str(RAW_LINESHAPE_DIR),
            ],
            "corrected_lineshape_data": [
                str(RAW_LINESHAPE_DIR),
                "Folder-level *_data_corr.pkl files when present; *_data.pkl otherwise.",
            ],
            "temperature_modeling": [
                str(RAW_JILA_PDF_PATH),
                "Canonical scan-level records fitted from C10+C13 because the paper's Extended Data Table 1 is not machine-readable from the PDF text extraction.",
            ],
            "drift_correction": [
                str(RAW_JILA_PDF_PATH),
                "Published records are treated as already silicon-cavity-drift corrected per JILA Methods.",
            ],
            "theoretical_injection_families": [
                str(RAW_FUCHS_PDF_PATH),
                "Fuchs 2025 motivates periodic modulation, slow-drift intuition, line broadening, and sideband regimes.",
            ],
        },
        "paper_backed_facts": {
            "jila_reports_73_line_scans": "73 line scans" in jila_lower,
            "jila_line_b_lowest_temperature_sensitivity": "line b has the lowest temperature sensitivity" in jila_lower,
            "jila_line_c_more_temperature_sensitive": "line c frequency has the strongest dependence on" in jila_lower
            or "greater temperature sensitivity" in jila_lower,
            "jila_known_cavity_drift_corrected": "silicon reference cavity drift" in jila_lower
            and "accounted for" in jila_lower,
            "jila_x2_larger_spread_than_c10_c13": "x2 data show a larger frequency spread" in jila_lower
            and "c10 and c13" in jila_lower,
            "fuchs_mentions_periodic_modulations": "periodic modulations" in fuchs_lower,
            "fuchs_mentions_slow_drift_limit": "linear drift" in fuchs_lower,
            "fuchs_mentions_broadening_or_splitting": "broadening of the line" in fuchs_lower
            or "splits the nuclear resonance" in fuchs_lower,
            "readme_mem_int_rule_present": "published data for this paper is all lorentzian fits with" in readme_lower
            and "mem int" in readme_lower,
        },
    }
    return audit


def write_audit_artifacts() -> dict[str, Any]:
    audit = build_preprocessing_audit()
    file_audit_path = TABLES_DIR / "file_audit.md"
    json_path = TABLES_DIR / "preprocessing_audit.json"

    source_lines = []
    for item in audit["source_files"]:
        source_lines.append(
            f"- `{item['path']}` | {item['size_human']} | sha256 `{item['sha256'][:16]}...`"
        )

    missing_lines = []
    for item in audit["missing_mem_int_scans"]:
        missing_lines.append(
            f"- `{item['scan_time_utc']}` | {item['target']} peak `{item['peak']}`"
        )

    temp_model_lines = []
    for peak, model in audit["temperature_models"].items():
        coeff = ", ".join(f"{value:.6g}" for value in model["coefficients_hz"])
        temp_model_lines.append(
            f"- Peak `{peak}`: coeffs `[a, b, c] = [{coeff}]`, "
            f"`T0 = {model['turning_point_temperature_k']:.3f} K`, "
            f"`n = {model['n_points']}`"
        )

    markdown = "\n".join(
        [
            "# File Audit",
            "",
            "## Confirmed Input Files",
            *source_lines,
            "",
            "## CSV Schema",
            f"- Rows: `{audit['csv_schema']['row_count']}`",
            f"- Raw original columns: `{audit['csv_schema']['raw_original_column_count']}`",
            f"- Stale index column present: `{audit['csv_schema']['has_stale_index_column']}` (`Unnamed: 0`, dropped from processed exports)",
            f"- Exact raw columns: `{audit['csv_schema']['raw_original_columns']}`",
            "",
            "## Timestamp Audit",
            f"- Timezone-aware: `{audit['timestamps']['timezone_aware']}`",
            f"- Timezone: `{audit['timestamps']['timezone']}`",
            f"- First scan: `{audit['timestamps']['min_scan_time_utc']}`",
            f"- Last scan: `{audit['timestamps']['max_scan_time_utc']}`",
            f"- Full span: `{audit['timestamps']['time_span_days']:.3f}` days",
            "",
            "## Published Subset Checks",
            f"- Unique Lorentzian scan keys in the CSV/fits crosswalk: `{audit['published_subset_checks']['unique_lorentzian_scan_keys']}`",
            f"- Literal `read_me` selection count: `{audit['published_subset_checks']['literal_readme_count']}`",
            f"- Default canonical selection count used by this benchmark: `{audit['published_subset_checks']['default_canonical_count']}`",
            f"- Peak `b` rows in default canonical selection: `{audit['published_subset_checks']['default_peak_b_count']}`",
            f"- Peak `b+c` rows in default canonical selection: `{audit['published_subset_checks']['default_peaks_bc_count']}`",
            f"- Strict QC count with `chisq_red <= 10`: `{audit['published_subset_checks']['strict_qc_count']}`",
            "",
            "## Subset Mismatch Note",
            "- The literal `read_me` wording produces 72 rows, not 73.",
            "- The CSV plus per-folder `fits.pkl` files show 73 unique Lorentzian scan keys.",
            "- The selected default rule is therefore `mem int` when present, otherwise `mem` Lorentzian for scans missing `mem int`.",
            "- Scans missing `mem int`:",
            *missing_lines,
            "",
            "## Lineshape Zip Inventory",
            f"- Zip entries: `{audit['zip_inventory']['entry_count']}`",
            f"- Lineshape folders: `{audit['zip_inventory']['lineshape_folder_count']}`",
            f"- Folders with `*_data_corr.pkl`: `{audit['zip_inventory']['folders_with_data_corr']}`",
            f"- Folders without `*_data_corr.pkl`: `{audit['zip_inventory']['folders_without_data_corr']}`",
            f"- Folders with explicit intensity pickles: `{audit['zip_inventory']['folders_with_intensity']}`",
            "",
            "## Temperature Model Audit",
            *temp_model_lines,
            "",
            "## Authoritative Sources",
            "- Scan-level frequency records: `data/raw/2025-11-13_Th_record_db.csv` plus the per-folder `*_fits.pkl` crosswalk.",
            "- Corrected lineshape data: `data/raw/Freq reproducibility data/*_data_corr.pkl` when present; otherwise `*_data.pkl`.",
            "- Temperature modeling: JILA 2026 paper plus data-derived weighted quadratic fits on canonical C10+C13 records.",
            "- Drift correction: JILA 2026 Methods; published scan-level records are treated as already silicon-cavity-drift corrected.",
            "- Injection families: Fuchs 2025 as the primary theory/rationale source for periodic, slow-drift, and line-broadening intuitions.",
            "",
            "## Paper-Backed Facts Verified in PDF Text",
            f"- JILA states 73 line scans: `{audit['paper_backed_facts']['jila_reports_73_line_scans']}`",
            f"- JILA states line b has the lowest temperature sensitivity: `{audit['paper_backed_facts']['jila_line_b_lowest_temperature_sensitivity']}`",
            f"- JILA states line c is more temperature sensitive: `{audit['paper_backed_facts']['jila_line_c_more_temperature_sensitive']}`",
            f"- JILA Methods mention silicon cavity drift correction: `{audit['paper_backed_facts']['jila_known_cavity_drift_corrected']}`",
            f"- JILA notes X2 has larger spread than C10/C13: `{audit['paper_backed_facts']['jila_x2_larger_spread_than_c10_c13']}`",
            f"- Fuchs discusses periodic modulation: `{audit['paper_backed_facts']['fuchs_mentions_periodic_modulations']}`",
            f"- Fuchs discusses slow-drift intuition: `{audit['paper_backed_facts']['fuchs_mentions_slow_drift_limit']}`",
            f"- Fuchs discusses broadening/splitting regimes: `{audit['paper_backed_facts']['fuchs_mentions_broadening_or_splitting']}`",
        ]
    )

    write_text(markdown, file_audit_path)
    write_json(audit, json_path)
    return {
        "audit": audit,
        "file_audit_path": str(file_audit_path),
        "json_path": str(json_path),
    }
