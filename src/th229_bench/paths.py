from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

BENCHMARK_DIR = PROJECT_ROOT / "benchmark"

RAW_CSV_PATH = RAW_DIR / "2025-11-13_Th_record_db.csv"
RAW_DOCX_PATH = RAW_DIR / "read_me.docx"
RAW_ZIP_PATH = RAW_DIR / "Freq reproducibility data.zip"
RAW_LINESHAPE_DIR = RAW_DIR / "Freq reproducibility data"
RAW_JILA_PDF_PATH = RAW_DIR / "JILA_Jan2026_paper.pdf"
RAW_FUCHS_PDF_PATH = RAW_DIR / "Fuchs_2025.pdf"
