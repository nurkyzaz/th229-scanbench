from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .paths import RAW_LINESHAPE_DIR


def scan_folders(lineshape_dir: Path = RAW_LINESHAPE_DIR) -> list[Path]:
    return sorted([path for path in lineshape_dir.iterdir() if path.is_dir()])


def load_lineshape_folder(folder: str | Path, lineshape_dir: Path = RAW_LINESHAPE_DIR) -> dict[str, Any]:
    folder_path = Path(folder)
    if not folder_path.is_absolute():
        folder_path = lineshape_dir / folder_path
    if not folder_path.exists():
        raise FileNotFoundError(folder_path)

    data_path = next(folder_path.glob("*_data.pkl"))
    corr_candidates = list(folder_path.glob("*_data_corr.pkl"))
    fits_path = next(folder_path.glob("*_fits.pkl"))
    intensity_candidates = list(folder_path.glob("*_intensity.pkl"))

    data = pd.read_pickle(data_path)
    corrected = pd.read_pickle(corr_candidates[0]) if corr_candidates else None
    fits = pd.read_pickle(fits_path)
    intensity = pd.read_pickle(intensity_candidates[0]) if intensity_candidates else None
    return {
        "folder": folder_path.name,
        "data": data,
        "corrected": corrected,
        "fits": fits,
        "intensity": intensity,
        "paths": {
            "data": str(data_path),
            "corrected": str(corr_candidates[0]) if corr_candidates else None,
            "fits": str(fits_path),
            "intensity": str(intensity_candidates[0]) if intensity_candidates else None,
        },
    }


def build_lineshape_manifest(lineshape_dir: Path = RAW_LINESHAPE_DIR) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for folder in scan_folders(lineshape_dir):
        loaded = load_lineshape_folder(folder)
        data = loaded["data"]
        corrected = loaded["corrected"]
        fits = loaded["fits"].reset_index()
        fit_times = pd.to_datetime(fits["_time"], utc=True)
        row = {
            "folder_name": folder.name,
            "scan_time_utc": fit_times.iloc[0].isoformat(),
            "target": fits["target"].iloc[0],
            "peak": fits["peak"].iloc[0],
            "data_rows": int(data.shape[0]),
            "data_detection_columns": int(data.shape[1]),
            "data_index_names": "|".join(str(name) for name in data.index.names),
            "has_corrected_data": corrected is not None,
            "corrected_rows": int(corrected.shape[0]) if corrected is not None else 0,
            "corrected_detection_columns": int(corrected.shape[1]) if corrected is not None else 0,
            "fits_rows": int(fits.shape[0]),
            "fit_corrections": ",".join(sorted(fits["correction"].astype(str).unique())),
            "has_intensity": loaded["intensity"] is not None,
            "usable_future_branch": bool(data.shape[1] == 200 and fits.shape[0] >= 2),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("scan_time_utc")

