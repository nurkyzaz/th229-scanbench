# File Audit

## Confirmed Input Files
- `data/raw/JILA_Jan2026_paper.pdf` | 3.0 MB | sha256 `fb2a270d0e527552...`
- `data/raw/Fuchs_2025.pdf` | 1.3 MB | sha256 `343b952acb669358...`
- `data/raw/2025-11-13_Th_record_db.csv` | 101.5 KB | sha256 `79622a26e9ce3310...`
- `data/raw/read_me.docx` | 104.2 KB | sha256 `07042fb2d5b48407...`
- `data/raw/Freq reproducibility data.zip` | 1.5 MB | sha256 `f43f7e6fd65e0406...`

## CSV Schema
- Rows: `428`
- Raw original columns: `19`
- Stale index column present: `True` (`Unnamed: 0`, dropped from processed exports)
- Exact raw columns: `['Unnamed: 0', '_time', 'correction', 'laser', 'Fitting function', 'Sweep range (kHz)', 'Linear baseline subtraction', 'peak', 'target', 'FWHM (kHz)', 'FWHM unc (kHz)', 'amp', 'amp_unc', 'baseline', 'baseline_unc', 'chisq_red', 'Freq (Hz)', 'Freq unc (Hz)', 'Temp (K)']`

## Timestamp Audit
- Timezone-aware: `True`
- Timezone: `UTC`
- First scan: `2024-05-30T04:13:20+00:00`
- Last scan: `2025-09-29T21:41:00+00:00`
- Full span: `487.728` days

## Published Subset Checks
- Unique Lorentzian scan keys in the CSV/fits crosswalk: `73`
- Literal `read_me` selection count: `72`
- Default canonical selection count used by this benchmark: `73`
- Peak `b` rows in default canonical selection: `55`
- Peak `b+c` rows in default canonical selection: `73`
- Strict QC count with `chisq_red <= 10`: `41`

## Subset Mismatch Note
- The literal `read_me` wording produces 72 rows, not 73.
- The CSV plus per-folder `fits.pkl` files show 73 unique Lorentzian scan keys.
- The selected default rule is therefore `mem int` when present, otherwise `mem` Lorentzian for scans missing `mem int`.
- Scans missing `mem int`:
- `2024-05-30T04:13:20+00:00` | X2 peak `b`
- `2024-05-30T07:58:29+00:00` | X2 peak `c`
- `2024-05-31T01:28:57+00:00` | X2 peak `c`
- `2024-05-31T06:06:02+00:00` | X2 peak `b`
- `2024-09-25T22:39:47+00:00` | X2 peak `c`

## Lineshape Zip Inventory
- Zip entries: `265`
- Lineshape folders: `73`
- Folders with `*_data_corr.pkl`: `42`
- Folders without `*_data_corr.pkl`: `31`
- Folders with explicit intensity pickles: `2`

## Temperature Model Audit
- Peak `b`: coeffs `[a, b, c] = [8.81349, -3453.52, 2.02041e+15]`, `T0 = 195.922 K`, `n = 27`
- Peak `c`: coeffs `[a, b, c] = [-13.819, -10318.5, 2.02041e+15]`, `T0 = -373.346 K`, `n = 7`

## Authoritative Sources
- Scan-level frequency records: `data/raw/2025-11-13_Th_record_db.csv` plus the per-folder `*_fits.pkl` crosswalk.
- Corrected lineshape data: `data/raw/Freq reproducibility data/*_data_corr.pkl` when present; otherwise `*_data.pkl`.
- Temperature modeling: JILA 2026 paper plus data-derived weighted quadratic fits on canonical C10+C13 records.
- Drift correction: JILA 2026 Methods; published scan-level records are treated as already silicon-cavity-drift corrected.
- Injection families: Fuchs 2025 as the primary theory/rationale source for periodic, slow-drift, and line-broadening intuitions.

## Paper-Backed Facts Verified in PDF Text
- JILA states 73 line scans: `True`
- JILA states line b has the lowest temperature sensitivity: `True`
- JILA states line c is more temperature sensitive: `True`
- JILA Methods mention silicon cavity drift correction: `True`
- JILA notes X2 has larger spread than C10/C13: `True`
- Fuchs discusses periodic modulation: `True`
- Fuchs discusses slow-drift intuition: `True`
- Fuchs discusses broadening/splitting regimes: `True`