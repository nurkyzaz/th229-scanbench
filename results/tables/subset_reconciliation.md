# Subset Reconciliation

The raw CSV has 73 unique Lorentzian scan keys after grouping by UTC scan time, crystal target, and peak.
A literal reading of `read_me.docx` selects Lorentzian `mem int` rows plus only the first May 2024 X2 `mem` fallback rows, which gives 72 rows.
The file-backed canonical subset used here selects Lorentzian `mem int` when available and otherwise uses the Lorentzian `mem` row for scans without a `mem int` row.
That rule yields 73 rows, 55 peak-b rows, and 73 peak-b+c rows.

The machine-readable companion table is `results/tables/subset_reconciliation_scan_keys.csv`. It lists every Lorentzian scan key, the correction rows present, the literal-readme inclusion flag, the canonical-fallback inclusion flag, and the inclusion reason.

The non-May reconciliation row is the 2024-09-25 X2 peak-c scan. It lacks a `mem int` row in the CSV and in its per-folder fits, so it is included only through the explicit `mem` fallback rule.

Peak c remains optional metadata and an exploratory ablation. It is not a headline dependency of the upgraded benchmark.