# Lineshape Audit

- Scan folders inspected: `73`.
- Folders with `*_data_corr.pkl`: `42`.
- Folders with explicit intensity pickle files: `2`.
- Folders passing the conservative future-branch manifest check: `73`.
- The pkl data are retained as a future-work branch only; the primary benchmark remains scan-level peak-b modulation detection.
- Within-scan lineshape reconstruction is not a headline task in this upgrade because corrected arrays are not uniformly present across all scans.

## Folder Counts By Target And Peak
```
target peak  count
   C10    b     15
   C10    c      3
   C13    b     12
   C13    c      4
    X2    b     28
    X2    c     11
```