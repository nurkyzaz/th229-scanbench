# Main Preprocessing Summary

## Row Counts
```
                        dataset  row_count
                   raw_csv_rows        428
raw_unique_lorentzian_scan_keys         73
     literal_readme_subset_rows         72
canonical_published_subset_rows         73
          strict_qc_subset_rows         41
            primary_peak_b_rows         55
        secondary_peaks_bc_rows         73
```

## Per-Crystal / Per-Peak Counts
```
target peak  row_count
   C10    b         15
   C10    c          3
   C13    b         12
   C13    c          4
    X2    b         28
    X2    c         11
```

## Drift Checks
```
             subset  n_rows  slope_hz_per_day  slope_stderr_hz_per_day  weighted_rms_hz  reduced_chi2
 all_processed_rows      73         -1.125056                 1.749888     11426.183638     92.889585
        peak_b_only      55          1.071022                 1.957586      2096.204637      3.905248
peak_b_c10_c13_only      27          0.549281                 2.561862      1289.282590      1.810707
     peak_b_x2_only      28          2.113082                 3.285190      2851.294630      6.213465
```

## Uncertainty Diagnostics
```
target peak  row_count  median_freq_unc_hz  p90_freq_unc_hz  residual_std_hz  median_abs_residual_hz
   C10    b         15         1148.562474      1965.927521      2156.310777             1459.763152
   C10    c          3         2510.672843      3409.541380      1915.647622              902.389396
   C13    b         12         1042.424348      1723.807925      1661.110908              454.782181
   C13    c          4         2055.564789      2830.831469      8831.412601             4478.699027
    X2    b         28         1616.896709      5458.234839      4334.616367             3051.713978
    X2    c         11         5342.809463      8957.400796    760235.579872             7671.803915
```