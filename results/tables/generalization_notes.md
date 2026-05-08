# Generalization Notes

- This is a secondary evaluation; it does not replace the main phase-index benchmark.
- The held-out periods are off the main period grid: 8.5 days, 17 days, 36 days, 75 days, 150 days, 300 days.
- Thresholds are copied from the main source-grid validation-null scores.
- Test metrics are computed only on held-out-period test examples.
- This quantifies whether score calibration transfers across period values, not whether the model can extrapolate beyond the total observation span.