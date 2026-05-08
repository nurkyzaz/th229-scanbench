# Drift Sanity Task Notes

- This task injects a linear trend in Hz/day on the same 55 peak-b timestamps.
- It is a secondary low-frequency stress test inspired by the draft, not the main benchmark.
- The score is the weighted RSS improvement from adding one linear slope term to crystal offsets.
- The detection threshold is set from validation null examples only.
- Minimum detectable absolute drift rate at 95% detection, interpolated over the tested grid: `14.375` Hz/day.