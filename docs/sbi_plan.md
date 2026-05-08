# SBI NPE Baseline Plan

## 1. Method Choice

Use single-round SNPE-C / NPE with a neural spline flow density estimator from `sbi`. The simulator is cheap, the parameter space is only three-dimensional, and the benchmark needs one amortized model that can score any length-55 residual vector without refitting. SNLE/SNRE would also be valid, but they make the detection statistic less direct because the desired score is posterior mass over amplitude. A single-round amortized NPE run also keeps `make reproduce` simple and avoids adaptive-round bookkeeping that could accidentally couple proposal updates to validation or test behavior.

## 2. Parameter And Prior Specification

Parameters are theta = `(A_hz, log_period_days, phi_rad)`, with `frequency_hz = 1 / (exp(log_period_days) * 86400)`.

- `A_hz`: mixture prior with `pi_null = 0.5` point mass at `A = 0` and `1 - pi_null = 0.5` log-uniform slab over `[50, 6500]` Hz. The point mass makes the null class explicit, matches the benchmark's 50/50 class balance, and keeps posterior mass near the decision boundary comparable to AUROC/AP baselines. The null class is exactly `A = 0`; the signal class for the SBI score is `A > 50 Hz`.
- `log_period_days`: continuous uniform over `[log(7), log(365)]`. This gives the NPE a physically meaningful metric: neighboring periods imply neighboring signal morphologies on the fixed 55 timestamps. Evaluation still uses the fixed benchmark test split and the score marginalizes period, so allowing continuous periods in SBI training does not change the scalar detection statistic.
- `phi_rad`: uniform on `[0, 2*pi]`.

The simulator config and `models/sbi_npe_v1.pt` metadata will record `pi_null = 0.5`, `A_min_hz = 50`, `A_max_hz = 6500`, `log_period_min = log(7)`, and `log_period_max = log(365)`.

Implementation note: the simulator stores and reports physical `A_hz`, but the NPE trains on the monotone coordinate `log1p(A_hz)` so a neural spline flow can resolve the point-null neighborhood near zero. The detection statistic is still reported as `P(A_hz > 50 | x)`, implemented as posterior mass above `log1p(50)` in the internal coordinate.

## 3. Detection Statistic

Use posterior probability mass `P(A_hz > 50 | x)` as the scalar detection score. This integrates over `log_period_days` and `phi_rad` by posterior sampling, treating both as nuisance parameters. Posterior mean amplitude is less calibrated near the point null, and a Bayes factor would require more careful accounting of mixed discrete-continuous prior mass. Thresholds for this scalar score will be calibrated only on validation null benchmark examples.

## 4. Simulator Definition

Each simulation uses the immutable 55 real peak-b rows from `data/processed/primary_peak_b.csv`: `seconds_since_first_observation`, `target`/crystal id, `freq_unc_hz`, and temperature metadata are fixed covariates. The simulator assumes the temperature correction has already been applied by preprocessing; it simulates residual frequency only and does not reapply any temperature model.

Pseudocode:

```text
load primary_peak_b rows in timestamp order
fit null models once via fit_null_models(primary_peak_b)
select the existing default training-fit null model
pi_null = 0.5
sample u ~ Uniform(0, 1)
if u < 0.5:
    A_hz = 0
else:
    log_A ~ Uniform(log(50), log(6500))
    A_hz = exp(log_A)
log_period_days ~ Uniform(log(7), log(365))
period_days = exp(log_period_days)
frequency_hz = 1 / (period_days * 86400)
phi_rad ~ Uniform(0, 2*pi)
t_sec = seconds_since_first_observation, with t=0 at first observed peak-b scan
signal_hz = A_hz * cos(2*pi*frequency_hz*t_sec + phi_rad)
noise_hz = null_model.sample(primary_peak_b, rng)
x = signal_hz + noise_hz
return theta, x
```

Noise comes only from `FittedNullModel.sample` in `src/th229_bench/null_models.py`, using parameters fitted by `fit_null_models` on its observation-level `null_fit_split == train` subset. The current default null may be `crystal_gaussian_x2_mixture`; if so, X2 outlier behavior is part of the training-fit simulator. Validation and test residuals are never sampled as empirical residuals.

## 5. Training Budget

Train on 100,000 simulator draws for the reproducibility target, with a pinned seed. The smoke/CI configuration uses 5,000 simulator draws, also with a pinned seed, so `make test` can exercise the SBI path quickly. The full budget is enough for a three-parameter posterior with a small neural spline flow while staying comfortably below the 30-minute reproduction budget on a single GPU; CPU should remain feasible for smoke mode. The first implementation should cap network size conservatively, e.g. 4-6 transforms and hidden width around 64-128, because the input is only length 55.

## 6. Leakage Audit

No benchmark examples are used for SBI training, including the benchmark training-split examples. The only observed data used before training are the 55 fixed peak-b covariate rows and the null-model parameters fitted inside `fit_null_models(primary_peak_b)` from `src/th229_bench/null_models.py`; that function assigns an observation-level chronological `null_fit_split` and fits parameters on `train` only. SBI simulator draws use fresh seeded RNG streams and `FittedNullModel.sample`, not stored residual vectors from `data/interim/benchmark_arrays.npz`.

Enforcement locations for implementation:

- `baselines/sbi_npe_simulator.py`: load `data/processed/primary_peak_b.csv`, call `fit_null_models`, freeze `default_name`, and expose simulator metadata including null model name, parameters hash, period grid, amplitude prior, and seed.
- `baselines/sbi_npe.py`: train only from simulator-produced `(theta, x)` pairs; never read any benchmark catalog rows, benchmark arrays, or benchmark labels for training.
- `run_benchmark.py --baseline sbi_npe`: scoring may read `data/interim/benchmark_catalog.csv` and `benchmark_arrays.npz`, but only after checkpoint training; validation null rows are used only for the 95th-percentile threshold.
- `tests/test_sbi_leakage.py`: assert the simulator uses the 55-row primary covariate table, assert null parameters match `fit_null_models(primary_peak_b)`, assert no benchmark split labels are consumed by training helpers, and assert threshold calibration masks `split == validation` and `label == 0`.

## 7. Evaluation Outputs

The SBI baseline will emit the same metrics and schemas as existing methods: test AUROC, test AP, test-null FPR at a validation-null 5% threshold, and per-period A95 with 16-84% bootstrap intervals. Output rows will use baseline name `sbi_npe` and columns compatible with `baseline_comparison*.csv`, `a95_vs_frequency_with_uncertainty*.csv`, and the representative-frequency Table 1 CSV. Figure 3 should accept the SBI A95 table as one more curve without changing the old baseline rows.

## 8. Compute Estimate

Target wall clock is 10-20 minutes for 100,000-draw training plus evaluation on a single consumer GPU, and under 30 minutes for `make reproduce` including existing baselines. CPU fallback should support the 5,000-draw smoke mode, while the reproducibility target should prefer GPU when available. The checkpoint will be saved to `models/sbi_npe_v1.pt` with training seed, `sbi` version, simulator config hash, null model name, `pi_null = 0.5`, and the continuous log-period prior specification.

Dependency pin: use `sbi==0.26.1`. On the current local Anaconda Python 3.13 environment, `sbi` installs cleanly with `torch==2.11.0`, but importing `sbi.inference` attempts to import `torch.utils.tensorboard`, which triggers a pre-existing TensorFlow segfault. The implementation installs a no-op TensorBoard `SummaryWriter` stub before importing `sbi.inference`; this avoids TensorFlow while still using the installed `sbi` inference stack.

Runtime note: direct posterior sampling with strict rejection outside the box prior can stall for some high-amplitude examples. Evaluation therefore uses direct samples with `reject_outside_prior=False` and records that choice in `results/sbi_npe_results.json`; the scalar statistic only uses the amplitude marginal above `log1p(50)`.

## Future Work Note

This slice uses the existing parametric training-fit null model. After the learned-null slice lands, retraining SBI with the learned-null simulator can be reported as a separate variant; it should not replace this first parametric-null SBI baseline.
