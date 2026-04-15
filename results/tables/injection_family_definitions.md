# Injection Family Definitions

## pure_sinusoid
- Equation: `nu(t) = A cos(omega t + phi)`
- Role: primary benchmark task
- Physics note: Motivated by time-varying fields that modulate the nuclear transition frequency.

## finite_coherence_sinusoid
- Equation: `nu(t) = A cos(omega t + phi_k) for t in coherence window k`
- Role: secondary stress test
- Physics note: Approximates finite coherence by phase re-randomization in fixed windows on the observed cadence.

## slow_linear_drift
- Equation: `nu(t) = slope * (t - mean(t))`
- Role: stress test, not a headline BSM model
- Physics note: Captures the slow-modulation limit where a sinusoid can look locally like a drift.

## gaussian_transient
- Equation: `nu(t) = A exp[-0.5 ((t - t0)/sigma_t)^2]`
- Role: controlled transient benchmark family
- Physics note: Stylized localized event; not claimed as a unique physical prediction.

## susy_note
- Equation: `not implemented as an injection family`
- Role: scope guard
- Physics note: Generic SUSY virtual corrections may motivate enhanced sensitivity, but do not by themselves define a time-domain morphology.
