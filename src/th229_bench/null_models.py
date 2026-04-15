from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import norm, t as student_t


def assign_null_fit_split(df: pd.DataFrame, train_fraction: float = 0.70) -> pd.DataFrame:
    """Assign a target-stratified chronological split for fitting null-model parameters."""
    out = df.sort_values(["target", "scan_time_utc"]).copy()
    out["null_fit_split"] = "holdout"
    for _, group in out.groupby("target", sort=False):
        n_train = max(1, int(np.floor(len(group) * train_fraction)))
        out.loc[group.index[:n_train], "null_fit_split"] = "train"
    return out.sort_values("scan_time_utc").reset_index(drop=True)


def _gaussian_nll(residual: np.ndarray, scale: np.ndarray) -> float:
    return float(-np.sum(norm.logpdf(residual, loc=0.0, scale=np.maximum(scale, 1e-9))))


def _student_nll(residual: np.ndarray, scale: np.ndarray, df: float) -> float:
    return float(-np.sum(student_t.logpdf(residual, df=df, loc=0.0, scale=np.maximum(scale, 1e-9))))


def _fit_gaussian_jitter(residual: np.ndarray, sigma: np.ndarray) -> float:
    max_scale = max(float(np.nanstd(residual) * 5.0), float(np.nanmax(sigma) * 5.0), 1.0)

    def objective(log_jitter: float) -> float:
        jitter = np.exp(log_jitter) - 1.0
        scale = np.sqrt(sigma**2 + jitter**2)
        return _gaussian_nll(residual, scale)

    result = minimize_scalar(objective, bounds=(np.log(1.0), np.log(max_scale + 1.0)), method="bounded")
    return float(np.exp(result.x) - 1.0)


def _fit_student_jitter(residual: np.ndarray, sigma: np.ndarray, df: float) -> float:
    max_scale = max(float(np.nanstd(residual) * 5.0), float(np.nanmax(sigma) * 5.0), 1.0)

    def objective(log_jitter: float) -> float:
        jitter = np.exp(log_jitter) - 1.0
        scale = np.sqrt(sigma**2 + jitter**2)
        return _student_nll(residual, scale, df=df)

    result = minimize_scalar(objective, bounds=(np.log(1.0), np.log(max_scale + 1.0)), method="bounded")
    return float(np.exp(result.x) - 1.0)


def _fit_student_jitter_and_df(residual: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    """Fit a simple Student-t tail parameter by grid search over df plus jitter MLE."""
    best_nll = np.inf
    best_jitter = 0.0
    best_df = 4.0
    for degrees in (2.25, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 40.0, 80.0):
        jitter = _fit_student_jitter(residual, sigma, df=degrees)
        scale = np.sqrt(sigma**2 + jitter**2)
        nll = _student_nll(residual, scale, df=degrees)
        if nll < best_nll:
            best_nll = nll
            best_jitter = jitter
            best_df = degrees
    return best_jitter, best_df


def _x2_mixture_nll(
    residual: np.ndarray,
    sigma: np.ndarray,
    core_jitter: float,
    outlier_probability: float,
    outlier_scale_multiplier: float,
) -> float:
    core_scale = np.sqrt(sigma**2 + core_jitter**2)
    outlier_scale = np.maximum(core_scale * outlier_scale_multiplier, core_scale + 1.0)
    log_core = np.log1p(-outlier_probability) + norm.logpdf(residual, loc=0.0, scale=core_scale)
    log_outlier = np.log(outlier_probability) + norm.logpdf(residual, loc=0.0, scale=outlier_scale)
    return float(-np.sum(logsumexp(np.vstack([log_core, log_outlier]), axis=0)))


def _fit_x2_mixture_params(residual: np.ndarray, sigma: np.ndarray, core_jitter: float) -> dict[str, float]:
    """Fit a conservative X2-only Gaussian outlier component by small grid search."""
    best = {
        "x2_outlier_probability": 0.05,
        "x2_outlier_scale_multiplier": 4.0,
        "x2_mixture_train_nll": np.inf,
    }
    for probability in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        for multiplier in (2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
            nll = _x2_mixture_nll(residual, sigma, core_jitter, probability, multiplier)
            if nll < best["x2_mixture_train_nll"]:
                best = {
                    "x2_outlier_probability": float(probability),
                    "x2_outlier_scale_multiplier": float(multiplier),
                    "x2_mixture_train_nll": float(nll),
                }
    return best


@dataclass
class FittedNullModel:
    name: str
    distribution: str
    parameters: dict[str, Any]
    train_nll_per_point: float
    holdout_nll_per_point: float
    holdout_coverage_1sigma: float
    holdout_coverage_2sigma: float

    def effective_sigma(self, df: pd.DataFrame) -> np.ndarray:
        formal = df["freq_unc_hz"].to_numpy(dtype=np.float64)
        if self.name == "formal_gaussian_scaled":
            return formal * float(self.parameters["global_uncertainty_scale"])
        jitters = self.parameters.get("crystal_jitter_hz", {})
        jitter = np.asarray([float(jitters.get(target, 0.0)) for target in df["target"]], dtype=np.float64)
        core_sigma = np.sqrt(formal**2 + jitter**2)
        if self.distribution == "x2_gaussian_mixture":
            is_x2 = df["target"].astype(str).to_numpy() == "X2"
            probability = float(self.parameters["x2_outlier_probability"])
            multiplier = float(self.parameters["x2_outlier_scale_multiplier"])
            mixture_sigma = core_sigma.copy()
            mixture_sigma[is_x2] = np.sqrt(
                (1.0 - probability) * core_sigma[is_x2] ** 2
                + probability * (multiplier * core_sigma[is_x2]) ** 2
            )
            return mixture_sigma
        return core_sigma

    def sample(self, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        scale = self.effective_sigma(df)
        if self.distribution == "gaussian":
            return rng.normal(loc=0.0, scale=scale)
        if self.distribution == "student_t":
            degrees = float(self.parameters["student_t_df"])
            return student_t.rvs(degrees, loc=0.0, scale=scale, random_state=rng)
        if self.distribution == "student_t_target_df":
            samples = np.zeros(len(df), dtype=np.float64)
            jitters = self.parameters["crystal_jitter_hz"]
            dfs = self.parameters["crystal_student_t_df"]
            for target, index in df.groupby("target").groups.items():
                sub = df.loc[index]
                formal = sub["freq_unc_hz"].to_numpy(dtype=np.float64)
                jitter = float(jitters.get(target, 0.0))
                target_scale = np.sqrt(formal**2 + jitter**2)
                samples[df.index.get_indexer(index)] = student_t.rvs(
                    float(dfs.get(target, 4.0)),
                    loc=0.0,
                    scale=target_scale,
                    random_state=rng,
                )
            return samples
        if self.distribution == "x2_gaussian_mixture":
            jitters = self.parameters["crystal_jitter_hz"]
            formal = df["freq_unc_hz"].to_numpy(dtype=np.float64)
            jitter = np.asarray([float(jitters.get(target, 0.0)) for target in df["target"]], dtype=np.float64)
            core_scale = np.sqrt(formal**2 + jitter**2)
            samples = rng.normal(loc=0.0, scale=core_scale)
            is_x2 = df["target"].astype(str).to_numpy() == "X2"
            probability = float(self.parameters["x2_outlier_probability"])
            multiplier = float(self.parameters["x2_outlier_scale_multiplier"])
            outlier = rng.random(len(df)) < probability
            replace = is_x2 & outlier
            samples[replace] = rng.normal(loc=0.0, scale=core_scale[replace] * multiplier)
            return samples
        raise ValueError(f"Unknown null distribution: {self.distribution}")

    def nll(self, df: pd.DataFrame) -> float:
        residual = df["residual_hz"].to_numpy(dtype=np.float64)
        scale = self.effective_sigma(df)
        if self.distribution == "gaussian":
            return _gaussian_nll(residual, scale)
        if self.distribution == "student_t":
            return _student_nll(residual, scale, df=float(self.parameters["student_t_df"]))
        if self.distribution == "student_t_target_df":
            total = 0.0
            jitters = self.parameters["crystal_jitter_hz"]
            dfs = self.parameters["crystal_student_t_df"]
            for target, group in df.groupby("target"):
                group_residual = group["residual_hz"].to_numpy(dtype=np.float64)
                group_sigma = group["freq_unc_hz"].to_numpy(dtype=np.float64)
                jitter = float(jitters.get(target, 0.0))
                group_scale = np.sqrt(group_sigma**2 + jitter**2)
                total += _student_nll(group_residual, group_scale, df=float(dfs.get(target, 4.0)))
            return float(total)
        if self.distribution == "x2_gaussian_mixture":
            jitters = self.parameters["crystal_jitter_hz"]
            total = 0.0
            for target, group in df.groupby("target"):
                group_residual = group["residual_hz"].to_numpy(dtype=np.float64)
                group_sigma = group["freq_unc_hz"].to_numpy(dtype=np.float64)
                jitter = float(jitters.get(target, 0.0))
                if target == "X2":
                    total += _x2_mixture_nll(
                        group_residual,
                        group_sigma,
                        jitter,
                        float(self.parameters["x2_outlier_probability"]),
                        float(self.parameters["x2_outlier_scale_multiplier"]),
                    )
                else:
                    group_scale = np.sqrt(group_sigma**2 + jitter**2)
                    total += _gaussian_nll(group_residual, group_scale)
            return float(total)
        raise ValueError(f"Unknown null distribution: {self.distribution}")

    def standardized(self, df: pd.DataFrame) -> np.ndarray:
        return df["residual_hz"].to_numpy(dtype=np.float64) / self.effective_sigma(df)


def _summarize_model(name: str, distribution: str, params: dict[str, Any], train: pd.DataFrame, holdout: pd.DataFrame) -> FittedNullModel:
    model = FittedNullModel(
        name=name,
        distribution=distribution,
        parameters=params,
        train_nll_per_point=0.0,
        holdout_nll_per_point=0.0,
        holdout_coverage_1sigma=0.0,
        holdout_coverage_2sigma=0.0,
    )
    train_nll = model.nll(train)
    holdout_nll = model.nll(holdout)
    z = model.standardized(holdout)
    model.train_nll_per_point = float(train_nll / max(len(train), 1))
    model.holdout_nll_per_point = float(holdout_nll / max(len(holdout), 1))
    model.holdout_coverage_1sigma = float(np.mean(np.abs(z) <= 1.0))
    model.holdout_coverage_2sigma = float(np.mean(np.abs(z) <= 2.0))
    return model


def fit_null_models(primary_peak_b: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, FittedNullModel], str]:
    """Fit formal, crystal-aware Gaussian, Student-t, and X2 mixture null models."""
    split_df = assign_null_fit_split(primary_peak_b)
    train = split_df.loc[split_df["null_fit_split"].eq("train")].copy()
    holdout = split_df.loc[split_df["null_fit_split"].eq("holdout")].copy()
    residual = train["residual_hz"].to_numpy(dtype=np.float64)
    sigma = train["freq_unc_hz"].to_numpy(dtype=np.float64)

    global_scale = max(1.0, float(np.sqrt(np.mean((residual / np.maximum(sigma, 1.0)) ** 2))))
    models: dict[str, FittedNullModel] = {}
    models["formal_gaussian_scaled"] = _summarize_model(
        "formal_gaussian_scaled",
        "gaussian",
        {"global_uncertainty_scale": global_scale},
        train,
        holdout,
    )

    gaussian_jitters: dict[str, float] = {}
    student_jitters: dict[str, float] = {}
    fitted_student_jitters: dict[str, float] = {}
    fitted_student_dfs: dict[str, float] = {}
    degrees = 4.0
    for target, group in train.groupby("target"):
        group_res = group["residual_hz"].to_numpy(dtype=np.float64)
        group_sig = group["freq_unc_hz"].to_numpy(dtype=np.float64)
        gaussian_jitters[target] = _fit_gaussian_jitter(group_res, group_sig)
        student_jitters[target] = _fit_student_jitter(group_res, group_sig, df=degrees)
        fitted_jitter, fitted_df = _fit_student_jitter_and_df(group_res, group_sig)
        fitted_student_jitters[target] = fitted_jitter
        fitted_student_dfs[target] = fitted_df

    models["crystal_gaussian_jitter"] = _summarize_model(
        "crystal_gaussian_jitter",
        "gaussian",
        {"crystal_jitter_hz": gaussian_jitters},
        train,
        holdout,
    )
    models["crystal_student_t_jitter"] = _summarize_model(
        "crystal_student_t_jitter",
        "student_t",
        {"crystal_jitter_hz": student_jitters, "student_t_df": degrees},
        train,
        holdout,
    )
    models["crystal_student_t_fitted_df"] = _summarize_model(
        "crystal_student_t_fitted_df",
        "student_t_target_df",
        {"crystal_jitter_hz": fitted_student_jitters, "crystal_student_t_df": fitted_student_dfs},
        train,
        holdout,
    )

    x2_train = train.loc[train["target"].eq("X2")].copy()
    if not x2_train.empty:
        mixture_params = _fit_x2_mixture_params(
            x2_train["residual_hz"].to_numpy(dtype=np.float64),
            x2_train["freq_unc_hz"].to_numpy(dtype=np.float64),
            gaussian_jitters.get("X2", 0.0),
        )
        models["crystal_gaussian_x2_mixture"] = _summarize_model(
            "crystal_gaussian_x2_mixture",
            "x2_gaussian_mixture",
            {"crystal_jitter_hz": gaussian_jitters, **mixture_params},
            train,
            holdout,
        )

    # Choose the default by held-out negative log likelihood, but require a clean model.
    default_name = min(models, key=lambda key: models[key].holdout_nll_per_point)
    return split_df, models, default_name


def null_model_comparison_frame(models: dict[str, FittedNullModel], default_name: str) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        rows.append(
            {
                "null_model": name,
                "is_default": name == default_name,
                "distribution": model.distribution,
                "parameters_json": model.parameters,
                "train_nll_per_point": model.train_nll_per_point,
                "holdout_nll_per_point": model.holdout_nll_per_point,
                "holdout_coverage_abs_z_le_1": model.holdout_coverage_1sigma,
                "holdout_coverage_abs_z_le_2": model.holdout_coverage_2sigma,
            }
        )
    return pd.DataFrame(rows).sort_values("holdout_nll_per_point")
