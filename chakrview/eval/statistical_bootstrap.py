"""
Statistical Bootstrap Estimators for CAND_01 Evaluation.

Implements:
1. Ordinary non-parametric bootstrap.
2. Secondary one-way clustered sensitivity estimator (clustered on target_val_token_id).
3. Provisional two-way crossed-cluster weighting estimator (Cameron, Gelbach, Miller 2011; Menzel 2021).
4. Kish effective sample size diagnostics.
5. Weight variance and degenerate replicate detection.
6. Synthetic Monte Carlo diagnostic simulations for crossed-cluster structures.

GOVERNANCE CLASSIFICATION:
- The two-way multiplicative weighting method is strictly
  PROVISIONAL_PENDING_STATISTICAL_VALIDATION.
- The one-way cluster bootstrap is a SECONDARY SENSITIVITY ESTIMATOR.
- It is NEVER automatically promoted to primary status.
- If the two-way estimator exceeds the registered degenerate replicate threshold (>10%),
  the uncertainty reporting status is formally classified as INCONCLUSIVE.
- Passing synthetic Monte Carlo simulations is diagnostic only and DOES NOT
  constitute empirical validation on real model predictions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ==============================================================================
# AUTHORITATIVE GOVERNANCE CONSTANTS (CAND_01 v1.3.1)
# ==============================================================================
DEFAULT_NUM_BOOTSTRAP_REPLICATES: int = 2000
DEFAULT_CONFIDENCE_LEVEL: float = 0.95
DEFAULT_RANDOM_SEED: int = 999042

# Degeneracy thresholds per v1.3.1 targeted corrections:
# 1. Effective sample size fraction: N_eff < 0.25 * N
# 2. Maximum normalized weight variance threshold: 10.0
# 3. Maximum acceptable degenerate replicate rate: 10% (0.10)
# 4. Minimum unique clusters drawn per dimension: 10
REGISTERED_MAX_DEGENERATE_RATE: float = 0.10
REGISTERED_MIN_NEFF_FRACTION: float = 0.25
REGISTERED_MAX_WEIGHT_VARIANCE: float = 10.0
REGISTERED_MIN_UNIQUE_CLUSTERS: int = 10


class EstimatorStatus(str, Enum):
    PROVISIONAL_PENDING_VALIDATION = "PROVISIONAL_PENDING_STATISTICAL_VALIDATION"
    SECONDARY_SENSITIVITY = "SECONDARY_SENSITIVITY_ESTIMATOR"
    CONVERGED_VALID = "CONVERGED_VALID"
    DEGENERATE_EXCEEDED = "DEGENERATE_EXCEEDED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class BootstrapResult:
    """Holds summary statistics and diagnostics for a bootstrap evaluation."""
    estimator_name: str
    status: EstimatorStatus
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    num_replicates_requested: int = DEFAULT_NUM_BOOTSTRAP_REPLICATES
    num_valid_replicates: int = 0
    num_degenerate_replicates: int = 0
    degenerate_rate: float = 0.0
    mean_effective_sample_size: float = 0.0
    mean_weight_variance: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimator_name": self.estimator_name,
            "status": self.status.value,
            "point_estimate": float(self.point_estimate),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "confidence_level": float(self.confidence_level),
            "num_replicates_requested": self.num_replicates_requested,
            "num_valid_replicates": self.num_valid_replicates,
            "num_degenerate_replicates": self.num_degenerate_replicates,
            "degenerate_rate": float(self.degenerate_rate),
            "mean_effective_sample_size": float(self.mean_effective_sample_size),
            "mean_weight_variance": float(self.mean_weight_variance),
            "notes": self.notes,
        }


def compute_kish_effective_sample_size(weights: np.ndarray) -> float:
    """
    Computes Kish's Effective Sample Size (Kish 1965):
    N_eff = (sum w)^2 / sum(w^2)

    Handles non-finite, negative, or all-zero weights robustly by returning 0.0.
    """
    if weights is None or len(weights) == 0:
        return 0.0
    if not np.all(np.isfinite(weights)):
        return 0.0
    if np.any(weights < 0):
        return 0.0

    sum_w = float(np.sum(weights))
    if sum_w <= 0.0:
        return 0.0
    sum_w_sq = float(np.sum(weights ** 2))
    if sum_w_sq <= 0.0:
        return 0.0
    return (sum_w ** 2) / sum_w_sq


def ordinary_nonparametric_bootstrap(
    values: np.ndarray,
    num_replicates: int = DEFAULT_NUM_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> BootstrapResult:
    """
    Standard non-parametric bootstrap resampling of independent observations.
    Serves as an unclustered baseline reference.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0 or not np.all(np.isfinite(values)):
        return BootstrapResult(
            estimator_name="OrdinaryNonparametricBootstrap",
            status=EstimatorStatus.INCONCLUSIVE,
            point_estimate=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            notes=["Empty or non-finite input array provided."],
        )

    rng = np.random.default_rng(random_seed)
    point_est = float(np.mean(values))

    replicates = []
    for _ in range(num_replicates):
        idx = rng.integers(0, n, size=n)
        replicates.append(np.mean(values[idx]))

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(replicates, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(replicates, 100 * (1.0 - alpha / 2.0)))

    return BootstrapResult(
        estimator_name="OrdinaryNonparametricBootstrap",
        status=EstimatorStatus.CONVERGED_VALID,
        point_estimate=point_est,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        num_replicates_requested=num_replicates,
        num_valid_replicates=num_replicates,
        num_degenerate_replicates=0,
        degenerate_rate=0.0,
        mean_effective_sample_size=float(n),
        mean_weight_variance=0.0,
        notes=["Assumes strictly i.i.d. observations; ignores crossed clustering."],
    )


def one_way_clustered_bootstrap(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    num_replicates: int = DEFAULT_NUM_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> BootstrapResult:
    """
    One-way cluster bootstrap resampling clusters with replacement.
    Typically clustered on target_val_token_id.

    GOVERNANCE POSTURE:
    Designated strictly as a SECONDARY SENSITIVITY ESTIMATOR.
    Never automatically promoted to primary status.
    """
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    n = len(values)
    if n == 0 or len(cluster_ids) != n or not np.all(np.isfinite(values)):
        return BootstrapResult(
            estimator_name="OneWayClusteredBootstrap",
            status=EstimatorStatus.INCONCLUSIVE,
            point_estimate=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            notes=["Invalid, non-finite, or mismatched input shapes."],
        )

    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    if n_clusters < 2:
        return BootstrapResult(
            estimator_name="OneWayClusteredBootstrap",
            status=EstimatorStatus.INCONCLUSIVE,
            point_estimate=float(np.mean(values)),
            ci_lower=float(np.min(values)),
            ci_upper=float(np.max(values)),
            notes=["Fewer than 2 unique clusters present."],
        )

    cluster_to_indices = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
    rng = np.random.default_rng(random_seed)
    point_est = float(np.mean(values))

    replicates = []
    degenerate_count = 0
    neff_list = []

    for _ in range(num_replicates):
        resampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sample_indices = []
        for c in resampled_clusters:
            sample_indices.extend(cluster_to_indices[c])
        sample_indices = np.array(sample_indices, dtype=int)

        if len(sample_indices) == 0:
            degenerate_count += 1
            continue

        rep_est = float(np.mean(values[sample_indices]))
        replicates.append(rep_est)
        neff_list.append(len(sample_indices))

    if len(replicates) == 0:
        return BootstrapResult(
            estimator_name="OneWayClusteredBootstrap",
            status=EstimatorStatus.INCONCLUSIVE,
            point_estimate=point_est,
            ci_lower=0.0,
            ci_upper=0.0,
            notes=["All bootstrap replicates collapsed."],
        )

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(replicates, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(replicates, 100 * (1.0 - alpha / 2.0)))

    return BootstrapResult(
        estimator_name="OneWayClusteredBootstrap",
        status=EstimatorStatus.SECONDARY_SENSITIVITY,
        point_estimate=point_est,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        num_replicates_requested=num_replicates,
        num_valid_replicates=len(replicates),
        num_degenerate_replicates=degenerate_count,
        degenerate_rate=degenerate_count / num_replicates,
        mean_effective_sample_size=float(np.mean(neff_list)) if neff_list else 0.0,
        mean_weight_variance=0.0,
        notes=[
            "Classified strictly as SECONDARY SENSITIVITY ESTIMATOR.",
            "Clustered on target identifier; key-token correlation structure is unmodeled.",
        ],
    )


def provisional_two_way_cluster_bootstrap(
    values: np.ndarray,
    key_ids: np.ndarray,
    val_ids: np.ndarray,
    num_replicates: int = DEFAULT_NUM_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_degenerate_rate: float = REGISTERED_MAX_DEGENERATE_RATE,
    min_effective_sample_fraction: float = REGISTERED_MIN_NEFF_FRACTION,
    max_weight_variance_threshold: float = REGISTERED_MAX_WEIGHT_VARIANCE,
    min_unique_clusters: int = REGISTERED_MIN_UNIQUE_CLUSTERS,
) -> BootstrapResult:
    """
    Provisional two-way crossed-cluster weighting bootstrap.
    Specification: docs/experiments/CAND_01_V1.3.1_TARGETED_CORRECTIONS.md Section 2.1.

    Observational weight: w_m = c_K(m) * c_V(m), where c_K and c_V are resample
    multiplicities of key and value tokens respectively.

    STATUS:
    PROVISIONAL_PENDING_STATISTICAL_VALIDATION.

    If degenerate replicates exceed max_degenerate_rate (default 10%),
    the estimator is deemed invalid and classified as INCONCLUSIVE.
    """
    values = np.asarray(values, dtype=float)
    key_ids = np.asarray(key_ids)
    val_ids = np.asarray(val_ids)
    n = len(values)

    if n == 0 or len(key_ids) != n or len(val_ids) != n or not np.all(np.isfinite(values)):
        return BootstrapResult(
            estimator_name="ProvisionalTwoWayClusterBootstrap",
            status=EstimatorStatus.INCONCLUSIVE,
            point_estimate=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            notes=["Input arrays are empty, non-finite, or dimensions mismatch."],
        )

    unique_keys = np.unique(key_ids)
    unique_vals = np.unique(val_ids)
    n_keys = len(unique_keys)
    n_vals = len(unique_vals)

    if n_keys < 2 or n_vals < 2:
        return BootstrapResult(
            estimator_name="ProvisionalTwoWayClusterBootstrap",
            status=EstimatorStatus.INCONCLUSIVE,
            point_estimate=float(np.mean(values)),
            ci_lower=float(np.min(values)),
            ci_upper=float(np.max(values)),
            notes=["Fewer than 2 unique keys or values present."],
        )

    rng = np.random.default_rng(random_seed)
    point_est = float(np.mean(values))

    valid_replicates: List[float] = []
    degenerate_count = 0
    neff_history: List[float] = []
    weight_var_history: List[float] = []

    # Pre-map token to indices for fast lookup
    key_to_idx = {k: i for i, k in enumerate(unique_keys)}
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    obs_key_idx = np.array([key_to_idx[k] for k in key_ids], dtype=int)
    obs_val_idx = np.array([val_to_idx[v] for v in val_ids], dtype=int)

    replicates_attempted = 0
    max_total_draws = num_replicates * 5

    while len(valid_replicates) < num_replicates and replicates_attempted < max_total_draws:
        replicates_attempted += 1

        # Resample key and value cluster sets with replacement
        sampled_key_indices = rng.integers(0, n_keys, size=n_keys)
        sampled_val_indices = rng.integers(0, n_vals, size=n_vals)

        c_K = np.bincount(sampled_key_indices, minlength=n_keys)
        c_V = np.bincount(sampled_val_indices, minlength=n_vals)

        weights = c_K[obs_key_idx] * c_V[obs_val_idx]
        sum_w = np.sum(weights)

        # Non-finite or non-positive weight check
        if not np.all(np.isfinite(weights)) or sum_w <= 0.0:
            degenerate_count += 1
            continue

        # Degeneracy diagnostic 1: Effective sample fraction vs N
        if sum_w < (min_effective_sample_fraction * n):
            degenerate_count += 1
            continue

        # Degeneracy diagnostic 2: Kish N_eff
        n_eff = compute_kish_effective_sample_size(weights)
        if n_eff < (min_effective_sample_fraction * n):
            degenerate_count += 1
            continue

        # Degeneracy diagnostic 3: Minimum unique clusters sampled
        unique_keys_sampled = np.count_nonzero(c_K)
        unique_vals_sampled = np.count_nonzero(c_V)
        # Adapt cluster check if population has fewer clusters than min_unique_clusters
        eff_min_clusters = min(min_unique_clusters, min(n_keys, n_vals))
        if unique_keys_sampled < eff_min_clusters or unique_vals_sampled < eff_min_clusters:
            degenerate_count += 1
            continue

        # Degeneracy diagnostic 4: Normalized weight variance
        mean_w = np.mean(weights)
        if mean_w <= 0.0 or not np.isfinite(mean_w):
            degenerate_count += 1
            continue
        norm_weights = weights / mean_w
        weight_variance = float(np.var(norm_weights))
        if not np.isfinite(weight_variance) or weight_variance > max_weight_variance_threshold:
            degenerate_count += 1
            continue

        # Valid replicate
        est = float(np.sum(weights * values) / sum_w)
        if not np.isfinite(est):
            degenerate_count += 1
            continue

        valid_replicates.append(est)
        neff_history.append(n_eff)
        weight_var_history.append(weight_variance)

    total_draws = len(valid_replicates) + degenerate_count
    degenerate_rate = degenerate_count / max(1, total_draws)

    notes = [
        "Classification: PROVISIONAL_PENDING_STATISTICAL_VALIDATION.",
        "Weight formula: w_m = c_K(m) * c_V(m).",
    ]

    # Authoritative governance threshold check: degenerate_rate > max_degenerate_rate
    if degenerate_rate > max_degenerate_rate or len(valid_replicates) < num_replicates:
        notes.append(
            f"DEGENERACY CRITERIA TRIPPED: degenerate rate {degenerate_rate:.2%} "
            f"exceeded registered threshold {max_degenerate_rate:.2%} or replicate target unmet. "
            "Primary uncertainty classified as INCONCLUSIVE."
        )
        status = EstimatorStatus.INCONCLUSIVE
    else:
        status = EstimatorStatus.PROVISIONAL_PENDING_VALIDATION

    if len(valid_replicates) > 0:
        alpha = 1.0 - confidence_level
        ci_lower = float(np.percentile(valid_replicates, 100 * (alpha / 2.0)))
        ci_upper = float(np.percentile(valid_replicates, 100 * (1.0 - alpha / 2.0)))
    else:
        ci_lower = 0.0
        ci_upper = 0.0

    return BootstrapResult(
        estimator_name="ProvisionalTwoWayClusterBootstrap",
        status=status,
        point_estimate=point_est,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        num_replicates_requested=num_replicates,
        num_valid_replicates=len(valid_replicates),
        num_degenerate_replicates=degenerate_count,
        degenerate_rate=degenerate_rate,
        mean_effective_sample_size=float(np.mean(neff_history)) if neff_history else 0.0,
        mean_weight_variance=float(np.mean(weight_var_history)) if weight_var_history else 0.0,
        notes=notes,
    )


def run_synthetic_crossed_simulation(
    num_simulations: int = 50,
    n_keys: int = 30,
    n_vals: int = 30,
    num_bootstrap_replicates: int = 200,
    nominal_confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    key_sd: float = 0.5,
    val_sd: float = 0.5,
    base_logit: float = 0.0,
    random_seed: int = 12345,
) -> Dict[str, Any]:
    """
    Diagnostic synthetic Monte Carlo simulation to evaluate coverage, bias,
    and degeneracy rate under known crossed random-effects data generating processes.

    IMPORTANT GOVERNANCE NOTICE:
    Passing this diagnostic simulation is an engineering sanity check and DOES NOT
    constitute formal statistical validation or accreditation of the estimator on
    real ChakrView model outputs.
    """
    rng = np.random.default_rng(random_seed)
    true_mean = 1.0 / (1.0 + np.exp(-base_logit))

    covered_ordinary = 0
    covered_oneway = 0
    covered_twoway = 0
    twoway_degeneracy_rates = []
    twoway_estimates = []

    for sim_idx in range(num_simulations):
        # Generate crossed random effects
        u_k = rng.normal(0, key_sd, size=n_keys)
        w_v = rng.normal(0, val_sd, size=n_vals)

        # Generate crossed observations
        keys = []
        vals = []
        outcomes = []
        for k_idx in range(n_keys):
            for v_idx in range(n_vals):
                prob = 1.0 / (1.0 + np.exp(-(base_logit + u_k[k_idx] + w_v[v_idx])))
                y = 1.0 if rng.random() < prob else 0.0
                keys.append(k_idx)
                vals.append(v_idx)
                outcomes.append(y)

        outcomes_arr = np.array(outcomes, dtype=float)
        keys_arr = np.array(keys, dtype=int)
        vals_arr = np.array(vals, dtype=int)

        seed = random_seed + sim_idx * 101

        # Ordinary bootstrap
        res_ord = ordinary_nonparametric_bootstrap(
            outcomes_arr,
            num_replicates=num_bootstrap_replicates,
            confidence_level=nominal_confidence,
            random_seed=seed,
        )
        if res_ord.ci_lower <= true_mean <= res_ord.ci_upper:
            covered_ordinary += 1

        # One-way cluster bootstrap
        res_one = one_way_clustered_bootstrap(
            outcomes_arr,
            cluster_ids=vals_arr,
            num_replicates=num_bootstrap_replicates,
            confidence_level=nominal_confidence,
            random_seed=seed,
        )
        if res_one.ci_lower <= true_mean <= res_one.ci_upper:
            covered_oneway += 1

        # Provisional two-way cluster bootstrap
        res_two = provisional_two_way_cluster_bootstrap(
            outcomes_arr,
            key_ids=keys_arr,
            val_ids=vals_arr,
            num_replicates=num_bootstrap_replicates,
            confidence_level=nominal_confidence,
            random_seed=seed,
        )
        if res_two.ci_lower <= true_mean <= res_two.ci_upper:
            covered_twoway += 1

        twoway_degeneracy_rates.append(res_two.degenerate_rate)
        twoway_estimates.append(res_two.point_estimate)

    # Compute Monte Carlo standard error for coverage: sqrt(p * (1 - p) / M)
    p_ord = covered_ordinary / num_simulations
    p_one = covered_oneway / num_simulations
    p_two = covered_twoway / num_simulations

    se_ord = float(np.sqrt(p_ord * (1.0 - p_ord) / num_simulations))
    se_one = float(np.sqrt(p_one * (1.0 - p_one) / num_simulations))
    se_two = float(np.sqrt(p_two * (1.0 - p_two) / num_simulations))

    return {
        "diagnostic_disclaimer": "Diagnostic simulation only. Does NOT accredit estimator for production.",
        "num_simulations": num_simulations,
        "nominal_confidence": nominal_confidence,
        "true_underlying_mean": float(true_mean),
        "coverage": {
            "ordinary_bootstrap_coverage": float(p_ord),
            "ordinary_mc_se": se_ord,
            "one_way_clustered_coverage": float(p_one),
            "one_way_mc_se": se_one,
            "provisional_two_way_coverage": float(p_two),
            "provisional_two_way_mc_se": se_two,
        },
        "mean_point_estimate": float(np.mean(twoway_estimates)),
        "empirical_bias": float(np.mean(twoway_estimates) - true_mean),
        "mean_two_way_degenerate_rate": float(np.mean(twoway_degeneracy_rates)),
    }
