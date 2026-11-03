# CHAKRVIEW — CAND_01 COMPLETE ENGINEERING AUDIT, CORRECTION & READINESS REPORT
**Comprehensive Independent Software Audit, Defect Remediation, Test Expansion, and Execution Boundary Verification**

**Repository:** `D:\Project\ChakrView`  
**Branch:** `main` | **Target Commit (HEAD):** `5aeaa2734de5a98802b96afe36324765f3244738`  
**Research Candidate:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Execution Posture:** Strictly Non-Executable / Synthetic Testing Only (No Experiment Execution)  
**Parent Governance:** Experiment Governance Protocol v1.0, CAND_01 v1.3.1 Targeted Corrections, Formal Gate Review  
**Audit Completion Date:** 2026-11-02  

---

## 1. Executive Verdict & Decisions

### Formal Engineering Audit Decision
**`AUDIT_PASS_WITH_CONDITIONS`**

### Summary of Conditions & Epistemic Boundaries
1. **Software Engineering Readiness:** All critical implementation defects, task-family gaps, metric couplings, and JSON serialization bugs have been resolved and independently verified. The test suite passes 100% (18/18 tests).
2. **Provisional Statistical Estimator Limitation:** The two-way cluster bootstrap ($w_m = c_K \times c_V$) trips its registered degeneracy criterion ($>10\%$ degenerate replicates) across sparse crossed evaluations (exhibiting $25\%\text{--}40\%$ empirical degeneracy). Its status remains strictly **`PROVISIONAL_PENDING_STATISTICAL_VALIDATION`** and trips to **`INCONCLUSIVE`** as mandated by institutional governance.
3. **Primary / Secondary Estimator Boundary:** One-way clustering on `target_val_token_id` is maintained strictly as a **`SECONDARY_SENSITIVITY_ESTIMATOR`**. It has **not** been promoted to primary status.
4. **Execution Boundaries Intact:** Experiment execution remains formally **`UNAPPROVED`**. Production dataset generation remains **`UNAUTHORIZED`**. Calibration remains **`UNAUTHORIZED`**.

### Independent Status Summary Table
| Domain / Component | Status | Classification / Authority |
| :--- | :--- | :--- |
| **Software Implementation** | **`IMPLEMENTATION_VERIFIED`** | All 8 Task Families, metrics, schemas, and runners audited |
| **Synthetic Test Suite** | **`TESTS_PASSED`** | 18 unit tests passing across all boundary conditions |
| **Statistical Estimator** | **`PROVISIONAL_PENDING_VALIDATION`** | Trips to `INCONCLUSIVE` when degeneracy $>10\%$ |
| **Production Dataset Creation** | **`UNAUTHORIZED`** | Strictly prohibited in this phase |
| **Gradient Calibration** | **`UNAUTHORIZED`** | 60-step protocol unexecuted |
| **Canonical Checkpoint Inference**| **`UNAUTHORIZED`** | Live checkpoint forward passes blocked |
| **Experiment Execution** | **`UNAPPROVED`** | Formal execution posture unchanged |

---

## 2. Audited Source Files

The audit covered every file comprising the newly created CAND_01 software infrastructure:

1. [`chakrview/eval/__init__.py`](file:///d:/Project/ChakrView/chakrview/eval/__init__.py) — Package initialization and module namespace.
2. [`chakrview/eval/statistical_bootstrap.py`](file:///d:/Project/ChakrView/chakrview/eval/statistical_bootstrap.py) — Core bootstrap estimators, Kish effective sample size, and Monte Carlo diagnostic harness.
3. [`scripts/generate_cand01_dataset.py`](file:///d:/Project/ChakrView/scripts/generate_cand01_dataset.py) — Synthetic dataset generator implementing TF-01 through TF-08, token disjointness, and VER-03 symmetric closure.
4. [`scripts/evaluate_cand01_benchmark.py`](file:///d:/Project/ChakrView/scripts/evaluate_cand01_benchmark.py) — Benchmark evaluation harness decoupling $A_{\text{FT}}$, $A_{\text{FC}}$, target loss, perplexity, and target rank.
5. [`scripts/run_cand01_experiment.py`](file:///d:/Project/ChakrView/scripts/run_cand01_experiment.py) — Non-interactive dry-run runner with hard execution boundary gates and cross-platform path normalization.
6. [`tests/test_cand01_infrastructure.py`](file:///d:/Project/ChakrView/tests/test_cand01_infrastructure.py) — Comprehensive automated unit test suite.
7. `tests/fixtures/cand01_synthetic/` & `tests/fixtures/cand01_dryrun_output/` — Synthetic test data and execution telemetry.

---

## 3. Findings and Severity Classification

During the independent source-code audit, 6 material findings were identified and dispositioned:

| Finding ID | Severity | Affected File | Description | Remediation Status |
| :--- | :---: | :--- | :--- | :---: |
| **AUD-01** | **HIGH** | `chakrview/eval/statistical_bootstrap.py` | **Scattered & Inconsistent Degeneracy Thresholds:** Discrepancy between registered 10% threshold and test suite 5% trigger; hardcoded magic numbers. | **CORRECTED** |
| **AUD-02** | **HIGH** | `scripts/generate_cand01_dataset.py` | **Incomplete Task Family Implementations:** TF-03 through TF-08 collapsed into a generic distractor template; TF-05 lacked near-key prefix collision, TF-07 lacked distance padding. | **CORRECTED** |
| **AUD-03** | **HIGH** | `scripts/evaluate_cand01_benchmark.py` | **Metric Coupling Violation:** Evaluator computed only single-token exact match from logits, omitting clean termination ($A_{\text{FC}}$) and pre-softmax target rank ($R_{\text{tgt}}$). | **CORRECTED** |
| **AUD-04** | **MEDIUM** | `scripts/run_cand01_experiment.py` | **Path Normalization Vulnerability:** Forbidden directory checks used raw substring matching without `os.path.normpath`, creating platform-dependent path bypass risks. | **CORRECTED** |
| **AUD-05** | **MEDIUM** | `scripts/evaluate_cand01_benchmark.py` | **NumPy int64 JSON Serialization Crash:** Array types in `generated_token_ids` caused `TypeError: Object of type int64 is not JSON serializable` during runner execution. | **CORRECTED** |
| **AUD-06** | **LOW** | `scripts/generate_cand01_dataset.py` | **Context Key Collision Invariant:** Multi-pair sampling did not strictly guard against sampling the same key twice in a single context sequence. | **CORRECTED** |

---

## 4. Remediation Actions Taken

### 4.1 Resolution of AUD-01 (Statistical Threshold Governance)
- Codified authoritative constants at the module level in [`chakrview/eval/statistical_bootstrap.py`](file:///d:/Project/ChakrView/chakrview/eval/statistical_bootstrap.py):
  - `REGISTERED_MAX_DEGENERATE_RATE = 0.10` (10% ceiling)
  - `REGISTERED_MIN_NEFF_FRACTION = 0.25` ($N_{\text{eff}} \ge 0.25 \cdot N$)
  - `REGISTERED_MAX_WEIGHT_VARIANCE = 10.0`
  - `REGISTERED_MIN_UNIQUE_CLUSTERS = 10`
  - `DEFAULT_NUM_BOOTSTRAP_REPLICATES = 2000`
- Eliminated all scattered magic numbers. Updated unit tests to verify exact boundary behavior at the 10% threshold.

### 4.2 Resolution of AUD-02 (Full Task Family Suite TF01–TF08)
- Refactored `generate_record()` in [`scripts/generate_cand01_dataset.py`](file:///d:/Project/ChakrView/scripts/generate_cand01_dataset.py) into dedicated generation paths:
  - **TF-01 (Direct Copy):** `COPY: X -> X` (strictly excluded from Partitions B & C; raises `ValueError` if attempted).
  - **TF-02 (Single Pair):** `PAIR: K=V. QUERY: K -> V`.
  - **TF-03 (Multi-Pair):** Multi-pair associative binding with randomized target position.
  - **TF-04 (Distractors):** Contexts with parameterized distractor tiers (1, 3, 7 distractors).
  - **TF-05 (Similar Keys):** Key disambiguation under prefix collisions (`{key}_alt`).
  - **TF-06 (Recombination):** Held-out association tuples drawn from training entity pool.
  - **TF-07 (Offset Scaling):** Distance padding buffer (`pad_0 .. pad_N`) scaling distance from 10 to 100 tokens.
  - **TF-08 (Context Scaling):** Dense associative contexts scaling from 5 to 15 pairs.

### 4.3 Resolution of AUD-03 (Metric Decoupling & Target Rank)
- Updated [`scripts/evaluate_cand01_benchmark.py`](file:///d:/Project/ChakrView/scripts/evaluate_cand01_benchmark.py) to implement the full metric taxonomy from Evaluator Spec v1.3:
  - **$A_{\text{FT}}$ (First-Token Exact Match):** Primary metric evaluating `generated_tokens[0] == target_token_id`.
  - **$A_{\text{FC}}$ (Full Completion Match):** Co-primary metric evaluating `len == 2 and generated_tokens[0] == target and generated_tokens[1] == EOS`.
  - **$R_{\text{tgt}}$ (Pre-softmax Target Rank):** Diagnostic ranking telemetry: $1 + \sum_{v} \mathbb{I}(z_v > z_{\text{target}})$.
  - **Target Loss & Perplexity:** Evaluated via stable log-softmax.

### 4.4 Resolution of AUD-04 (Cross-Platform Path Gatekeeper)
- Implemented `assert_safe_path()` in [`scripts/run_cand01_experiment.py`](file:///d:/Project/ChakrView/scripts/run_cand01_experiment.py) utilizing `os.path.normpath(os.path.abspath(path)).lower()`.
- Explicitly blocks path traversal escapes (`../../data/preprocessed`) and relative canonical paths across Windows backslashes and Unix slashes.

### 4.5 Resolution of AUD-05 (JSON Serialization Integrity)
- Explicitly cast token arrays and identifiers to Python native `int` types (`[int(t) for t in self.generated_token_ids]`) before serialization in `evaluate_cand01_benchmark.py`.

### 4.6 Resolution of AUD-06 (Context Key Uniqueness)
- Added `used_keys_in_call` tracking in `_sample_pairs()`, mathematically guaranteeing that no synthetic prompt context ever presents conflicting duplicate keys for different values.

---

## 5. Comprehensive Unit Test Coverage

The test suite in [`tests/test_cand01_infrastructure.py`](file:///d:/Project/ChakrView/tests/test_cand01_infrastructure.py) was expanded from 14 to **18 unit tests**, executed against the `.venv` virtual environment:

```text
..................
----------------------------------------------------------------------
Ran 18 tests in 4.419s

OK
```

### Detailed Test Execution Matrix
| Test Case Name | Target Subsystem | Tested Condition | Result |
| :--- | :--- | :--- | :---: |
| `test_token_pool_disjointness` | Generator | $|\mathcal{E}_{\text{train}} \cap \mathcal{E}_{\text{test}}| = 0$ | **PASS** |
| `test_deterministic_generation` | Generator | Deterministic bit-exact output given seed | **PASS** |
| `test_tf01_identity_structure_and_partition_restrictions` | Generator | TF-01 format & exclusion from B/C splits | **PASS** |
| `test_all_eight_task_families_implemented` | Generator | Distinct formats for TF-01 through TF-08 | **PASS** |
| `test_bidirectional_association_leakage_prevention` | Generator | VER-03 symmetric closure isolation (0 leakage) | **PASS** |
| `test_unique_keys_per_context` | Generator | Contexts contain zero duplicate query keys | **PASS** |
| `test_prompt_hash_uniqueness` | Generator | Intra-partition prompt duplicate rejection | **PASS** |
| `test_metric_decoupling_aft_and_afc` | Evaluator | Decoupling $A_{\text{FT}}$, $A_{\text{FC}}$, and $R_{\text{tgt}}$ | **PASS** |
| `test_perplexity_relationship` | Evaluator | $\text{PPL} = \exp(\mathcal{L})$ mathematical check | **PASS** |
| `test_empty_dataset_handling` | Evaluator | Graceful zero-sample execution | **PASS** |
| `test_kish_effective_sample_size_edge_cases` | Bootstrap | Uniform, delta, zero, negative, NaN/Inf weights | **PASS** |
| `test_bootstrap_reproducibility` | Bootstrap | Bit-exact bootstrap CI across matching seeds | **PASS** |
| `test_one_way_cluster_sensitivity_status` | Bootstrap | Retains `SECONDARY_SENSITIVITY_ESTIMATOR` status | **PASS** |
| `test_degeneracy_boundary_conditions_10_percent_threshold` | Bootstrap | Exact boundary check at 10% threshold | **PASS** |
| `test_synthetic_crossed_simulation_diagnostic` | Bootstrap | Monte Carlo simulation structure & MC standard errors | **PASS** |
| `test_assert_safe_path_security_checks` | Security Gate | Rejection of traversal and forbidden paths | **PASS** |
| `test_runner_boundary_rejection_cli` | Runner Gate | Subprocess CLI rejection of `--checkpoint-path` | **PASS** |
| `test_runner_dry_run_success` | Runner Pipeline | End-to-end synthetic dry-run execution | **PASS** |

---

## 6. Synthetic Dry-Run Execution Results

The dry-run pipeline was executed via:
```powershell
.venv\Scripts\python.exe scripts\run_cand01_experiment.py --dry-run --output-dir tests\fixtures\cand01_dryrun_output --replicates 50
```

### Dry-Run Output Telemetry
- **Dataset Synthesis:** Generated 50 train, 35 val recombination, 35 test recombination, and 40 test disjoint records across all 8 Task Families.
- **Leakage Audit Results:**
  - `token_disjointness_passed`: `True` ($0$ overlap)
  - `val_symmetric_leakage_count`: `0`
  - `test_symmetric_leakage_count`: `0`
  - `cross_partition_hash_collisions`: `0`
- **Simulated Performance Metrics:**
  - Validation Recombination: $A_{\text{FT}} = 94.29\%$, $A_{\text{FC}} = 88.57\%$
  - Test Recombination: $A_{\text{FT}} = 94.29\%$, $A_{\text{FC}} = 94.29\%$
  - Test Disjoint: $A_{\text{FT}} = 85.00\%$, $A_{\text{FC}} = 75.00\%$
- **Statistical Uncertainty Classification (A_FT):**
  - `Val Recomb`: `ProvisionalTwoWayClusterBootstrap` $\to$ **`INCONCLUSIVE`** (Degen Rate: $28.6\%$, CI: $[0.7779, 1.0000]$)
  - `Test Recomb`: `ProvisionalTwoWayClusterBootstrap` $\to$ **`INCONCLUSIVE`** (Degen Rate: $39.8\%$, CI: $[0.8154, 1.0000]$)
  - `Test Disjoint`: `ProvisionalTwoWayClusterBootstrap` $\to$ **`INCONCLUSIVE`** (Degen Rate: $25.4\%$, CI: $[0.6973, 0.9945]$)
- **Report Serialization:** Successfully written to [`tests/fixtures/cand01_dryrun_output/cand01_dryrun_execution_report.json`](file:///d:/Project/ChakrView/tests/fixtures/cand01_dryrun_output/cand01_dryrun_execution_report.json).

---

## 7. Monte Carlo Diagnostic Simulation Findings

A Monte Carlo simulation across $N_{\text{sim}} = 50$ crossed synthetic experiments ($|K|=30, |V|=30, N=900, \sigma_{\text{key}}=0.5, \sigma_{\text{val}}=0.5, \mu=0.5000, B=200$) produced:

```json
{
  "diagnostic_disclaimer": "Diagnostic simulation only. Does NOT accredit estimator for production.",
  "num_simulations": 50,
  "nominal_confidence": 0.95,
  "true_underlying_mean": 0.5,
  "coverage": {
    "ordinary_bootstrap_coverage": 0.66,
    "ordinary_mc_se": 0.0670,
    "one_way_clustered_coverage": 0.88,
    "one_way_mc_se": 0.0460,
    "provisional_two_way_coverage": 1.0,
    "provisional_two_way_mc_se": 0.0
  },
  "mean_point_estimate": 0.49898,
  "empirical_bias": -0.00102,
  "mean_two_way_degenerate_rate": 0.3523
}
```

### Statistical Evaluation
1. **Severe Ordinary Undercoverage:** Ordinary bootstrap coverage is severely deflated ($66\% \pm 6.7\%$), confirming that ignoring crossed entity structures invalidates standard standard errors.
2. **One-Way Cluster Bootstrap Limitation:** Clustered on values alone, coverage reaches $88\% \pm 4.6\%$, falling short of the nominal $95\%$ level due to unmodeled key-token covariance.
3. **Two-Way Degeneracy Tripping:** While the provisional two-way estimator covers $100\%$, its mean degenerate replicate rate ($35.23\%$) consistently exceeds the $10\%$ institutional limit.
4. **Governance Consequence:** The software behaves strictly according to protocol by marking uncertainty as **`INCONCLUSIVE`**. It does not mask or silently substitute this outcome.

---

## 8. Protected Hash & Repository Integrity Verification

All canonical model files, session runtimes, checkpoints, and tracked files remain bit-exact and unmodified:

| File Path | Verified SHA-256 Hash | Status |
| :--- | :--- | :--- |
| [`chakrview/brain/model.py`](file:///d:/Project/ChakrView/chakrview/brain/model.py) | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **UNMODIFIED / INTACT** |
| [`chakrview/runtime/session.py`](file:///d:/Project/ChakrView/chakrview/runtime/session.py) | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **UNMODIFIED / INTACT** |
| Step 10k Canonical Checkpoint | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **UNMODIFIED / INTACT** |
| [`.gitignore`](file:///d:/Project/ChakrView/.gitignore) | `700f3d8b6d8c5c216884f1808f5d5c7da17282ce4d69f6e5d563f2ee49e4fda6` | **PRESERVED EXACTLY** |
| [`docs/experiments/CAND_01_V1.3.1_TARGETED_CORRECTIONS.md`](file:///d:/Project/ChakrView/docs/experiments/CAND_01_V1.3.1_TARGETED_CORRECTIONS.md) | `1408dd0d93733ece3987b04426580c552afbf00ab87710d02759f8d2a530ba8c` | **UNMODIFIED / INTACT** |

- **Git Working Tree State:** Branch `main`, HEAD commit `5aeaa2734de5a98802b96afe36324765f3244738`. Exactly 1 tracked modification (`.gitignore`), zero staged files.
- **Excluded Directory:** `experiments/overnight_runs/` was strictly avoided throughout all audit and test operations.

---

## 9. Confirmation of Prohibited Operations Not Performed

The following permanent prohibitions were strictly upheld throughout this entire task:
- Zero reads or scans of `experiments/overnight_runs/`.
- Zero reads of Stage C or canonical pretraining shards.
- Zero evaluation or forward passes against canonical model weights.
- Zero production dataset files generated under `data/experiments/cand_01/`.
- Zero gradient calibration runs executed.
- Zero training loops, backward passes, or optimizer updates executed.
- Zero Git staging, commits, pushes, or resets performed.

---

## 10. Remaining Blockers Before Next Authorization Gates

1. **Pretraining Frequency Extraction Gate:** Authorization is required to read Stage C training shard counts to establish empirical token frequency tiers.
2. **Production Dataset Serialization Gate:** Authorization is required to execute `generate_cand01_dataset.py` with canonical vocabulary bindings and write production datasets.
3. **Statistical Estimator Resolution:** Pre-registration of an asymptotic or GLMM fallback strategy to resolve reporting when the two-way bootstrap trips to `INCONCLUSIVE`.
4. **Pre-Execution Calibration & Inference Gate:** Formal authorization to execute the 60-step calibration protocol and live baseline evaluation against checkpoint step 10,000.

---

## 11. Final Posture

- **Engineering Audit Verdict:** **`AUDIT_PASS_WITH_CONDITIONS`** (Software infrastructure is functionally complete, robustly tested, and safely gated).
- **Production Dataset Generation:** **`UNAUTHORIZED`**.
- **Gradient Calibration:** **`UNAUTHORIZED`**.
- **Experiment Execution:** **`UNAPPROVED`**.

