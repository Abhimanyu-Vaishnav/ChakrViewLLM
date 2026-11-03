# ChakrView Experiment Registration Template v1.0
## Reusable Proposal, Gating, and Execution Tracking Template

**Repository:** `ChakrView`  
**Location:** `docs/experiments/EXPERIMENT_REGISTRATION_TEMPLATE_v1.0.md`  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md`  
**Version:** 1.0.0  
**Effective Date:** 2026-10-31  
**Instructions:** Copy this template into a new file `docs/experiments/EXPXX_[NAME]_REPORT.md` when proposing an experiment. Populate all placeholders (`[...]`). Placeholders must never be filled with invented values.

---

```markdown
# CHAKRVIEW — EXPERIMENT REPORT: [EXPERIMENT_ID]
## [Experiment Title]

**Experiment ID:** EXPXX_[NAME]  
**Research Owner:** [Name / Role]  
**Run Mode:** [EXPLORATORY / CONFIRMATORY]  
**Status:** [PLANNED / RUNNING / COMPLETED / ARCHIVED]  
**Date:** YYYY-MM-DD  

---

### 1. Scientific Formulation
- **Research Question:** [Specific, narrow scientific inquiry]
- **Primary Hypothesis ($H_1$):** [Mechanistic assertion describing why the intervention will alter model behavior]
- **Null Hypothesis ($H_0$):** [Default falsifiable assertion stating no measurable difference from baseline]
- **Mechanistic Rationale:** [Theoretical, architectural, or gradient pathway explaining the expected effect]
- **Independent Variable(s):** [The exact parameter, loss function, or dataset condition being manipulated]
- **Dependent Variable(s):** [Specific quantitative metrics measuring the effect]
- **Controlled Variables:** [Parameters held strictly constant across all experimental arms]
- **Potential Confounders:** [Identified alternative explanations, biases, or artifacts]
- **Pre-Registered Success Criteria:** [Quantitative threshold required to reject H0, locked prior to execution]
- **Rejection Criteria:** [Explicit conditions under which the intervention is deemed ineffective or harmful]

---

### 2. Lineage and Configuration
- **Parent Checkpoint Relative Path:** `[e.g. experiments/batch16_21_combined/.../training_checkpoint_step_10000.pt]`
- **Parent Checkpoint SHA-256:** `[Full 64-hex SHA-256 digest]`
- **Dataset ID(s):** `[From docs/data/DATASET_GOVERNANCE_INDEX.md, e.g. DS-EXP35-TRAIN]`
- **Dataset File Path:** `[Relative path to JSONL]`
- **Dataset SHA-256:** `[Full 64-hex SHA-256 digest]`
- **Dataset Governance Status:** [VERIFIED / INTERNAL_ONLY / HOLD_PROVENANCE / HOLD_LICENSE]
- **Dataset Reproducibility Tier:** [GENERATION_DETERMINISTIC / SPLIT_DETERMINISTIC]
- **Optimizer & LR Schedule:** [e.g. AdamW, peak lr=1e-4, linear warmup (50 steps), cosine decay]
- **Batch Size & Accumulation:** [Micro-batch: N, Gradient Accumulation: M, Effective Batch: N*M]
- **Sequence Length:** [e.g. 128 / 256 / 512 tokens]
- **Max Optimization Steps:** [Hard cap, e.g. 500 steps]
- **Seed List:** `[e.g. Single seed: [42] for Exploratory; Multi-seed: [42, 137, 2026] for Confirmatory]`
- **Hardware & CPU Threads:** [e.g. Local CPU, torch.set_num_threads(8)]

---

### 3. Experimental Controls & Factorial Arms Matrix
| Arm ID | Description | Independent Variable State | Controlled Parameters | Seed List |
| :--- | :--- | :--- | :--- | :--- |
| **Arm A (Control)** | Baseline configuration | State 0 | Locked | `[...]` |
| **Arm B (Intervention)** | Proposed intervention | State 1 | Locked | `[...]` |
| **Arm C (Optional)** | Factorial condition 1 | State 2 | Locked | `[...]` |
| **Arm D (Optional)** | Factorial combined | State 3 | Locked | `[...]` |

---

### 4. Evaluator Identity & Validation Suite
- **Evaluator Script Path:** `[e.g. scripts/eval_bench.py]`
- **Evaluator Script SHA-256:** `[Full 64-hex SHA-256 digest]`
- **Benchmark Dataset Path:** `[e.g. experiments/expXX/benchmark.json]`
- **Benchmark SHA-256:** `[Full 64-hex SHA-256 digest]`
- **Scoring Logic & Boundary Rule:** [Explicit word-boundary regex or AST parser specification; no unanchored substring searches]
- **Pre-Flight Validation Suite:** [Tested against N known correct, M known incorrect, and K edge cases]
- **Stratified Spot-Check Summary:** [Manual audit of N cases; observed error rate and binomial uncertainty reported]

---

### 5. Canonical Non-Inferiority & Regression Gates
| Critical Capability Family | Benchmark Test Suite | Parent Baseline Score | Pre-Registered Non-Inferiority Margin | Allowed Degradation Threshold |
| :--- | :--- | :---: | :---: | :---: |
| Python Syntax Adherence | Canonical Coding Suite | `[...]` | $\Delta \ge 0.0\%$ | Strict zero tolerance |
| Generation Stability | Canonical Repetition Suite | `[...]` | $\text{Margin} \le \pm 1.0\%$ | Scientifically justified margin |
| Context Retention | Canonical Capability Suite | `[...]` | $\Delta \ge 0.0\%$ | Strict zero tolerance |

---

### 6. Resource Quotas & Automated Hard-Stop Triggers
- **Max RAM Ceiling:** `[e.g. 16.0 GB]`
- **Max Disk Allocation:** `[e.g. 2.0 GB]`
- **Max Wall-Clock Duration:** `[e.g. 4.0 hours]`
- **Maximum Retry Limit:** `[1 retry for transient OS error; 0 retries for script code crash]`
- **Automated Hard-Stop Triggers:**
  - Loss value equals `NaN` or `Inf`.
  - Gradient norm exceeds clipping threshold for 5 consecutive steps.
  - Resident RAM exceeds registered ceiling.
  - Disk free space drops below 5.0 GB.
  - Model outputs repetitive token collapse for $\ge 20$ tokens on diagnostic probe.
  - Any unauthorized modification of tracked files or canonical assets.

---

### 7. Evaluation & Statistical Results Matrix
| Capability Dimension | Evaluated Metric | Parent Baseline | Candidate (Arm A) | Candidate (Arm B) | Mean Delta | p-value / 95% CI | Gate Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Optimization | Final Train Loss | `[...]` | `[...]` | `[...]` | `[...]` | N/A | Logged |
| Generation Stability | Repetition Rate | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | [PASS / FAIL] |
| Target Task | Primary Accuracy | `[...]` | `[...]` | `[...]` | `[...]` | `[...]` | [PASS / FAIL] |

#### Per-Seed Performance Disclosure (Confirmatory Runs)
| Arm ID | Seed 42 Score | Seed 137 Score | Seed 2026 Score | Mean $\pm$ SEM | Outlier / Excluded Seeds |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Arm A | `[...]` | `[...]` | `[...]` | `[...]` | None |
| Arm B | `[...]` | `[...]` | `[...]` | `[...]` | None |

---

### 8. Qualitative Failure Analysis
- **Error Distribution:** [Classification of top error modes]
- **Sample Failure 1:** [Input prompt $\rightarrow$ Model completion $\rightarrow$ Target expected $\rightarrow$ Root cause analysis]
- **Sample Failure 2:** [Input prompt $\rightarrow$ Model completion $\rightarrow$ Target expected $\rightarrow$ Root cause analysis]

---

### 9. Four-Dimensional Reproducibility Status
- `DATASET_REPRODUCIBILITY`: [GENERATION_DETERMINISTIC / SPLIT_DETERMINISTIC / REPRODUCIBILITY INCOMPLETE]
- `TRAINING_REPRODUCIBILITY`: [STATISTICALLY_REPRODUCIBLE / BITWISE_DETERMINISTIC_RUN]
- `CHECKPOINT_INTEGRITY`: `VERIFIED_HASH_MATCH` (SHA-256: `[Output checkpoint hash]`)
- `EVALUATION_REPRODUCIBILITY`: `VERIFIED_DETERMINISTIC_GREEDY` (Evaluator SHA-256 locked)

---

### 10. Pre-Execution Approval Record (GO / NO-GO Gate)
*Must be signed by independent reviewer / maintainer prior to launching training script.*
- **Reviewer / Maintainer Name:** `[Name / Role]`
- **Review Date:** YYYY-MM-DD
- **Pre-Execution Verdict:** `[GO / NO-GO]`
- **Approved Run Mode:** `[EXPLORATORY / CONFIRMATORY]`
- **Approved Lineage & Resource Bounds:** Verified parent hash `[...]`, RAM ceiling `[...]` GB, seed list `[...]`.
- **Reviewer Signature:** `[Signed / Authorized]`

---

### 11. Post-Execution Decision & Terminal Verdict
- **Terminal Decision:** `[PROMOTE / CONTINUE / REJECT / INCONCLUSIVE]`
- **Scientific Justification:** [Synthesis of empirical evidence directly against pre-registered criteria; explanation of why H0 was rejected or failed to be rejected]
- **Canonical Asset Protection Confirmed:** Parent checkpoint, `model.py`, and `session.py` verified bit-exact identical.
- **Git State Confirmed:** Zero staged files, zero remote pushes, clean untracked state preserved.
- **Project Maintainer Sign-Off:** `[Signed / Authorized]` | **Date:** YYYY-MM-DD
```

