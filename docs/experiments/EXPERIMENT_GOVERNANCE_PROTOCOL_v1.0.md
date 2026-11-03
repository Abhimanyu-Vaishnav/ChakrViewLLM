# ChakrView Experiment Governance Protocol v1.0
## Controlled Scientific Lifecycle, Evidence Standards & Repository Protection

**Repository:** `ChakrView`  
**Location:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md`  
**Purpose:** Formal, reproducible, and verifiable experiment execution protocol for all research and training across ChakrView.  
**Version:** 1.0.0 (Clarification Patch v1.1 applied)  
**Effective Date:** 2026-10-31  
**Status:** Canonical Governance Standard  

---

## 1. Purpose and Scope

ChakrView is an indigenous, modular, low-resource, CPU-first neural language model engineered from first principles. Its continuous advancement depends on rigorous scientific inquiry, reproducible experimental configurations, and uncompromising verification of actual computational and linguistic capabilities.

### 1.1 The Core Scientific Invariant
> [!IMPORTANT]
> **AN EXPERIMENT IS NOT SUCCESSFUL MERELY BECAUSE TRAINING LOSS DECREASES.**  
> Lower training loss proves only that an optimizer fitted weights to training tokens. It does not prove intelligence, instruction following, generalization, multi-turn reasoning, or factual grounding.

### 1.2 Operational Scope
This protocol governs all machine learning research, pre-training extensions, supervised fine-tuning (SFT), continued pre-training (CPT), dynamic prompt conditioning, decoding intervention experiments, and benchmark evaluations conducted in the ChakrView repository.

- **Covered Activities:**
  - Any model initialization, optimizer stepping, fine-tuning, or adaptation.
  - Controlled A/B training comparisons (e.g. loss masking, curriculum ordering, regularized retention).
  - Diagnostic evaluations, benchmark suites, and probe assessments.
  - Creation, storage, and retirement of experimental checkpoints.
- **Outside Scope:**
  - Pure software development and unit testing of utility modules that do not alter model state or evaluate model capability.
  - Pure repository documentation maintenance (governed by repository management rules).

### 1.3 Relationship to Dataset Governance
This protocol directly integrates with and enforces the five canonical dataset governance registers in `docs/data/`:
1. `DATASET_GOVERNANCE_INDEX.md`
2. `DATASET_PROVENANCE_REGISTER.md`
3. `DATASET_LICENSE_REGISTER.md`
4. `DATASET_REPRODUCIBILITY.md`
5. `DATASET_RELEASE_CHECKLIST.md`

No experiment may ingest, train upon, or evaluate against a dataset whose lineage, schema, and split boundaries violate the dataset governance standards. Unresolved provenance or licensing holds on a dataset strictly restrict the experiment to internal research and bar any associated model checkpoint from public release.

---

## 2. Mandatory Experiment Lifecycle

Every experiment in ChakrView must advance sequentially through twelve mandatory phases. No phase may be bypassed, merged without documentation, or retroactively fabricated.

```
[1. Proposal] ───────────────► [2. Hypothesis Registration] ───► [3. Dataset & Evidence Review]
                                                                        │
[6. Baseline Capture] ◄─────── [5. Pre-Execution Approval] ◄───────────┘
       │
       ▼
[7. Training Execution] ─────► [8. Checkpoint Capture] ───────► [9. Independent Evaluation]
                                                                        │
[12. Decision & Archive] ◄─── [11. Forensic Audit] ◄──────────── [10. Analysis & Synthesis]
```

### Phase Details and Gates:

1. **Proposal:** Define the operational necessity, resource estimate, run mode, and architectural scope.
   - *Inputs:* Problem statement or diagnostic finding from earlier milestones.
   - *Action:* Draft candidate specification.
   - *Gate to Proceed:* Must address a known failure mode or capability objective without violating CPU-first constraints.
2. **Hypothesis Registration:** Formalize the scientific hypothesis and null hypothesis.
   - *Inputs:* Draft proposal.
   - *Action:* Complete formal Section 3 preregistration fields.
   - *Gate to Proceed:* Must establish measurable dependent variables and falsifiable rejection criteria.
3. **Evidence and Dataset Review:** Audit training and evaluation data.
   - *Inputs:* Proposed JSONL datasets and evaluation suites.
   - *Action:* Verify against `docs/data/` manifests, check split boundaries, confirm leak-free isolation.
   - *Gate to Proceed:* Must have verified SHA-256 hashes and zero split contamination.
4. **Experimental Design:** Define controlled variables, baselines, and arms.
   - *Inputs:* Registered hypothesis and audited data.
   - *Action:* Define identical initializations, compute budgets, batch sizes, and seeds.
   - *Gate to Proceed:* Single-variable intervention or explicit factorial design.
5. **Pre-Execution Approval (GO/NO-GO Gate):** Formal review against the Section 14 Checklist.
   - *Inputs:* Complete experimental spec and resource limits.
   - *Action:* Audit pre-flight gates; issue written GO or NO-GO verdict.
   - *Gate to Proceed:* Explicit written sign-off by research owner/maintainer.
6. **Baseline Capture:** Record parent checkpoint metrics before training.
   - *Inputs:* Parent checkpoint path.
   - *Action:* Run full evaluation suite on the parent model to establish zero-point reference.
   - *Gate to Proceed:* Baseline scores logged in immutable JSON.
7. **Training Execution:** Execute bounded optimization with active telemetry.
   - *Inputs:* Pre-flight approved scripts and locked seeds.
   - *Action:* Run training with bounded step count, logging step-level loss, grad norm, and memory.
   - *Gate to Proceed:* Normal convergence without NaN, Inf, exploding gradients, or hardware thrashing.
8. **Checkpoint Capture:** Persist candidate model weights and execution snapshots.
   - *Inputs:* Trained model state.
   - *Action:* Save checkpoints separately from canonical assets; compute SHA-256 immediately.
   - *Gate to Proceed:* Verify weight tensors match expected architecture and are non-empty.
9. **Independent Evaluation:** Run locked evaluation benchmarks without ground-truth leakage.
   - *Inputs:* Checkpoints and isolated evaluation suites.
   - *Action:* Run independent scoring scripts with temperature=0.0 (greedy decoding) unless sampling is explicitly under test.
   - *Gate to Proceed:* Raw predictions preserved for every task.
10. **Statistical and Qualitative Analysis:** Evaluate effect sizes, error categories, and failure samples.
    - *Inputs:* Raw generation logs and benchmark JSON outputs.
    - *Action:* Calculate delta over baseline, classify failure modes, evaluate distribution shift.
    - *Gate to Proceed:* Comparative matrix completed.
11. **Forensic Audit:** Conduct independent sanity review of evaluator behavior.
    - *Inputs:* Evaluation outputs and scoring code.
    - *Action:* Check for regex artifacts, substring false positives, prompt formatting mismatch.
    - *Gate to Proceed:* Evaluator integrity confirmed.
12. **Decision and Archival (Post-Execution Decision):** Assign one of four standardized decisions and lock artifacts.
    - *Inputs:* Audit findings and comparative analysis.
    - *Action:* Render verdict (`PROMOTE`, `CONTINUE`, `REJECT`, `INCONCLUSIVE`), update registers, preserve Git tree.
    - *Gate to Proceed:* Project maintainer sign-off.

---

## 3. Hypothesis Registration Standards

Unverified hypotheses and mechanistic speculations must never be documented as empirical facts.

### 3.1 Mandatory Experiment Run Modes
Every registered proposal must explicitly designate one of two mutually exclusive run modes:

- **`EXPLORATORY`:**
  - **Purpose:** Used for hypothesis generation, initial debugging, architectural probing, and feasibility assessment.
  - **Rules:** May utilize single-seed pilots and flexible diagnostic probing. Results may motivate future hypotheses but **cannot** independently establish a verified scientific claim and **cannot** directly justify model promotion (`PROMOTE`).
- **`CONFIRMATORY`:**
  - **Purpose:** Rigorous verification of a pre-registered scientific hypothesis.
  - **Rules:** Requires pre-registered primary and null hypotheses, fixed evaluation plans, multi-seed validation ($\ge 3$ seeds), explicit non-inferiority margins, formal statistical testing, and pre-declared decision rules. Any change made after training initiates must be formally logged as an experimental deviation and cannot be claimed as pre-registered.

*Rule: An exploratory finding may motivate a confirmatory experiment, but the confirmatory run must be registered and approved as a distinct, independent experiment.*

### 3.2 Registration Fields
Every registered experiment must record:
- **Experiment ID:** Unique identifier following convention `EXP<number>_<descriptor>` (e.g. `EXP34_ATTRACTOR_SUPPRESSION`).
- **Run Mode:** `EXPLORATORY` or `CONFIRMATORY`.
- **Research Question:** Explicit question regarding model learning, representation, or behavior.
- **Primary Hypothesis ($H_1$):** Mechanistic assertion stating why a specific intervention will alter model behavior.
- **Null Hypothesis ($H_0$):** Default assertion stating that the intervention produces no measurable capability improvement over the baseline.
- **Mechanistic Rationale:** Explanation of the expected neural or representational pathway.
- **Expected Observable Behavior:** Quantitative and qualitative changes anticipated in generations.
- **Independent Variable(s):** The single parameter, loss term, or data condition being manipulated.
- **Dependent Variables:** Specific metrics measuring the effect.
- **Controlled Variables:** Variables held strictly constant across all experimental arms.
- **Potential Confounders:** Known alternative explanations.
- **Expected Failure Modes:** Plausible degeneration patterns.
- **Pre-Registered Success Criteria:** Quantitative thresholds required to reject $H_0$, defined prior to execution.
- **Rejection Criteria:** Conditions under which the intervention is deemed ineffective or harmful.

---

## 4. Baseline and Experiment Lineage

Every experimental run must maintain unbroken provenance back to an established canonical checkpoint.

### 4.1 Canonical Artifact Invariance
- **Protected Checkpoint Policy:** The canonical Step 10,000 checkpoint (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`, SHA-256 `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`) is **immutable**. It must never be overwritten, modified in-place, or moved.
- **Protected Architecture Code:** Core neural definitions in `chakrview/brain/model.py` (SHA-256 `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361`) and session management in `chakrview/runtime/session.py` (SHA-256 `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`) are strictly protected.
- **Storage Separation:** All experimental candidate checkpoints must reside in experiment-isolated subdirectories (e.g. `experiments/expXX/arm_X/checkpoints/`).

### 4.2 Lineage Metadata Tracking
Every checkpoint must be accompanied by an immutable manifest recording:
- Parent checkpoint relative path and full 64-hex SHA-256 digest.
- Active Git commit HEAD hash at run start.
- Architecture parameters (layers, heads, embed dimension, vocabulary size).
- Tokenizer files, vocabulary sizes, and tokenizer hashes.
- Optimizer type, learning rate schedule, warm-up steps, weight decay.
- Batch size, micro-batch size, gradient accumulation steps, sequence length.
- Total optimizer steps and total tokens ingested.
- Exact random seeds (`seed=42`).
- Python version, PyTorch version, operating system, and hardware architecture.
- UTC execution timestamp.

---

## 5. Dataset and Split Governance

### 5.1 Dataset Eligibility
Training may only commence with datasets registered in `docs/data/DATASET_GOVERNANCE_INDEX.md`. The dataset must possess a verified SHA-256 digest matching its manifest.

### 5.2 Strict Split Isolation
- **No Shared Records:** Training, validation, and evaluation benchmarks must be partitioned with zero record overlap.
- **No Prompt Leakage:** Evaluation benchmark prompt strings must not exist within training data.
- **No Template Leakage:** Evaluation tasks must not simply swap slot entities into identical prompt templates used during training unless measuring pure in-distribution fitting.
- **Out-of-Distribution (OOD) Protection:** True OOD benchmarks must utilize unseen entity combinations, distinct syntactic structures, and separate relation pairs.
- **Benchmark Retirement Policy:** If an independent benchmark is inadvertently exposed to training or used iteratively for checkpoint tuning, it is permanently retired as an independent benchmark and redesignated as a development set. A new, unexposed benchmark must be authored and frozen before capability claims can be made.

---

## 6. Experimental Controls, Factorial Design & Seed Governance

### 6.1 Principles of Controlled Comparison
To attribute an observed effect to a specific intervention, experimental designs must enforce strict parity:
- **Initialization Parity:** All comparison arms must branch from the exact same parent checkpoint weights.
- **Data Parity:** Ingestion order, token count, and data splits must be identical across arms, unless data mixture is the independent variable.
- **Compute Parity:** Optimizer steps and effective batch sizes must be identical.
- **Decoding Parity:** Post-training evaluation across arms must use identical decoding parameters (temperature, top-p, repetition penalty, context framing).
- **Factorial Design for Multi-Variable Interventions:** If an experiment combines multiple interventions (e.g. prompt-loss masking AND inhibitor prefixes), testing them in a single combined arm creates an uninterpretable confounder. Multi-variable interventions must execute a $2 \times 2$ factorial matrix:
  - *Arm 1 (Control):* Baseline prompt + unmasked loss.
  - *Arm 2 (Intervention A):* Baseline prompt + masked loss.
  - *Arm 3 (Intervention B):* Inhibitor prefix + unmasked loss.
  - *Arm 4 (Combined A+B):* Inhibitor prefix + masked loss.

### 6.2 Seed Governance & Aggregation Policy
- **Exploratory Runs:** Single-seed execution is permitted for initial pilot feasibility.
- **Confirmatory Runs:** Minimum of **three (3) independent random seeds** are required (e.g. `seeds: [42, 137, 2026]`).
- **Seed Policy Parity:** The exact same seed list must be executed across all comparative arms.
- **Aggregation Rules:**
  - Both individual seed results and aggregated statistics (mean $\pm$ standard error of the mean) must be reported.
  - Selective removal, cherry-picking, or post-hoc exclusion of poor-performing seeds is strictly prohibited.
  - If a seed run fails due to hardware crash or resource abort, the failure must be documented in the report.

---

## 7. Evaluation Governance, Evaluator Integrity & Acceptance Standards

Capability assessment must evaluate multi-dimensional language model behavior rather than relying on a single aggregate metric.

### 7.1 Required Capability Dimensions
Every capability audit must report on:
1. **Training & Validation Loss:** Optimization dynamics, convergence trajectory, and validation perplexity.
2. **Language Generation Stability:** EOS token prediction, repetition loop resistance, attractor basin avoidance, and sequence length variance.
3. **Instruction Adherence:** Format following (JSON, markdown, bulleted lists), constraint satisfaction, and negative constraint compliance.
4. **Context & Multi-Turn Retention:** State tracking across turns, pronoun resolution, and history recall.
5. **Entity Copying & Grounding:** Extracting and accurately copying target substrings directly from prompt context.
6. **Reasoning & Arithmetic:** Step-by-step logic, operator recognition, and verified mathematical calculation.
7. **Syntax & Code Structure:** Valid syntax generation (Python parsing without syntax error).
8. **Generalization to Unseen Distributions:** Performance on OOD entity pairs and novel syntactic phrasings.
9. **Resource Efficiency:** Inference throughput (tokens/second on CPU) and memory consumption.

### 7.2 Evaluator Integrity and Acceptance Standards
Automated evaluation scripts are susceptible to measurement errors that can masquerade as model intelligence.

Every evaluator script must be accompanied by an **Evaluator Identity & Validation Record**:
- **Evaluator Identity:** Exact script path, Git commit/version, and full 64-hex SHA-256 digest.
- **Scoring Logic Specification:** Explicit documentation of normalization, whitespace handling, and case folding.
- **Mandatory Boundary Matching:** Word-boundary regex (e.g. `\bTARGET\b`) or token-level alignment is required. Unanchored substring matching (`target in prediction`) is **strictly prohibited**.
- **Edge-Case Handling:** Explicit rules for handling empty outputs, malformed responses, refusals, and repetitive loops.
- **Validation Suite:** Evaluators must be pre-tested against a test suite containing known correct completions, known distractors, near-miss substrings, and malformed texts.
- **Stratified Spot-Checking:** Manual adjudication of a stratified sample (minimum 20 cases across task families).
  - *Scientific Guard on Small Samples:* Observing 0 errors in a 20-sample spot-check does **not** prove zero underlying error rate. The report must state the sample size and acknowledge binomial uncertainty.
- **Blocking Defect Threshold:** Any known answer leakage, benchmark contamination, or systematic false-acceptance flaw in the evaluator **immediately blocks** experimental evaluation until the evaluator is corrected and re-audited.

### 7.3 Canonical Non-Inferiority & Regression Governance
To prevent catastrophic forgetting during domain fine-tuning, candidate checkpoints must undergo canonical regression testing:
- **Baseline Metric Freeze:** The comparison baseline must use the exact parent checkpoint evaluated under identical decoding conditions. Historical baseline numbers must not be assumed without verified configuration parity.
- **Pre-Registered Non-Inferiority Margins:** The experiment proposal must pre-register acceptable performance deltas on critical canonical tasks:
  - *Critical Core Capabilities (e.g. Syntax parsing, EOS termination):* Zero regression tolerance ($\Delta \ge 0.0\%$).
  - *Secondary Benchmarks:* Non-inferiority margin pre-defined and scientifically justified based on benchmark sample size and baseline variance.
- **No Aggregate Concealment:** An experiment cannot claim non-inferiority based on an aggregate score if a critical capability family suffers a severe regression.

---

## 8. Training and Resource Governance

ChakrView is committed to CPU-first, reproducible execution on commodity hardware.

### 8.1 Resource Quotas & Operational Limits
Every experiment proposal must pre-register resource boundaries calibrated to available hardware:
- **CPU Quota:** Maximum CPU thread allocation (e.g. `torch.set_num_threads(8)`).
- **Memory Ceiling:** Maximum allowable RAM allocation (e.g. 16 GB).
- **Disk Quota:** Maximum disk footprint allocated for checkpoints and logs (e.g. $\le 2.0$ GB).
- **Execution Timeout:** Maximum wall-clock training duration and step bound.
- **Run Limit:** Maximum number of automated retry attempts (maximum 1 retry for transient system errors; 0 retries for code crashes).

### 8.2 Safe Stopping Triggers (Hard Abort Conditions)
Training runners must implement automated process termination upon encountering:
- Loss value becomes `NaN` or `Inf`.
- Gradient norm exceeds clipping threshold for 5 consecutive steps.
- Resident memory exceeds the registered RAM ceiling.
- Disk free space drops below critical safety threshold (e.g. 5 GB remaining).
- Complete mode collapse (e.g. generation probe outputs identical repetitive token loop for 20 tokens).
- Unintended modification of tracked files or canonical checkpoints.

*Rule: Any hard-stop event must be fully documented in the experiment report; aborted runs must never be silently deleted or omitted.*

### 8.3 Checkpoint Preservation Policy
Automated cleanup scripts must **never** delete previous checkpoints or historical experiment runs. Checkpoints must be preserved until formal review categorizes them as archived or retired.

---

## 9. Statistical and Scientific Interpretation

To maintain intellectual honesty, analysis must respect rigorous semantic boundaries:
- **Loss vs. Intelligence:** Lower cross-entropy loss indicates better probability assignment over training tokens. It does not prove logical reasoning or semantic understanding.
- **Stability vs. Capability:** A model that reliably outputs a period and `<EOS>` has achieved generation stability; this must not be conflated with conversational competence.
- **Token Copying vs. Comprehension:** Verbatim reproduction of prompt tokens indicates positional or attention routing; it does not establish factual deduction.
- **Syntax vs. Semantics:** Generating syntactically parseable Python code does not establish that the code correctly solves the intended algorithmic problem.
- **Statistical Significance Requirements:**
  - No comparative claim may assert statistical significance without an explicit test (e.g. paired t-test for normal metrics, Wilcoxon signed-rank or bootstrap resampling for non-normal task accuracy).
  - Multiple comparison corrections (e.g. Holm-Bonferroni) must be applied when evaluating multiple task families simultaneously.
  - Both effect sizes (Cohen's $d$ or absolute accuracy delta) and confidence intervals must be reported.
- **Statistical vs. Practical Significance:** A statistically significant gain on a micro-metric that produces no observable change in human dialogue quality does not justify candidate promotion.

---

## 10. Experiment Decision Framework

Upon completion of training, evaluation, and forensic audit, every experiment must be formally categorized into exactly one of four terminal verdicts:

```
                          [Completed Experiment]
                                    │
       ┌────────────────┬───────────┴───────────┬────────────────┐
       ▼                ▼                       ▼                ▼
   [PROMOTE]       [CONTINUE]               [REJECT]       [INCONCLUSIVE]
```

### 1. `PROMOTE`
- **Definition:** The intervention conclusively validated its primary hypothesis, exceeded all pre-registered success criteria, produced zero regressions beyond pre-registered non-inferiority margins on canonical baselines, and passed all integrity audits.
- **Prerequisites:** Must be a `CONFIRMATORY` run with multi-seed verification; all datasets must have verified lineage; evaluator must have passed integrity audit.
- **Operational Action:** Checkpoint is designated as a milestone candidate.
- **Restriction:** Promotion does *not* authorize public distribution until dataset licensing, clean-tree staging, and owner approvals pass.

### 2. `CONTINUE`
- **Definition:** The intervention demonstrated promising evidence or partial hypothesis validation, but residual confounding factors, hyperparameter sensitivities, or unverified seeds require further controlled experimentation.
- **Operational Action:** Formulate follow-up experiment specification referencing the current run.

### 3. `REJECT`
- **Definition:** The primary hypothesis was falsified ($H_0$ could not be rejected), the intervention produced unacceptable capability regressions, or the approach proved computationally infeasible.
- **Operational Action:** Experiment is permanently retired and archived with full failure forensics.

### 4. `INCONCLUSIVE`
- **Definition:** Experimental execution was compromised by evaluator defects, data contamination, hardware disruption, or unforeseen confounders that prevent scientific interpretation.
- **Operational Action:** Results are invalidated. Flaws must be remediated before any subsequent run.

---

## 11. Reproducibility Taxonomy & Artifact Integrity

Scientific reproducibility requires orthogonal accounting across four distinct domains. Conflating these dimensions under a single ambiguous label is prohibited.

### 11.1 The Four Reproducibility Dimensions
1. **Dataset Reproducibility (`DS_REPRODUCIBILITY`):**
   - Refers strictly to data generation and partitioning.
   - *Tiers:* `GENERATION_DETERMINISTIC`, `SPLIT_DETERMINISTIC`, `BIT_EXACT_VERIFIED`, `REPRODUCIBILITY INCOMPLETE`.
   - *Invariance:* `GENERATION_DETERMINISTIC` describes dataset compilation only; it must **never** be applied as a label for model training.
2. **Training Reproducibility (`TRAIN_REPRODUCIBILITY`):**
   - Refers to model weight optimization.
   - *Tiers:* `STATISTICALLY_REPRODUCIBLE` (multi-seed metrics within confidence intervals), `BITWISE_DETERMINISTIC_RUN` (exact gradient and weight match under identical CPU thread pins), or `NONDETERMINISTIC_RUN`.
3. **Checkpoint Integrity (`CHECKPOINT_INTEGRITY`):**
   - Refers to the physical model weight artifacts.
   - *Verification:* Verified full 64-hex SHA-256 match, uncorrupted state dict, verified non-zero parameter tensors.
4. **Evaluation Reproducibility (`EVAL_REPRODUCIBILITY`):**
   - Refers to scoring benchmark repeatability.
   - *Verification:* Evaluator script SHA-256 locked, greedy decoding ($T=0.0$), identical context boundaries producing identical task scores on re-run.

### 11.2 Artifact Bundle Requirements
Every experiment archive must contain:
1. Experiment specification and completion report.
2. Training configuration snapshot (`json`).
3. Pre-training and post-training evaluation matrices (`json`).
4. Raw generation samples and task-level predictions (`json`).
5. Step-level training telemetry logs (`json`).
6. Cryptographic SHA-256 digests for all datasets, parent checkpoints, and generated checkpoints.

---

## 12. Git and Repository Governance

Experimentation must not degrade repository hygiene.

### 12.1 Mandatory Git Audit Sequence
- **Pre-Flight Check:** Record current branch, HEAD SHA, staged files, tracked modifications, and untracked entries before executing any script.
- **Post-Run Check:** Confirm that no unintended files were created, no tracked files were modified, and the index remains unstaged.

### 12.2 Prohibited Actions
- `git add .` or broad wildcard staging is strictly prohibited.
- Model weight files (`*.pt`) must never be committed to Git (governed by root `.gitignore`).
- No experiment script or autonomous subagent may execute `git commit`, `git push`, `git reset`, `git checkout`, or history rewrites without explicit project maintainer authorization.

### 12.3 Absolute Protection of Overnight Runs
The local directory `experiments/overnight_runs/` (~5.12 GB, ~25,470 files) is strictly local-only and protected by `.gitignore`. No experiment script, audit runner, or maintenance routine may scan, read, write, or access this directory.

---

## 13. Experiment Documentation Template

All future experiments must document their design, telemetry, and findings using the following standardized markdown template:

```markdown
# CHAKRVIEW — EXPERIMENT REPORT: [EXPERIMENT_ID]
## [Experiment Title]

**Experiment ID:** EXPXX_NAME  
**Research Owner:** [Name / Role]  
**Run Mode:** [EXPLORATORY / CONFIRMATORY]  
**Status:** [PLANNED / RUNNING / COMPLETED / ARCHIVED]  
**Date:** YYYY-MM-DD  

### 1. Scientific Formulation
- **Research Question:** [Specific scientific inquiry]
- **Primary Hypothesis ($H_1$):** [Mechanistic assertion]
- **Null Hypothesis ($H_0$):** [Falsifiable baseline assertion]
- **Mechanistic Rationale:** [Theoretical or architectural basis]
- **Independent Variable(s):** [Controlled intervention]
- **Dependent Variable(s):** [Measured metrics]
- **Controlled Variables:** [Locked parameters across arms]
- **Potential Confounders:** [Identified confounding factors]
- **Pre-Registered Success Criteria:** [Quantitative thresholds to reject H0]
- **Rejection Criteria:** [Explicit failure thresholds]

### 2. Lineage and Configuration
- **Parent Checkpoint Path:** `path/to/checkpoint.pt`
- **Parent Checkpoint SHA-256:** `[Full 64-hex SHA-256]`
- **Dataset ID(s):** `DS-NAME` (from `docs/data/DATASET_GOVERNANCE_INDEX.md`)
- **Dataset SHA-256:** `[Full 64-hex SHA-256]`
- **Dataset Governance Status:** [VERIFIED / HOLD_PROVENANCE / HOLD_LICENSE]
- **Dataset Reproducibility Tier:** `GENERATION_DETERMINISTIC`
- **Optimizer & LR Schedule:** [e.g. AdamW, lr=1e-4, linear warmup, cosine decay]
- **Batch Size & Accumulation:** [Micro-batch, accumulation steps, effective batch]
- **Max Steps & Seed List:** `steps=[N]`, `seeds: [42, 137, 2026]`
- **Hardware & CPU Threads:** [e.g. 8 threads, CPU-only]

### 3. Experimental Controls & Factorial Arms
| Arm ID | Description | Independent Variable State | Control Settings | Seed List |
| :--- | :--- | :--- | :--- | :--- |
| Arm A (Control) | Baseline formulation | State 0 | Locked | [42, ...] |
| Arm B (Candidate) | Proposed intervention | State 1 | Locked | [42, ...] |

### 4. Evaluator Identity & Validation Suite
- **Evaluator Script Path:** `scripts/eval_bench.py`
- **Evaluator Script SHA-256:** `[Full 64-hex SHA-256]`
- **Benchmark Name & SHA-256:** `benchmark.json` (`[Full 64-hex SHA-256]`)
- **Scoring Logic & Word Boundary Rule:** [e.g. Exact word-boundary regex \bTARGET\b]
- **Validation Spot-Check Summary:** [Audited N cases; observed 0 false positives]

### 5. Canonical Non-Inferiority & Regression Gates
| Capability Dimension | Benchmark Suite | Baseline (Parent) | Non-Inferiority Margin | Allowed Degradation |
| :--- | :--- | :---: | :---: | :---: |
| Syntax Adherence | Canonical Coding Suite | [Baseline Score] | $\Delta \ge 0.0\%$ | Strict zero tolerance |
| Repetition Stability | Repetition Benchmark | [Baseline Score] | Margin $\le \pm 1.0\%$ | Justified margin |

### 6. Resource Quotas & Hard-Stop Triggers
- **Max RAM Ceiling:** [e.g. 16 GB]
- **Max Disk Footprint:** [e.g. 2.0 GB]
- **Max Step Bound:** [e.g. 500 steps]
- **Hard-Stop Triggers:** [NaN loss, gradient norm > 5.0 for 5 steps, RAM exhaustion]

### 7. Evaluation & Statistical Results Matrix
| Capability Dimension | Metric | Baseline (Parent) | Candidate (Arm A) | Candidate (Arm B) | Mean Delta | p-value / CI | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Optimization | Final Loss | | | | | N/A | |
| Target Task | Accuracy / F1 | | | | | | |

### 8. Qualitative Failure Analysis
- Sample failure cases and error classification.

### 9. Four-Dimensional Reproducibility Status
- `DATASET_REPRODUCIBILITY`: `GENERATION_DETERMINISTIC`
- `TRAINING_REPRODUCIBILITY`: `STATISTICALLY_REPRODUCIBLE`
- `CHECKPOINT_INTEGRITY`: `VERIFIED_HASH_MATCH` (SHA-256: `[Hash]`)
- `EVALUATION_REPRODUCIBILITY`: `VERIFIED_DETERMINISTIC_GREEDY`

### 10. Pre-Execution GO/NO-GO Approval Record
- **Reviewer / Maintainer:** [Name / Role]
- **Review Date:** YYYY-MM-DD
- **Pre-Execution Verdict:** [GO / NO-GO]
- **Approval Notes:** [Verification of lineage, evaluator, and quotas]

### 11. Final Decision & Verdict
- **Decision:** [PROMOTE / CONTINUE / REJECT / INCONCLUSIVE]
- **Scientific Justification:** [Synthesis of evidence against pre-registered criteria]
- **Owner Sign-Off:** [Signature / Date]
```

---

## 14. Pre-Execution Checklist (GO / NO-GO Gate)

Before executing any training or optimization command, the researcher or agent must verify and check every item. The Pre-Execution Gate is binary: **GO** (all pass) or **NO-GO** (any fail).

- [ ] **Gate 1: Run Mode & Hypothesis Registered:** Run mode (`EXPLORATORY` vs `CONFIRMATORY`), $H_1$, $H_0$, and measurable success criteria are locked.
- [ ] **Gate 2: Parent Lineage Confirmed:** Parent checkpoint path and SHA-256 are verified; canonical assets are protected.
- [ ] **Gate 3: Dataset Governance Validated:** Training dataset is recorded in `DATASET_GOVERNANCE_INDEX.md` with verified SHA-256 and compatible license role.
- [ ] **Gate 4: Split Isolation Verified:** Zero overlap between train, validation, and evaluation benchmarks.
- [ ] **Gate 5: Evaluator Audited:** Evaluator script SHA-256 locked, word-boundary regex verified, zero answer leakage confirmed.
- [ ] **Gate 6: Controls & Seeds Locked:** Comparison arms have identical compute, initialization, and locked multi-seed lists ($\ge 3$ for confirmatory).
- [ ] **Gate 7: Non-Inferiority Margins Set:** Critical capability families and regression margins pre-registered.
- [ ] **Gate 8: Resource Limits Configured:** Hard memory caps, thread quotas, and abort conditions defined.
- [ ] **Gate 9: Protected Directory Safe:** Scripts confirm no access or writes to `experiments/overnight_runs/`.
- [ ] **Gate 10: Git Baseline Recorded:** Clean working tree recorded; zero staged files.
- [ ] **Gate 11: Written GO Approval:** Formal pre-execution sign-off by research maintainer.

*Execution is blocked if any check fails.*

---

## 15. Post-Experiment Audit Checklist

Before declaring an experiment completed or rendering a decision verdict:

- [ ] **Gate 1: Checkpoint Hash Recorded:** Full 64-hex SHA-256 calculated for all output checkpoints.
- [ ] **Gate 2: Raw Predictions Preserved:** Complete task-by-task model predictions saved to disk.
- [ ] **Gate 3: Baseline Parity Confirmed:** Comparison delta calculated against verified parent model baseline under matched decoding settings.
- [ ] **Gate 4: Multi-Dimensional Evaluation:** Results report on generation stability, reasoning, and task accuracy rather than loss alone.
- [ ] **Gate 5: Evaluator Integrity Audited:** Stratified spot-check of evaluation predictions confirms zero false-positive substring matches and zero false negatives.
- [ ] **Gate 6: Non-Inferiority Verified:** No unapproved regressions on critical canonical baselines.
- [ ] **Gate 7: Statistical Significance Reported:** P-values, confidence intervals, and per-seed variance disclosed without selective omission.
- [ ] **Gate 8: Four-Tier Reproducibility Declared:** Separate reporting of dataset, training, checkpoint, and evaluation reproducibility tiers.
- [ ] **Gate 9: Failure Analysis Documented:** Qualitative review of failure modes and error patterns completed.
- [ ] **Gate 10: Decision Justified:** Final decision (`PROMOTE`, `CONTINUE`, `REJECT`, `INCONCLUSIVE`) strictly follows Section 10 criteria.
- [ ] **Gate 11: Git Tree Clean:** Verified that no untracked binaries or unintended file edits occurred.

---

## 16. Relationship to Future ChakrView Milestones & EXP36 Governance Caution

This protocol strictly governs all future experimental proposals in the ChakrView roadmap.

### Note Regarding Candidate Proposals (e.g. Exp36 AIP)
Following the forensic findings of Milestone M5.1 regarding prompt conditioning and attractor dynamics, ideas for subsequent research—such as an Attractor-Inhibited Preconditioning (AIP) experiment (conceptually designated Exp36)—have been explored in discussion notes and paper-based simulations.

> [!CAUTION]
> **EXP36 IS STRICTLY UNAPPROVED AND UNEXECUTED**  
> Neither this document nor any previous compliance audit grants approval to implement, train, or execute Exp36. Any paper-based simulation conducted during governance stress-testing is a structural template simulation only and does not establish verified scientific facts.

### Specific Governance Requirements for any Future Exp36 Proposal:
If a research engineer seeks to formally propose an Exp36 experiment, the proposal must remediate several critical design considerations:
1. **Parent Lineage Justification:** Earlier M5.1 findings derived from continued pre-training branches (e.g. M3 Step 250). Proposing to branch from the canonical Step 10,000 checkpoint represents a major lineage departure and must be scientifically justified as a distinct research question.
2. **Confounded Intervention (Factorial Design Requirement):** Combining an inhibitor prefix and prompt-loss masking in a single experimental arm creates an uninterpretable confound. A formal Exp36 proposal must execute a $2 \times 2$ factorial design (Control, Masking Only, Prefix Only, Combined) to isolate main effects and interaction.
3. **Baseline Repetition Metric Parity:** Historical evaluations recorded canonical Step 10,000 repetition rate as 0.2540 on earlier test sets. Any proposal citing a baseline must verify the exact evaluator script, task set, and decoding parameters rather than adopting unverified figures.
4. **Preregistration of Justified Thresholds:** Success criteria (e.g. extraction accuracy targets, non-inferiority margins) must be justified by task requirements rather than adopting hypothetical simulation numbers.
5. **Pre-Execution Gate Adherence:** Zero execution may commence until the formal Section 13 proposal passes every gate of Section 14 and receives explicit written GO authorization.
