# ChakrView Experiment Governance Adoption Plan v1.0
## Operationalizing Protocol v1.0 across Research, Training, and Evaluation

**Repository:** `ChakrView`  
**Location:** `docs/experiments/EXPERIMENT_GOVERNANCE_ADOPTION_PLAN_v1.0.md`  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md`  
**Target Milestone:** Post-M5.1 Research Operations  
**Version:** 1.0.0  
**Effective Date:** 2026-10-31  
**Status:** Living Operational Plan  

---

## 1. Overview and Operational Objective

The ChakrView Experiment Governance Protocol v1.0 (`EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md`) establishes non-negotiable scientific and repository invariants for all experimental work. This Adoption Plan defines the practical operational mechanisms, workflows, gatekeepers, and verification tools required to apply Protocol v1.0 to every future experiment conducted within the ChakrView repository.

### Fundamental Scientific Premise
All research within ChakrView operates under the foundational principle:
> [!IMPORTANT]
> **AN EXPERIMENT IS NOT SUCCESSFUL MERELY BECAUSE TRAINING LOSS DECREASES.**  
> Optimization dynamics reflect token-level cross-entropy minimization; they do not establish language intelligence, contextual reasoning, factual extraction, or multi-turn conversational competence.

---

## 2. Stage-by-Stage Governance Implementation

Protocol v1.0 defines a 12-stage scientific lifecycle. The adoption workflow operationalizes these stages into four clear working phases:

```
[ PHASE I: PRE-EXECUTION ]
  1. Proposal ────────► 2. Hypothesis Registration ────────► 3. Dataset Audit ────────► 4. Design & GO/NO-GO Gate
                                                                                               │
[ PHASE II: RUNTIME EXECUTION ]                                                                │
  5. Baseline Freeze ◄─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  6. Bounded Optimization ───► 7. Active Telemetry Monitoring ───► 8. Checkpoint & Asset Capture
                                                                           │
[ PHASE III: POST-EXECUTION ]                                             │
  9. Isolated Evaluation ◄─────────────────────────────────────────────────┘
         │
         ▼
  10. Statistical & Qualitative Audit ───► 11. Evaluator Forensics ───► 12. Terminal Decision Verdict
```

---

## 3. Detailed Governance Procedures

### 3.1 Pre-Execution Registration & Run Mode Governance
Every prospective experiment must be formally registered in `docs/experiments/` using `EXPERIMENT_REGISTRATION_TEMPLATE_v1.0.md` before any script is executed.

- **Mandatory Registration Attributes:**
  - `Experiment ID`: Unique canonical identifier (e.g. `EXP36_CONTROLLED_ENTITY_COPYING`).
  - `Run Mode`: Explicit selection of `EXPLORATORY` (pilot probing, single-seed, hypothesis discovery) or `CONFIRMATORY` (rigorous multi-seed hypothesis verification, pre-declared margins).
  - `Research Question`: Specific inquiry addressing a demonstrated limitation from M3–M5.1.
  - `Primary ($H_1$) and Null ($H_0$) Hypotheses`: Clear mechanistic assertion contrasted against a falsifiable null condition.
  - `Lineage Anchors`: Parent checkpoint path and verified full 64-hex SHA-256 digest.
  - `Dataset Integrity`: Training dataset relative path, manifest SHA-256, and governance status in `docs/data/DATASET_GOVERNANCE_INDEX.md`.
  - `Evaluator Identity`: Script path, Git version, SHA-256 digest, and word-boundary scoring rule.
  - `Pre-Registered Success and Rejection Criteria`: Quantitative thresholds locked prior to training.
  - `Resource Budget`: Hard allocations for CPU threads, RAM ceiling, disk footprint, and step bounds.

### 3.2 Formal Pre-Execution GO / NO-GO Gating
Execution is strictly prohibited until the 11 gates of the Pre-Execution Checklist are verified by an independent reviewer or project maintainer:
- A missing parent checkpoint hash, unverified dataset lineage, missing evaluator identity, unanchored substring matching defect, or unbudgeted resource allocation constitutes a **mandatory NO-GO**.
- **Run Mode Enforcement:** An `EXPLORATORY` run cannot be used to justify model promotion. A `CONFIRMATORY` run cannot proceed without at least three pre-registered independent seeds (`seeds: [42, 137, 2026]`).
- **Written GO Record:** The experiment document must feature a signed approval block documenting reviewer name, date, approved run mode, and verified asset hashes.

### 3.3 Runtime Execution Discipline & Telemetry
During training and optimization, runners must maintain continuous telemetry:
- **Seed Parity:** In confirmatory runs, all comparison arms must execute the identical pre-registered seed list. Cherry-picking or selective exclusion of poor-performing seeds is prohibited.
- **Resource Boundary Compliance:** Memory usage must remain bounded below the registered RAM ceiling. CPU core allocation must respect `torch.set_num_threads`.
- **Automated Hard-Stop Triggers:** Processes must terminate immediately upon encountering:
  1. Loss value equal to `NaN` or `Inf`.
  2. Gradient norm explosion exceeding clipping thresholds for 5 consecutive steps.
  3. Disk free space dropping below safety thresholds ($< 5.0$ GB).
  4. Repetitive mode collapse (e.g. continuous single-token loops on generation probes).
  5. Any unintended modification of tracked repository files or canonical artifacts.
- *Preservation Rule:* Aborted or failed runs must remain documented with failure forensics; they must never be deleted to mask negative outcomes.

### 3.4 Post-Execution Evaluation, Statistics & Decision
Post-training evaluation must enforce strict scientific and repository safeguards:
- **Baseline Parity:** Candidate models must be evaluated against the exact parent checkpoint evaluated under identical decoding settings ($T=0.0$, greedy decoding).
- **Evaluator Integrity Audit:** Automated benchmark scores must undergo a stratified spot-check (minimum 20 cases) confirming zero false-positive substring matches and zero answer leakage. Small-sample uncertainty must be explicitly acknowledged.
- **Canonical Non-Inferiority Testing:** Regressions on critical core capabilities (e.g. Python syntax validity, EOS stopping) have zero tolerance ($\Delta \ge 0.0\%$). Regressions on secondary benchmarks must not exceed pre-registered, justified margins. No aggregate metric may conceal a severe capability failure.
- **Terminal Decision Framework:** Every experiment must be classified into exactly one of four canonical verdicts:
  - `PROMOTE`: Validated confirmatory hypothesis, passed all regression and integrity gates. Designated as a milestone candidate (does not grant public release).
  - `CONTINUE`: Promising evidence with residual confounding factors or hyperparameter sensitivity requiring follow-up.
  - `REJECT`: Falsified hypothesis, computational infeasibility, or unacceptable capability regression. Permanently archived.
  - `INCONCLUSIVE`: Compromised by evaluator defect, data contamination, or hardware disruption. Invalidated and archived.

---

## 4. Operational Governance Adoption Checklist

This checklist must be executed sequentially for every experimental run in ChakrView:

### Phase I: Pre-Execution Approval (GO / NO-GO)
- [ ] **1. Template Initialized:** Proposal initialized using `EXPERIMENT_REGISTRATION_TEMPLATE_v1.0.md`.
- [ ] **2. Run Mode Declared:** Marked explicitly as `EXPLORATORY` or `CONFIRMATORY`.
- [ ] **3. Falsifiable Hypotheses Locked:** $H_1$, $H_0$, and quantitative success criteria defined.
- [ ] **4. Parent Checkpoint Verified:** Path and 64-hex SHA-256 match recorded lineage (Canonical Step 10k: `96357705a683482e...`).
- [ ] **5. Dataset Governance Verified:** Training dataset registered in `docs/data/DATASET_GOVERNANCE_INDEX.md` with verified SHA-256.
- [ ] **6. Split Isolation Confirmed:** Zero overlap between train, validation, and independent benchmark splits.
- [ ] **7. Evaluator Locked:** Script path, Git commit, SHA-256, and boundary matching rule verified.
- [ ] **8. Factorial Arms Formulated:** Multi-variable interventions structured as $2 \times 2$ factorial designs.
- [ ] **9. Resource Quotas Allocated:** CPU thread count, RAM ceiling, and step caps defined.
- [ ] **10. Git Baseline Clean:** Working tree clean, `.gitignore` hash verified, `experiments/overnight_runs/` isolated.
- [ ] **11. Written GO Authorized:** Maintainer sign-off recorded.

### Phase II: Execution & Checkpoint Capture
- [ ] **12. Telemetry Active:** Loss, gradient norm, and memory footprint logged per step.
- [ ] **13. Checkpoint Persisted Separately:** Stored in isolated experiment directory; full SHA-256 calculated immediately.
- [ ] **14. Hardware Abort Inactive:** No NaN, Inf, gradient explosion, or resource breaches occurred.

### Phase III: Evaluation & Decision Verdict
- [ ] **15. Matched Decoding Applied:** Greedy decoding ($T=0.0$) used across parent baseline and candidate arms.
- [ ] **16. Raw Predictions Preserved:** Full task-level outputs saved to JSON.
- [ ] **17. Evaluator Spot-Check Passed:** Stratified audit confirms zero substring false positives.
- [ ] **18. Non-Inferiority Verified:** No unapproved regression on canonical capabilities.
- [ ] **19. Multi-Seed Statistics Reported:** Mean, standard error, and significance tests reported without cherry-picking.
- [ ] **20. Four-Tier Reproducibility Declared:** Separate reporting of dataset, training, checkpoint, and evaluation reproducibility.
- [ ] **21. Terminal Verdict Rendered:** Assigned `PROMOTE`, `CONTINUE`, `REJECT`, or `INCONCLUSIVE`.
- [ ] **22. Git Tree Clean:** Zero staged changes, no binary check-ins, zero remote pushes.

