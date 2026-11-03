# CHAKRVIEW — EXPERIMENT REGISTRATION: CAND_01_CONTROLLED_ENTITY_COPYING v1.2 (REVIEWED)
## Controlled Entity-Copying & Associative Binding Registration Document

**Document ID:** `REG-EXP-CAND01-003`  
**Location:** `docs/experiments/CAND_01_CONTROLLED_ENTITY_COPYING_REGISTRATION_v1.2_REVIEWED.md`  
**Supersedes:** `docs/experiments/CAND_01_CONTROLLED_ENTITY_COPYING_REGISTRATION_v1.1_REVIEWED.md`, `docs/experiments/CAND_01_CONTROLLED_ENTITY_COPYING_REGISTRATION_v1.0.md`  
**Experiment ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Research Track:** Track 1 (Entity Association and Copying)  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md` Section 13  
**Research Owner:** ChakrView Core ML Research Team  
**Run Mode:** `CONFIRMATORY`  
**Status:** `READY_FOR_PRE_EXECUTION_REVIEW` (Pre-Execution Registration — Execution Strictly Prohibited)  
**Execution Authorization:** **`UNAPPROVED`**  
**Date:** 2026-10-31  

---

### 1. Scientific Formulation

- **Research Question:**
  Can the baseline ChakrMicro model checkpoint learn to reliably retrieve and copy an associated value token when queried with its corresponding key token from context into its completion, under controlled synthetic conditions with minimal template syntax and target-focused loss masking?
- **Primary Hypothesis ($H_1$):**
  Training on minimal-syntax key-value pairs (`PAIR: [KEY]=[VAL]. QUERY: [KEY] -> [VAL]`) with prompt tokens masked will produce a multi-seed mean first-token exact-match accuracy $A_{\text{FT}} \ge 75.0\%$ [PROVISIONAL] and full completion match $A_{\text{FC}} \ge 70.0\%$ [PROVISIONAL] across three independent random seeds (`seeds: [42, 137, 2026]`) on held-out test splits, demonstrating controlled associative retrieval behavior.
- **Null Hypothesis ($H_0$):**
  Model copy accuracy on held-out test splits does not exceed empirical control baselines (specifically, key-shuffled permutation baseline and unconditioned token prior baseline, bounded provisionally at $\le 5.0\%$), indicating that the optimization failed to induce functional associative retrieval.
- **Independent Variable(s):**
  1. Context Framing (Minimal Key-Value Syntax vs. Conversational Natural Language).
  2. Loss Masking Distribution (Target-Only Masked vs. Full Sequence Unmasked).
- **Dependent Variable(s):**
  1. First-Token Exact-Match Accuracy on Partition C Recombination split ($A_{\text{FT,recomb}}$).
  2. First-Token Exact-Match Accuracy on Partition D Disjoint OOD split ($A_{\text{FT,ood}}$).
  3. Full Completion Match with clean EOS termination ($A_{\text{FC}}$).
  4. Prompt probability sensitivity delta ($\Delta p$).
  5. Teacher-forced target token rank ($R_{\text{tgt}}$) as diagnostic telemetry.
  6. Granular error category breakdown (`ERR-01` through `ERR-06`).
- **Controlled Variables:**
  - Base Model Architecture: ChakrMicro (3,443,136 parameters, 6L / 192D / 6H, RoPE, tied embeddings).
  - Base Checkpoint: Step 10,000 canonical pretraining checkpoint.
  - Vocabulary: `data/experiments/vocab_4096` (4,093 active tokens).
  - Seed List: Exactly three independent seeds `[42, 137, 2026]` evaluated for all arms.
  - Evaluation Decoding: Deterministic greedy decoding ($T=0.0$, `max_new_tokens=16`).
- **Potential Confounders & Mitigations:**
  1. *Token Frequency Prior Bias:* Mitigated via log-frequency tier stratification across entity pools.
  2. *Subword Tokenizer Fragmentation:* Mitigated via strict single-token entity guarantee.
  3. *Ordinal Position Bias:* Mitigated by uniformly distributing target pair position across context slots.
  4. *Evaluator Substring Inflation:* Mitigated by strict token-ID equality checking and sequence length gating.
  5. *Cluster Dependence across Prompts:* Mitigated via clustered standard errors grouped by target token ID.
- **Pre-Registered Success Criteria (Confirmatory):**
  - Primary Gate 1: Multi-seed mean $A_{\text{FT}} \ge 75.0\%$ on both Partition C and Partition D.
  - Primary Gate 2: Multi-seed mean $A_{\text{FC}} \ge 70.0\%$ on both Partition C and Partition D.
  - Secondary Gate: Multi-seed mean distractor resistance (TF-04, 3 pairs) $\ge 60.0\%$.
  - Secondary Gate: Multi-seed mean prompt sensitivity delta $\Delta p \ge 0.20$.
- **Rejection Criteria:**
  - Multi-seed mean $A_{\text{FT}} < 25.0\%$ or $\Delta p < 0.05$.

---

### 2. Lineage and Configuration Provenance

| Configuration Field | Proposed Setting | Provenance Classification | Pre-Flight Validation Mandate |
|---|---|---|---|
| **Parent Checkpoint Path** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `VERIFIED` | SHA-256 verified bit-exact: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **Model Architecture** | ChakrMicro 6L / 192D / 6H / RoPE / Tied Embeddings | `VERIFIED` | Matches `chakrview/brain/model.py` and `ModelConfig` |
| **Dataset ID** | `DS-CAND01-SYNTH-V1` | `PROPOSED` | Requires generation, SHA-256 recording, and index registration |
| **Dataset File Paths** | Partitions A, B, C, D under `experiments/cand01_controlled_entity_copying/data/` | `PROPOSED` | Must verify 4-partition schema validation and split disjointness |
| **Dataset Governance** | `INTERNAL_ONLY` / `SPECIFICATION_ONLY` | `PROPOSED` | Restricted to internal research; no public release |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}, \text{wd}=0.01$) | `HISTORICAL` | Pretraining baseline AdamW configuration |
| **Learning Rate** | Peak $1.0 \times 10^{-4}$, linear warmup 50 steps, cosine decay to $1.0 \times 10^{-5}$ | `PROVISIONAL` | Requires 20-step pre-flight gradient telemetry check |
| **Micro-batch / Accum** | Micro-batch 2, Accumulation 4 (Effective batch 8) | `PROVISIONAL` | Balances CPU memory throughput and gradient stability |
| **Sequence Length** | $T_{\text{max}} = 512$ tokens | `VERIFIED` | Matches `collator.py` and `ModelConfig.max_context` |
| **Step Budget** | Exactly 500 steps ($2.048 \times 10^6$ tokens) | `PROVISIONAL` | Bounded compute budget for diagnostic run |
| **Device & Precision** | Single-node CPU, deterministic float32 | `VERIFIED` | Matches project CPU-first mandate |

---

### 3. Factorial Experimental Arms & Seed Controls

All four arms will be executed sequentially using identical seeds `seeds: [42, 137, 2026]`:

| Arm ID | Description | Context Framing | Loss Masking Policy | Entity Pool Matching | Seeds |
|---|---|---|---|---|---|
| **Arm A (Intervention)** | Core Treatment | Minimal Syntax (`K=V`) | Target Only (`mask_prompt=True`) | Single-token pool $\mathcal{E}$ | `[42, 137, 2026]` |
| **Arm B (Masking Control)** | Masking Ablation | Minimal Syntax (`K=V`) | Full Sequence Unmasked | Single-token pool $\mathcal{E}$ | `[42, 137, 2026]` |
| **Arm C (Framing Control)** | Framing Ablation | Natural Language | Target Only (`mask_prompt=True`) | Matched single-token pool $\mathcal{E}$ | `[42, 137, 2026]` |
| **Arm D (Historical Reference)** | Historical Baseline | Natural Language | Full Sequence Unmasked | Multi-token subwords (Exp35 replica) | `[42, 137, 2026]` |

---

### 4. Evaluator Identity & Validation Suite

- **Evaluator Script (Prospective):** `scripts/evaluate_cand01_benchmark.py`
- **Evaluator Unit Test (Prospective):** `tests/test_cand01_evaluator.py`
- **Benchmark Partitions (Prospective):**
  - Partition C (Test Recombination): `cand01_benchmark_test_recomb_500.json`
  - Partition D (Test Disjoint OOD): `cand01_benchmark_test_ood_500.json`
- **Scoring Directives:** Coupled first-token exact match ($A_{\text{FT}}$) and full completion match ($A_{\text{FC}}$).
- **Negative Controls:** Empty context, wrong-key query, and shuffled-mapping permutations.
- **Statistical Aggregation:** Clustered bootstrap confidence intervals grouped by `target_val_token_id`.

---

### 5. Canonical Non-Inferiority & Regression Gates

In compliance with Protocol v1.0 Section 6, the candidate checkpoints must not induce catastrophic regressions on general baseline capabilities:

| Capability Dimension | Benchmark Suite | Baseline (Parent Checkpoint) | Non-Inferiority Margin | Allowed Degradation |
|---|---|---|---|---|
| **Language Modeling Perplexity** | Canonical Pretraining Validation Stream | Step 10,000 Validation Loss | $\Delta \text{loss} \le +0.15$ | Strict tolerance |
| **Repetition Stability** | Synthetic Repetition Benchmark | Parent Baseline Repetition Rate | $\Delta \text{rep} \le +2.0\%$ | Justified margin |
| **Coding Adherence** | Canonical Coding Suite | Parent Baseline Syntax Score | $\Delta \ge 0.0\%$ | Zero tolerance |

---

### 6. Post-Execution Decision Taxonomy (Protocol v1.0 Section 10)

Post-run analysis maps strictly to the four protocol states:
- **`PROMOTE`:** Both $A_{\text{FT}} \ge 75.0\%$ and $A_{\text{FC}} \ge 70.0\%$ met across 3 seeds on both Partitions C and D; zero canonical regressions.
- **`CONTINUE`:** Partial capability demonstrated ($25.0\% \le A_{\text{FT}} < 75.0\%$); formulate follow-up specification.
- **`REJECT`:** Primary hypothesis falsified ($A_{\text{FT}} < 25.0\%$); archive failure forensics and document architectural limit.
- **`INCONCLUSIVE`:** Run compromised by hardware disruption, NaN divergence, or evaluator defect.

---

### 7. Pre-Execution Approval Checklist & Governance Posture

- **Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`
- **Current Candidate Status:** `READY_FOR_PRE_EXECUTION_REVIEW`
- **Execution Authorization Status:** **`UNAPPROVED`**

| Pre-Execution Gate | Verification State | Verification Evidence / Condition |
|---|---|---|
| Git working tree clean | **VERIFIED** | Only documentation files created; zero staged changes |
| Protected files unaltered | **VERIFIED** | Hashes for `model.py`, `session.py`, checkpoint match canonical values |
| Canonical checkpoint verified | **VERIFIED** | SHA-256 `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| Dataset generator implemented | **PENDING** | Requires `scripts/generate_cand01_dataset.py` (Gap 01) |
| 4-partition dataset verified | **PENDING** | Requires manifest hash recording and split disjointness verification |
| Evaluator implemented & tested | **PENDING** | Requires `scripts/evaluate_cand01_benchmark.py` & unit tests (Gap 02 & 03) |
| Non-interactive runner created | **PENDING** | Requires `scripts/run_cand01_experiment.py` (Gap 04) |
| Pre-flight calibration completed | **PENDING** | Requires 20-step gradient telemetry check on disposable model copy |
| Research Owner Sign-off | **PENDING** | Formal Pre-Execution GO Record signature required |

*(In accordance with Experiment Governance Protocol v1.0 Section 14, execution is strictly prohibited until all pending gates are cleared and signed.)*

