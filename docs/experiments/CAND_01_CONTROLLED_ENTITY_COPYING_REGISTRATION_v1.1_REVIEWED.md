# CHAKRVIEW — EXPERIMENT REGISTRATION: CAND_01_CONTROLLED_ENTITY_COPYING v1.1 (REVIEWED)
## Controlled Entity-Copying & Associative Binding Registration Document

**Document ID:** `REG-EXP-CAND01-002`  
**Location:** `docs/experiments/CAND_01_CONTROLLED_ENTITY_COPYING_REGISTRATION_v1.1_REVIEWED.md`  
**Supersedes:** `docs/experiments/CAND_01_CONTROLLED_ENTITY_COPYING_REGISTRATION_v1.0.md`  
**Experiment ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Research Track:** Track 1 (Entity Association and Copying)  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md` Section 13  
**Research Owner:** ChakrView Core ML Research Team  
**Run Mode:** `CONFIRMATORY`  
**Status:** `READY_FOR_REVIEW` (Pre-Execution Registration — Execution Strictly Prohibited)  
**Execution Authorization:** **`UNAPPROVED`**  
**Date:** 2026-10-31  

---

### 1. Scientific Formulation

- **Research Question:**
  Can the baseline ChakrMicro model checkpoint learn to reliably retrieve and copy an arbitrary value token associated with a given key token from context into its completion when natural language conversational template noise is eliminated and loss is masked to 100% on the target value?
- **Primary Hypothesis ($H_1$):**
  Training on minimal-syntax key-value pairs (`PAIR: [KEY]=[VAL]. QUERY: [KEY] -> [VAL]`) with prompt and template tokens masked will produce $\ge 75.0\%$ exact-match copy accuracy on held-out test splits across three independent random seeds (`seeds: [42, 137, 2026]`), demonstrating that the model can learn associative token retrieval behavior under low-noise conditions.
- **Null Hypothesis ($H_0$):**
  Model copy accuracy on held-out test splits does not exceed empirical control baselines (specifically, key-shuffled permutation baseline and unconditioned token prior baseline, bounded provisionally at $\le 5.0\%$), indicating that the optimization failed to induce functional associative retrieval.
- **Mechanistic Rationale:**
  In standard SFT (Exp35), 78.96% of gradients updated syntactic boilerplate tokens, starving key-value routing heads of gradient signals. Masking prompt and template tokens directs 100% of gradient updates through the causal attention path connecting query tokens to context key-value representations.
- **Independent Variable(s):**
  1. Context Framing (Minimal Key-Value Syntax vs. Conversational Natural Language).
  2. Loss Masking Distribution (Target-Only Masked vs. Full Sequence Unmasked).
- **Dependent Variable(s):**
  1. First-Token Exact-Match Accuracy on held-out test split ($A_{\text{FT}}$).
  2. Full Completion Match with clean EOS termination ($A_{\text{FC}}$).
  3. Mean target token probability rank ($R_{\text{tgt}}$) in pre-softmax logit distribution.
  4. Prompt probability sensitivity delta ($\Delta p$).
  5. Error category breakdown (Distractor Intrusion, Key Echo, Syntax Echo, Repetition).
- **Controlled Variables:**
  - Base Model Architecture: ChakrMicro (3,443,136 parameters, 6L / 192D / 6H, RoPE, tied embeddings).
  - Base Checkpoint: Step 10,000 canonical pretraining checkpoint.
  - Vocabulary: `data/experiments/vocab_4096` (4,093 active tokens).
  - Seed List: Exactly three independent seeds `[42, 137, 2026]` evaluated for all arms.
  - Evaluation Decoding: Deterministic greedy decoding ($T=0.0$).
- **Potential Confounders:**
  1. Token frequency bias in base vocabulary (controlled via frequency-band stratification).
  2. Subword fragmentation (mitigated via single-token entity guarantee).
  3. Ordinal shortcut exploitation (mitigated by randomizing pair positions in context).
  4. Unanchored substring evaluator false positives (mitigated via exact token ID matching).
- **Pre-Registered Success Criteria (Confirmatory):**
  - Primary: Multi-seed mean test first-token exact-match accuracy $\ge 75.0\%$.
  - Secondary: Multi-seed mean distractor resistance (3 distractors) $\ge 60.0\%$.
  - Secondary: Multi-seed mean prompt sensitivity delta $\Delta p \ge 0.20$.
  - Secondary: Multi-seed mean target token logit rank $\le 5$ out of 4,096.
- **Rejection Criteria:**
  - Multi-seed mean test exact-match accuracy $< 25.0\%$.
  - Multi-seed mean prompt sensitivity delta $\Delta p < 0.05$.
  - Multi-seed mean target token logit rank $> 50$.

---

### 2. Lineage and Configuration Provenance

| Configuration Field | Proposed Setting | Provenance Status | Verification Required Prior to Execution |
|---|---|---|---|
| **Parent Checkpoint Path** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | **VERIFIED FROM IMPLEMENTATION** | SHA-256 verified: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **Model Architecture** | ChakrMicro 6L / 192D / 6H / RoPE / Tied Embeddings | **VERIFIED FROM IMPLEMENTATION** | Matches `chakrview/brain/model.py` and `ModelConfig` |
| **Dataset ID** | `DS-CAND01-SYNTH-V1` | **PROPOSED** | Requires generation, SHA-256 recording, and index registration |
| **Dataset File Path** | `experiments/cand01_controlled_entity_copying/data/cand01_train.jsonl` | **PROPOSED** | Must verify file existence and schema validation |
| **Dataset Governance** | `INTERNAL_ONLY` / `SPECIFICATION_ONLY` | **PROPOSED** | Clear internal research classification; no public release |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}, \text{wd}=0.01$) | **HISTORICALLY OBSERVED** | Standard pretraining AdamW settings from Phase 7 |
| **Learning Rate** | Peak $1.0 \times 10^{-4}$, linear warmup 50 steps, cosine decay to $1.0 \times 10^{-5}$ | **PROPOSED** | Conservative rate for 500-step SFT; requires pilot validation |
| **Micro-batch / Accum** | Micro-batch 2, Accumulation 4 (Effective batch 8) | **PROPOSED** | Balances CPU memory throughput and gradient stability |
| **Sequence Length** | $T_{\text{max}} = 512$ tokens | **VERIFIED FROM IMPLEMENTATION** | Matches `collator.py` and `ModelConfig.max_context` |
| **Step Budget** | Exactly 500 steps ($2.048 \times 10^6$ tokens) | **PROPOSED** | Bounded compute budget for diagnostic run |
| **Device & Precision** | Single-node CPU, deterministic float32 | **VERIFIED FROM IMPLEMENTATION** | Matches project CPU-first mandate |

---

### 3. Factorial Experimental Arms & Seed Controls

All four arms will be executed sequentially using identical seeds `seeds: [42, 137, 2026]`:

| Arm ID | Description | Context Framing | Loss Masking Policy | Seeds |
|---|---|---|---|---|
| **Arm A (Intervention)** | Core Treatment | Minimal Syntax (`K=V`) | Target Only (`mask_prompt=True`) | `[42, 137, 2026]` |
| **Arm B (Masking Control)** | Supervision Ablation | Minimal Syntax (`K=V`) | Unmasked Full Sequence | `[42, 137, 2026]` |
| **Arm C (Framing Control)** | Framing Ablation | Natural Language | Target Only (`mask_prompt=True`) | `[42, 137, 2026]` |
| **Arm D (Historical Reference)** | Historical Reference | Natural Language | Unmasked Full Sequence | `[42, 137, 2026]` |

---

### 4. Evaluator Identity & Validation Suite

- **Evaluator Script (Prospective):** `scripts/evaluate_cand01_benchmark.py`
- **Evaluator SHA-256:** To be locked upon script completion.
- **Benchmark Dataset (Prospective):** `cand01_test_benchmark_500.json`
- **Benchmark SHA-256:** To be locked upon generation completion.
- **Scoring Logic:** First-token exact token ID match ($A_{\text{FT}}$) and full completion match ($A_{\text{FC}}$).
- **Negative Controls:** Empty context, wrong-key query, and shuffled-mapping permutations.
- **Unit-Test Requirement:** Must pass `tests/test_cand01_evaluator.py` before execution.

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
- **`PROMOTE`:** All success criteria met across 3 seeds; zero canonical regressions. Advances to Track 2.
- **`CONTINUE`:** Partial capability demonstrated ($25.0\% \le A_{\text{FT}} < 75.0\%$); formulate follow-up specification.
- **`REJECT`:** Primary hypothesis falsified ($A_{\text{FT}} < 25.0\%$); archive failure forensics and document architectural limit.
- **`INCONCLUSIVE`:** Run compromised by hardware disruption, NaN divergence, or evaluator defect.

---

### 7. Pre-Execution Approval Checklist & Governance Posture

- **Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`
- **Current Candidate Status:** `READY_FOR_REVIEW`
- **Execution Authorization Status:** **`UNAPPROVED`**

| Pre-Execution Gate | Verification State | Verification Evidence / Condition |
|---|---|---|
| Git working tree clean | **VERIFIED** | Only documentation files created; no staged changes |
| Protected files unaltered | **VERIFIED** | Hashes for `model.py`, `session.py`, checkpoint match canonical values |
| Canonical checkpoint verified | **VERIFIED** | SHA-256 `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| Dataset generator implemented | **PENDING** | Requires `scripts/generate_cand01_dataset.py` (Gap 01) |
| Dataset manifest & audit passed | **PENDING** | Requires manifest hash recording and leakage test |
| Evaluator implemented & tested | **PENDING** | Requires `scripts/evaluate_cand01_benchmark.py` & unit tests (Gap 02) |
| Non-interactive runner created | **PENDING** | Requires `scripts/run_cand01_experiment.py` (Gap 03) |
| Research Owner Sign-off | **PENDING** | Formal Pre-Execution GO Record signature required |

*(In accordance with Experiment Governance Protocol v1.0 Section 14, execution is strictly prohibited until all pending gates are cleared and signed.)*

