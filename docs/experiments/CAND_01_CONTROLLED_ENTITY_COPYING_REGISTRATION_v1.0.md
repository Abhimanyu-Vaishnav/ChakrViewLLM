# CHAKRVIEW — EXPERIMENT REGISTRATION: CAND_01_CONTROLLED_ENTITY_COPYING
## Controlled Entity-Copying & Associative Binding Registration Document

**Document ID:** `REG-EXP-CAND01-001`  
**Location:** `docs/experiments/CAND_01_CONTROLLED_ENTITY_COPYING_REGISTRATION_v1.0.md`  
**Experiment ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Research Track:** Track 1 (Entity Association and Copying)  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md` Section 13  
**Research Owner:** ChakrView Core ML Research Team  
**Run Mode:** `CONFIRMATORY`  
**Status:** `READY_FOR_REVIEW` (Pre-Execution Registration — Execution Prohibited)  
**Date:** 2026-10-31  

---

### 1. Scientific Formulation

- **Research Question:**
  Can the baseline ChakrMicro model checkpoint learn to reliably retrieve and copy an arbitrary value token associated with a given key token from context into its completion when natural language conversational template noise is eliminated and loss is masked to 100% on the target value?
- **Primary Hypothesis ($H_1$):**
  Training on minimal-syntax key-value pairs (`PAIR: [KEY]=[VAL]. QUERY: [KEY] -> [VAL]`) with prompt and template loss masked will produce $\ge 75.0\%$ exact-match copy accuracy on held-out unseen tokens, demonstrating that the 6-layer architecture can form functional induction circuits.
- **Null Hypothesis ($H_0$):**
  Model copy accuracy on held-out unseen tokens remains $< 5.0\%$, indicating an architectural incapacity to route attention for associative token retrieval.
- **Mechanistic Rationale:**
  In standard SFT (Exp35), 78.96% of gradients updated syntactic boilerplate tokens, starving key-value routing heads of gradient signals. Masking prompt and template tokens directs 100% of gradient updates through the causal attention path connecting query tokens to context key-value representations.
- **Independent Variable(s):**
  1. Context Framing (Minimal Key-Value Syntax vs. Conversational Natural Language).
  2. Loss Masking Distribution (Target-Only Masked vs. Full Sequence Unmasked).
- **Dependent Variable(s):**
  1. Exact-Match Copy Accuracy on held-out test split ($A_{\text{EM}}$).
  2. Mean target token probability rank ($R_{\text{tgt}}$) in logit distribution.
  3. Prompt probability sensitivity delta ($\Delta p$).
  4. Error category breakdown (Distractor Intrusion, Key Echo, Syntax Echo, Repetition).
- **Controlled Variables:**
  - Base Model Architecture: ChakrMicro (3,443,136 parameters, 6L / 192D / 6H, RoPE, tied embeddings).
  - Base Checkpoint: Step 10,000 canonical pretraining checkpoint.
  - Vocabulary: `data/experiments/vocab_4096` (4,093 active tokens).
  - Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.98, \text{wd}=0.01$).
  - Learning Rate: Peak $1.0 \times 10^{-4}$ with 50-step linear warmup and cosine decay to $1.0 \times 10^{-5}$.
  - Optimization Budget: Exactly 500 steps, batch size 8 (micro-batch 2, accum 4).
  - Random Seeds: Data generation seed `42`, Training seed `42`.
- **Potential Confounders:**
  1. Token frequency bias in base vocabulary.
  2. Subword fragmentation (mitigated by single-token entity guarantee).
  3. Ordinal shortcut exploitation (mitigated by randomizing pair positions in context).
  4. Unanchored substring evaluator false positives (mitigated by exact first-token matching).
- **Pre-Registered Success Criteria:**
  - Primary: Held-out test exact-match accuracy $\ge 75.0\%$.
  - Secondary: Distractor resistance (3 distractors) $\ge 60.0\%$.
  - Secondary: Prompt sensitivity delta $\Delta p \ge 0.20$.
  - Secondary: Target token logit rank $\le 5$ out of 4,096.
- **Rejection Criteria:**
  - Test exact-match accuracy $< 25.0\%$.
  - Prompt sensitivity delta $\Delta p < 0.05$.
  - Average target token logit rank $> 50$.

---

### 2. Lineage and Configuration

- **Parent Checkpoint Relative Path:**
  `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Parent Checkpoint SHA-256:**
  `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Checkpoint Source Tag / Commit:**
  Tag: `v0.8.0`, Commit: `ae96c645eb08c438186ea4bdfd6b1d6e1591f249`
- **Architectural Configuration:**
  - Layers: 6
  - Hidden Dimension: 192
  - Attention Heads: 6 (Head Dimension: 32)
  - Vocabulary Size: 4,096 (Active: 4,093)
  - Positional Embeddings: Rotary Position Embeddings (RoPE)
  - Weight Tying: Input embedding and output projection matrices are strictly tied.
- **Dataset Specification Reference:**
  `docs/experiments/CAND_01_DATASET_SPECIFICATION_v1.0.md`
- **Dataset ID (Prospective):** `DS-CAND01-SYNTH-V1`
- **Dataset Governance Status:** `SPECIFICATION_ONLY` (To be verified upon generation)
- **Dataset Reproducibility Tier:** `GENERATION_DETERMINISTIC` (Single seed 42)
- **Training Budget:**
  - Micro-batch size: 2
  - Gradient accumulation steps: 4
  - Effective batch size: 8
  - Sequence length: 512 tokens
  - Total steps: 500
  - Total token budget: $500 \times 8 \times 512 = 2,048,000$ tokens
  - Device: CPU (Deterministic float32)

---

### 3. Factorial Experimental Arms

To ensure complete scientific attribution, the run allocates compute across four arms:

1. **Arm A (Intervention):** Minimal Key-Value Syntax + Target-Only Loss Masking.
2. **Arm B (Masking Ablation):** Minimal Key-Value Syntax + Unmasked Autoregressive Loss.
3. **Arm C (Framing Ablation):** Natural Language Conversational Framing + Target-Only Loss Masking.
4. **Arm D (Historical Control):** Natural Language Conversational Framing + Unmasked Loss (Exp35 replica).

---

### 4. Gating and Approval Record

- **Proposal Status:** `READY_FOR_REVIEW`
- **Pre-Execution Check Completed:**
  - Git working directory verified clean: [x]
  - No untracked code modifications: [x]
  - Parent checkpoint SHA-256 verified: [x]
  - Core architecture files verified: [x]
  - Evaluator specification completed: [x]
  - Safety constraints acknowledged: [x]
- **Execution Approval:** **`UNAPPROVED`**
  *(In accordance with Experiment Governance Protocol v1.0 Section 14, execution is strictly prohibited until a formal Pre-Execution GO Record is reviewed and signed by the research lead.)*
