# CHAKRVIEW — CAND_01 SCIENTIFIC DESIGN SPECIFICATION v1.3 (REVIEWED)
## Controlled Entity-Copying & Associative Binding Under Low-Noise Conditions

**Document ID:** `DOC-EXP-CAND01-SCI-004`  
**Location:** `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.3_REVIEWED.md`  
**Supersedes:** `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.2_REVIEWED.md`, `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.1_REVIEWED.md`, `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Research Track:** Track 1 (Entity Association and Copying)  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md` (v1.0.0)  
**Research Priority:** Priority 1 (Diagnostic Gate for Extractive Tasks)  
**Target Architecture:** ChakrMicro Baseline (3,443,136 parameters, 6L / 192D / 6H, RoPE, tied embeddings)  
**Target Parent Checkpoint:** Canonical Step 10,000 (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`, SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)  
**Status:** `READY_FOR_PRE_EXECUTION_REVIEW` (Specification Only — Execution Strictly Unapproved)  
**Execution Authorization:** **`UNAPPROVED`**  
**Date:** 2026-10-31  

---

## 1. Revision History & Audit Scope (v1.3 Targeted Redesign)

This document represents the formal v1.3 scientific redesign of `CAND_01_SCIENTIFIC_DESIGN_v1.2_REVIEWED.md`, resolving findings ADV-01 through ADV-07 from the independent adversarial verification:
- **Correction ADV-01 (TF-01 Cardinality & Recombination Split Scope):** Formally restricted TF-01 (Direct Copy) strictly to Partitions A (Train) and D (Disjoint-Token). Eliminated TF-01 from Partitions B and C (Recombination Splits) because identity copying involves no relational pairing to recombine. Reduced TF-01 training count to 350 examples ($< 450$ available training tokens), mathematically guaranteeing prompt-disjointness ($\text{SHA256}(\text{Prompt}_i) \neq \text{SHA256}(\text{Prompt}_j)$) across all splits.
- **Correction ADV-02 (Frequency Stratification Status):** Replaced fabricated logarithmic cutoffs with an empirical percentile policy. Marked frequency tier cutoffs as `BLOCKED_PENDING_DATA` until histogram extraction on tokenized pretraining data is authorized and executed.
- **Correction ADV-03 & ADV-04 (Statistical Design Realism & GLMM Identifiability):** Acknowledged the inferential limits of 3 seeds ($n=3$, minimum two-sided sign permutation $p = 0.25$). Dropped impossible $p < 0.01$ permutation tests. Replaced with descriptive multi-seed reporting ($\text{mean} \pm \text{range}$, SEM) and two-way clustered bootstrap uncertainty ($B=2,000$) grouped on both `key_token_id` and `target_val_token_id`. Removed the unidentifiable prompt random intercept from the GLMM formula.
- **Correction ADV-05 (Factorial Arm Comparability):** Formally classified the Arm A vs. Arm C contrast as a **composite packaging intervention** (syntax boilerplate + sequence length expansion + RoPE distance shift + softmax entropy dilution), explicitly barring it from being cited as an isolated syntax-framing test.
- **Correction ADV-06 (Gradient Telemetry Contract):** Defined the pre-flight calibration contract explicitly for **unclipped global gradient $L_2$ norm**, extended calibration length to 60 steps to cover the warmup peak (step 50), and labeled all training hyperparameters strictly as `PROVISIONAL`.
- **Correction ADV-07 (OOD Terminology Realism):** Formally replaced the misleading term "out-of-distribution generalization" with **"disjoint-token in-context copying"**, acknowledging that the target token is physically present in the prompt context.

---

## 2. Research Objective & Epistemic Boundaries

### 2.1 Research Objective
The sole objective of CAND_01 is to determine whether the ChakrMicro baseline checkpoint, under controlled synthetic conditions with minimal template syntax and target-focused loss masking, can **behaviorally retrieve and reproduce an associated value token** when queried with its corresponding key token from local context.

### 2.2 Epistemic Boundaries: What This Experiment Does and Does Not Establish

```
+---------------------------------------------------------------------------------------------------+
|                                      EPISTEMIC BOUNDARIES                                         |
+---------------------------------------------------------------------------------------------------+
|  WHAT CAND_01 CAN DEMONSTRATE:                       WHAT CAND_01 CANNOT DEMONSTRATE:             |
|  - In-context associative token retrieval            - General natural language reasoning         |
|  - Direct token copying without relation             - Open-domain factual knowledge extraction   |
|  - Distractor interference decay profile             - Multi-turn conversational grounding        |
|  - Positional degradation under RoPE                 - Proof of "induction circuits"              |
|  - Sensitivity to prompt key swap (Delta p)          - Universal architectural capacity           |
|  - In-context copying of disjoint token IDs          - Semantic lexical concept generalization    |
+---------------------------------------------------------------------------------------------------+
```

- **Behavioral Evidence vs. Mechanistic Proof:**  
  Achieving high accuracy on held-out test splits demonstrates that the model's forward pass successfully emits the bound value token under the tested prompt format. It does **not** prove:
  1. The physical emergence of dedicated two-layer attention induction circuits.
  2. A specific linear subspace binding mechanism in feedforward or attention layers.
  3. That the model possesses semantic understanding of "keys" and "values".
- **Falsification vs. Architectural Incapacity:**  
  If the model achieves accuracy no better than chance controls, it demonstrates that the tested training procedure (AdamW, 500 steps, learning rate schedule, target-only loss) failed to induce associative behavior. It does **not** prove that the 3.44M parameter architecture is fundamentally incapable of associative retrieval under different curricula, objectives, or initializations.
- **Hypothesis Status of Milestone 5 Diagnoses:**  
  The explanation that static template tokens starved entity tokens of useful gradient updates in Milestone 5 remains a **working hypothesis**. CAND_01 tests this hypothesis; it does not treat it as established dogma prior to execution.

---

## 3. Core Research Questions & Formal Hypotheses

### 3.1 Primary Research Question
Under minimal synthetic prompt framing (`PAIR: K=V. QUERY: K -> V`) and 100% target-focused loss masking, can the ChakrMicro baseline checkpoint achieve statistically significant associative retrieval on held-out test splits compared to empirical reference baselines?

### 3.2 Formal Hypotheses
- **Alternative Hypothesis ($H_1$):**  
  When trained on synthetic key-value pairs with minimal syntactic framing and 100% loss applied strictly to target answer tokens, ChakrMicro will achieve a multi-seed mean first-token exact-match accuracy $A_{\text{FT}} \ge 75.0\%$ [PROVISIONAL] and full completion match $A_{\text{FC}} \ge 70.0\%$ [PROVISIONAL] across three independent random seeds (`seeds: [42, 137, 2026]`) on held-out test splits, demonstrating controlled associative retrieval behavior.
- **Null Hypothesis ($H_0$):**  
  ChakrMicro achieves exact-match accuracy on held-out test splits that does not exceed empirical reference baselines (specifically, the key-shuffled permutation baseline and unconditioned token prior baseline, bounded provisionally at $\le 5.0\%$), indicating that the optimization failed to induce functional associative retrieval.

### 3.3 Empirical Reference Baselines (Replacing Arbitrary Chance Thresholds)
1. **Unconditioned Base Model Prior Control ($B_{\text{prior}}$):**  
   Measure the pre-softmax probability and emission rate of the target token when prompt context contains zero key-value definitions (`QUERY: ->`). Measures intrinsic vocabulary prior.
2. **Key-Shuffled Permutation Control ($B_{\text{perm}}$):**  
   Query with a random key not present in context, or shuffle the query key relative to context pairs. Measures whether output emission depends on specific key-value binding or unconditioned context copying.
3. **Frequency-Matched Token Control ($B_{\text{freq}}$):**  
   Evaluate retrieval accuracy across frequency tiers to detect embedding norm bias.

---

## 4. The Eight Core Task Families

| Family ID | Task Name | Canonical Prompt Structure | Expected Output | Core Capability Tested |
|---|---|---|---|---|
| **TF-01** | Direct Token Copy | `COPY: X3M1 ->` | `X3M1` | Direct contextual replication; baseline sanity check (Train & Disjoint OOD only). |
| **TF-02** | Single Key-Value Association | `PAIR: K7P9=X3M1. QUERY: K7P9 ->` | `X3M1` | Single-step key-value relational binding without distractors. |
| **TF-03** | Multi-Pair Association | `PAIRS: K7=X3, M2=L8. QUERY: M2 ->` | `L8` | Selective retrieval among multiple valid bindings in local context. |
| **TF-04** | Distractor Resistance | `PAIRS: D1=N4, K7=X3, D2=R9. QUERY: K7 ->` | `X3` | Resistance to irrelevant context distractors (1, 3, 7 pairs). |
| **TF-05** | Similar-Key Discrimination | `PAIRS: K7P1=X3, K7P2=M8. QUERY: K7P1 ->` | `X3` | Disambiguation when keys differ by Levenshtein distance 1. |
| **TF-06** | Unseen Pair Recombination | `PAIRS: K1=V2, K2=V1. QUERY: K1 ->` | `V2` | Recombination of familiar keys and values in novel pairings. |
| **TF-07** | Context-Position Variation | `PAIRS: K1=V1 [PAD x N] QUERY: K1 ->` | `V1` | Positional robustness across token offsets (10, 50, 100, 200 tokens). |
| **TF-08** | Context-Length Scaling | `[Dense context with N pairs] QUERY: K_tgt ->` | `V_tgt` | Capacity scaling under context saturation (total tokens: 64 to 400). |

---

## 5. Factorial Experimental Arms & Confound Isolation

To evaluate the causal drivers of learning, CAND_01 implements a $2 \times 2$ factorial matrix:
- **Factor 1: Context Framing** (Minimal Key-Value Syntax vs. Natural Language Conversational Framing).
- **Factor 2: Loss Masking** (Target-Only Masked Loss vs. Unmasked Autoregressive Loss).

### 5.1 Formal Factorial Arms

| Arm ID | Context Framing | Loss Masking Policy | Supervision Allocation | Primary Comparison & Classification |
|---|---|---|---|---|
| **Arm A (Core Intervention)** | Minimal Syntax (`K=V`) | Target Only (`mask_prompt=True`) | 100% on target token | Baseline intervention: minimal noise. |
| **Arm B (Masking Control)** | Minimal Syntax (`K=V`) | Full Sequence (`mask_prompt=False`) | ~15% target, 85% syntax | **Arm A vs. B:** Cleanly isolates main effect of loss masking under matched minimal syntax. |
| **Arm C (Framing Control)** | Natural Language | Target Only (`mask_prompt=True`) | 100% on target token | **Arm A vs. C:** Evaluates **composite packaging intervention** (syntax + length + distance + softmax dilution). |
| **Arm D (Historical Reference)** | Natural Language | Full Sequence (Standard SFT unmasked) | ~16% entity, 84% syntax | **Exploratory Reference:** Historical Exp35 replica; conflates syntax, subwords, and unmasked loss. |

> [!IMPORTANT]
> **Scientific Classification of Arm A vs. Arm C (Correction ADV-05):**  
> Conversational natural language framing necessarily expands prompt length ($3\times$ to $5\times$), increases RoPE positional distance between key and query, and dilutes attention softmax distribution across neutral syntax tokens. Therefore, the contrast between Arm A and Arm C **must not be claimed as an isolated test of syntax framing**. It is formally designated as a **composite packaging intervention**.

---

## 6. Dataset Allocation & Partition Cardinality (Resolution ADV-01)

### 6.1 Mathematical Cardinality & Feasibility Proof
To resolve the combinatorial collapse identified in Finding ADV-01:
- Entity Pool: $|\mathcal{E}_{\text{train}}| = 450$ tokens, $|\mathcal{E}_{\text{test}}| = 150$ tokens.
- **TF-01 Re-scoping:** Task Family TF-01 (Direct Copy) is **strictly excluded from Partitions B and C** (Recombination Splits) because identity mapping ($X \to X$) has no relational pair to recombine.
- **TF-01 Allocation:** Partition A (Train) allocates exactly **350 examples** to TF-01. Partition D (Disjoint-Token) allocates **50 examples** to TF-01.
- Total unique tokens required for TF-01: $350 \le 450$ ($\mathcal{E}_{\text{train}}$) and $50 \le 150$ ($\mathcal{E}_{\text{test}}$).
- **Mathematical Invariant:** $\text{SHA256}(\text{Prompt}_i) \neq \text{SHA256}(\text{Prompt}_j)$ is **100% satisfied** across all partitions with zero duplicate leakage.

### 6.2 Partition Overview (Total Volume: 5,500 Records)
1. **Partition A (Train — 4,000 records):**  
   Keys/values from $\mathcal{E}_{\text{train}}$ (450 tokens). TF-01: 350 records; TF-02 through TF-05: balanced 912–913 records/task.
2. **Partition B (Validation Recombination — 500 records):**  
   Keys/values from $\mathcal{E}_{\text{train}}$. TF-02 through TF-08 (~71 records/task). Strictly held-out combinations. Diagnostic monitoring only.
3. **Partition C (Test Recombination — 500 records):**  
   Keys/values from $\mathcal{E}_{\text{train}}$. TF-02 through TF-08 (~71 records/task). Held-out combinations. Yields in-distribution score: $\mathbf{A_{\text{test,recomb}}}$.
4. **Partition D (Test Disjoint-Token In-Context Copying — 500 records):**  
   Keys/values strictly from $\mathcal{E}_{\text{test}}$ (150 tokens). All 8 task families (~62–63 records/task). Yields disjoint-token score: $\mathbf{A_{\text{test,disjoint}}}$.

---

## 7. Statistical Design, Uncertainty & Multi-Seed Policy (Resolutions ADV-03 & ADV-04)

### 7.1 Inferential Limits of Three Seeds ($n=3$)
With $n=3$ independent training runs (`seeds: [42, 137, 2026]`), the minimum possible two-sided sign-permutation p-value is $p = 0.25$. Therefore:
- Formal null-hypothesis significance testing (NHST) at $\alpha = 0.05$ or $\alpha = 0.01$ across training seeds is **mathematically unachievable**.
- Seed-level variation will be reported **descriptively** as individual seed scores, mean, standard error of the mean (SEM), and range ($\min, \max$).
- Claims of confirmatory statistical significance apply strictly within-model across clustered prompt evaluations, not as population-level assertions over all possible training initializations.

### 7.2 Two-Way Clustered Uncertainty Estimation
Because evaluation instances reuse key and value tokens from finite pools, prompts sharing a `key_token_id` or `target_val_token_id` are statistically dependent:
1. **Two-Way Clustered Bootstrap:** Compute 95% confidence intervals using a two-way cluster bootstrap ($B = 2,000$ resamples) resampled simultaneously over unique key token IDs and unique target token IDs.
2. **Mixed-Effects Logistic Regression (GLMM):**
   $$\text{logit}(P(Y_{ij} = 1)) = \beta_0 + \beta_{\text{arm}} + u_{\text{key}(i)} + w_{\text{val}(j)} + \epsilon_{ij}$$
   Where $u_{\text{key}}$ and $w_{\text{val}}$ represent random intercepts for keys and values. The unidentifiable prompt random effect ($v_{\text{prompt}}$) is formally removed.

---

## 8. Hyperparameter Provenance & Calibration Protocol (Resolutions ADV-04 & ADV-06)

| Parameter | Proposed Setting | Provenance Classification | Pre-Flight Validation Mandate |
|---|---|---|---|
| **Base Architecture** | 6L / 192D / 6H / RoPE / Tied Embeddings | `VERIFIED` | Matches `chakrview/brain/model.py` and canonical checkpoint |
| **Sequence Length** | $T_{\text{max}} = 512$ tokens | `VERIFIED` | Matches `collator.py` bounds check |
| **Device & Precision** | Single-node CPU, float32 | `VERIFIED` | Matches project CPU-first mandate |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.98, \text{wd}=0.01$) | `HISTORICAL` | Inherited from Stage C pretraining |
| **Learning Rate** | Peak $1.0 \times 10^{-4}$, linear warmup 50 steps, cosine decay to $1.0 \times 10^{-5}$ | `PROVISIONAL` | Requires 60-step pre-flight calibration |
| **Effective Batch** | Micro-batch 2, Accumulation 4 ($B_{\text{eff}} = 8$) | `PROVISIONAL` | Matches CPU memory limits; requires stability check |
| **Step Budget** | Exactly 500 optimization steps | `PROVISIONAL` | Bounded diagnostic compute budget |

### 8.1 Separately Authorized Pre-Flight Gradient Calibration Protocol
To verify numerical stability under target-only loss before launching confirmatory runs:
- **Duration:** Exactly **60 steps** (ensuring the optimizer traverses the warmup peak at step 50).
- **Metric:** Log the **unclipped global gradient $L_2$ norm** ($\|\mathbf{g}\|_2 = \sqrt{\sum_i \|\nabla_{\theta_i} \mathcal{L}\|_2^2}$) prior to applying clipping.
- **Stability Criterion:** Unclipped $\|\mathbf{g}\|_2 \le 5.0$ on $\ge 80\%$ of post-warmup steps (steps 51–60).
- **Safety Boundary:** Operates exclusively on a disposable scratch checkpoint copy; canonical assets remain untouched.
- *Strict Rule:* This calibration is a training activity. It is **not authorized** by this design document.

---

## 9. Evaluator Metric Gating & Decoding Contract

### 9.1 Two-Token Completion Contract
All evaluations use deterministic greedy decoding (`temperature=0.0`, `top_k=1`, `top_p=1.0`, `max_new_tokens=16`):
- **First-Token Exact Match ($A_{\text{FT}}$):** $\hat{t}_0 == \text{target\_token\_id}$.
- **Full Completion Match with Clean Termination ($A_{\text{FC}}$):**
  $$A_{\text{FC}} = \mathbb{I}\left(\hat{t}_0 == \text{target\_token\_id} \land \hat{t}_1 == \text{eos\_id} \land \text{len}(\hat{\mathbf{t}}) == 2\right)$$
- If the model emits the target token followed by trailing repetition or syntax words, it scores $1.0$ on $A_{\text{FT}}$ but $0.0$ on $A_{\text{FC}}$, triggering `ERR-04`.

---

## 10. Pre-Registered Post-Execution Decision Thresholds (Protocol v1.0 Section 10)

```
                  [ CAND_01 DECISION THRESHOLDS ]
                                 |
    +----------------------------+----------------------------+
    |                            |                            |
[ PROMOTE ]                  [ CONTINUE ]                 [ REJECT ]
• Mean A_FT >= 75.0%         • Mean A_FT: 25.0% - 74.9%   • Mean A_FT < 25.0%
• Mean A_FC >= 70.0%         • Mean A_FC: 20.0% - 69.9%   • Mean Sensitivity < 0.05
• Distractor (3) >= 60.0%    • Distractor (3) >= 20.0%    • Mean Target Rank > 50
• Sensitivity Delta >= 0.20  • Sensitivity: 0.05 - 0.19   • No seed improvement
• Mean Target Rank <= 5      • Rank: 6 - 50 / 4096        • Falsifies tested procedure
• Zero canonical regression  • Formulate follow-up        • Document limitation
```

1. **`PROMOTE`:**  
   Multi-seed mean $A_{\text{FT}} \ge 75.0\%$ and $A_{\text{FC}} \ge 70.0\%$ on both Recombination and Disjoint-Token splits; distractor resistance (3 pairs) $\ge 60.0\%$; zero regression on canonical validation loss ($\Delta \text{loss} \le +0.15$). Validates architecture for Track 2.
2. **`CONTINUE`:**  
   Multi-seed mean $A_{\text{FT}}$ between $25.0\%$ and $74.9\%$. Partial capability noted; requires follow-up hyperparameter/curriculum specification.
3. **`REJECT`:**  
   Multi-seed mean $A_{\text{FT}} < 25.0\%$ or $\Delta p < 0.05$. Tested training procedure falsified; archive forensics.
4. **`INCONCLUSIVE`:**  
   Run compromised by hardware interruption, divergence, or evaluator defect.

---

## 11. Execution Governance & Approval Posture

- **Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`
- **Proposal Status:** `READY_FOR_PRE_EXECUTION_REVIEW`
- **Execution Authorization Status:** **`UNAPPROVED`**
- In accordance with Experiment Governance Protocol v1.0 Section 14, execution is strictly forbidden until:
  1. This reviewed v1.3 design package completes formal peer scientific review.
  2. Standalone dataset generators and benchmark evaluators are validated.
  3. Pre-flight 60-step gradient calibration is separately authorized and executed.
  4. A formal Pre-Execution GO checklist is signed by the research lead.

