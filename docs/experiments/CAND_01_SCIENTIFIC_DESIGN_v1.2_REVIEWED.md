# CHAKRVIEW — CAND_01 SCIENTIFIC DESIGN SPECIFICATION v1.2 (REVIEWED)
## Controlled Entity-Copying & Associative Binding Under Low-Noise Conditions

**Document ID:** `DOC-EXP-CAND01-SCI-003`  
**Location:** `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.2_REVIEWED.md`  
**Supersedes:** `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.1_REVIEWED.md`, `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Research Track:** Track 1 (Entity Association and Copying)  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md` (v1.0.0)  
**Research Priority:** Priority 1 (Blocker for Extractive Downstream Tasks)  
**Target Architecture:** ChakrMicro Baseline (3,443,136 parameters, 6L / 192D / 6H, RoPE, tied embeddings)  
**Target Parent Checkpoint:** Canonical Step 10,000 (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`, SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)  
**Status:** `READY_FOR_PRE_EXECUTION_REVIEW` (Specification Only — Execution Strictly Unapproved)  
**Execution Authorization:** **`UNAPPROVED`**  
**Date:** 2026-10-31  

---

## 1. Revision History & Audit Scope (v1.2)

This document represents the formal v1.2 scientific revision of `CAND_01_SCIENTIFIC_DESIGN_v1.1_REVIEWED.md`, incorporating resolutions to the independent scientific audit findings (F-01 through F-07):
- **Resolution F-01 (Partitioning Symmetry & Dual Test Architecture):** Replaced the asymmetric split structure with an explicit 4-way evaluation architecture that cleanly separates in-distribution seen-token recombination ($A_{\text{test,recomb}}$) from disjoint out-of-distribution token generalization ($A_{\text{test,ood}}$). Prohibited using validation loss as an estimator of OOD performance.
- **Resolution F-02 (Cluster Dependence & Clustered Standard Errors):** Explicitly documented that $N=500$ prompts across 150 unique tokens violate i.i.d. assumptions. Formally required clustered bootstrap and multiway clustered variance estimation grouped by target token ID.
- **Resolution F-03 (Factorial Cleanliness & Arm D Demarcation):** Audited the $2 \times 2$ factorial matrix. Cleanly isolated the syntax framing effect (Arm A vs. C) using matched single-token entities. Explicitly demarcated Arm D as an exploratory historical reference that conflates syntax with subword tokenization.
- **Resolution F-04 (Hyperparameter Provenance & Calibration Protocol):** Classified all training parameters (`HISTORICAL`, `PROVISIONAL`, `VERIFIED`). Specified an isolated, separately authorized 20-step gradient telemetry calibration on a disposable model copy.
- **Resolution F-05 (Coupled Evaluator Metric Gating):** Formally coupled the primary behavioral success metric $A_{\text{FT}} \ge 75.0\%$ with a clean sequence completion threshold $A_{\text{FC}} \ge 70.0\%$, preventing models with repetition loops or trailing syntax leakage from being scored as successful.
- **Resolution F-06 (Corpus Frequency Stratification):** Defined token pool stratification criteria based on base pretraining corpus frequencies, isolating token prior bias.
- **Resolution F-07 (Autoregressive Primacy over Teacher Forcing):** Established that free autoregressive greedy generation is the sole basis for behavioral success; teacher-forced target rank is demoted strictly to secondary diagnostic telemetry.

---

## 2. Research Objective & Epistemic Boundaries

### 2.1 Research Objective
The sole objective of CAND_01 is to determine whether the ChakrMicro baseline checkpoint, under controlled synthetic conditions with minimal template syntax and target-focused loss masking, can **behaviorally retrieve and reproduce an associated value token** when queried with its corresponding key token from local context.

### 2.2 Epistemic Boundaries: What This Experiment Does and Does Not Establish

```
+---------------------------------------------------------------------------------------+
|                                EPISTEMIC BOUNDARY                                    |
+---------------------------------------------------------------------------------------+
|  WHAT CAND_01 CAN DEMONSTRATE:                   WHAT CAND_01 CANNOT DEMONSTRATE:    |
|  - In-context associative token retrieval        - General natural language reasoning |
|  - Direct token copying without relation         - Factual knowledge extraction       |
|  - Distractor interference decay profile         - Multi-turn conversational memory   |
|  - Positional degradation under RoPE             - Proof of "induction circuits"      |
|  - Sensitivity to prompt key swap (Delta p)      - Universal architectural capacity   |
+---------------------------------------------------------------------------------------+
```

- **Behavioral Evidence vs. Mechanistic Claim:**  
  Achieving high accuracy on held-out test splits demonstrates that the model's forward pass successfully emits the bound value token under the tested prompt format. It does **not** prove:
  1. The physical emergence of dedicated two-layer attention induction circuits.
  2. A specific linear subspace binding mechanism in feedforward or attention layers.
  3. That the model possesses semantic understanding of "keys" and "values".
- **Falsification vs. Architectural Incapacity:**  
  If the model achieves accuracy no better than chance controls, it demonstrates that the tested training procedure (AdamW, 500 steps, learning rate schedule, target-only loss) failed to induce associative behavior. It does **not** prove that the 3.44M parameter architecture is fundamentally incapable of associative retrieval under different curricula, objectives, or initializations.

---

## 3. Core Research Questions & Formal Hypotheses

### 3.1 Primary Research Question
Under minimal synthetic prompt framing (`PAIR: K=V. QUERY: K -> V`) and 100% target-focused loss masking, can the ChakrMicro baseline checkpoint achieve statistically significant associative retrieval on held-out test splits compared to empirical reference baselines?

### 3.2 Formal Hypotheses
- **Alternative Hypothesis ($H_1$):**  
  When trained on synthetic key-value pairs with minimal syntactic framing and 100% loss applied strictly to target answer tokens, ChakrMicro will achieve a multi-seed mean first-token exact-match accuracy $A_{\text{FT}} \ge 75.0\%$ [PROVISIONAL] and full completion match $A_{\text{FC}} \ge 70.0\%$ [PROVISIONAL] on held-out test splits across three independent random seeds (`seeds: [42, 137, 2026]`), demonstrating controlled associative retrieval behavior.
- **Null Hypothesis ($H_0$):**  
  ChakrMicro achieves exact-match accuracy on held-out test splits that does not exceed empirical reference baselines (specifically, the key-shuffled permutation baseline and unconditioned token prior baseline, bounded provisionally at $\le 5.0\%$), indicating that the optimization failed to induce functional associative retrieval.

### 3.3 Empirical Reference Baselines (Replacing Arbitrary Chance Thresholds)
To prevent reliance on unverified uniform random distribution assumptions:
1. **Unconditioned Base Model Prior Control ($B_{\text{prior}}$):**  
   Measure the pre-softmax probability and emission rate of the target token when prompt context contains zero key-value definitions (`QUERY: ->`). Measures the intrinsic vocabulary prior.
2. **Key-Shuffled Permutation Control ($B_{\text{perm}}$):**  
   Query with a random key not present in context, or shuffle the query key relative to context pairs. Measures whether output emission depends on specific key-value binding or unconditioned context copying.
3. **Frequency-Matched Token Control ($B_{\text{freq}}$):**  
   Evaluate retrieval accuracy across high-, medium-, and low-frequency vocabulary tiers to detect embedding norm bias.

---

## 4. The Eight Core Task Families

CAND_01 structures evaluation across eight orthogonal task families:

| Family ID | Task Name | Canonical Prompt Structure | Expected Output | Core Capability Tested |
|---|---|---|---|---|
| **TF-01** | Direct Token Copy | `COPY: X3M1 ->` | `X3M1` | Direct contextual replication; baseline sanity check. |
| **TF-02** | Single Key-Value Association | `PAIR: K7P9=X3M1. QUERY: K7P9 ->` | `X3M1` | Single-step key-value relational binding without distractors. |
| **TF-03** | Multi-Pair Association | `PAIRS: K7=X3, M2=L8. QUERY: M2 ->` | `L8` | Selective retrieval among multiple valid bindings in local context. |
| **TF-04** | Distractor Resistance | `PAIRS: D1=N4, K7=X3, D2=R9. QUERY: K7 ->` | `X3` | Resistance to irrelevant context distractors (1, 3, 7 pairs). |
| **TF-05** | Similar-Key Discrimination | `PAIRS: K7P1=X3, K7P2=M8. QUERY: K7P1 ->` | `X3` | Disambiguation when keys differ by Levenshtein distance 1. |
| **TF-06** | Unseen Pair Recombination | `PAIRS: K1=V2, K2=V1. QUERY: K1 ->` | `V2` | Recombination of familiar keys and values in novel pairings. |
| **TF-07** | Context-Position Variation | `PAIRS: K1=V1 [PAD x N] QUERY: K1 ->` | `V1` | Positional robustness across token offsets (10, 50, 100, 200 tokens). |
| **TF-08** | Context-Length Scaling | `[Dense context with N pairs] QUERY: K_tgt ->` | `V_tgt` | Capacity scaling under context saturation (total tokens: 64 to 400). |

---

## 5. Factorial Experimental Arms & Confound Isolation

To determine the causal drivers of learning, CAND_01 implements a $2 \times 2$ factorial matrix across two independent variables:
- **Factor 1: Context Framing** (Minimal Key-Value Syntax vs. Natural Language Conversational Framing).
- **Factor 2: Loss Masking** (Target-Only Masked Loss vs. Unmasked Autoregressive Loss).

### 5.1 Formal Factorial Arms

| Arm ID | Context Framing | Loss Masking Policy | Supervision Allocation | Primary Comparison & Isolating Purpose |
|---|---|---|---|---|
| **Arm A (Core Intervention)** | Minimal Syntax (`K=V`) | Target Only (`mask_prompt=True`) | 100% on target token | Baseline intervention: optimal associative routing. |
| **Arm B (Masking Control)** | Minimal Syntax (`K=V`) | Full Sequence (`mask_prompt=False`) | ~15% target, 85% syntax | **Arm A vs. B:** Isolates main effect of loss masking under matched minimal syntax. |
| **Arm C (Framing Control)** | Natural Language | Target Only (`mask_prompt=True`) | 100% on target token | **Arm A vs. C:** Isolates main effect of syntax framing under matched single-token entities. |
| **Arm D (Historical Reference)** | Natural Language | Full Sequence (Standard SFT unmasked) | ~16% entity, 84% syntax | **Exploratory Reference:** Approximates Exp35 conditions; conflates syntax and subwords. |

> [!IMPORTANT]
> **Confound Isolation in Arm C vs. Arm D:**  
> In Arm C, all entity slots must use the **exact same single-token entity pool** as Arms A and B. This ensures that the comparison between Arm A and Arm C cleanly isolates natural language sentence boilerplate without tokenizer subword fragmentation. Arm D retains natural language multi-token entities strictly as an exploratory historical reference; it must **never** be cited as a clean causal comparator.

---

## 6. Dataset Allocation & Dual Test Architecture (Resolution F-01)

To eliminate the validation/test distribution shift identified in Audit Finding F-01, CAND_01 establishes an explicit four-partition dataset structure:

```
                              [ VOCABULARY POOL: 600 SINGLE TOKENS ]
                                                |
                 +------------------------------+------------------------------+
                 |                                                             |
    [ Training Token Pool: 450 Tokens ]                           [ OOD Token Pool: 150 Tokens ]
                 |                                                             |
   +-------------+-------------+                                               |
   |                           |                                               |
[ Partition A: Train ]   [ Partition B: Val Recomb ]   [ Partition C: Test Recomb ]    [ Partition D: Test OOD ]
  4,000 examples           500 examples                  500 examples                    500 examples
  Training pairs           Held-out combinations         Held-out combinations           Strictly disjoint tokens
  Supervised training      Model checkpoint tuning       In-distribution test            Zero-shot OOD test
```

### Partition Definitions
1. **Partition A (Train Split — 4,000 examples):**  
   Keys and values sampled from $\mathcal{E}_{\text{train}}$ (450 tokens). Associations $(K_i, V_j)$ are designated training tuples.
2. **Partition B (Validation Recombination Split — 500 examples):**  
   Keys and values sampled from $\mathcal{E}_{\text{train}}$, but associations $(K_i, V_j)$ are strictly held out from Partition A. Used strictly for hyperparameter monitoring and early stopping diagnostics.
3. **Partition C (Test Recombination Split — 500 examples):**  
   Keys and values sampled from $\mathcal{E}_{\text{train}}$, with associations held out from Partitions A and B. Yields the in-distribution recombination score: $\mathbf{A_{\text{test,recomb}}}$.
4. **Partition D (Test OOD Entity Split — 500 examples):**  
   Keys and values sampled strictly from $\mathcal{E}_{\text{test}}$ (150 tokens). Yields the out-of-distribution token generalization score: $\mathbf{A_{\text{test,ood}}}$.

*Directives:*
- $A_{\text{test,recomb}}$ and $A_{\text{test,ood}}$ must be reported as separate, non-aggregated metrics.
- Validation loss/accuracy on Partition B must **never** be claimed as an estimate of Partition D performance.

---

## 7. Statistical Design, Cluster Dependence, and Seed Policy (Resolution F-02)

### 7.1 Cluster Dependence Hazard
Because $N=500$ evaluation instances are generated from 150 unique entity tokens across 8 task families, individual prompt instances share underlying token representations. Prompts containing the same target token $V$ or query key $K$ are statistically dependent. Treating them as 500 independent Bernoulli trials artificially deflates standard errors.

### 7.2 Required Statistical Analysis Protocol
1. **Multi-Seed Execution:** Exactly three independent random seeds (`seeds: [42, 137, 2026]`) executed across all four arms.
2. **Clustered Bootstrap Estimation:** Compute 95% confidence intervals using a cluster bootstrap resampled at the `target_val_token_id` level ($B = 2,000$ resamples).
3. **Hierarchical Mixed-Effects Model:** Model binary test success via mixed-effects logistic regression:
   $$\text{logit}(P(Y_{ijk} = 1)) = \beta_0 + \beta_{\text{arm}} + u_{\text{token}(j)} + v_{\text{prompt}(k)} + \epsilon_{ijk}$$
   Where $u_{\text{token}}$ represents random intercepts for entity token IDs.
4. **Hypothesis Testing:** Evaluate differences across arms using paired permutation tests across seed-matched runs, applying Bonferroni correction for family-wise error rate control ($\alpha = 0.01$).

---

## 8. Hyperparameter Provenance & Pre-Flight Calibration (Resolution F-04)

All training hyperparameters are formally audited and classified:

| Parameter | Proposed Setting | Provenance Classification | Pre-Flight Validation Mandate |
|---|---|---|---|
| **Base Architecture** | 6L / 192D / 6H / RoPE / Tied Embeddings | `VERIFIED` | Matches `chakrview/brain/model.py` and canonical checkpoint |
| **Sequence Length** | $T_{\text{max}} = 512$ tokens | `VERIFIED` | Matches `collator.py` bounds check |
| **Device & Precision** | Single-node CPU, float32 | `VERIFIED` | Matches project CPU-first mandate |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.98, \text{wd}=0.01$) | `HISTORICAL` | Inherited from Stage C pretraining |
| **Learning Rate** | Peak $1.0 \times 10^{-4}$, linear warmup 50 steps, cosine decay to $1.0 \times 10^{-5}$ | `PROVISIONAL` | **Requires pre-flight calibration (Section 8.1)** |
| **Effective Batch** | Micro-batch 2, Accumulation 4 ($B_{\text{eff}} = 8$) | `PROVISIONAL` | Matches CPU cache limits; requires memory stability check |
| **Step Budget** | Exactly 500 optimization steps | `PROVISIONAL` | Bounded diagnostic compute budget |

### 8.1 Isolated Pre-Flight Calibration Protocol (Separately Authorized Activity)
Target-only loss concentrates 100% of gradient magnitude onto a single token position ($[VAL] + [EOS]$). To prevent gradient explosion or immediate logit saturation:
- A dedicated, separately authorized 20-step gradient telemetry run must be conducted on a **disposable scratch copy** of the checkpoint.
- Telemetry will log: per-step gradient $L_2$ norm, gradient clipping frequency, and maximum pre-softmax logit drift.
- If gradient norms exceed 5.0 on $>20\%$ of steps, the learning rate must be calibrated downward (e.g. to $5.0 \times 10^{-5}$) prior to confirmatory execution.
- *Strict Rule:* This calibration is a training activity. It is **not authorized** by this design document.

---

## 9. Diagnostic Limitations: Mechanistic Interpretability

- **Attention Flow Diagnostics:** Attention weight measurements ($W_{\text{attn}}$) between query and context tokens in Layer 5 are observational probes. They do **not** establish causal information routing without intervention ablations (e.g. activation patching).
- **Layer Specificity:** Layer 5 is examined as a primary candidate based on historical transformer depth heuristics; however, all 6 layers will be logged to avoid confirmation bias.
- **RoPE Distance Curves:** Accuracy degradation across token offsets ($D \in \{10, 50, 100, 200\}$) measures positional stability under RoPE; it does not isolate frequency decay without rotary base frequency controls.

---

## 10. Pre-Registered Post-Execution Decision Thresholds (Protocol v1.0 Section 10)

Post-execution outcomes map strictly to the four canonical states defined in Protocol v1.0:

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
   - Multi-seed mean first-token match $A_{\text{FT}} \ge 75.0\%$ on both Recombination and OOD splits.
   - Multi-seed mean full completion match $A_{\text{FC}} \ge 70.0\%$.
   - Multi-seed mean distractor resistance (TF-04, 3 pairs) $\ge 60.0\%$.
   - Multi-seed mean prompt sensitivity delta $\Delta p \ge 0.20$.
   - Zero regression beyond non-inferiority margin on canonical pretraining validation loss ($\Delta \text{loss} \le +0.15$).
   *Action:* Candidate architecture validated for associative copying; advances to Track 2.
2. **`CONTINUE`:**
   - Multi-seed mean $A_{\text{FT}}$ between $25.0\%$ and $74.9\%$.
   *Action:* Partial capability demonstrated; formulate follow-up specification exploring learning rate, warmup length, or curriculum staging.
3. **`REJECT`:**
   - Multi-seed mean $A_{\text{FT}} < 25.0\%$ or $\Delta p < 0.05$.
   *Action:* Tested training procedure falsified; archive results with failure forensics.
4. **`INCONCLUSIVE`:**
   - Run compromised by hardware failure, NaN loss divergence, or cross-split leakage.
   *Action:* Results invalidated; remediate defect before re-execution.

---

## 11. Execution Governance & Approval Posture

- **Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`
- **Proposal Status:** `READY_FOR_PRE_EXECUTION_REVIEW`
- **Execution Authorization Status:** **`UNAPPROVED`**
- Under Experiment Governance Protocol v1.0 Section 14, execution is strictly forbidden until:
  1. This reviewed v1.2 design package completes formal peer scientific review.
  2. Standalone dataset generators and benchmark evaluators are validated.
  3. Pre-flight gradient calibration is separately authorized and executed.
  4. A formal Pre-Execution GO checklist is signed by the research lead.

