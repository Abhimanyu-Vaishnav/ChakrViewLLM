# CHAKRVIEW — CAND_01 SCIENTIFIC DESIGN SPECIFICATION v1.1 (REVIEWED)
## Controlled Entity-Copying & Associative Binding Under Low-Noise Conditions

**Document ID:** `DOC-EXP-CAND01-SCI-002`  
**Location:** `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.1_REVIEWED.md`  
**Supersedes:** `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Research Track:** Track 1 (Entity Association and Copying)  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md` (v1.0.0)  
**Research Priority:** Priority 1 (Blocker for Extractive Downstream Tasks)  
**Target Architecture:** ChakrMicro Baseline (3,443,136 parameters, 6L / 192D / 6H, RoPE, tied embeddings)  
**Target Parent Checkpoint:** Canonical Step 10,000 (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`, SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)  
**Status:** `READY_FOR_SCIENTIFIC_REVIEW` (Specification Only — Execution Strictly Unapproved)  
**Execution Authorization:** **`UNAPPROVED`**  
**Date:** 2026-10-31  

---

## 1. Revision History & Audit Scope

This document represents the formal v1.1 scientific revision of `CAND_01_SCIENTIFIC_DESIGN_v1.0.md`. It resolves critical and high-severity scientific, statistical, and governance issues identified during independent forensic review:
- **Correction 1 (Behavioral vs. Mechanistic Scope):** Removed unsupported assertions that successful behavioral task execution "proves the existence of induction circuits" or that failure "proves architectural incapacity." Reframed conclusions strictly in terms of empirical behavioral task performance and attention routing hypotheses.
- **Correction 2 (H0 & Empirical Reference Baselines):** Eliminated the arbitrary 5% uniform chance threshold. Replaced with explicit empirical reference controls (unconditioned prior baseline, key-shuffled permutation baseline, and frequency-matched control).
- **Correction 3 (Dataset Independence & Factor Isolation):** Formally distinguished unseen-token evaluation, unseen-pair recombination, distractor scaling, and positional scaling. Defined the unit of statistical independence.
- **Correction 4 (Statistical Rigor & Multi-Seed Requirement):** Corrected the false N=500 power (>0.9999) and ±1.94% error margin claims. Established proper binomial worst-case confidence limits (±4.38%), accounted for cluster dependence, and mandated a 3-seed confirmatory evaluation protocol.
- **Correction 5 (Arm D Historical Characterization):** Clarified that Arm D is a related historical reference arm, not an exact bit-level replica of Exp35.
- **Correction 6 (Decision Taxonomy Alignment):** Replaced obsolete categories (`ITERATE`, `ABANDON`, `DEFECT`) with canonical Protocol v1.0 Section 10 post-execution states: `PROMOTE`, `CONTINUE`, `REJECT`, `INCONCLUSIVE`.

---

## 2. Executive Summary & Problem Context

During the Milestone 5 (M5 / Exp35) evaluation of dynamic property-conditioned extraction (DPCE), the fine-tuned ChakrMicro model achieved an extraction accuracy of exactly **0/200 (0.00%)**, despite training loss decreasing smoothly from 5.08 to 1.098. Subsequent Milestone 5.1 (M5.1) forensic investigations diagnosed four observed factors:
1. **Syntactic Template Domination (Observed):** $78.96\%$ of supervised loss tokens were static boilerplate words (`"At station"`, `"resides in"`). The model optimized syntactic sequence probabilities while leaving entity representation unconstrained.
2. **Attention Sensitivity Absence (Observed):** Prompt perturbation probes revealed a mean next-token probability shift ($\Delta p$) of only $0.00014$ when switching entity slot values.
3. **Subword Fragmentation (Observed):** Natural language entity names fragmented into 3 to 6 subword tokens under the 4,096-token BPE tokenizer.
4. **Confounded Task Framing (Methodological Limitation):** Natural language conversational framing conflated three distinct capabilities: (a) syntactic template fluency, (b) context-entity retrieval/copying, and (c) key-value relational binding.

Candidate 1 (`CAND_01_CONTROLLED_ENTITY_COPYING`) is engineered to isolate this behavioral capability. By stripping away prose framing, natural language syntax, and multi-token entity names, CAND_01 addresses the foundational empirical question: **Can the frozen ChakrMicro transformer architecture learn to route causal attention and reproduce an arbitrary key-value binding under clean, synthetic, low-noise conditions?**

---

## 3. Core Research Questions & Hypotheses

### 3.1 Primary Research Question
Under minimal synthetic prompt framing and 100% target-focused loss masking, can the ChakrMicro baseline checkpoint learn to associate an arbitrary key token with an arbitrary value token and reproduce the exact value token when queried with the key?

### 3.2 Formal Hypotheses
- **Alternative Hypothesis ($H_1$):**  
  When trained on synthetic key-value pairs with minimal syntactic framing and 100% loss applied strictly to the target answer token, ChakrMicro will achieve a statistically significant improvement over empirical chance baselines on held-out test splits, exceeding a provisional exact-match threshold of $75.0\%$ across three random seeds, supporting the hypothesis that the architecture can learn associative retrieval behavior under zero-noise conditions.
- **Null Hypothesis ($H_0$):**  
  ChakrMicro achieves exact-match copy accuracy on held-out test splits that does not exceed empirical control baselines (specifically, the key-shuffled permutation baseline and unconditioned token prior baseline, bounded provisionally at $\le 5.0\%$), indicating that the optimization under test failed to establish functional associative retrieval.

### 3.3 Epistemic Boundary: Behavioral Evidence vs. Mechanistic Claim
- **Behavioral Confirmation:** Achieving $A_{\text{EM}} \ge 75.0\%$ confirms that the model's end-to-end forward pass reliably outputs the bound value token given the key query.
- **Mechanistic Non-Equivalence:** Behavioral success does **not** prove:
  1. The existence of formal 2-layer induction circuits.
  2. A specific internal routing pathway or clean semantic vector binding.
  3. That attention heads are performing pure causal copying rather than complex non-linear heuristic matching.
- **Failure Non-Equivalence:** Achieving $< 5.0\%$ indicates that the tested optimization procedure (AdamW, LR schedule, 500 steps) failed to induce associative retrieval; it does **not** prove that the 3.44M parameter architecture is mathematically incapable of doing so under alternative curricula, objectives, or initializations.

---

## 4. Disentanglement of Capability Dimensions & Task Families

CAND_01 separates associative extraction into distinct behavioral dimensions across eight task families:

```
                      [ CAND_01 CAPABILITY SPECTRUM ]
                                     |
    +--------------------------------+--------------------------------+
    |                                |                                |
[Direct Copying]         [Associative Binding]             [Interference Resistance]
• TF-01: Direct Copy     • TF-02: Single Pair              • TF-04: Distractor Scaling
  Input: X -> X            Input: K1=V1; Q: K1 -> V1         (1, 3, 7 distractor pairs)
• Baseline sanity check  • 1-to-1 arbitrary binding        • TF-05: Similar-Key Disambig
                                                              (Levenshtein dist = 1)
                                     |
    +--------------------------------+--------------------------------+
    |                                                                 |
[Recombination & OOD Tokens]                             [Positional Stability]
• TF-06: Unseen Recombination                            • TF-07: Query Offset Scaling
  (Seen keys + Seen vals in novel pairings)                (10, 50, 100, 200 tokens)
• TF-08: Context Length Scaling                          • Token-Disjoint Test Set
  (Dense packing: 64 to 400 tokens)                        (Disjoint token IDs)
```

| Family ID | Task Name | Canonical Prompt Structure | Expected Completion | Capability Tested |
|---|---|---|---|---|
| **TF-01** | Direct Token Copy | `COPY: X3M1 ->` | `X3M1` | Direct contextual replication; baseline sanity check. |
| **TF-02** | Single Key-Value Association | `PAIR: K7P9=X3M1. QUERY: K7P9 ->` | `X3M1` | Single-step key-value relational binding without distractors. |
| **TF-03** | Multi-Pair Association | `PAIRS: K7=X3, M2=L8. QUERY: M2 ->` | `L8` | Selection among multiple valid bindings in local context. |
| **TF-04** | Distractor Resistance | `PAIRS: D1=N4, K7=X3, D2=R9. QUERY: K7 ->` | `X3` | Resistance to irrelevant context distractors (1, 3, 7 pairs). |
| **TF-05** | Similar-Key Discrimination | `PAIRS: K7P1=X3, K7P2=M8. QUERY: K7P1 ->` | `X3` | Prefix collision disambiguation (Levenshtein distance = 1). |
| **TF-06** | Unseen Combination Rebinding | `PAIRS: K1=V2, K2=V1. QUERY: K1 ->` | `V2` | Dynamic binding of familiar keys to familiar values in novel pairs. |
| **TF-07** | Context-Position Variation | `PAIRS: K1=V1 [PAD x N] QUERY: K1 ->` | `V1` | Positional robustness across token offsets (10, 50, 100, 200 tokens). |
| **TF-08** | Context-Length Scaling | `[Dense context with N pairs] QUERY: K_tgt ->` | `V_tgt` | Capacity scaling under context saturation (total tokens: 64 to 400). |

---

## 5. Experimental Arms & Factorial Matrix

To determine the causal drivers of learning, CAND_01 implements a $2 \times 2$ factorial matrix across two independent variables:
- **Factor 1: Context Framing** (Minimal Key-Value Syntax vs. Natural Language Conversational Framing).
- **Factor 2: Loss Masking** (Target-Only Masked Loss vs. Unmasked Autoregressive Loss).

### 5.1 Formal Factorial Arms

| Arm ID | Context Framing | Loss Masking Policy | Supervision Focus | Primary Hypothesis Tested |
|---|---|---|---|---|
| **Arm A (Core Intervention)** | Minimal Syntax (`K=V`) | Target Only (`mask_prompt=True`, ignore template) | 100% on target token | Optimal associative routing without syntax interference. |
| **Arm B (Masking Control)** | Minimal Syntax (`K=V`) | Full Sequence (`mask_prompt=False`, prompt supervised) | ~15% target, 85% syntax | Tests if template gradient dominates even with minimal syntax. |
| **Arm C (Framing Control)** | Natural Language | Target Only (`mask_prompt=True`, entity supervised) | 100% on target entity token | Tests if natural language syntactic fluff impairs attention routing. |
| **Arm D (Historical Reference)** | Natural Language | Full Sequence (Standard SFT unmasked) | ~16% entity, 84% syntax | Related historical reference approximating Exp35 conditions. |

> [!NOTE]
> Arm D is designated as a **related historical reference arm**, not an exact bit-level replica of Exp35. True bit-exact replication would require matching the exact historical prompt ordering, unaligned batch padding, and legacy tokenizer state. Arm D provides qualitative baseline continuity within the randomized trial.

---

## 6. Empirical Baselines & Control Conditions

To prevent reliance on unverified chance assumptions, model performance will be compared against three empirical reference controls:

1. **Unconditioned Base Model Prior Control:**
   Measure the pre-softmax probability and emission rate of the target token when prompt context is empty or neutralized (`QUERY: ->`). Measures the base language prior.
2. **Key-Shuffled Permutation Control:**
   Randomly shuffle the query key while leaving context key-value pairs intact. Measures whether the model's output depends on key-specific attention routing or unconditioned context extraction.
3. **Frequency-Matched Token Control:**
   Evaluate performance when target tokens are drawn from high-frequency vs. low-frequency vocabulary bands in `data/experiments/vocab_4096`.

---

## 7. Statistical Analysis Plan & Sample Size Justification

### 7.1 Statistical Confidence Limits (N = 500)
- In binary classification with $N = 500$ observations, the standard error is given by $SE = \sqrt{p(1-p)/N}$.
- At $p = 0.50$ (maximum variance), $SE = \sqrt{0.25 / 500} \approx 0.02236$. The worst-case 95% Wilson confidence interval half-width is approximately:
  $$\text{Margin} \approx 1.96 \times 0.02236 \approx 4.38\%$$
- Claims of $\pm 1.94\%$ margin apply only under specific high-accuracy conditions ($p = 0.75$) assuming strictly independent and identically distributed (i.i.d.) observations.

### 7.2 Independence Unit and Cluster Dependence
- The fundamental unit of statistical independence is the **unique key-value mapping**.
- Repeated prompts using the same key-value pairs across different task families introduce cluster correlation. Analysis of variance (ANOVA) and mixed-effects logistic regression with random effects for token ID will be used to account for token-level variance.

### 7.3 Confirmatory Multi-Seed Protocol
In strict compliance with Protocol v1.0 Section 3.4, confirmatory evaluation mandates:
- **Seed List:** Exactly three independent random seeds: `seeds: [42, 137, 2026]`.
- **Reporting:** Every metric will report mean $\pm$ standard error across seeds ($n=3$).
- **Significance Testing:** Two-tailed paired t-tests (or Wilcoxon signed-rank tests for non-normal distributions) comparing Arm A vs. Arm B, Arm C, and Arm D at $\alpha = 0.01$ with Bonferroni correction for multiple arm comparisons.

---

## 8. Diagnostic Limitations: Mechanistic Interpretability

- **Attention Weight Limitation:** Tracking attention weights ($W_{\text{attn}}$) between query and value tokens is an observational diagnostic, **not** causal proof of information routing. Attention concentration can occur without downstream heads utilizing the attended representation.
- **Layer Independence:** While Layer 5 attention is tracked as a diagnostic probe, no specific layer is assumed to be scientifically privileged. Analysis will examine all 6 layers.
- **Positional Patterns:** Observed accuracy decay across token offsets ($D \in \{10, 50, 100, 200\}$) characterizes positional stability under RoPE; it does not isolate the exact rotary frequency component without targeted rotational ablations.

---

## 9. Pre-Registered Decision Thresholds (Protocol v1.0 Section 10)

Outcomes map strictly to the four post-execution governance states defined in Protocol v1.0:

```
                  [ CAND_01 DECISION THRESHOLDS ]
                                 |
    +----------------------------+----------------------------+
    |                            |                            |
[ PROMOTE ]                  [ CONTINUE ]                 [ REJECT ]
• Test Acc >= 75.0% (mean)   • Test Acc: 25.0% - 74.9%    • Test Acc < 25.0% (mean)
• Distractor (3) >= 60.0%    • Distractor (3) >= 20.0%    • Sensitivity Delta < 0.05
• Delta p >= 0.20 (mean)     • Delta p: 0.05 - 0.19       • Target Rank > 50 / 4096
• Target Rank <= 5 / 4096    • Target Rank: 6 - 50        • No seed-level improvement
• No canonical regression    • Partial capability noted   • Architectural limitation
```

1. **`PROMOTE`:**
   - Multi-seed mean test exact-match accuracy $A_{\text{EM}} \ge 75.0\%$.
   - Multi-seed mean distractor resistance (TF-04, 3 pairs) $\ge 60.0\%$.
   - Multi-seed mean prompt sensitivity delta $\Delta p \ge 0.20$.
   - Mean target token logit rank $\le 5$ out of 4,096.
   - Zero regression beyond non-inferiority margin on canonical pretraining loss.
   *Action:* Candidate architecture validated for associative copying; eligible for Track 2 (Curriculum & Multihop Binding).

2. **`CONTINUE`:**
   - Multi-seed mean test exact-match accuracy between $25.0\%$ and $74.9\%$.
   - Mean target token logit rank between 6 and 50 out of 4,096.
   *Action:* Formulate follow-up experiment specification investigating learning rate, warmup length, curriculum staging (e.g. TF-01 prior to TF-02), or RoPE base frequency.

3. **`REJECT`:**
   - Multi-seed mean test exact-match accuracy $< 25.0\%$.
   - Multi-seed mean prompt sensitivity delta $\Delta p < 0.05$.
   - Mean target token logit rank $> 50$ out of 4,096.
   *Action:* Reject candidate intervention. Archive results with full failure forensics. Document baseline architectural limitation.

4. **`INCONCLUSIVE`:**
   - Execution compromised by hardware disruption, NaN loss divergence, evaluator defects, or cross-split data leakage.
   *Action:* Invalidate results. Remediate defect before re-running.

---

## 10. Execution Governance & Approval Posture

- **Proposal Status:** `READY_FOR_SCIENTIFIC_REVIEW`
- **Execution Authorization:** **`UNAPPROVED`**
- Under Experiment Governance Protocol v1.0 Section 14, execution is strictly forbidden until:
  1. This reviewed v1.1 design package completes formal peer scientific review.
  2. Standalone dataset generators and benchmark evaluators are validated.
  3. A formal Pre-Execution GO checklist is signed by the research lead.

