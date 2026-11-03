# CHAKRVIEW — CAND_01 ADVERSARIAL DESIGN REVIEW v1.3 (REVIEWED)
## Critical Stress-Testing, Methodological Defenses, and Vulnerability Analysis

**Document ID:** `REV-EXP-CAND01-ADV-004`  
**Location:** `docs/experiments/CAND_01_ADVERSARIAL_DESIGN_REVIEW_v1.3_REVIEWED.md`  
**Supersedes:** `docs/experiments/CAND_01_ADVERSARIAL_DESIGN_REVIEW_v1.2_REVIEWED.md`, `docs/experiments/CAND_01_ADVERSARIAL_DESIGN_REVIEW_v1.1_REVIEWED.md`, `docs/experiments/CAND_01_ADVERSARIAL_DESIGN_REVIEW_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Review Type:** ADVERSARIAL SCIENTIFIC AND METHODOLOGICAL STRESS TEST  
**Reviewer Role:** Independent Skeptical ML Peer Reviewer & Statistical Auditor  
**Status:** `REVIEW_COMPLETED`  
**Execution Authorization:** **`UNAPPROVED`**  
**Date:** 2026-10-31  

---

## 1. Adversarial Purpose & Mandate (v1.3 Targeted Redesign)

Scientific experiments often fail not because the central hypothesis is false, but because subtle experimental artifacts, evaluation loopholes, or data leakage create illusions of capability or uninformative failures. This adversarial review interrogates `CAND_01_CONTROLLED_ENTITY_COPYING` against twelve demanding critique vectors, incorporating the resolutions to findings ADV-01 through ADV-07 to ensure that the experimental specification is mathematically sound, statistically defensible, and robust to scrutiny.

---

## 2. Twelve Adversarial Inquiries & Methodological Defenses

### Inquiry 1: Could the model achieve 75% accuracy simply by learning an unanchored "recency bias" (always copying the most recent token)?
- **Adversarial Challenge:**
  In single-pair sequences (`PAIR: K=V. QUERY: K -> V`), the value $V$ is often the most recent entity token prior to the query delimiter. A trivial 1-layer attention head could attend exclusively to position $T-3$ and achieve $100\%$ accuracy without learning associative binding.
- **Methodological Defense:**
  Task Family TF-03 (Multi-Pair Association) and TF-04 (Distractor Resistance) randomize the position of the target pair across positions 1, 2, 3, and 4 in context. In TF-04 with 3 distractors, the target pair appears at the final position only $25\%$ of the time. Recency bias yields only $25\%$ accuracy on TF-04. The pre-registered promotion criterion mandates $\ge 60\%$ on TF-04, mathematically disproving recency bias.

### Inquiry 2: How does v1.3 resolve the TF-01 combinatorial collapse and duplicate leakage (Finding ADV-01)?
- **Adversarial Challenge:**
  In v1.2, TF-01 required 924 examples from $\mathcal{E}_{\text{train}}$, but there were only 450 unique training tokens. Duplicate prompts had to leak across splits.
- **Methodological Defense:**
  In v1.3, Task Family TF-01 (Direct Copy: `COPY: X -> X`) is **strictly excluded from Partitions B and C** (Validation and Test Recombination). Direct copying has no key-value relational mapping to recombine. Furthermore, Partition A (Train) allocates exactly **350 examples** to TF-01, and Partition D allocates **50 examples**. Because $350 \le 450$ and $50 \le 150$, **prompt-level disjointness is 100% mathematically guaranteed** with zero duplication across splits.

### Inquiry 3: Why is Partition D described as "disjoint-token in-context copying" rather than "zero-shot generalization"?
- **Adversarial Challenge:**
  The token being copied is explicitly printed in the prompt context. Describing this as "zero-shot OOD generalization" deceives reviewers into believing the model learned external concepts.
- **Methodological Defense:**
  v1.3 formally adopts the precise term **"disjoint-token in-context copying"**. It measures whether the attention routing mechanism can copy a token identity that was not exposed to gradient updates during fine-tuning. It makes zero claims of semantic generalization or open-domain concept reasoning.

### Inquiry 4: Does the prompt delimiter (` -> `) act as an artificial trigger that cannot generalize?
- **Adversarial Challenge:**
  The model might overfit to the exact delimiter ` -> ` and fail if a colon or newline is used.
- **Methodological Defense:**
  Standardizing the prompt delimiter is necessary in Confirmatory mode to eliminate syntax parsing variance. The research question is not "can the model parse arbitrary punctuation", but "can the attention heads route information from key to value". Formatting robustness is deferred to Track 4 (Format Robustness).

### Inquiry 5: Could high training loss reduction occur while generation still outputs repetitive gibberish?
- **Adversarial Challenge:**
  In Exp35, training loss dropped from 5.08 to 1.098 while generation scored 0/200. Could CAND_01 repeat this delusion?
- **Methodological Defense:**
  In Exp35, 78.96% of loss tokens were static boilerplate words. In CAND_01 Arm A, 100% of the loss is applied strictly to the target value token ($[VAL] + [EOS]$). There are zero template tokens in the loss function. Furthermore, the evaluation harness mandates both $A_{\text{FT}} \ge 75.0\%$ and clean sequence completion $A_{\text{FC}} \ge 70.0\%$; repetitive generation after token 0 scores $0.0$ on $A_{\text{FC}}$ and triggers `ERR-04`.

### Inquiry 6: Why was the paired seed-level permutation test removed (Finding ADV-03)?
- **Adversarial Challenge:**
  v1.2 prescribed a paired permutation test across 3 seeds demanding $p < 0.01$. But for $n=3$, the minimum possible two-sided p-value is $0.25$.
- **Methodological Defense:**
  v1.3 formally removes this mathematically impossible test. It acknowledges that $n=3$ seeds can only provide **descriptive uncertainty** ($\text{mean} \pm \text{range}$, SEM). Inferential statistical testing is confined to within-model evaluations using two-way cluster bootstraps over key and target token IDs.

### Inquiry 7: Why was the prompt random effect removed from the GLMM formula (Finding ADV-04)?
- **Adversarial Challenge:**
  v1.2 included a random intercept $v_{\text{prompt}(k)}$ where each prompt had only a single binary observation.
- **Methodological Defense:**
  A prompt random effect with single observations is structurally unidentifiable in binary logistic regression (confounded with latent residual variance). v1.3 removes $v_{\text{prompt}}$ and specifies random intercepts strictly for crossed factors with repeated measures: $u_{\text{key}}$ and $w_{\text{val}}$.

### Inquiry 8: Why is Arm A vs. Arm C classified as a "composite packaging intervention" (Finding ADV-05)?
- **Adversarial Challenge:**
  Converting minimal syntax into conversational framing increases context length $4\times$, increases RoPE positional distance, and dilutes softmax mass. How can it isolate syntax framing alone?
- **Methodological Defense:**
  v1.3 concedes that Arm A vs. Arm C evaluates a bundled composite intervention. The document explicitly bars citing Arm A vs. Arm C as an isolated test of syntax framing, documenting the sequence length and RoPE confounds transparently.

### Inquiry 9: Why was the pre-flight gradient calibration extended to 60 steps (Finding ADV-06)?
- **Adversarial Challenge:**
  Step 20 of training is still inside the 50-step linear warmup, where the learning rate is $< 4 \times 10^{-5}$. A 20-step check cannot detect instability at the step 50 peak ($10^{-4}$).
- **Methodological Defense:**
  v1.3 extends the calibration duration to **60 steps**, explicitly capturing post-warmup dynamics at maximum learning rate velocity. Furthermore, it specifies logging the **unclipped global gradient $L_2$ norm**.

### Inquiry 10: How are frequency tier boundaries handled without fabricated logarithmic cutoffs (Finding ADV-02)?
- **Adversarial Challenge:**
  v1.2 mandated $\log_{10} C(t) \ge 4.0$ ($C(t) \ge 10,000$) on a 465k corpus after excluding stopwords, which is mathematically impossible.
- **Methodological Defense:**
  v1.3 discards arbitrary logarithmic cutoffs and marks frequency tier boundaries as **`BLOCKED_PENDING_DATA`**. It specifies that empirical percentile cutoffs (top 33%, middle 33%, bottom 34%) will be computed directly from tokenized pretraining data when script execution is authorized.

### Inquiry 11: Could the model exploit cross-example leakage across batches?
- **Adversarial Challenge:**
  If the same key appears across multiple training examples with different values, the model might suffer catastrophic interference or learn a static prior.
- **Methodological Defense:**
  The dataset specification enforces unique key assignment within each batch and strictly disjoint token pools between training and test sets.

### Inquiry 12: What is the most dangerous misinterpretation of a successful CAND_01 run?
- **Adversarial Challenge:**
  If Arm A scores 90% accuracy, how might the project deceive itself?
- **Methodological Defense:**
  The most dangerous trap would be declaring that ChakrMicro has "mastered reasoning" or "proven the existence of induction circuits." CAND_01 tests synthetic associative lookup under zero-noise formatting. It proves that the architecture can route attention to copy tokens under ideal conditions; it does not prove natural language extraction, conversational grounding, or multi-hop factual inference.

---

## 3. Residual Risks and Mitigations Summary

| Residual Risk | Severity | Mitigation Strategy |
|---|---|---|
| **Softmax Temperature Sensitivity** | Low | Evaluation is locked to deterministic greedy decoding ($T=0.0$). |
| **Micro-batch Noise** | Medium | Gradient accumulation of 4 steps yields effective batch size of 8, smoothing gradient variance on CPU. |
| **Two-Way Cluster Dependence** | Medium | Two-way cluster bootstrap resampled simultaneously across keys and values ($B=2,000$). |
| **Pre-Flight LR Divergence** | Medium | 60-step unclipped gradient norm telemetry check on disposable model copy prior to full training. |

---

## 4. Adversarial Review Conclusion & Status

- **Review Recommendation:** **`READY_FOR_PRE_EXECUTION_REVIEW`**
- **Documentation Assessment:** All identified mathematical contradictions, statistical impossibilities, split leakages, and overclaims from v1.2 have been systematically resolved in the v1.3 specification suite.
- **Execution Mandate:** This review assesses scientific design documentation only. **Execution authorization remains strictly UNAPPROVED.**

