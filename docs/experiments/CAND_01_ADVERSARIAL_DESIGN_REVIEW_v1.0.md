# CHAKRVIEW — CAND_01 ADVERSARIAL DESIGN REVIEW v1.0
## Critical Stress-Testing, Failure Modes, and Methodological Defenses

**Document ID:** `REV-EXP-CAND01-ADV-001`  
**Location:** `docs/experiments/CAND_01_ADVERSARIAL_DESIGN_REVIEW_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Review Type:** ADVERSARIAL SCIENTIFIC AND METHODOLOGICAL STRESS TEST  
**Reviewer Role:** Adversarial ML Peer Reviewer & Statistical Auditor  
**Status:** `REVIEW_COMPLETED`  
**Date:** 2026-10-31  

---

## 1. Adversarial Purpose and Review Mandate

Scientific experiments often fail not because the underlying hypothesis is false, but because subtle experimental artifacts, evaluation loopholes, or data leakage create illusions of success or uninformative failures. This adversarial review interrogates `CAND_01_CONTROLLED_ENTITY_COPYING` against twelve demanding critique vectors to ensure that results—whether positive or negative—are definitive and impervious to methodological challenge.

---

## 2. Twelve Adversarial Stress-Test Inquiries & Methodological Defenses

### Inquiry 1: Could the model achieve 75% accuracy simply by learning an unanchored "recency bias" (always copying the most recent token)?
- **Adversarial Challenge:**
  In simple single-pair sequences (`PAIR: K=V. QUERY: K -> V`), the value $V$ is often the most recent entity token prior to the query delimiter. A trivial 1-layer attention head could attend exclusively to position $T-3$ and achieve $100\%$ accuracy without learning associative binding.
- **Methodological Defense:**
  Task Family TF-03 (Multi-Pair Association) and TF-04 (Distractor Resistance) randomize the position of the target pair across positions 1, 2, 3, and 4 in context. In TF-04 with 3 distractors, the target pair appears at the final position only $25\%$ of the time. Recency bias yields only $25\%$ accuracy on TF-04. The pre-registered promotion criterion mandates $\ge 60\%$ on TF-04, mathematically disproving recency bias.

### Inquiry 2: Could the model exploit vocabulary frequency artifacts or token ID clustering?
- **Adversarial Challenge:**
  If target value tokens are concentrated in specific BPE frequency bands or numerical ID ranges, the model's output projection bias vector ($b_{\text{out}}$) could memorize this cluster.
- **Methodological Defense:**
  ChakrMicro uses tied input and output embeddings without separate output bias vectors ($W_{\text{out}} = W_{\text{emb}}^T$). Furthermore, target tokens are uniformly sampled across the valid single-token vocabulary range $[3, 4092]$, and the test split uses strictly disjoint token IDs ($\mathcal{E}_{\text{train}} \cap \mathcal{E}_{\text{test}} = \emptyset$).

### Inquiry 3: How do we know single-token entities reflect real associative capability rather than toy memorization?
- **Adversarial Challenge:**
  Real extraction tasks involve natural language strings (e.g. "Mumbai", "Dr. Vikram"). Restricting entities to single synthetic tokens trivializes the task.
- **Methodological Defense:**
  Milestone 5.1 proved that natural language SFT conflated token fragmentation with induction attention failure. CAND_01 is deliberately an isolation experiment. If a 3.44M model cannot associate $1 \to 1$ discrete token mappings, it is mathematically impossible for it to perform multi-token compound induction. Establishing single-token capability is an indispensable prerequisite before re-introducing tokenizer complexity in Track 2.

### Inquiry 4: Does the prompt delimiter (` -> `) act as an artificial trigger that cannot generalize?
- **Adversarial Challenge:**
  The model might overfit to the exact delimiter ` -> ` and fail if a colon or newline is used.
- **Methodological Defense:**
  Standardizing the prompt delimiter is necessary in Confirmatory mode to eliminate syntax parsing variance. The research question is not "can the model parse arbitrary punctuation", but "can the attention heads route information from key to value". Formatting robustness is deferred to Track 4 (Format Robustness).

### Inquiry 5: Could high training loss reduction occur while generation still outputs repetitive gibberish?
- **Adversarial Challenge:**
  In Exp35, training loss dropped from 5.08 to 1.098 while generation scored 0/200. Could CAND_01 repeat this delusion?
- **Methodological Defense:**
  In Exp35, 78.96% of loss tokens were static boilerplate words. In CAND_01 Arm A, 100% of the loss is applied strictly to the target value token ($[VAL] + [EOS]$). There are zero template tokens in the loss function. A drop in training loss directly reflects increased probability on the true target token. Furthermore, the primary metric is greedy autoregressive generation accuracy ($A_{\text{EM}}$), not teacher-forced loss.

### Inquiry 6: Could the evaluator produce false positives through substring matching?
- **Adversarial Challenge:**
  Many benchmark scripts check `if target in completion`. If the model generates a repetitive loop of characters, the target substring may appear by coincidence.
- **Methodological Defense:**
  The CAND_01 evaluator specification strictly forbids substring searching. It requires exact first-token equality (`generated_tokens[0] == target_token_id`) or anchored regex (`^target\b`). A repetition loop generates an immediate failure (`ERR-04`).

### Inquiry 7: Could Rotary Position Embeddings (RoPE) prevent generalizing across context lengths?
- **Adversarial Challenge:**
  RoPE rotates query and key representations based on relative position. If training sequences are short (e.g. 30 tokens), the model may fail completely when queries appear at offset 100 or 200.
- **Methodological Defense:**
  Task Family TF-07 explicitly trains and evaluates positional offsets at 10, 50, 100, and 200 tokens. The distance decay curve is a tracked secondary metric. If accuracy decays rapidly with distance, RoPE base frequency attenuation is immediately identified as the causal mechanism.

### Inquiry 8: Could 500 optimization steps cause catastrophic forgetting of base pretraining representations?
- **Adversarial Challenge:**
  Fine-tuning 3.44M parameters for 500 steps on pure synthetic data could destroy general language representations or cause representation collapse.
- **Methodological Defense:**
  CAND_01 operates on a branch checkpoint; the canonical Step 10,000 checkpoint is read-only and preserved permanently. Furthermore, CAND_01 is an exploratory/confirmatory diagnostic probe of architectural limits, not a final production release model. If representation collapse occurs, it will be detected via secondary attention entropy metrics.

### Inquiry 9: Is a test sample size of 500 records statistically sufficient?
- **Adversarial Challenge:**
  Is $N=500$ large enough to definitively reject $H_0$ without high uncertainty?
- **Methodological Defense:**
  The gap between $H_0$ ($<5\%$) and $H_1$ ($\ge 75\%$) is 70 percentage points. For $N=500$, the 95% confidence interval half-width is $\pm 1.94\%$. The margin of error is over 35 times smaller than the effect size. $N=500$ provides statistical power $> 0.9999$ at $\alpha = 0.001$.

### Inquiry 10: Could key similarity (TF-05) cause attention head saturation?
- **Adversarial Challenge:**
  Keys differing by only 1 character (e.g. `K7P1` vs `K7P2`) may have nearly identical embedding representations, confusing the query projection.
- **Methodological Defense:**
  TF-05 is designed specifically to test this boundary. If the model achieves 90% on TF-02 (distinct keys) but 10% on TF-05 (similar keys), it proves that the model's query-key attention projection lacks fine-grained discriminative resolution.

### Inquiry 11: Could the model exploit cross-example leakage across batches?
- **Adversarial Challenge:**
  If the same key appears across multiple training examples with different values, the model might suffer catastrophic interference or learn a static prior.
- **Methodological Defense:**
  The dataset specification enforces unique key assignment within each batch and strictly disjoint token pools between training and test sets.

### Inquiry 12: What happens if all 4 arms fail to reach 25% accuracy?
- **Adversarial Challenge:**
  If even Arm A fails, does that invalidate the entire ChakrView project?
- **Methodological Defense:**
  No. It definitively answers the scientific question: a standard 6-layer / 192-dimension transformer without recurrent mechanisms or explicit pointer heads cannot perform zero-shot associative token routing at 3.44M parameters. This falsification directly informs architecture design, directing research resources toward hybrid pointer networks or recurrence rather than wasting compute on further fine-tuning.

---

## 3. Residual Risks and Mitigations Summary

| Residual Risk | Severity | Mitigation Strategy |
|---|---|---|
| **Softmax Temperature Sensitivity** | Low | Evaluation is locked to deterministic greedy decoding ($T=0.0$). |
| **Micro-batch Noise** | Medium | Gradient accumulation of 4 steps yields effective batch size of 8, smoothing gradient variance on CPU. |
| **Token ID Prior Imbalance** | Medium | Uniform random sampling of entity tokens from neutral pool $\mathcal{E}$; ablation across disjoint test set. |

