# CHAKRVIEW — CAND_01 ADVERSARIAL DESIGN REVIEW v1.2 (REVIEWED)
## Critical Stress-Testing, Methodological Defenses, and Vulnerability Analysis

**Document ID:** `REV-EXP-CAND01-ADV-003`  
**Location:** `docs/experiments/CAND_01_ADVERSARIAL_DESIGN_REVIEW_v1.2_REVIEWED.md`  
**Supersedes:** `docs/experiments/CAND_01_ADVERSARIAL_DESIGN_REVIEW_v1.1_REVIEWED.md`, `docs/experiments/CAND_01_ADVERSARIAL_DESIGN_REVIEW_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Review Type:** ADVERSARIAL SCIENTIFIC AND METHODOLOGICAL STRESS TEST  
**Reviewer Role:** Independent Skeptical ML Peer Reviewer & Statistical Auditor  
**Status:** `REVIEW_COMPLETED`  
**Execution Authorization:** **`UNAPPROVED`**  
**Date:** 2026-10-31  

---

## 1. Adversarial Purpose & Mandate (v1.2)

Scientific experiments often fail not because the central hypothesis is false, but because subtle experimental artifacts, evaluation loopholes, or data leakage create illusions of capability or uninformative failures. This adversarial review interrogates `CAND_01_CONTROLLED_ENTITY_COPYING` against twelve demanding critique vectors, incorporating the independent audit findings to ensure that results—whether positive or negative—are definitive and impervious to challenge.

---

## 2. Twelve Adversarial Inquiries & Methodological Defenses

### Inquiry 1: Could the model achieve 75% accuracy simply by learning an unanchored "recency bias" (always copying the most recent token)?
- **Adversarial Challenge:**
  In single-pair sequences (`PAIR: K=V. QUERY: K -> V`), the value $V$ is often the most recent entity token prior to the query delimiter. A trivial 1-layer attention head could attend exclusively to position $T-3$ and achieve $100\%$ accuracy without learning associative binding.
- **Methodological Defense:**
  Task Family TF-03 (Multi-Pair Association) and TF-04 (Distractor Resistance) randomize the position of the target pair across positions 1, 2, 3, and 4 in context. In TF-04 with 3 distractors, the target pair appears at the final position only $25\%$ of the time. Recency bias yields only $25\%$ accuracy on TF-04. The pre-registered promotion criterion mandates $\ge 60\%$ on TF-04, mathematically disproving recency bias.

### Inquiry 2: Could the model exploit vocabulary frequency artifacts or token ID clustering?
- **Adversarial Challenge:**
  If target value tokens are concentrated in specific BPE frequency bands or numerical ID ranges, the model's output projection bias vector ($b_{\text{out}}$) could memorize this cluster.
- **Methodological Defense:**
  ChakrMicro uses tied input and output embeddings without separate output bias vectors ($W_{\text{out}} = W_{\text{emb}}^T$). Furthermore, target tokens are stratified across log-frequency tiers (Resolution F-06), and the OOD test split uses strictly disjoint token IDs ($\mathcal{E}_{\text{train}} \cap \mathcal{E}_{\text{test}} = \emptyset$).

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
  In Exp35, 78.96% of loss tokens were static boilerplate words. In CAND_01 Arm A, 100% of the loss is applied strictly to the target value token ($[VAL] + [EOS]$). There are zero template tokens in the loss function. Furthermore, the evaluation harness mandates both $A_{\text{FT}} \ge 75.0\%$ and clean sequence completion $A_{\text{FC}} \ge 70.0\%$ (Resolution F-05); repetitive generation after token 0 scores $0.0$ on $A_{\text{FC}}$ and triggers `ERR-04`.

### Inquiry 6: Could the evaluator produce false positives through substring matching?
- **Adversarial Challenge:**
  Many benchmark scripts check `if target in completion`. If the model generates a repetitive loop of characters, the target substring may appear by coincidence.
- **Methodological Defense:**
  The CAND_01 evaluator specification strictly forbids substring searching. It requires exact first-token equality (`generated_tokens[0] == target_token_id`) and full completion validation ($A_{\text{FC}}$). Repetition loops or extraneous generation are detected and scored as failures.

### Inquiry 7: Could Rotary Position Embeddings (RoPE) prevent generalizing across context lengths?
- **Adversarial Challenge:**
  RoPE rotates query and key representations based on relative position. If training sequences are short (e.g. 30 tokens), the model may fail completely when queries appear at offset 100 or 200.
- **Methodological Defense:**
  Task Family TF-07 explicitly trains and evaluates positional offsets at 10, 50, 100, and 200 tokens. The distance decay curve is a tracked secondary metric. If accuracy decays rapidly with distance, RoPE base frequency attenuation is immediately identified as the causal mechanism.

### Inquiry 8: Could 500 optimization steps cause catastrophic forgetting of base pretraining representations?
- **Adversarial Challenge:**
  Fine-tuning 3.44M parameters for 500 steps on pure synthetic data could destroy general language representations or cause representation collapse.
- **Methodological Defense:**
  CAND_01 operates on a branch checkpoint; the canonical Step 10,000 checkpoint is read-only and preserved permanently. Furthermore, CAND_01 incorporates canonical non-inferiority regression gates on pretraining validation loss, syntax adherence, and repetition stability (Protocol v1.0 Section 6).

### Inquiry 9: Why does the statistical plan now mandate clustered standard errors and cluster bootstrap?
- **Adversarial Challenge:**
  The audit found that $N=500$ evaluation instances are generated from only 150 unique test token IDs. If the model fails on a specific token embedding, that error repeats across multiple prompts. Doesn't standard binomial testing grossly exaggerate statistical confidence?
- **Methodological Defense:**
  Yes. In v1.2, this vulnerability is formally closed (Resolution F-02). The evaluator script is required to compute clustered standard errors and a $B=2,000$ cluster bootstrap resampled at the `target_val_token_id` level, providing honest confidence intervals that reflect token-level variance.

### Inquiry 10: How does v1.2 resolve the validation vs. test split asymmetry?
- **Adversarial Challenge:**
  In v1.1, validation examples used seen tokens recombined, while test examples used disjoint unseen tokens. Early stopping on validation would overfit to recombination and fail on OOD tokens.
- **Methodological Defense:**
  v1.2 explicitly separates evaluation into a 4-partition architecture (Resolution F-01). The benchmark evaluates and reports Partition C (Test Recombination: $A_{\text{test,recomb}}$) and Partition D (Test Disjoint OOD: $A_{\text{test,ood}}$) as separate metrics. Validation loss on Partition B is never claimed as an estimator of OOD performance.

### Inquiry 11: Could target-only loss with learning rate $10^{-4}$ cause immediate gradient explosion?
- **Adversarial Challenge:**
  Concentrating 100% of loss on a single token position without template dilution increases per-token gradient magnitude by $>4\times$. Standard AdamW at $10^{-4}$ may cause numerical instability.
- **Methodological Defense:**
  v1.2 classifies the learning rate strictly as `PROVISIONAL` (Resolution F-04) and specifies a separately authorized 20-step pre-flight gradient telemetry check on a disposable copy before launching confirmatory runs.

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
| **Token ID Clustering** | Medium | Uniform log-frequency stratification; cluster bootstrap grouped by token ID. |
| **Pre-Flight LR Divergence** | Medium | 20-step gradient norm telemetry check on disposable model copy prior to full training. |

---

## 4. Adversarial Review Conclusion & Status

- **Review Recommendation:** **`READY_FOR_PRE_EXECUTION_REVIEW`**
- **Documentation Assessment:** All identified methodological vulnerabilities, cluster dependence issues, split asymmetries, and overclaims have been resolved in the v1.2 specification suite.
- **Execution Mandate:** This review assesses scientific design documentation only. **Execution authorization remains strictly UNAPPROVED.**

