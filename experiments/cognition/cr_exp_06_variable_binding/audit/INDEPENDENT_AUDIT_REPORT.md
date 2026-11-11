# CHAKRVIEW — CR-EXP-06 INDEPENDENT SCIENTIFIC AUDIT REPORT

- **Audit Date**: 2026-11-06  
- **Auditor Role**: Independent Scientific Auditor  
- **Audit Target**: CR-EXP-06 Targeted Variable-Binding & Relational Curriculum Regularization  
- **Target Repository**: `D:\Project\ChakrView`  
- **Git Branch**: `exp/cbr-01-conversational-breakthrough` | **Git HEAD**: `564472a42245a47a7bf6f78ca6410e523383d2ce`  
- **Execution Mode**: Strictly Read-Only Scientific Audit (0 training steps, 0 model mutations, 0 git changes)

---

## 1. Authoritative Audit Decision

> **`CR-EXP-06 AUDIT PARTIALLY PASSED — LIMITATIONS FOUND`**

### Summary of Audit Verdict:
1. **Numerical Results Reproduce Exact Bit-for-Bit**:
   All reported accuracy scores (Control Arm A: 10.0%, Treatment Arm B: 12.0%, OOD Entities: 0.0%, OOD Templates: 0.0%, Multi-Hop: 0.0%, Conversational Preservation: 58.33% vs. 41.67%) reproduce exactly from raw test generations.
2. **Experimental Equivalence & Loss Implementation Fully Validated**:
   Both training arms initialized from the identical conversational pilot checkpoint (`ea6ee570...`), processed identical data in identical sequence, and executed 150 optimizer steps under identical AdamW schedules. The weighted loss implementation was verified by exact hand calculation ($8 \times 10^{-8}$ numerical discrepancy). Zero dataset contamination exists across OOD splits.
3. **Crucial Scientific Limitation Uncovered (False Positive Variable Binding)**:
   A deep qualitative inspection of the 10 raw hits in Control Arm A and 12 raw hits in Treatment Arm B revealed that **only 2 hits (2.0%)** represented genuine variable binding where the model correctly linked the prompt's subject to the prompt's city. The remaining 8–10 hits were **subject-substitution hallucinations** where the model merely emitted a familiar city string (`"Chennai"`) while substituting an unmentioned training person (e.g. asked about Deepa, the model generated `"Kabir lives in Chennai"`).
4. **Statistical Inconclusiveness of Treatment Weighting**:
   McNemar's paired test comparing Control Arm A (10%) and Treatment Arm B (12%) yielded $\chi^2 = 0.125, p = 0.7237$ with heavily overlapping 95% confidence intervals. The reported 2-point difference is statistically indistinguishable from noise.

---

## 2. Checkpoint Lineage & Integrity Verification (Audit A)

All SHA-256 digests were computed from disk buffers:

| Checkpoint / Artifact | File Path | Actual SHA-256 Digest | Status |
|---|---|---|---|
| **Frozen Conversational Parent** | `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **PASS (Bit-Exact)** |
| **Conversational Pilot Candidate** | `experiments/cognition/autonomous_12h/conversational_pilot/checkpoints/conversational_pilot_best_val.pt` | `ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd` | **PASS (Bit-Exact)** |
| **Control Arm Best Val Checkpoint** | `experiments/cognition/cr_exp_06_variable_binding/control_arm/checkpoints/control_arm_best_val.pt` | `729d5812c1ec02d217ef8cc342453a008ab3046a550e627b16b341c4a458c21f` | **PASS (Bit-Exact)** |
| **Treatment Arm Best Val Checkpoint**| `experiments/cognition/cr_exp_06_variable_binding/treatment_arm/checkpoints/treatment_arm_best_val.pt` | `2d8cea93ba8dbe06e1c714d5563ad89b18600f7a4f766fbb7cfc563c947a9d90` | **PASS (Bit-Exact)** |
| **Canonical Baseline Model Init** | `chakrview/brain/model.py` (seed=42) | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **PASS (Bit-Exact, $\Delta W = 0$)** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **PASS (Bit-Exact)** |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **PASS (Bit-Exact)** |
| **Tokenizer Config** | `data/experiments/vocab_4096/config.json` | `0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589` | **PASS (Bit-Exact)** |
| **Runtime Session** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS (Untouched)** |

Both experimental arms started from the exact identical parent state (`ea6ee570...`). Total model parameter count is 3,443,136 parameters. Weight drift from initialization was bounded ($L_2 = 1.346$ for Control, $1.412$ for Treatment).

---

## 3. Training Equivalence Audit (Audit B)

A parameter-by-parameter audit of configurations, code, and logs verified that:
- **Optimizer**: AdamW (`weight_decay=0.01`, `grad_clip=0.5`).
- **Schedule**: Cosine annealing ($3 \times 10^{-5} \to 5 \times 10^{-6}$ over 150 steps).
- **Batch Size & Data**: 8 samples per step, identical seed (42), identical mini-batch ordering.
- **Total Steps**: Exactly 150 optimizer updates completed per arm.
- **Only Discovered Difference**: Relational target token multiplier was $1.0\times$ in Control Arm A and $5.0\times$ in Treatment Arm B. Full experimental parity confirmed.

---

## 4. Weighted Loss Implementation Audit (Audit C)

Audit of `prepare_batch_tensors()` and loss reduction confirmed:
- Prompt tokens are strictly masked with `-100` and assigned weight $0.0$.
- Marked target entity tokens receive exactly weight $5.0$.
- The denominator strictly equals $\sum w_t$ across active response tokens.
- Hand-calculated synthetic loss matched the PyTorch implementation with an exact discrepancy of **$8 \times 10^{-8}$**, ruling out normalization or span-offset bugs.

---

## 5. Dataset Leakage & Contamination Audit (Audit D)

- **Exact Prompt Overlap**: 0 / 100 in `test_ood_entities`, 0 / 100 in `test_ood_templates`, 0 / 60 in `test_multihop`.
- **Entity Pool Disjointness**: The 12 persons, 10 cities, and 9 objects in the OOD splits share **0% overlap** with the training pool.
- **Structural Similarity vs Contamination**: `test_in_distribution` intentionally shares the entity pool and syntax with train to evaluate in-distribution interpolation, while OOD splits enforce strict disjointness.

---

## 6. Evaluation Reproduction & Qualitative Audit (Audit E & F)

### 6.1 Recalculated Accuracy Parity

| Split / Metric | Reported Control | Audited Control | Reported Treatment | Audited Treatment | Reproduction Status |
|---|---|---|---|---|---|
| **In-Distribution Retrieval** | 10.0% | **10.0%** | 12.0% | **12.0%** | **REPRODUCED** |
| **Strict OOD Entities** | 0.0% | **0.0%** | 0.0% | **0.0%** | **REPRODUCED** |
| **Strict OOD Templates** | 0.0% | **0.0%** | 0.0% | **0.0%** | **REPRODUCED** |
| **Multi-Hop Relational** | 0.0% | **0.0%** | 0.0% | **0.0%** | **REPRODUCED** |
| **Conversational Preservation** | 58.33% | **58.33%** | 41.67% | **41.67%** | **REPRODUCED** |
| **Multi-Turn Horizon** | 0.0% | **0.0%** | 0.0% | **0.0%** | **REPRODUCED** |

### 6.2 Qualitative Finding: Subject-Substitution Hallucination
When auditing the actual generated strings:
- **Control Arm A (10 raw hits)**:
  - Genuine Subject-Value Bindings: **2 / 10 (2.0% of total split)**
  - Subject-Substitution Hallucinations: **8 / 10**
- **Treatment Arm B (12 raw hits)**:
  - Genuine Subject-Value Bindings: **2 / 12 (2.0% of total split)**
  - Subject-Substitution Hallucinations: **10 / 12**
- **Scientific Significance**: The model did not learn dynamic variable binding. Rather, upon recognizing the query cadence `"Where does X live?"`, it emitted familiar city tokens (`"Chennai"`, `"Pune"`) paired with whichever subject token happened to be statically salient in its memory (`"Kabir"`, `"Priya"`).

### 6.3 Statistical Reassessment
- **Retrieval Comparison**: McNemar's paired test yields $\chi^2 = 0.125, p = 0.7237$. The difference between 10% and 12% is statistically insignificant.
- **Conversational Degradation**: The 58.33% vs 41.67% gap corresponds to a difference of only 2 probes on an $n=12$ sample, with overlapping 95% Wilson confidence intervals ([31.95%, 80.67%] vs [19.33%, 68.05%]).

---

## 7. Reassessment of Scientific Hypotheses & Claims

1. **Did CR-EXP-06 demonstrate genuine variable binding?**  
   **No.** True zero-shot variable binding requires extracting the value bound to the specific key in the prompt. Scoring 0.0% on OOD entities and achieving only 2% genuine subject-value binding on familiar entities proves that genuine variable binding was not acquired.
2. **Could familiar retrieval gains be memorization?**  
   **Yes.** The 10–12% raw hits represent surface pattern matching and static co-occurrence memorization from training data.
3. **Is standard SFT insufficient for multi-hop reasoning in a 3.44M Transformer?**  
   **Yes.** Across 60 multi-hop probes, accuracy was exactly 0.0%. Unaugmented autoregressive attention in a compact dense model cannot compose multi-step relational hops.
4. **Is dedicated architectural routing necessary?**  
   **Supported.** The evidence from CR-EXP-03, CR-EXP-05, and CR-EXP-06 indicates that standard dense Transformer attention suffers from residual stream dilution and lack of key-value routing. Architectural enhancements (such as pointer-augmented attention or explicit relational memory circuits) appear strictly necessary for reliable variable binding.
5. **Is persona retention more than exact phrase reproduction?**  
   **Partially.** The model reliably reproduces its self-introduction, but falls into syntax hallucinations when extended beyond 25 tokens.

---

## 8. Final Audit Sign-Off

The audit concludes that CR-EXP-06 was executed with high experimental rigor and complete reproducibility of numerical metrics. However, scientific claims of "relational retrieval improvement" must be qualified: the model learned an answering cadence and memorized static entity patterns, but failed genuine variable binding and out-of-distribution transfer.

