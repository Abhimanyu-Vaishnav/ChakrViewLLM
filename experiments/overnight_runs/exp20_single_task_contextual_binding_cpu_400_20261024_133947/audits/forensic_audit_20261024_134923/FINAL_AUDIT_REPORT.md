# ChakrView Exp20 — Final Forensic Audit & Verification Report

**Audit Identifier**: `forensic_audit_20261024_134923`  
**Target Experiment**: `exp20_single_task_contextual_binding_cpu_400_20261024_133947`  
**Repository**: `D:\Project\ChakrView`  
**Git Commit**: `f30136a5dc02934180a503fdf0a4c287469bf106`  
**Audit Directory**: `experiments/overnight_runs/exp20_single_task_contextual_binding_cpu_400_20261024_133947/audits/forensic_audit_20261024_134923/`  

---

## 1. Executive Verdict: `AUDIT_VERIFIED`

The reported results of **Exp20** (`exp20_single_task_contextual_binding_cpu_400`) have been independently reproduced and verified with complete cryptographic and empirical fidelity.

- **Integrity**: 14/14 cryptographic hash checks passed. All 9 canonical protected repository assets (step 10,000 checkpoint, vocab, merges, `session.py`, `model.py`, Exp17 splits) remain bit-exact and untouched.
- **Data Isolation**: 0 entity overlap between train, validation, and test splits. Test set parity with Exp19 confirmed 20/20 field-by-field.
- **Evaluation Reproduction**: 20/20 test completions reproduced bit-for-bit using `sft_best_checkpoint.pt`. Exact task accuracy is **0/20 (0.0%)**, and EOS completion rate is **20/20 (100.0%)**.
- **Evaluator Soundness**: Zero correct answers were falsely rejected. The evaluator operates strictly as designed.
- **Discrepancy Resolved**: The prose reference to `"384 embedding dim"` in `EXP20_COMPARISON.md` was a typographical error; checkpoint weights and configuration prove that canonical ChakrMicro (`d_model = 192`, 3,443,136 parameters) was strictly instantiated and trained.
- **Regression Suite**: 126/126 unit tests passed in 16.99 seconds.

---

## 2. Summary Table of Verification Checks

| Verification Check | Target / Asset | Method | Reported Result | Audited Result | Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Canonical Checkpoint Hash** | `training_checkpoint_step_10000.pt` | SHA-256 | `96357705...` | `96357705...` | **PASS** |
| **Architecture Seed 42 Digest** | Parameter byte hashing | Python simulation | `c5571c9c...` | `c5571c9c...` | **PASS** |
| **Protected Code & Data** | `session.py`, `model.py`, Exp17 splits | SHA-256 | Bit-exact | Bit-exact | **PASS** |
| **Split Isolation** | Train (400), Val (50), Test (20) | Entity string intersection | 0 overlap | 0 overlap | **PASS** |
| **Test Set Parity vs Exp19** | 20 context extraction records | Field-by-field equality | 20 / 20 match | 20 / 20 match | **PASS** |
| **Optimizer Steps** | Training log & checkpoint metadata | Log parsing & state inspect | Exactly 400 steps | Exactly 400 steps | **PASS** |
| **Best Step & Val Loss** | Checkpoint state dictionary | Direct tensor inspection | Step 240 (1.7447) | Step 240 (1.7447) | **PASS** |
| **Exact Task Accuracy** | 20 frozen held-out test records | Deterministic re-inference | 0.0% (0/20) | 0.0% (0/20) | **PASS** |
| **EOS Completion Rate** | 20 frozen held-out test records | Deterministic re-inference | 100.0% (20/20) | 100.0% (20/20) | **PASS** |
| **Syntax Frame Accuracy** | 20 frozen held-out test records | Syntax string parsing | 95.0% (19/20) | 95.0% (19/20) | **PASS** |
| **Regression Test Suite** | Full repository test suite | `unittest` runner | 126 / 126 pass | 126 / 126 pass | **PASS** |
| **Architecture Width Check** | Instantiated tensor parameters | Weight tensor shapes | Discrepant in text | `d_model=192` verified | **RESOLVED** |

---

## 3. Confirmed Observations vs Unresolved Hypotheses

### Confirmed Scientific Observations
1. **Task Isolation Eliminates Surface Monoculture**: Isolating training to `context_extraction` drove EOS termination to 100% and reduced CPU query latency to 108.62 ms, with 95% syntax frame compliance.
2. **0/20 Exact Accuracy Is Factual**: On novel held-out entities, the model scored 0/20 across all evaluation runs.
3. **Failure Mechanism Is Entity Hallucination / Memorization**: The model failed because it filled the name slot using blended tokens from the 50 training names (`'Hanna Lin'`, `'Handry Lindry'`), rather than copying the name from the context.

### Hypotheses Requiring Discriminating Tests
1. **Capacity Limit Hypothesis**: Does ChakrMicro (3.44M parameters) inherently lack the parameter capacity to form induction copy circuits?
2. **Context-Format Hypothesis**: Does the prompt format `Context: ...\nQuestion: ...\nAnswer:` present the entity too early in the context window for single-head attention copying?
3. **Familiarity vs Novelty Hypothesis**: Can ChakrMicro copy entities if the entity vocabulary is shared between training and test (testing the copy mechanism in isolation from OOD token generalization)?

---

## 4. Recommended Next Experiment: Exp21 Proposal

**Title**: `exp21_synthetic_contextual_copy_probe_cpu`  
**Nature**: Controlled Diagnostic Probe (No external architecture changes yet).

### Research Question
Can ChakrMicro learn an exact copy-paste attention circuit on a synthetic single-token key-value extraction task where tokens are familiar, or is the 3.44M architecture incapable of induction-head copying even in a simplified environment?

### Experimental Design
- **Single Intervention**: Simplify the extraction target to single-token alphanumeric identifiers (e.g. `System Alpha assigned code X-4`).
- **Data Splits**:
  - Training: 300 synthetic single-token context-key extraction pairs.
  - Test Set A (Familiar tokens, novel pairings): 20 records.
  - Test Set B (Completely novel held-out tokens): 20 records.
- **Primary Metric**: Exact token copy accuracy. Success threshold: $\ge 50\%$ on Set A.
- **Resource Estimate**: ~20 seconds on CPU (300 steps).
- **Discriminating Value**:
  - If Set A passes and Set B fails $\to$ Tokenization / embedding generalization is the bottleneck.
  - If Set A fails $\to$ Capacity limitation is conclusively proven; scaling to ChakrMini (~15M parameters) becomes mathematically and empirically mandatory.

*Approval Notice: This proposal is documented for user review. No training or canonical modification has been initiated.*
