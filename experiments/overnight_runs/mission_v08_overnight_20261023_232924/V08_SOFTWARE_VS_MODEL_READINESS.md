# CHAKRVIEW v0.8: SOFTWARE READINESS VS MODEL READINESS

**Assessment Date**: 2026-10-24  
**Target Release Candidate**: ChakrView v0.8.0-rc1  
**Authoritative Evaluator**: TaskEvaluator v0.1 (`chakrview/evaluation/task_evaluator.py`)  

---

## 1. Core Principle: Decoupling Software Scope from Experimental Weights

A foundational release principle of the ChakrView architecture is that software runtime components, evaluation harnesses, and dataset generators are evaluated on their own engineering merits, independent of whether experimental neural fine-tuning runs have converged to state-of-the-art capability.

| Component Track | Status | Verdict | Release Readiness |
| :--- | :---: | :---: | :--- |
| **ChakrView v0.8 Software Runtime & Core** | **PASS** | **READY** | Full software candidate ready for packaging / release. |
| **Experimental SFT Checkpoints (`sft_best`)**| **HOLD** | **EXCLUDED** | Do NOT promote or release. Retain canonical `step_10000.pt`. |
| **Exp18 Official Datasets** | **HOLD** | **EXCLUDED** | Retain in-memory / quarantine until 190 items reviewed. |

---

## 2. Software Readiness Audit (100% Verified)

The software components of the repository meet all v0.8 release gates:
1. **Repository Test Suite**:
   - 123 / 123 tests pass across `tests/` (`python -m unittest discover -s tests`).
   - 17 / 17 tests pass in `tests/test_task_evaluator.py`.
   - 15 / 15 tests pass in `test_exp18_generator.py`.
2. **Dedicated Neural Runtime Adapter**:
   - Verified CPU execution with strict cold-start caching.
   - Pre- and post-execution Delta-W = 0 verification guarantees weight immutability.
   - Truthful metadata extraction without credential exposure.
3. **TaskEvaluator v0.1**:
   - Corrected legacy hardcoded entity heuristics in `_score_factual_qa`.
   - 100% precision on expected ground-truth answers across all 1,330 scorable records.
4. **Protected Artifact Integrity**:
   - All 8 protected assets bitwise match their frozen SHA-256 hashes.
   - Zero modifications to `session.py`, `model.py`, `step_10000.pt`, or Exp17 datasets.

---

## 3. Model Weight Readiness Audit (EXCLUDED)

The experimental SFT weights generated in `sft_exp18_run` are **NOT** ready for release:
1. **Task Accuracy Failure**: Scorable task accuracy is 1.43% (2/140) compared to the baseline's 2.14% (3/140).
2. **Structural Formatting**: 0% valid RFC 8259 JSON outputs under the 45-token generation limit.
3. **Training Conditioning Defect**: The training pipeline did not apply the prompt conditioning template, producing a format distribution shift at inference.

**Rule**: Canonical pretrained weights (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`) remain the authoritative and exclusive weights for ChakrView v0.8.

---

## 4. Final Recommendation

Tag and stage **ChakrView v0.8** strictly as:
$$\mathbf{V08\_SOFTWARE\_RELEASE\_CANDIDATE\_READY}$$
with the explicit release note that neural weights remain frozen at canonical Step 10,000 baseline, while experimental conversation SFT is deferred to v0.9 upon completion of the training prompt-conditioning alignment and human semantic dataset review.
