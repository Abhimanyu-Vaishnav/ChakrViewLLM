# ChakrView Exp17 Validation — Next-Experiment Decision & Recommendation

**Auditor**: Senior ML Systems Engineer & Experiment Auditor  
**Date**: 2026-10-23  

---

## 1. Primary Recommendation

**`RECOMMENDED PRIMARY ACTION: B. IMPROVE TASK-LEVEL EVALUATION AND SCORING BEFORE ANY FURTHER TRAINING`**

---

## 2. Evidence Supporting the Decision

1. **The Evaluation Metric Blind Spot**:
   - The primary metric used to celebrate Exp17 was **clean turn termination (`EOS%` rate)**, which jumped from **0% to 45%** on the held-out test split and **5% to 60%** on Normal Chat Gate v0.1.
   - However, our independent task-level re-evaluation revealed that **actual task correctness remained 0.0%**. In `structured_formatting`, 100% of outputs emitted EOS, but 0% were valid JSON. In arithmetic, the model stopped cleanly but hallucinated incorrect sums.
   - Proceeding to further training (e.g. Exp18 or Exp17 v2) without automated, task-level semantic grading scripts will cause future runs to optimize blindly for early EOS emission (e.g. emitting EOS immediately) rather than task fulfillment.

2. **The Dataset is Structurally Sound**:
   - The 400-example official dataset (`train_exp17_v1.jsonl`, `val_exp17_v1.jsonl`, `test_exp17_v1.jsonl`) passed all structural, schema, tokenization, absence-proof, and split-leakage gates.
   - The issue is not that the training data is broken, but that the evaluation pipeline lacks automated semantic scorers.

---

## 3. Scope of Work for the Recommended Phase

### Phase 1: Implement Category-Specific Automatic Scorers
Create an automated offline evaluation module `chakrview/evaluation/task_evaluator.py` that evaluates:
1. `structured_formatting`: Validates RFC 8259 syntax via `json.loads` and verifies required schema keys.
2. `basic_arithmetic`: Extracts generated numeric tokens and checks numerical equality with expected values.
3. `uncertainty_abstention`: Evaluates refusal detection on unanswerables and exact fact extraction on controls.
4. `instruction_following`: Validates case conversions, string reversals, and word count constraints.

### Phase 2: Define Dual-Score Gate Criteria
Establish that future SFT experiments must meet **two separate threshold gates**:
- **Turn Boundary Gate**: EOS Rate $\ge 50\%$.
- **Semantic Task Accuracy Gate**: Task Correctness $\ge 25\%$ on in-distribution test prompts.

---

## 4. Secondary Next Actions (Deferred)

- **Secondary Action C (Curriculum Loss Weighting)**: After automated task scoring is in place, introduce weighted cross-entropy loss in `sft_engine.py` to boost gradient signal on complex categories (`context_extraction` and `instruction_following`).
- **Secondary Action D (Decoding Exploration)**: Test if min-token constraints (`min_new_tokens=5`) prevent the model from emitting premature EOS tokens on structured formatting prompts.

---

## 5. Architectural & Canonical Invariants

During the next phase, the following must remain frozen:
- Canonical base checkpoint (`experiments/.../training_checkpoint_step_10000.pt`)
- Canonical tokenizer (`vocab.json`, `merges.json`)
- Protected runtime (`chakrview/runtime/session.py`)
- Protected model architecture (`chakrview/brain/model.py`)
- Official Exp17 dataset files (`data/train_exp17_v1.jsonl`, `val_exp17_v1.jsonl`, `test_exp17_v1.jsonl`)
