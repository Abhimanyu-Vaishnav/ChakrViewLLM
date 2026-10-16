# Experiment Protocol: Batches 1421–1460

## Independent Validation of Domain Adaptation and Generalization

### 1. Mission Mandate and Objective

In Batch 1381–1420, a controlled 80-step domain adaptation of `ChakrMicro` on 21 technical documents demonstrated a marked reduction in cross-entropy loss (5.0630 to 4.3371) and perplexity (158.07 to 76.49) on a 27-document held-out test split.

The primary objective of Batches 1421–1460 is to subject these findings to **rigorous independent validation**:
1. Audit the previous experiment's evaluation harness, tokenizer alignment, and data isolation.
2. Build an **independent, multi-domain evaluation corpus** completely unexposed in previous training or test splits, spanning:
   - Operating Systems & Hardware Architecture.
   - Python & Distributed Systems Engineering.
   - Distributed Consensus & Database Internals.
   - General Technical Explanatory Prose.
   - Mathematical Formulations & Numerical Statements.
3. Evaluate BOTH the canonical baseline and the adapted candidate on this new independent corpus to determine if improvements generalize across unseen technical domains.
4. Conduct seed replication experiments starting from identical base weights and permitted data to quantify optimization variance.
5. Execute systematic qualitative generation probes across multi-style prompts to audit repetition, syntax validity, and potential forgetting.

---

### 2. Invariants and Guardrails

1. **Checkpoint Preservation:**
   - Canonical Base: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (SHA-256: `96357705a6...`) - READ ONLY.
   - Candidate: `experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt` (SHA-256: `15d48176...`) - READ ONLY.
2. **Strict Test Corpus Isolation:**
   - The new independent evaluation corpus must never be used for gradient computation, hyperparameter selection, or checkpoint selection.
3. **Deterministic Evaluation:**
   - Identical context length, causal masking, tokenizer, and token target alignment.
4. **Workspace Protection:**
   - `chakrview/runtime/session.py` remains unstaged and untracked.
