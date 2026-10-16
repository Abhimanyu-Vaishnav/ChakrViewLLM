# Experiment Protocol: Batches 1381–1420

## Controlled Domain Adaptation, Overfitting Detection and Checkpoint Validation

### 1. Objective and Scientific Question

Can a controlled, low-learning-rate continual domain adaptation run on `ChakrMicro` improve held-out technical language modeling without:
1. Overfitting the tiny 3,212-token training corpus?
2. Inducing catastrophic forgetting on baseline pretraining capabilities?
3. Degenerating into repetitive generation loops?

---

### 2. Hypotheses and Success Criteria

- **Hypothesis $H_1$:** Fine-tuning on 25 training-split technical documents with low learning rate ($\eta = 5\times 10^{-5}$) and cosine decay will reduce held-out validation cross-entropy from $5.0630$ to $< 4.7000$ (Perplexity from $158.07$ to $< 110.0$).
- **Null Hypothesis $H_0$:** Due to the extremely small training volume (3,212 tokens), training will rapidly overfit (training loss collapses while validation loss rises or plateaus), or the model will suffer catastrophic forgetting.
- **Overfitting Warning Criterion:** Any step where training loss drops by $> 1.0$ nats while validation loss fails to decrease or increases by $> 0.10$ nats.
- **Early-Stopping Rule:** Evaluate validation loss every 10 steps. If validation loss fails to improve for 4 consecutive evaluations (patience = 40 steps), halt training immediately.

---

### 3. Checkpoint & Lineage Safety Invariants

1. **Canonical Checkpoint Immutability:**
   - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
   - Canonical SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
   - **MUST NEVER BE MODIFIED OR OVERWRITTEN.**
2. **Experimental Candidate Destination:**
   - Path: `experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt`
3. **Strict Split Isolation:**
   - The 27 held-out test documents (3,456 target tokens) are strictly reserved for final held-out evaluation.
   - The 25 training documents are split into:
     - 20 training documents (~2,560 tokens)
     - 5 validation / early-stopping documents (~650 tokens)
   - Zero test data is used for gradients, validation loss, or checkpoint selection.
4. **Protected Workspace Code:**
   - `chakrview/runtime/session.py` remains unstaged and untracked.
