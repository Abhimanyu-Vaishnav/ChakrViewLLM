# Research Experiment Specification: Replay-Regularized Multi-Task Pilot with Frozen Stage C Reference Anchor

**Document Version:** 1.0.0  
**Status:** PROPOSED EXPERIMENT SPECIFICATION (PREPARED ONLY — NOT AUTHORIZED FOR EXECUTION)  
**Training Gate:** `TRAINING READINESS: NOT READY FOR TRAINING` ($\Delta W = 0$)  
**Target Model:** ChakrMicro ($L=6, d=384, H=6, V=4096$, 3,443,136 parameters)  
**Reference Frozen Baseline:** Stage C 9k Checkpoint (`checkpoint_0009000.pt`, SHA-256: `3e03ae0a134da40439038d965278f4da2996b1ac2e8f959d6059859600b0bb23`)

---

## 1. Research Motivation & Core Hypothesis

### 1.1 Problem Statement
The Stage D 500-step pilot checkpoint demonstrated catastrophic mode collapse: when trained purely on arithmetic scratchpads without general language retention regularization, the model lost Stage C's diverse grammatical representations and collapsed into emitting a single arithmetic template across all prompt domains (`The answer is 200.`, `So he has 200*2=<<2*2=200>>12`).

### 1.2 Core Research Hypothesis
*Hypothesis:* Jointly optimizing an active student model on high-quality, founder-authorized QA records while regularizing output distributions against a permanently frozen Stage C reference model (via output-token KL-divergence anchor loss) alongside a 20% general language replay buffer will preserve base syntactic fluency and mitigate template mode collapse, while achieving measurable objective accuracy on domain questions.

### 1.3 Measurable Success Criteria
1. **Zero Template Mode Collapse:** Output template repetition rate across non-arithmetic prompts falls from $>80\%$ (Stage D baseline) to $<10\%$.
2. **Lexical Diversity Preservation:** Long-horizon unique-token ratio remains $\ge 0.35$ and 4-gram repetition ratio stays $\le 0.40$ on the open continuation benchmark.
3. **Target Task Accuracy:** Evaluated on the benchmark suite v2.1, deterministic accuracy reaches $\ge 15\%$ on factual extraction / arithmetic tasks, compared to $0.0\%$ currently.
4. **Perplexity Invariant:** Perplexity on the held-out Stage C validation set increases by no more than $+15\%$ relative to frozen Stage C reference perplexity.

---

## 2. Data Admission Prerequisites & Split Isolation

### 2.1 Founder Review Admission Gate (Mandatory Precondition)
No training experiment may commence until:
- The 30 Stage D QA candidates in `checkpoints/experiments/neural_learning_v02/stage_d_design/qa_review_queue_v0.3.0.jsonl` are reviewed and explicitly approved by the Founder using `tools/qa_review/compile_reviewed_dataset.py`.
- Admission manifest is compiled with zero validation/test split leakage.
- Independent cryptographic verification confirms the official training dataset contains only founder-approved items.

### 2.2 Split Isolation & Contamination Defense
- **QA Dataset:**
  - Training split: strictly records from the approved training partition.
  - Validation split: strictly from `stage_d_qa_pilot_manifest.json` val partition (72 items).
  - Test split: strictly quarantined (68 items); never seen by optimizer or loss calculation.
- **Replay Dataset:**
  - Sampled strictly from the pre-tokenized Stage C training shard (`data/manifests/stage_c_manifest.json`).
  - Held-out Stage C validation lines are isolated to measure catastrophic forgetting.

### 2.3 Replay Proportion Justification
- **Proposed Ratio:** $80\%$ task data (QA tokens) / $20\%$ replay data (Stage C general pretraining tokens).
- **Justification:** Prior literature in low-capacity neural continual learning indicates that below $10\%$ replay, parameter drift and language degeneration occur rapidly in small models (<10M params); above $35\%$ replay, the model under-fits target prompt formats. The $20\%$ ratio is treated as an empirical hypothesis to be validated against $0\%$ and $30\%$ controls.

---

## 3. Mathematical Objective: KL Anchor Regularization

Let $\theta$ denote the parameters of the trainable student model, and let $\theta_{\text{ref}}$ denote the permanently frozen parameters of the Stage C 9k reference model.

For each token step $t$ in a sequence:
$$p_{\theta}(y_t \mid x_{<t}) = \text{softmax}(z_t / T)$$
$$q_{\text{ref}}(y_t \mid x_{<t}) = \text{softmax}(z_t^{\text{ref}} / T)$$

The total loss per batch is formulated as:
$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{task-NLL}}(\theta) + \alpha \cdot \mathcal{L}_{\text{replay-NLL}}(\theta) + \beta \cdot D_{\text{KL}}\left( q_{\text{ref}}(\cdot \mid x) \,\|\, p_{\theta}(\cdot \mid x) \right)$$

Where:
- $\mathcal{L}_{\text{task-NLL}}$ is the cross-entropy loss over authorized QA target tokens.
- $\mathcal{L}_{\text{replay-NLL}}$ is the cross-entropy loss over general language replay tokens ($\alpha = 0.25$).
- $D_{\text{KL}}$ is the Kullback-Leibler divergence between the frozen reference logits and student logits over non-special vocabulary tokens.
- $\beta$ is the regularization anchor weight (default pilot: $\beta = 0.10$).
- **Reference Model Freezing:** $\theta_{\text{ref}}$ is set to `eval()` mode with `requires_grad = False` and checked via weight hash assertions before every epoch.

---

## 4. Controlled Experimental Matrix & Ablations

| Run ID | Name | QA Data | Replay Stream | Stage C KL Anchor ($\beta$) | Purpose |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **RUN-A0** | Unregularized Control | Approved QA (100%) | None (0%) | 0.0 | Replicate Stage D failure mode under approved data |
| **RUN-A1** | Replay-Only Ablation | Approved QA (80%) | Stage C (20%) | 0.0 | Measure effect of token replay without logit anchoring |
| **RUN-A2** | Anchor-Only Ablation | Approved QA (100%) | None (0%) | 0.10 | Measure effect of logit anchoring without replay tokens |
| **RUN-EXP** | Joint Multi-Task (Proposed) | Approved QA (80%) | Stage C (20%) | 0.10 | Full regularized continual learning architecture |

---

## 5. Checkpoint Lifecycle, Budget & Rollback Strategy

1. **Compute Budget:**
   - 1,000 steps max per run.
   - Batch size = 8, sequence length = 128.
   - CPU estimated runtime: $\sim 15$ minutes per run on 4 threads; GPU (if enabled): $<2$ minutes.
2. **Checkpoint Schedule:**
   - Saved at steps 250, 500, 750, 1000.
   - Checkpoint format: includes `model_state_dict`, `optimizer_state_dict`, `model_hash`, `reference_model_hash`, `step`, `loss_components`.
3. **Rollback & Stop Conditions (Circuit Breakers):**
   - **Degeneration Breaker:** If unique token ratio on validation prompts drops below $0.20$ at any evaluation checkpoint, abort run immediately.
   - **Explosion Breaker:** If $D_{\text{KL}} > 5.0$ or training loss exhibits NaN/Inf, trigger immediate rollback to previous checkpoint.
   - **Negative Learning Breaker:** If Stage C validation loss exceeds $+30\%$ relative to baseline, abort run.

---

## 6. Training Readiness Gate
- **Status:** `TRAINING READINESS: NOT READY FOR TRAINING`
- **Execution Authorized:** `FALSE`
- This specification establishes scientific readiness. Actual training execution requires explicit founder authorization following formal review of dataset candidates.
