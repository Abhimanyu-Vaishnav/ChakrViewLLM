# Training Experiment Specification (Batches 1621–1660)

## 1. Experiment Overview & Candidate Checkpoint
- **Objective:** Train a newly adapted candidate checkpoint (`chakrmicro_controlled_candidate_step100.pt`) from the canonical step-10,000 checkpoint on the 120-example controlled corpus.
- **Base Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  
  SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Immutable).
- **Output Candidate:** `experiments/batch1621_1660/checkpoints/chakrmicro_controlled_candidate_step100.pt`.
- **Hardware Profile:** CPU-compatible training; dynamic batch sequence packing.

---

## 2. Hyperparameter Grid & Invariants

| Hyperparameter | Value | Rationale |
|:---|:---|:---|
| **Base Model Architecture** | ChakrMicro (1.5M parameters) | Fixed architecture; zero parameter modifications |
| **Tokenizer** | Byte-level BPE (4,096 vocab) | `release/chakrview-v0.5.0/data/experiments/vocab_4096` |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.95$) | Standard technical text pretraining / fine-tuning |
| **Learning Rate** | $5.0 \times 10^{-5}$ | Conservative rate to prevent catastrophic forgetting |
| **Weight Decay** | 0.01 | Modest L2 regularization |
| **LR Schedule** | Linear Warmup (10 steps) + Cosine Decay | Smooth convergence on small corpus |
| **Total Optimizer Steps** | 100 steps | Controlled adaptation regime |
| **Batch Size** | 4 sequences per step | Sequence length capped at 64 tokens |
| **Evaluation Cadence** | Every 10 steps | Independent validation loss tracked |
| **Stopping Rule** | Checkpoint with lowest validation loss | Zero tuning or selection against test set |
| **Fixed Seed** | 42 | Full CPU determinism |
