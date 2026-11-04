# CHAKRVIEW OVERNIGHT TRAINING EXPERIMENTS & LINEAGE
## Comprehensive Record of Controlled SFT Experiments & Checkpoints

---

### 1. Research Lineage Traceability

The conversational lineage of ChakrView follows this strictly tracked trajectory:

```
Canonical Pretraining (Step 10,000)
  SHA-256: 96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee
    │
    ▼
CBR-SFT-02 (Conversational Initialization)
    │
    ▼
CBR-SFT-03B (Greeting & Identity Balancing)
    │
    ▼
CBR-SFT-04 (Conversational Grounding, Step 200)
  SHA-256: 327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153
    │
    ▼
CBR-SFT-06A (Refusal Attractor Mitigation, Step 25)
  SHA-256: 0589fd886853a2e65ffa1c03af0e4708cb7dc3dfd40b4db6ccc179c5bf1045f1
    │
    ▼
CBR-SFT-07 (Overnight Conversational Breakthrough, Step 50 [SELECTED])
  SHA-256: 0cb5df5fa840d20cea737da47758d96cfe884782c9801f28e51660bb236de86b
```

---

### 2. Controlled Experiment Specifications

| Parameter | Configuration |
| :--- | :--- |
| **Model** | ChakrMicro (3,443,136 parameters, 6 layers, 6 heads, $d_{model}=192$, tied embeddings) |
| **Parent Checkpoint** | `experiments/cbr01_sft/runs/cbr_sft_06a/checkpoints/sft_checkpoint_step_25.pt` |
| **Parent SHA-256** | `0589fd886853a2e65ffa1c03af0e4708cb7dc3dfd40b4db6ccc179c5bf1045f1` |
| **Training Steps** | 200 steps |
| **Micro-batch Size** | 2 |
| **Gradient Accumulation** | 2 (Effective batch size = 4) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay = 0.01) |
| **Learning Rate** | $8.0 \times 10^{-5}$ decaying cosine to $1.0 \times 10^{-5}$ |
| **Gradient Clipping** | 1.0 |
| **Hardware** | Pure CPU execution |
| **Dataset** | `data/sft/cbr_sft_07/splits/cbr_sft_07_train_228.jsonl` (228 balanced conversational examples) |
| **Validation Dataset** | `data/sft/cbr_sft_07/splits/cbr_sft_07_val_22.jsonl` (22 held-out validation examples) |

---

### 3. Checkpoint Manifest & Integrity Hashes

All checkpoints saved during the run have been cryptographically verified:

| Checkpoint Name | Step | Path | SHA-256 |
| :--- | :--- | :--- | :--- |
| `sft_checkpoint_step_25.pt` | 25 | `experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_25.pt` | `000383afaa217545bf3b017d928d60c29238d2f88d3bb3180bb48ef1f0fdaabc` |
| **`sft_checkpoint_step_50.pt`** | **50** | **`experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_50.pt`** | **`0cb5df5fa840d20cea737da47758d96cfe884782c9801f28e51660bb236de86b`** |
| `sft_checkpoint_step_75.pt` | 75 | `experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_75.pt` | `5d1c297b23ac69e76b110b0e7705e0449f7729c4c51eb84f721923ff8ca66904` |
| `sft_checkpoint_step_100.pt` | 100 | `experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_100.pt` | `9b51ef12520e7b7b188f60ce9e86265d3e256dae1bb175d1d1069870d4993f44` |
| `sft_checkpoint_step_150.pt` | 150 | `experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_150.pt` | `a75980a514c0131d83829676ff49536a97f30d34764092fa8f9c01a629280470` |
| `sft_checkpoint_step_200.pt` | 200 | `experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_200.pt` | `335f311d4e528306a3894212cc6cc241d5e7cc7105f842c79bd38436a4e31425` |
