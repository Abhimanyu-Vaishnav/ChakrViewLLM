# NEURAL ROUTER TRAINING REPORT
## Batches 1661–1700 Controlled Adaptation Run (Step 10,000 -> Step 10,100)

### 1. Training Setup & Hyperparameters

| Parameter | Setting / Value |
| :--- | :--- |
| **Model Base** | Canonical Step 10,000 Checkpoint (`96357705...`) |
| **Architecture** | ChakrMicro v0.1 (`3,443,136` unique parameters) |
| **Output Checkpoint** | `experiments/batch1661_1700/checkpoints/chakrmicro_router_candidate_step100.pt` |
| **Candidate SHA-256** | `e221d18057ad7b27a2173ddf9cd1c74b7a74ff44e23556195503615734d04a57` |
| **Training Dataset** | `train.jsonl` (150 examples, uniform 6-way intent balance) |
| **Validation Dataset**| `val.jsonl` (30 examples) |
| **Optimizer** | AdamW (`lr=5e-5, betas=(0.9, 0.98), weight_decay=0.01`) |
| **Batch Size** | 4 |
| **Sequence Length** | Max 256 tokens |
| **Training Steps** | 100 optimizer steps |
| **Hardware** | Intel Core i5-12450HX CPU (0 GPUs) |
| **Wallclock Time** | 8.18 seconds |

---

### 2. Loss & Perplexity Trajectory

| Step | Train Cross-Entropy | Validation Loss | Validation Perplexity | Notes |
| :---: | :---: | :---: | :---: | :--- |
| **10** | 4.0498 | 4.0570 | 57.80 | Model starts picking up JSON formatting tokens (`{"intent":`) |
| **20** | 3.1812 | 3.4425 | 31.27 | Transition from generic prose to structured formatting |
| **30** | 2.7350 | 3.0298 | 20.69 | Rapid loss reduction on JSON punctuation |
| **40** | 2.7034 | 2.7289 | 15.32 | Steady convergence |
| **50** | 2.2623 | 2.4982 | 12.16 | Perplexity drops below 15 |
| **60** | 1.9977 | 2.3272 | 10.25 | Predicts common arguments keys |
| **70** | 1.5454 | 2.1865 | 8.90 | Continuing convergence |
| **80** | 1.3041 | 2.0773 | 7.98 | Monotonic validation descent |
| **90** | 1.4874 | 1.9993 | 7.38 | Minor batch variance |
| **100** | **1.2982** | **1.9405** | **6.96** | Final candidate saved |

---

### 3. Training Dynamics Analysis

1. **Validation Convergence:** Validation loss decreased steadily from $4.0570$ to $1.9405$, and perplexity reached an impressive $6.96$.
2. **The "Perplexity Mirage":**
   Even though validation perplexity reached $6.96$ on the validation distribution, the model produced valid JSON on only **14.17%** of held-out test tasks.
   Why? Because autoregressive language models without constrained decoding frequently generate trailing garbage, unmatched braces, or prematurely truncate JSON blocks when confronted with paraphrased inputs.
3. **Reproducibility:** The run is fully deterministic under random seed `42` and can be reproduced on CPU in under 10 seconds.
