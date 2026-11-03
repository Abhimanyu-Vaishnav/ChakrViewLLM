# ChakrView Exp34 — Controlled A/B Comparison Report
## Arm A (Masked Prompt) vs. Arm B (Unmasked Prompt)

**Experiment:** Exp34 Attractor Suppression / Prompt Masking Evaluation  
**Dataset:** `exp34_instruction_mixture_train_v3.jsonl` (Stratified 900 train / 100 val split)  
**Base Checkpoint:** Canonical Step 10,000 (`96357705a683...`)  
**Arm A Checkpoint (Step 50):** `sft_checkpoint_step_50.pt` (`150f6214862eefe0baa4f33fbf9f075594e7f00c61b09834446cc25b0c54cecf`)  
**Arm B Checkpoint (Step 50):** `sft_checkpoint_step_50.pt` (`ab4e61e1ba95c5705ff009e2b14962e3f8f103d1b303e13bff92da449aecfb99`)  
**Optimizer Budget:** Exactly 50 optimizer steps per arm (AdamW, lr 5e-5 -> 5e-6, exact token-weighted gradient accumulation, batch size 2 x accum 2 = 4 sequences/step).

---

## 1. Executive Comparison Matrix

| Evaluation Dimension | Canonical Step 10,000 Baseline | Arm A: `mask_prompt=True` (ChakrView Standard) | Arm B: `mask_prompt=False` (Full Causal Control) | Delta ($\Delta$ Arm A vs. Canon) | Delta ($\Delta$ Arm A vs. Arm B) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Intelligence Pass Rate (32 Tasks)** | 0/32 (0.0%) | **1/32 (3.1%)** | **1/32 (3.1%)** | +3.1% | 0.0% |
| **Attractor Collapse Rate (32 Tasks)** | 17/32 (53.1%) | **11/32 (34.4%)** | 14/32 (43.8%) | **-18.7% (Better)** | **-9.4% (Better)** |
| **Coding Benchmark Pass Rate (12 Tasks)** | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0.0% | 0.0% |
| **Coding Syntax Validity (12 Tasks)** | 3/12 (25.0%) | **4/12 (33.3%)** | **4/12 (33.3%)** | +8.3% | 0.0% |
| **Mean 2-Gram Repetition Ratio** | 0.3340 | **0.2773** | 0.3008 | **-5.67%** | **-2.35%** |
| **Mean 3-Gram Repetition Ratio** | 0.2821 | **0.2233** | 0.2627 | **-5.88%** | **-3.94%** |
| **Mean 4-Gram Repetition Ratio** | 0.2540 | **0.1862** | 0.2333 | **-6.78%** | **-4.71%** |
| **Mean Unique Token Ratio** | 0.5314 | 0.5663 | **0.5681** | +3.49% | -0.18% |
| **EOS Terminations (30 Prompts)** | 2/30 | 2/30 | **4/30** | 0 | -2 |
| **Initial Validation Loss** | N/A | 4.6659 | 4.7891 | N/A | -0.1232 |
| **Final Step 50 Validation Loss** | N/A | **4.5724** | 4.6471 | N/A | **-0.0747** |
| **Validation Perplexity** | N/A | **96.78** | 104.28 | N/A | **-7.50** |
| **Throughput (Tokens/Second)** | 239.3 | **316.8** | 283.3 | +77.5 | +33.5 |

---

## 2. Key Findings & Scientific Takeaways

1. **Attractor Loop Suppression:**
   - Arm A (`mask_prompt=True`) drove a **18.7% reduction in attractor collapse** relative to the canonical checkpoint (falling from 53.1% to 34.4%).
   - Arm B (`mask_prompt=False`) only reduced collapse to 43.8% (a 9.3% reduction). Supervising prompt tokens caused the model to memorize prompt prefix templates, reinforcing loop attractor behavior.
2. **Repetition N-Grams:**
   - Arm A achieved the lowest n-gram repetition across all metrics (4-gram repetition dropped from 25.4% to **18.6%**, vs 23.3% in Arm B).
3. **Loss Dynamics:**
   - Both arms demonstrated steady in-distribution cross-entropy loss reduction across the 50 steps.
   - Arm A achieved both lower final validation loss (4.5724 vs 4.6471) and lower perplexity (96.78 vs 104.28).
4. **Generalization Constraints of 50-Step Pilot:**
   - Task accuracy on the 44 held-out downstream tasks remains low (1/32 intelligence, 0/12 coding), consistent with expectations for a 50-step pilot on 3.44M parameters.

