# UCIP-18 Controlled Conversational Learning Pilot Report

**Mission ID:** UCIP-18  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Base Checkpoint:** `experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt`  
- SHA-256: `8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943`  
**Execution Timestamp:** 2026-11-20T18:54:00+05:30  
**Pilot Status:** COMPLETED (300 STEPS)  
**Pilot Authorization:** Controlled pilot only (max 300 steps). Full training prohibited.  

---

## 1. Executive Summary & Verdict

- **Final Experimental Verdict:** **FAIL** (Instruction Following Not Acquired; SFT Loss Decreased but Generalization Remains at 0%).
- **Promotion / Release Authorization:** **DENIED**. Zero promotion to production or release candidate.
- **Root Observation:**
  While the decoupled gradient objective ($\mathcal{L} = 0.85 \cdot L_{\text{SFT}} + 0.15 \cdot L_{\text{rep}}$) and reconstructed UCIP-17 dataset successfully lowered SFT validation loss from **5.6904** to **3.9228** ($-1.7676$ points) and eliminated degenerate `###` punctuation attractor loops (dropping attractor rate from **52%** to **5%** while raising EOS emission from **0%** to **47%**), the 1.5M-parameter ChakrMicro model **failed to achieve semantic instruction following** (strict semantic accuracy remains **0.0%** across 100 dev probes). Instead of following instructions, the model migrated to an English pretraining phrase attractor ("*The centry of the passcop...*").

---

## 2. Experimental Setup & Configuration

- **Dataset:** UCIP-17 Reconstructed Dataset (`data/sft/ucip17/sft_train_12000.jsonl`, SHA-256: `0ad8840e...`).
- **Foundational Replay:** Stage C pretraining binary shards (`data/tokenized/stage_c/train/`).
- **Batch Composition:** 12 SFT sequences (dynamic batching, active assistant token mask) + 1 Stage C replay sequence (512 tokens).
- **Objective:** Decoupled token-normalized loss:
  $$\mathcal{L} = 0.85 \cdot \left(\frac{1}{N_{\text{SFT}}}\sum \ell_{\text{SFT}}\right) + 0.15 \cdot \left(\frac{1}{N_{\text{rep}}}\sum \ell_{\text{rep}}\right)$$
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.95, \text{weight\_decay}=0.01$).
- **Schedule:** Peak learning rate $2.5 \times 10^{-5}$ with 30-step linear warmup, cosine decay to $2.5 \times 10^{-6}$ over 300 steps.
- **Hardware & Runtime:** Windows 11 CPU, single-thread deterministic execution. Total runtime: 92.9 seconds.

---

## 3. Training & Validation Trajectory

| Step Interval | Total Loss | SFT Train Loss | Replay Train Loss | SFT Val Loss | Stage C Retention Loss | Learning Rate | Grad Norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Step 000 (Init)** | — | — | — | **5.6904** | **3.9544** | $0.00$ | — |
| **Step 050** | 5.0162 | 5.0089 | 5.0576 | 5.0888 | 3.9493 | $2.47 \times 10^{-5}$ | 2.982 |
| **Step 100** | 4.4216 | 4.6493 | 3.1317 | 4.4934 | 3.9763 | $2.15 \times 10^{-5}$ | 3.237 |
| **Step 150** | 4.1815 | 4.1696 | 4.2488 | 4.2136 | 3.9748 | $1.57 \times 10^{-5}$ | 2.643 |
| **Step 200** | 3.7795 | 3.7137 | 4.1524 | 4.1636 | 3.9653 | $9.29 \times 10^{-6}$ | 2.596 |
| **Step 250** | 3.9255 | 3.8345 | 4.4411 | 4.0660 | 3.9675 | $4.35 \times 10^{-6}$ | 3.497 |
| **Step 300 (Best/Final)** | 3.9186 | 3.7788 | 4.7107 | **3.9228** | **3.9681** | $2.50 \times 10^{-6}$ | 2.725 |

- **SFT Validation Loss Delta:** $-1.7676$ ($-31.1\%$ relative improvement).
- **Stage C Retention Loss Delta:** $+0.0137$ points (from 3.9544 to 3.9681). Zero catastrophic forgetting of language modeling capability.

---

## 4. Strict Behavioral Evaluation Comparison (100 Stratified Probes)

Evaluated on 100 stratified development probes across all 18 categories and 4 languages (Hindi, Hinglish, Sanskrit, English) using exact semantic criteria with zero length-only or superficial fallbacks:

| Evaluation Metric | Baseline (Step 800) | UCIP-18 Pilot (Step 300) | Delta / Trend |
|---|---:|---:|:---:|
| **Strict Semantic Accuracy** | **0.00%** | **0.00%** | Unchanged (0 / 100) |
| **Attractor Loop Rate** | **52.0%** | **5.0%** | **-47.0% (Massive reduction in `#` repetition)** |
| **EOS Emission Rate** | **0.0%** | **47.0%** | **+47.0% (Learned proper termination)** |
| **Exact Matches** | 0 | 0 | 0 |
| **Partial Matches** | 0 | 0 | 0 |
| **Incorrect Responses** | 48 | 95 | Shifted from attractor loop to non-instruction text |

### Sample Qualitative Generations:

1. **Category:** `HING` (Hinglish QA)  
   - **Prompt:** `User: Git me commit ka kya matlab hai?\nAssistant: `  
   - **Target:** `Git me commit ka matlab repository ke snapshot ko permanently version history me save karna hota hai.`  
   - **Baseline:** `################################################` (Attractor Loop, No EOS)  
   - **Pilot:** `The centry of the passcop-of the passcop-of the passcop-of the passcop-of the passcops.` (Emitted EOS)

2. **Category:** `PYTHON`  
   - **Prompt:** `User: Can you write a function to reverse a string.\nAssistant: `  
   - **Target:** `\`\`\`python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n\`\`\``  
   - **Baseline:** `The show of the show of the show of the show of the show...`  
   - **Pilot:** `The centry of the passcopes of the passcops.` (Emitted EOS)

3. **Category:** `CONV` (Hindi Greeting)  
   - **Prompt:** `User: नमस्ते, आपका क्या कार्य है?\nAssistant: `  
   - **Target:** `नमस्ते! मेरा उद्देश्य स्पष्ट, निष्पक्ष और संप्रभु भाषा सहायता प्रदान करना है।`  
   - **Baseline:** `################################################` (Attractor Loop, No EOS)  
   - **Pilot:** `[0]\n    's <ref name="nb" />\n\n    's </ref>` (Wikipedia XML fragment from Stage C)

---

## 5. Failure Diagnosis & Capacity Boundary

1. **Punctuation Attractor Solved, But Semantic Attractor Formed:**  
   UCIP-17's token diversity and prompt-masking eliminated the degenerate hash attractor `#`, and taught the model when to stop via `<EOS>`. However, because the underlying base model is a 1.5M parameter micro-model with limited representation capacity, fine-tuning gradients shifted probability mass into low-entropy English Wikipedia/Stage C n-grams (`The centry of the passcop...`) rather than learning associative cross-attention mappings from prompt tokens to response semantics.
2. **Capacity Boundary:**  
   ChakrMicro (4 layers, 128 hidden dim, 4 heads) does not possess sufficient parameter capacity to generalize across 18 conversational categories and multiple languages via 300 steps of SFT.

---

## 6. Checkpoint & Artifact Manifest

- **Pilot Best Checkpoint:** `experiments/ucip18/pilot/checkpoint_best.pt`  
  - SHA-256: `c22bd77e00ce97671dec31b47c4a674f689633df6622a7777acfb9a3cc0d346a`
- **Pilot Final Checkpoint (Step 300):** `experiments/ucip18/pilot/checkpoint_step_0300.pt`  
  - SHA-256: `1d1046d1362f2c8f9ea2099d9a08a337b9264b07df70c58023eca7e259a9b725`
- **Baseline Step 800 Checkpoint:** `experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt`  
  - SHA-256: `8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943`
- **Protected Runtime `session.py`:** `chakrview/runtime/session.py`  
  - SHA-256: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` (VERIFIED UNMODIFIED)
- **Canonical Baseline Seed 42 In-Memory:**  
  - Hash: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` (VERIFIED UNMODIFIED)
