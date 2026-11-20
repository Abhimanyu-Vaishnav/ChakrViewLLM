# UCIP-16 Generation Forensic Audit Report
## Systematic Investigation of Neural Generation Breakdown in ChakrView Conversation Alpha

**Mission ID:** UCIP-16  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Starting HEAD:** `acfe331`  
**Auditor:** Antigravity Senior Forensic Auditor  
**Investigation Timestamp:** 2026-11-20T18:24:30+05:30  

---

## 1. Executive Summary & Forensic Verdict

### Forensic Verdict: **C. NO RELIABLE CONVERSATIONAL CAPABILITY YET**

Supervised fine-tuning in UCIP-15 achieved **mathematical loss reduction** on training sequences and successfully **suppressed raw base punctuation degeneration** (`#####...`), but it **failed to establish semantic instruction following**. 

When subjected to independent, strict semantic evaluation (eliminating the invalid $>15$-character fallback heuristic):
- **Exact Correct Rate:** **0%** ($0/100$)
- **Partial Semantic Correctness:** **0%** ($0/100$)
- **Invalid Attractor / Degenerate Loops:** **98%** ($98/100$)

The model does not follow instructions, cannot perform basic arithmetic, cannot bind entities, cannot copy tokens, and cannot converse in English, Hindi, or Sanskrit. Under greedy decoding, it gets trapped in one of two dominant attractor basins: an English phrase loop (`"The first of the first of..."`) or a Web URL string loop (`"[https://www.wiv.gov/www.html Sublic..."`).

---

## 2. Checkpoint & Inference Architecture Parity (Phase 1)

Both models were loaded and evaluated under identical, deterministic CPU inference:
- **Base Checkpoint:** `experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt` (SHA-256: `8759ccb4...`, Step 800).
- **SFT Checkpoint:** `releases/chakrview_conversation_alpha/chakrview_conversation_alpha.pt` (SHA-256: `004ca38f...`, Step 550).
- **Architecture Invariants:** Exactly 3,443,136 parameters, 6 layers, 6 heads, width 192, context 512, RoPE positional encoding, tied embeddings (`lm_head.weight is embedding.weight`), vocabulary 4,096.
- **Inference Wrapper Parity:** Standardized prefix wrapper `User: <prompt>\nAssistant: `.

---

## 3. Controlled Generation Experiment across 100 Probes (Phase 2)

A battery of 100 non-sealed development probes was evaluated across multiple decoding regimes:

| Decoding Configuration | Base Checkpoint (Step 800) | SFT Checkpoint (Step 550) | Observed Output Characteristics |
|---|---|---|---|
| **Greedy ($T=0.0$)** | 54% `####` loops, 46% repetitive text | 98% attractor loops, 2% ungrounded text | Punctuation loops replaced by phrase attractors |
| **Sampling ($T=0.7$, Top-$k=40$)** | Random token salad | Disconnected token sequences | Unstable subword sequences; no semantic coherence |
| **Sampling + Rep Penalty ($1.2$)** | Repetitive fragments dispersed | Pseudo-random English/Devanagari fragments | Breaks the exact token loop, but produces gibberish |
| **Early EOS Termination** | 0 / 100 | 0 / 100 | Neither model ever emits `<EOS>` during unconstrained generation |

### Key Token-Level Diagnostic:
- When prompted with `User: ...\nAssistant: `, `<EOS>` (Token 1) has a logit of **$-0.73$** and rank **375** (probability: **0.02%**). The model overwhelmingly prefers token 898 (`"The "`, logit 5.36, prob 10.37%).
- Once the model emits token 898 (`"The "`), it enters a self-reinforcing autoregressive loop where token 828 (`"fi"`), 117 (`"r"`), 525 (`"st "`), 770 (`"of "`), and 559 (`"the "`) form a cyclic Markov attractor.
- Conversely, when the ground-truth target is fully provided in the context, `<EOS>` (Token 1) has logit **7.32** and rank **1** (probability: **35.89%**). 
- **Finding:** The model learned when a turn ends *if teacher-forced*, but has **zero generative probability of ending a turn autonomously from prompt tokens**.

---

## 4. Root Cause Analysis

The total failure of conversational capability stems from three interrelated engineering flaws:

1. **Severe Dataset Template Repetition (Phase 4):**
   - The 12,000-sample training dataset contains only **1,851 unique responses** (84.6% duplication).
   - In 13 out of 18 categories, fewer than 10 unique response templates exist (e.g., 841 `EXPL` records contain only 3 unique responses; 828 `QA` records contain only 4 unique responses).
   - The single sentence *"Gravity exerts a continuous centripetal force..."* was trained **491 times**.
   - The model was not taught generalized instruction-following; it was trained on extreme template collapse.
2. **Replay Imbalance and Gradient Asymmetry (Phase 5):**
   - The training mixture drew 8 SFT sequences and 2 Stage C replay sequences per step.
   - However, because prompt tokens are masked in SFT (`-100`), SFT sequences only contributed ~150 active tokens per sample (~1,200 active tokens per step).
   - Meanwhile, Stage C replay chunks were unmasked (512 tokens per sequence = 1,024 active tokens per step).
   - Across training, **75.5% of all supervised tokens were raw Stage C pretraining text**, causing pretraining statistical n-grams (Wikipedia URLs, common article phrases) to dominate over instruction compliance.
3. **Capacity Constraints (Phase 7):**
   - At 3.44M parameters and a hidden width of 192, the model lacks the representational capacity to store world knowledge while simultaneously learning conditional dialogue routing without external memory.
