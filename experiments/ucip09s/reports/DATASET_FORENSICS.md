# Stage C — Dataset Forensics Report

**Date:** 2026-11-17  
**Mission:** UCIP-09S Controlled Retention & Capability Isolation Study  
**Datasets Analyzed:**
- `experiments/cognition/ucip_09/data/ucip09_train.jsonl` (Parent baseline)
- `experiments/cognition/ucip_09/data/ucip09_validation.jsonl`
- `experiments/cognition/ucip_09r/data/ucip09r_train.jsonl` (Degraded fine-tune)
- `experiments/cognition/ucip_09r/data/ucip09r_val.jsonl`
**Diagnostic Analysis Output:** `experiments/ucip09s/evaluations/dataset_forensic_stats.json`

---

## 1. Executive Summary & Core Discovery

A forensic comparative analysis of the training datasets reveals the mathematical cause of the catastrophic forgetting in UCIP-09R:

1. **Massive Training Set Skew towards a Single Domain:**
   - In **UCIP-09**, training data was balanced across 7 balanced domains (English conversation, Hindi, Hinglish, explanations, copying, arithmetic, instruction following), with arithmetic occupying only **16.9%** (549 / 3,240).
   - In **UCIP-09R**, the dataset was abruptly reduced from 3,240 samples to 885 samples, and arithmetic was expanded to **88.5%** (783 / 885).
2. **Virtual Elimination of Retention Anchors (Catastrophic Deletion):**
   - **English conversation:** Slashed from 448 samples (13.8%) down to **12 samples (1.35%)** — a **97.3% reduction**.
   - **Devanagari Hindi:** Slashed from 545 samples (16.8%) down to **12 samples (1.35%)** — a **97.8% reduction**.
   - **Hinglish:** Slashed from 635 samples (19.6%) down to **6 samples (0.68%)** — a **99.1% reduction**.
   - **Instruction following:** Slashed from 259 samples (8.0%) down to **0 samples (0.0%)** — **100% eliminated**.
3. **Severe Token Frequency Distortion & Artificial Output Bias:**
   - Because 783 of 885 samples in UCIP-09R were arithmetic word problems or number generation, the target output space became heavily dominated by specific numeric tokens: `21` (51 instances), `27` (50 instances), `15` (46 instances), `18` (44 instances), `24` (43 instances), and template phrases like `remaining.` (178 instances) and `total.` (55 instances).
   - This concentrated token probability mass into rigid numeric bi-grams, entirely explaining why UCIP-09R answered `21` and `22 cards in total` to almost every arithmetic question in owner testing.

---

## 2. Quantitative Dataset Comparison

| Metric | UCIP-09 Train | UCIP-09R Train | Absolute Change | Percentage Shift |
|---|:---:|:---:|:---:|:---:|
| **Total Samples** | 3,240 | 885 | -2,355 | -72.7% |
| **Unique Prompts** | 3,229 (99.7%) | 881 (99.5%) | -2,348 | -72.7% |
| **Unique Targets** | 593 | 257 | -336 | -56.7% |
| **Avg Prompt Length (words)** | 11.2 | 14.8 | +3.6 | +32.1% |
| **Avg Target Length (words)** | 13.9 | 6.4 | -7.5 | -54.0% |

### Category Distribution Breakdown

| Category | UCIP-09 Count | UCIP-09 Share | UCIP-09R Count | UCIP-09R Share | Structural Change |
|---|:---:|:---:|:---:|:---:|:---:|
| **Arithmetic / Math** | 549 | 16.9% | 783 | 88.5% | **+423% Relative Dominance** |
| **Exact Copying** | 355 | 11.0% | 50 | 5.6% | -85.9% |
| **Devanagari Hindi** | 545 | 16.8% | 12 | 1.4% | **-97.8% (Virtual Elimination)** |
| **English Conversation** | 448 | 13.8% | 12 | 1.4% | **-97.3% (Virtual Elimination)** |
| **English Explanations** | 449 | 13.9% | 22 | 2.5% | -95.1% |
| **Hinglish** | 635 | 19.6% | 6 | 0.7% | **-99.1% (Virtual Elimination)** |
| **Instruction Following** | 259 | 8.0% | 0 | 0.0% | **-100.0% (Total Elimination)** |
| **Total** | 3,240 | 100.0% | 885 | 100.0% | — |

---

## 3. Lexical and Token-Level Forensics

### UCIP-09 Target Vocabulary Profile
In UCIP-09, the highest frequency target words were generic connective tokens across multiple languages:
- English: `you` (820), `and` (787), `have` (641), `will` (598), `the` (477), `to` (455), `i` (396)
- Hinglish: `aur` (640), `hai` (575), `ko` (440), `hai.` (383)
- Hindi: `है।` (433), `के` (383)
- Math: `=` (549)

This preserved balanced semantic embeddings across English, Hindi, and conversational contexts.

### UCIP-09R Target Vocabulary Distortion
In UCIP-09R, the target vocabulary underwent an extreme collapse:
- Top words: `you` (241), `have` (237), `will` (233), `remaining.` (178), `in` (61), `total.` (55)
- Numeric fixation: `21` (51), `27` (50), `15` (46), `18` (44), `24` (43), `30` (41), `33` (36), `12` (34), `22` (28)
- Hindi & Hinglish tokens: Completely dropped outside the top 50 target tokens!

When backpropagating over 885 samples where 88.5% of targets follow the rigid template `"You will have <N> <item> remaining."` or `"<N> <item> in total."`, the model's output projection layer learned that almost every prompt should route into the numbers `21`, `22`, or `remaining.`

---

## 4. Forensic Finding

The dataset design of UCIP-09R violated the fundamental principle of continual learning and retention:
**Attempting to fix a narrow capability (arithmetic) by fine-tuning on an 88.5% arithmetic dataset with 0% instruction following, 1.4% Hindi, and 1.4% English conversation caused severe catastrophic gradient overwriting.** The pre-trained conversational and multilingual representations were effectively replaced by a narrow, repetitive arithmetic template lookup table.

This confirms the first critical pillar of the UCIP-09S study: **Any targeted capability repair MUST incorporate balanced replay / anchor retention data from the parent distribution.**
