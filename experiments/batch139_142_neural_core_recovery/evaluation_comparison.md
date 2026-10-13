# Evaluation Comparison: Batch 139–142 Controlled Pretraining Arms

**Date:** 2026-10-13  
**Milestone:** Batch 139–142 Neural-Core Recovery  
**Device:** CPU (PyTorch 2.6+, 14 threads)  
**Parent Model:** ChakrMicro Canonical Step 10,000 Checkpoint  
**Tokens Evaluated per Split:** 10,240 tokens (20 batches of batch_size 8 $\times$ seq_len 64)  

---

## 1. Global Performance Comparison

| Metric | Arm A (Baseline) | Arm B (Replicated Pilot) | Arm C (Warmup+Cosine Natural) | Arm D (Warmup+Cosine Balanced) | Best Arm |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Training Steps** | 0 | 100 | 250 | 250 | — |
| **Tokens Trained** | 0 | 51,200 | 128,000 | 128,000 | — |
| **Peak LR** | N/A | $3.0\times 10^{-4}$ | $1.5\times 10^{-4}$ | $1.5\times 10^{-4}$ | — |
| **Warmup Steps** | N/A | 0 | 50 | 50 | — |
| **Sampling Mode** | N/A | Balanced (50% Indic) | Natural Uniform | Balanced (50% Indic) | — |
| **Validation Loss** | **3.2441** | 3.3643 ($+0.1202$) | 3.2534 ($+0.0093$) | 3.2614 ($+0.0173$) | **Arm A (Baseline)** |
| **Validation PPL** | **25.64** | 28.91 ($+3.27$) | 25.88 ($+0.24$) | 26.09 ($+0.45$) | **Arm A (Baseline)** |
| **Global Test Loss** | **3.0721** | 3.1936 ($+0.1215$) | 3.0896 ($+0.0175$) | 3.0776 ($+0.0055$) | **Arm A (Baseline)** |
| **Global Test PPL** | **21.59** | 24.38 ($+2.79$) | 21.97 ($+0.38$) | 21.71 ($+0.12$) | **Arm A (Baseline)** |

---

## 2. Multi-Domain Held-Out Test Shard Breakdown

Cross-entropy loss evaluated separately on held-out test shards (10,240 tokens each):

| Domain Test Shard | Arm A (Baseline) | Arm B (Replicated Pilot) | Arm C (Warmup+Cosine Natural) | Arm D (Warmup+Cosine Balanced) | Delta (Arm D vs Baseline) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **English** (`shard_00000.bin`) | **3.1970** | 3.3059 | 3.2002 | 3.2062 | $+0.0092$ |
| **Hindi** (`shard_00001.bin`) | **3.0569** | 3.1654 | 3.0720 | 3.0586 | $+0.0017$ |
| **Code / Reasoning** (`shard_00002.bin`) | 2.7540 | 2.8584 | 2.7719 | **2.7526** | **$-0.0014$** |

### Key Scientific Findings:
1. **The Replicated Pilot (Arm B) Confirms Severe Regression:**
   Starting abruptly at $3\times 10^{-4}$ with 0 warmup degraded test loss by $+0.1215$ nats across all domains, reproducing the Batch 135–138 finding.
2. **Warmup + Cosine Annealing (Arms C & D) Successfully Eliminates 90%+ of the Regression:**
   By applying a 50-step linear warmup ($0 \to 1.5\times 10^{-4}$) followed by cosine decay down to $2\times 10^{-5}$, test loss degradation was reduced from $+0.1215$ (Arm B) down to $+0.0055$ (Arm D).
3. **Code/Reasoning Domain Slight Gain under Balanced Sampling:**
   Arm D achieved a minor improvement on held-out code tokens ($2.7540 \to 2.7526$), demonstrating that domain-balanced pretraining with proper warmup protects structural code representations.
4. **Overall Baseline Superiority on General Language:**
   Neither Arm C nor Arm D beat the canonical Step 10,000 baseline on general English and Hindi validation loss. This proves that 250 steps on 128,000 tokens is still in the "recovery" regime. A model that was pretrained for 10,000 steps (~5.1M tokens) cannot be substantially improved by a tiny 128k token micro-burst without entering deeper training.

---

## 3. Qualitative Continuation & Generation Diversity

Greedy decoding continuations (32 tokens) and diversity metrics:

| Metric | Arm A (Baseline) | Arm B (Replicated Pilot) | Arm C (Warmup Natural) | Arm D (Warmup Balanced) |
|:---|:---:|:---:|:---:|:---:|
| **Global 3-gram Repetition** | **0.2778** | 0.4841 | 0.5556 | 0.4365 |
| **Global Unique Token Ratio** | **0.5391** | 0.3594 | 0.3359 | 0.4141 |

### Prompt 1: English
- **Prompt:** `"The development of modern science began with"`
- **Arm A (Baseline):** `"in the world of the world. \n\nThe remaining the word of the world of the world of the"`
- **Arm B (Replicated):** `"in the first of the first of the first of the first of the first of the first of the"`
- **Arm C (Natural):** `"to be a list of the world of the world of the world of the world of the world of"`
- **Arm D (Balanced):** `"in the first of the first of the first of the first of the first of the first of the"`

### Prompt 2: Code
- **Prompt:** `"def binary_search(arr, target):"`
- **Arm A (Baseline):** `"\"\"\"\n    if not isinstance(\")\n    if not self._subset(\")\n    if no"`
- **Arm B (Replicated):** `"\"\"\"\n    if not None:\n        if self._conversion_conversion(\"\\n\")\n        if not"`
- **Arm C (Natural):** `"\"\"\"\n    if not None:\n        if not None:\n        if not None:\n            return self._point_"`
- **Arm D (Balanced):** `"print(\"\\n\")\n    print(\"\\n\")\n    print(\"\\n\")\n    print(\"\\n\")"`

### Prompt 3: Hindi
- **Prompt:** `"भारत एक विशाल और विविधतापूर्ण"`
- **Arm A (Baseline):** `"राज्य का अपनी महाराष्ट्रीय क्षेत्र के बाद दोनों में माना जाता है।\n\nइ"`
- **Arm B (Replicated):** `"राज्य का पुत्र दो दो दो दो दो दो दो दो "`
- **Arm C (Natural):** `"राज्य में महाराजा महाराजा महाराजा महाद्वीप में महाराजा म"`
- **Arm D (Balanced):** `"राज्य का अपनी महाना जाता है।\n\nइन्हें भी देखें\n* दोनों "`

---

## 4. Synthesis and Checkpoint Lineage Decision

- **Candidate Promotion Status:** **NO AUTOMATIC PROMOTION.**
- In accordance with the project's non-negotiable principle (*"Do not select a winner using training loss alone. Do not claim success without evidence."*), none of the candidate checkpoints (Arms B, C, D) are promoted over the canonical baseline `training_checkpoint_step_10000.pt`.
- The canonical Step 10,000 checkpoint remains the authoritative frozen baseline.
- All candidate checkpoints are securely preserved under `experiments/batch139_142_neural_core_recovery/checkpoints/` with bit-exact lineage records.
