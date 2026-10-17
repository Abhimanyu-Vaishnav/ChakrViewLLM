# ChakrView Controlled Continued-Pretraining Experiment Plan — Batches 2021–2060
**Milestone:** Batches 2021–2060  
**Branch:** `dev/v0.6-v0.7-validation`  
**Status:** Pre-Registered (Execution Deferred to Future Milestone)  
**Date:** October 2026  

---

## 1. Pre-Registration Statement & Non-Negotiable Constraint

This document serves as the formal pre-registered specification for the next continued-pretraining (CPT) run of ChakrMicro.

**NON-NEGOTIABLE CONSTRAINT:**  
As stipulated by the mission guidelines, **no model training is executed during Batches 2021–2060**. This document freezes the experimental protocol, datasets, hyperparameters, evaluation criteria, and transfer probes prior to running any training steps, completely preventing post-hoc parameter manipulation.

---

## 2. Experimental Setup & Corpus Manifests

### 2.1 Starting Checkpoints
1. **Canonical Checkpoint (Primary):**  
   - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
   - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
   - Status: Untouched from-scratch pretraining baseline (10,000 steps).
2. **Historical 150-Step Baseline (Secondary Reference):**  
   - Path: `experiments/batch1861_1900/cpt_runs/run_seed42/checkpoint_step_150.pt`

### 2.2 Tokenizer
- **Path:** `release/chakrview-v0.5.0/data/experiments/vocab_4096`
- **Vocabulary Size:** 4,096 tokens (byte-level BPE).
- **Invariance:** Strictly frozen. Zero token additions or merges.

### 2.3 Dataset Partitions (Corpus v2)
- **Manifest:** `experiments/batch2021_2060/CORPUS_V2_MANIFEST.json`
- **Split Manifest:** `experiments/batch2021_2060/SPLIT_V2_MANIFEST.json`
- **Train Split:** 782 documents, 147,476 tokens across 6 source families (`gutenberg_tagore_gitanjali`, `ietf_rfc793_tcp`, `cpython_algorithms_heapq`, `wikipedia_hi_premchand`, `wikipedia_en_turing_machine`, `wikipedia_hi_chemistry`).
- **Validation Split:** 186 documents, 32,617 tokens across 3 source families (`google_hinglish_top_dialogues`, `cpython_algorithms_bisect`, `wikipedia_hi_ganit`).
- **Test Split (Frozen Benchmark):** 290 documents, 65,665 tokens across 4 source families (`gutenberg_loney_coordinate_geometry`, `wikipedia_hi_sanskrit_sahitya`, `wikipedia_sa_bhagavad_gita`, `wikipedia_en_information_theory`).
- **Source Isolation:** Strictly source-family disjoint ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$). Zero 8-gram leakage.

---

## 3. Training Hyperparameters & Execution Budget

| Parameter | Value | Rationale |
|:---|:---|:---|
| **Context Length ($T$)** | 256 tokens | Standard ChakrMicro context window |
| **Batch Size ($B$)** | 4 sequences per step | Fits comfortably in standard CPU cache |
| **Gradient Accumulation** | 2 steps | Effective batch size of 8 sequences (2,048 tokens/step) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$) | Standard transformer optimizer |
| **Weight Decay** | 0.01 | Gentle regularization against subword overfitting |
| **Peak Learning Rate** | $1.5 \times 10^{-4}$ | Reduced learning rate for CPT to prevent representation collapse |
| **Learning Rate Schedule** | Cosine decay to $1.5 \times 10^{-5}$ with 20-step linear warmup | Smooth transition from canonical pretraining |
| **Total Training Steps** | 200 steps | Balanced step budget avoiding the 300-step overfitting regime |
| **Checkpoint Interval** | Every 25 steps (steps 25, 50, 75, 100, 125, 150, 175, 200) | Granular trajectory monitoring |
| **Random Seeds** | 42, 1337, 2026 (3 distinct runs) | Triplicate validation to establish multi-seed variance |
| **Hardware Target** | CPU-first (Intel/AMD x86_64, 4-8 threads) | Hardware-adaptive CPU training |
| **RAM Budget** | $\le 2.0$ GB RSS | Lightweight memory footprint |
| **Expected Runtime** | ~12–15 minutes per seed on modern multicore CPU | High iteration speed without GPU requirement |

---

## 4. Checkpoint Selection & Early-Stopping Criteria

1. **Selection Split:** The best checkpoint $\theta^*$ must be chosen **strictly and exclusively** by minimizing cross-entropy loss on the **Validation Split** (`SPLIT_V2_MANIFEST.json` validation partition).
2. **Absolute Blindness to Test Split:** The Test Split must remain untouched and un-evaluated until $\theta^*$ is selected and permanently frozen.
3. **Early-Stopping Condition:** If validation cross-entropy fails to improve for 3 consecutive evaluation checkpoints (patience = 75 steps), training terminates early to prevent overfitting.
4. **Regression Threshold:** If at any point validation loss exceeds the baseline canonical loss $+0.25$ CE, the run is aborted as divergent.

---

## 5. Multi-Domain Evaluation Suite & Success Criteria

### 5.1 Per-Domain, Per-Language Evaluation Metrics
Upon selection of $\theta^*$, evaluation will be computed on the frozen Test Split across both aggregate and disaggregated slices:
- **Aggregate Metrics:**
  - Token-weighted Cross-Entropy & Perplexity
  - Equal-source Macro Cross-Entropy & Perplexity
- **Domain Slices:**
  - Mathematics / Analytical Geometry (`gutenberg_loney_coordinate_geometry`)
  - Theoretical Mathematics / Information Theory (`wikipedia_en_information_theory`)
  - Hindi Literary / Historical Exposition (`wikipedia_hi_sanskrit_sahitya`)
  - Classical Sanskrit Verses (`wikipedia_sa_bhagavad_gita`)

### 5.2 Capability Transfer Probes (Distinguishing Neural Core vs Tools)
Cross-entropy improvements alone do not demonstrate downstream reasoning capability. To evaluate genuine learning without tool confounding, the evaluation suite incorporates deterministic transfer probes:

1. **Pure Neural Arithmetic Probes (No Calculator Tooling):**
   - 20 unseen 2-digit and 3-digit addition and multiplication queries.
   - Evaluated under greedy decoding ($T=0$).
   - Scoring: Exact string match of final numerical output.
2. **Pure Neural Code Syntax Completion:**
   - 15 function signature and block completion probes from standard library paradigms.
   - Scoring: Syntactic validity via Python AST parsing (`ast.parse`).
3. **Pure Neural Factual Completion:**
   - 25 cloze-style factual completions across history, science, and computing.
   - Scoring: Strict normalized token match against target fact.
4. **Tool-Assisted Comparison Baseline:**
   - Run the same probes through ChakrView's Governed Neuro-Symbolic Hybrid runtime.
   - Explicitly document the delta between pure model-only generation and hybrid tool-assisted execution.

---

## 6. Pre-Registered Hypotheses

- **$H_1$ (Natural Generalization):** CPT on Corpus v2 will achieve lower validation CE loss compared to the canonical 10,000-step checkpoint without regressing on any individual test domain.
- **$H_2$ (Overfitting Suppression):** Unlike the Batch 1861–1900 synthetic CPT run, validation loss on natural text will remain stable past step 100 rather than degrading.
- **$H_3$ (Linguistic Retention):** Hindi and Sanskrit loss will improve or remain within $\pm 0.05$ CE of canonical, demonstrating zero-shot multilingual retention.
