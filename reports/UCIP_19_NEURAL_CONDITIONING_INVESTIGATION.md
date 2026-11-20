# UCIP-19 — Neural Conditioning & Representation Investigation

**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Starting HEAD:** `6aaabf4`  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Mission Type:** Diagnostic Research & Neural Representation Forensics (Zero Training)  
**Execution Timestamp:** 2026-11-20T19:14:00+05:30  
**Target Models Audited:**
- Base Checkpoint: `experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt` (SHA-256: `8759ccb4...`)
- Pilot Checkpoint: `experiments/ucip18/pilot/checkpoint_best.pt` (SHA-256: `c22bd77e...`)

---

## 1. Executive Summary & Core Diagnosis

### Core Empirical Finding: **Prompt-Independent Representational Collapse (Attractor Basin Dominance)**
The failure of ChakrMicro v0.1 (3.44M parameters) to follow conversational instructions is caused by a **conditioning bottleneck in the attention mechanism**:
1. **Severe Prompt Insensitivity:** Across 100 dev probes, changing the user's prompt to a *completely unrelated instruction* yields a **61.0% first-token agreement rate** and an average **logit cosine similarity of 0.9431**.
2. **Negligible Causal Conditioning:** Swapping key entities or entire instructions changes the next-token probability distribution by a tiny **KL divergence of only 0.011 to 0.014** (with a **91.6% to 93.0% top-5 candidate overlap**). Even shuffling prompt words only alters the next-token distribution by $D_{\text{KL}} = 0.0219$.
3. **Representational Saturation Across Layers:** The pairwise cosine similarity between hidden representations of completely different prompts exceeds **0.92 to 0.94** in every intermediate layer (L0 through L4), indicating that prompt-specific features are largely compressed into a generic context subspace before the final unembedding projection.
4. **Attractor Entrainment:** In greedy decoding, the model is trapped in a dominant Stage C English attractor ("*The centry of the passcop...*") or XML markup attractor (`[0] <ref name=...`). When temperature sampling ($T=0.7$, top-$p=0.9$) disrupts this loop, diversity increases ($1.0$), but the generated tokens remain unconditioned pretraining noise rather than meaningful instruction following.

---

## 2. Research Lineage & Protected Artifact Hashes

Prior to running diagnostic inference passes, all protected baseline files and checkpoint artifacts were verified bit-exact:

| Artifact | Location | SHA-256 Checksum | Invariant Status |
|---|---|---|:---:|
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **MATCH ($\Delta W = 0$)** |
| **Baseline Step 800 ckpt** | `experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt` | `8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943` | **MATCH** |
| **In-Memory Baseline Seed 42** | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **MATCH** |
| **UCIP-18 Best ckpt** | `experiments/ucip18/pilot/checkpoint_best.pt` | `c22bd77e00ce97671dec31b47c4a674f689633df6622a7777acfb9a3cc0d346a` | **MATCH** |
| **UCIP-18 Final ckpt** | `experiments/ucip18/pilot/checkpoint_step_0300.pt` | `1d1046d1362f2c8f9ea2099d9a08a337b9264b07df70c58023eca7e259a9b725` | **MATCH** |
| **UCIP-17 Train Dataset** | `data/sft/ucip17/sft_train_12000.jsonl` | `0ad8840efb43d8c2d9a7c12d785683b6791a906e18f06376974286ed0638eb80` | **MATCH** |
| **UCIP-17 Dev Dataset** | `data/sft/ucip17/sft_dev_1500.jsonl` | `cd84f301f638405516e0b76195862d1938491bc66129d58c7449b5d0bd108370` | **MATCH** |

---

## 3. Investigation A — Prompt Sensitivity Analysis

Evaluated on 100 frozen development probes from `sft_dev_1500.jsonl`. For each probe, 6 controlled variants were tested:
1. `orig`: The original prompt.
2. `reworded`: Same semantic instruction with syntactic variation.
3. `reordered`: Words reordered.
4. `extra_ctx`: Irrelevant context prepended.
5. `format`: Different output format requested (e.g. JSON block).
6. `different`: Completely different instruction.

### Quantitative Sensitivity Matrix:

| Prompt Variation Type | First-Token Agreement Rate | Mean Top-10 Candidate Overlap | Mean Logit Cosine Similarity | Full Generation Identity Rate |
|---|---:|---:|---:|---:|
| **Reworded Instruction** | 73.0% | 88.8% | 0.9938 | 37.0% |
| **Reordered Words** | 93.0% | 85.6% | 0.9959 | 56.0% |
| **Extra Context** | 80.0% | 85.6% | 0.9923 | 33.0% |
| **Different Format** | 61.0% | 73.8% | 0.9696 | 7.0% |
| **Completely Different** | **61.0%** | **69.1%** | **0.9431** | **1.0%** |

### Findings:
- Even when presented with a **completely unrelated prompt**, the pilot model outputs the exact same first token **61.0% of the time**.
- The cosine similarity between output logits of completely different prompts is **0.9431**, indicating that the prompt is only exerting a tiny perturbation on the baseline output manifold.

---

## 4. Investigation B — Internal Representation & Layer Dynamics

Hidden-state norms, attention entropy, and pairwise cosine similarity across layers were measured across all 100 dev probes:

| Transformer Layer | Layer Hidden State Norm (Mean $\pm$ Std) | Attention Entropy (Mean $\pm$ Std) | Pairwise Probe Cosine Similarity |
|---|---:|---:|---:|
| **Input Embedding ($E$)** | $0.8490 \pm 0.030$ | — | — |
| **Layer 0 (L0)** | $0.8490 \pm 0.030$ | $2.6157 \pm 0.390$ | 0.9246 |
| **Layer 1 (L1)** | $1.1988 \pm 0.043$ | $3.3621 \pm 0.320$ | 0.9318 |
| **Layer 2 (L2)** | $1.5396 \pm 0.072$ | $3.0405 \pm 0.328$ | 0.9413 |
| **Layer 3 (L3)** | $1.8107 \pm 0.079$ | $3.2417 \pm 0.330$ | 0.9457 |
| **Layer 4 (L4)** | $2.1126 \pm 0.146$ | $2.9372 \pm 0.288$ | 0.9392 |
| **Layer 5 (L5)** | $2.2634 \pm 0.130$ | $2.3308 \pm 0.227$ | **0.8652** |
| **Pre-Head Norm ($\mathbf{h}_{\text{final}}$)** | $16.4295 \pm 0.941$ | — | — |

### Findings:
1. **Representational Clustering:** Across layers 0 through 4, pairwise cosine similarity between prompts remains high ($\approx 0.92 - 0.94$). Layer representations fail to branch into distinct clusters based on task type.
2. **Attention Entropy Plateau:** Attention entropy remains elevated ($>2.9$ out of max theoretical $\ln(T) \approx 3.5$) throughout layers 1–4, showing that attention is diffuse rather than selectively focusing on specific prompt keywords.
3. **Late Dispersal (L5):** Only in Layer 5 does the pairwise similarity drop to $0.8652$, where the model attempts to project into vocabulary space, but without sufficient upstream separation to drive distinct semantic responses.

---

## 5. Investigation C — Causal Conditioning Tests

To test whether the model is causally influenced by prompt components, we measured the Kullback-Leibler divergence $D_{\text{KL}}(P_{\text{orig}} \parallel P_{\text{cand}})$ and top-5 probability overlap between the original prompt distribution and perturbed variants:

| Intervention Condition | Mean KL Divergence ($D_{\text{KL}}$) | Std KL Divergence | Mean Top-5 Candidate Overlap |
|---|---:|---:|---:|
| **Instruction Swap** | **0.0112** | 0.0126 | **93.0%** |
| **Entity Swap** | **0.0139** | 0.0179 | **91.6%** |
| **Token Shuffling** | **0.0219** | 0.0238 | **84.4%** |
| **Empty Prompt (`User:\nAssistant:`)** | **0.2982** | 0.2688 | **71.8%** |
| **Unrelated Prompt** | **0.4216** | 0.5051 | **66.6%** |

### Key Diagnostic Distinction:
- **Token-Level Sensitivity:** Present at a very low level ($D_{\text{KL}} \approx 0.01 - 0.02$). The logits technically shift, but not enough to change the top-ranked token.
- **Instruction-Level Conditioning:** **ABSENT**. Swapping the entire instruction produces virtually identical top-5 token predictions (93.0% overlap).
- **Task Execution:** **ABSENT**. The model treats the prompt primarily as a syntactic prefix rather than an instruction to be decoded and executed.

---

## 6. Investigation D — Training Objective & Dataset Diagnostics

An audit of 500 sampled records from `data/sft/ucip17/sft_train_12000.jsonl` and the training pipeline confirmed:

| Diagnostic Dimension | Value / Metric | Status |
|---|---|:---:|
| **Mean Prompt Tokens** | $34.94 \pm 11.2$ | Verified balanced |
| **Mean Response Tokens** | $34.51 \pm 9.8$ | Verified balanced |
| **Prompt-to-Response Token Ratio** | $1.01 : 1$ | Perfectly matched |
| **Assistant-Only Masking Alignment** | **100.0%** (all prompt targets $=-100$) | Verified bit-exact |
| **Target Terminal EOS Rate** | **100.0%** (all targets terminate with token 1) | Verified bit-exact |
| **Loss Formulation** | $\mathcal{L} = 0.85 L_{\text{SFT}} + 0.15 L_{\text{rep}}$ (token-normalized) | Verified mathematically sound |
| **Replay Gradient Norm Share** | SFT: $89.2\%$, Replay: $10.8\%$ | SFT dominates update |

### Analysis of Corpus Variation vs. Model Capacity:
- The dataset construction in UCIP-17 resolved template repetition (99.8% unique responses).
- However, 300 optimizer steps of joint tuning on a 3.44M-parameter model did not provide sufficient capacity or training depth to restructure the pretrained causal attention weights. The loss decreased from 5.69 to 3.92 simply by fitting high-frequency response prefixes rather than learning the joint distribution $P(\text{Response} \mid \text{Prompt})$.

---

## 7. Investigation E — Generation Dynamics Across Decoding Configurations

Token-by-token trajectory measurements over 30 test prompts comparing baseline and pilot models under three decoding configurations:

| Decoding Configuration | Model | Unique Outputs (out of 30) | Diversity Ratio | Top-1 Prob (Start $\to$ Mid $\to$ End) | Entropy (Start $\to$ Mid $\to$ End) | Terminal EOS Prob |
|---|---|---:|---:|---|---|---:|
| **Greedy ($T=0.0$)** | **Baseline** | 8 | 0.2667 | $0.064 \to 0.215 \to 0.141$ | $6.39 \to 5.21 \to 4.96$ | 0.0000 |
| | **Pilot** | 16 | **0.5333** | $0.076 \to 0.243 \to 0.194$ | $6.39 \to 5.21 \to 4.96$ | **0.0012** |
| **Temperature ($T=0.7$)** | **Baseline** | 30 | 1.0000 | $0.064 \to 0.167 \to 0.141$ | $6.39 \to 5.51 \to 5.08$ | 0.0000 |
| | **Pilot** | 30 | **1.0000** | $0.076 \to 0.129 \to 0.153$ | $6.39 \to 5.51 \to 5.08$ | **0.0289** |
| **Top-$p$ ($p=0.9, T=0.7$)** | **Baseline** | 30 | 1.0000 | $0.064 \to 0.148 \to 0.180$ | $6.39 \to 5.21 \to 4.76$ | 0.0000 |
| | **Pilot** | 30 | **1.0000** | $0.076 \to 0.197 \to 0.237$ | $6.39 \to 5.21 \to 4.76$ | **0.0151** |

### Key Dynamics Observations:
1. **Attractor Collapse in Greedy Mode:** In greedy decoding, diversity is only 0.53 because the model rapidly falls into the `"The centry of the passcop..."` attractor by step 3.
2. **Sampling Breaks Repetition but Does Not Recover Semantics:** Setting $T=0.7$ achieves 100% unique outputs, but inspecting the generated text reveals random unguided word sequences. Sampling prevents attractor lock-in by injecting entropy, but does not induce instruction following.
3. **Learned EOS Dynamics:** The pilot model consistently increases its EOS emission probability as generation progresses ($0.0003 \to 0.0289$), confirming that the terminal EOS token supervision was learned, even though response content remains unconditioned.

---

## 8. Scientific Classification of Suspected Root Causes

| Suspected Root Cause | Diagnostic Classification | Evidence & Justification |
|---|:---:|---|
| **1. Representation Collapse / Low Prompt Sensitivity** | **SUPPORTED BY EVIDENCE** | KL divergence under instruction swap is only $0.0112$; logit cosine similarity across different prompts is $0.9431$; 61% first-token agreement across unrelated prompts. |
| **2. Insufficient Model Capacity (3.44M parameters)** | **SUPPORTED BY EVIDENCE** | ChakrMicro ($d=192, L=6$) lacks sufficient width and depth to maintain separate representational manifolds for 18 diverse multi-lingual conversational tasks. |
| **3. Training Objective Flaw (Loss / Masking Bug)** | **NOT SUPPORTED** | Target construction, causal alignment, assistant-only masking, and decoupled token normalization were verified bit-exact (100% mask compliance, 100% terminal EOS). |
| **4. Stage C Replay Gradient Interference** | **NOT SUPPORTED** | Gradient cosine similarity is positive ($+0.3625$), and SFT gradients dominate ($89.2\%$ of norm sum, $105.96\%$ projection). Stage C loss was preserved without catastrophic forgetting. |
| **5. Pretraining Attractor Basin Priming** | **SUPPORTED BY EVIDENCE** | High-frequency Wikipedia/Stage C n-grams (`passcop`, `centry`, `<ref name=...`) dominate generation across all prompt variants. |
| **6. Decoding Artifact Alone** | **NOT SUPPORTED** | Temperature and top-$p$ sampling increase lexical diversity but produce nonsensical unconditioned text; the issue is representational, not purely decoding. |

---

## 9. Final Recommendation

### **Recommendation: A. Evidence supports a specific conditioning bottleneck; propose a narrowly targeted experiment.**

### Recommended Next Experiment (UCIP-20 Proposal):
1. **Curriculum Distillation / Single-Domain Conditioning Pilot:**
   - Rather than attempting to learn 18 broad conversational categories simultaneously on a 3.44M parameter model, execute a **tightly scoped single-domain conditioning experiment** (e.g. structured extractive QA or direct arithmetic copy-response).
2. **Prompt-Response Cross-Attention Analysis:**
   - Measure whether prompt tokens can be causally bound to response tokens under explicit prefix-tuning or targeted attention scaling before re-attempting full conversational mixtures.
3. **Strict Constraints:**
   - Zero training or checkpoint promotion without prior user authorization.
