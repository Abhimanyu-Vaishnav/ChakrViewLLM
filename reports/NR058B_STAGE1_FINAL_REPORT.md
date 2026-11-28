# CHAKRVIEW RESEARCH MISSION NR-058B STAGE 1 REPORT
## One-Epoch Controlled Pretraining: ChakrGenesis17M vs. ChakrGenesis38M

**Date:** 2026-11-28  
**Mission ID:** NR-058B Stage 1  
**Lead Investigator:** Antigravity Autonomous Research Assistant  
**Hardware Platform:** NVIDIA RTX A1000 6GB Laptop GPU (CUDA 12.6, PyTorch 2.14.1+cu126, Python 3.14 venv)  
**Corpus:** Corpus V6 Train (`data/corpus_v6/splits/corpus_v6_train.jsonl`) & Val (`data/corpus_v6/splits/corpus_v6_val.jsonl`)  
**Tokenizer:** Canonical Byte-Level BPE 4096 (`data/experiments/vocab_4096`)  
**Artifact Isolation Root:** [experiments/stage1_nr058b/](file:///d:/Project/ChakrView/experiments/stage1_nr058b/)

---

## 1. Executive Summary & Core Verdict

Under explicit bounded authorization, Mission NR-058B Stage 1 executed exactly **one complete epoch** (1,305 optimizer steps each) of pretraining from fresh scratch initialization for both **ChakrGenesis17M** (17,063,808 parameters) and **ChakrGenesis38M** (37,499,392 parameters).

### Key Takeaways:
1. **Mathematical Training Convergence:**
   - **ChakrGenesis17M** dropped validation loss from **8.3890** (PPL: 4,398.56, Acc: 0.03%) to **3.1004** (PPL: **22.21**, Acc: **39.80%**).
   - **ChakrGenesis38M** dropped validation loss from **8.4240** (PPL: 4,554.99, Acc: 0.03%) to **2.9562** (PPL: **19.22**, Acc: **41.84%**).
2. **Architecture Showdown Verdict:**
   - **ChakrGenesis38M decisively outperformed ChakrGenesis17M across every single domain and aggregate metric:**
     - Overall Validation Loss: **2.9562 vs. 3.1004** ($\Delta = -0.1442$)
     - Overall Validation Perplexity: **19.22 vs. 22.21** ($\Delta = -2.99$)
     - Overall Next-Token Prediction Accuracy: **41.84% vs. 39.80%** ($\Delta = +2.04\%$)
   - The 38M architecture demonstrated superior language and code modeling capacity while consuming only 2.057 GB of VRAM (leaving > 3.9 GB of VRAM headroom on the 6GB GPU).
3. **Hardware Efficiency:**
   - ChakrGenesis17M completed 1 epoch in **195.0s (3.25 min)** at **21,283.4 tokens/sec** (6.69 steps/sec).
   - ChakrGenesis38M completed 1 epoch in **280.2s (4.67 min)** at **14,810.3 tokens/sec** (4.66 steps/sec).
   - Total pretraining compute time for both models was **7.92 minutes**, running stably without NaN, non-finite loss, or gradient explosion.
4. **Preservation & Non-Interference:**
   - Canonical checkpoints (`RC1`, `RC2`, `RC3`), canonical tokenizer, Benchmark V3/V2, and test split (`corpus_v6_test.jsonl`) remained 100% frozen and untouched.
   - All models were initialized deterministically from scratch (`seed=42`), eliminating any checkpoint contamination or weight inheritance.

---

## 2. Phase 0 & 1: Preflight Verification & Token Reconciliation

Prior to launch, the token count discrepancy between manifest reports and chunked training batches was resolved and mathematically verified:
- **Manifest Reported Tokens:** `4,163,555`
- **EOS Tokens Added at Document Boundaries:** `+9,843` (exactly 1 EOS token per document)
- **Dropped Short Trailing Fragments (< 16 tokens):** `-2,731`
- **Net Active Chunks Produced (len=256):** `20,865`
- **Net Active Training Tokens:** `4,170,667`
$$\text{Reconciliation Identity: } 4,163,555 + 9,843 - 2,731 = 4,170,667 \quad (\Delta = 7,112)$$
- At batch size 16: $\lfloor 20,865 / 16 \rfloor = 1,304$ full batches + 1 boundary batch (9 chunks) = **1,305 steps per epoch**.
- Documented in: [token_reconciliation_report.json](file:///d:/Project/ChakrView/experiments/stage1_nr058b/token_reconciliation_report.json).

---

## 3. Training Dynamics & Step Progression

Both models followed a strict 1-epoch schedule:
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}, \text{weight\_decay}=0.01$)
- **Precision:** AMP FP16 Autocast with GradientScaler, gradient clipping at 1.0
- **Schedule:** 100 warmup steps, cosine decay to 10% minimum LR at step 1305
- **Periodic Validation:** Evaluated at steps 200, 400, 600, 800, 1000, 1200, and 1305.

### Step-by-Step Validation Tracking

| Step | 17M Val Loss | 17M Val PPL | 17M Val Acc (%) | 38M Val Loss | 38M Val PPL | 38M Val Acc (%) | Delta Loss (38M - 17M) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0 (Init)** | 8.3890 | 4398.56 | 0.03% | 8.4240 | 4554.99 | 0.03% | +0.0350 |
| **200** | 5.6569 | 286.27 | 6.87% | 5.5895 | 267.61 | 7.22% | -0.0674 |
| **400** | 4.4819 | 88.40 | 21.95% | 4.3499 | 77.47 | 23.04% | -0.1320 |
| **600** | 3.7723 | 43.48 | 30.87% | 3.6134 | 37.09 | 32.79% | -0.1589 |
| **800** | 3.4235 | 30.68 | 35.21% | 3.2766 | 26.49 | 36.99% | -0.1469 |
| **1000** | 3.2393 | 25.52 | 37.77% | 3.0907 | 21.99 | 39.68% | -0.1486 |
| **1200** | 3.1381 | 23.06 | 39.26% | 2.9870 | 19.83 | 41.37% | -0.1511 |
| **1305 (Final)** | **3.1004** | **22.21** | **39.80%** | **2.9562** | **19.22** | **41.84%** | **-0.1442** |

---

## 4. Domain-Specific Breakdown (Validation Split)

The Corpus V6 validation split contains 546 documents across 7 active domains totaling 191,132 active tokens. 

> [!NOTE]
> **Mathematics Domain Reality:** Corpus V6 validation split contains 0 mathematics documents because all math documents were allocated across train and held-out test splits. We report honest empirical results on the existing validation domains without fabricating synthetic scores.

| Domain | Token Count | 17M Loss | 17M PPL | 17M Acc (%) | 38M Loss | 38M PPL | 38M Acc (%) | 38M vs 17M Acc Delta |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **corpus_v4_curated** | 27,172 | 1.4870 | 4.42 | 72.14% | **1.3026** | **3.68** | **74.96%** | **+2.82%** |
| **code_python_stdlib** | 72,357 | 3.0054 | 20.19 | 41.04% | **2.8861** | **17.92** | **42.74%** | **+1.70%** |
| **wiki_en_stem_humanities** | 62,094 | 3.4396 | 31.17 | 33.25% | **3.3348** | **28.07** | **34.81%** | **+1.56%** |
| **wiki_hi_devanagari** | 28,227 | 4.0257 | 56.02 | 21.25% | **3.7700** | **43.38** | **24.48%** | **+3.23%** |
| **local_hindi** | 301 | 5.5845 | 266.25 | 10.30% | **5.3519** | **211.00** | **12.29%** | **+1.99%** |
| **local_sanskrit** | 555 | 6.4689 | 644.79 | 3.78% | **6.3216** | **556.46** | **5.59%** | **+1.81%** |
| **local_english** | 426 | 5.2580 | 192.09 | 18.31% | **5.1663** | **175.26** | **19.01%** | **+0.70%** |
| **OVERALL VALIDATION** | **191,132** | **3.1004** | **22.21** | **39.80%** | **2.9562** | **19.22** | **41.84%** | **+2.04%** |

### Domain Observations:
- **Curated Foundation:** Reached remarkable proficiency after only 1 epoch (1.3026 loss, 74.96% accuracy), reflecting high structure in foundational conversational and factual pairs.
- **Python Code:** Achieved solid syntax comprehension (2.8861 loss, 42.74% accuracy), capturing standard library idioms and formatting tokens.
- **English STEM & Humanities:** Consistent progression down to 28.07 PPL and 34.81% accuracy.
- **Hindi Devanagari:** The 38M model demonstrated significantly higher multiligual capacity than 17M (+3.23% accuracy, 43.38 PPL vs 56.02 PPL).

---

## 5. Artifact Provenance & Checkpoint Hashes

All Stage 1 artifacts were generated and saved strictly inside `experiments/stage1_nr058b/`:

| Checkpoint Identifier | Model | Step | File Size | SHA-256 Hash |
|:---|:---:|:---:|:---:|:---|
| `chakr_chakrgenesis17m_stage1_step200.pt` | 17M | 200 | 195.40 MB | `c82fe35676ee526a6c2084ea4626127fc17c093a8c3d820b411d31d0fd8eeef2` |
| `chakr_chakrgenesis17m_stage1_step400.pt` | 17M | 400 | 195.40 MB | `50df94dd68eb107fb7733cc856cbf0946b518c7bf9e34e5699478f654fe45d8b` |
| `chakr_chakrgenesis17m_stage1_step600.pt` | 17M | 600 | 195.40 MB | `a1ee16b0ecad724ebfd45c117d91e3e7fcebfa94ba325bdfa92e1069fcb6bb23` |
| `chakr_chakrgenesis17m_stage1_step800.pt` | 17M | 800 | 195.40 MB | `24f52d32840ef3dfdf771804f5cb1b9970966f91cb943cf641d4022a84a65c40` |
| `chakr_chakrgenesis17m_stage1_step1000.pt` | 17M | 1000 | 195.40 MB | `e505df620ee6819fe7ea1316b1ca8f76e3391754093ff9fc9e32a67bc479ae93` |
| `chakr_chakrgenesis17m_stage1_step1200.pt` | 17M | 1200 | 195.40 MB | `d9cfd4c84279090b8f36c53569ee7dc7537b8aa073fa8e718ae32ce892eb527b` |
| `chakr_chakrgenesis17m_stage1_step1305.pt` | 17M | 1305 | 195.40 MB | `ff638236b77227d82b4fb711c1b18d867c4e5114798efcb5c4a17ce3da1c0d45` |
| **`chakr_chakrgenesis17m_stage1_final.pt`** | **17M** | **1305** | **195.40 MB** | **`2584b9a486e8c1c0c9725e42cebda16d0240a25079967b179c1a592fb1e2f09b`** |
| `chakr_chakrgenesis38m_stage1_step200.pt` | 38M | 200 | 429.29 MB | `b2370e9286d5e7a9b19dfa43fb9873ffabf60ee3930b58e7a0505106a7350711` |
| `chakr_chakrgenesis38m_stage1_step400.pt` | 38M | 400 | 429.29 MB | `c7cd8363750ebce5b3e2ec3138b1d9bf8ce3a0050868f0808cf70cae6cf46b2b` |
| `chakr_chakrgenesis38m_stage1_step600.pt` | 38M | 600 | 429.29 MB | `939034d7b9ea3ee74fbc19c9e557b7f14bfa4f16b08709424694ccb659c47cf7` |
| `chakr_chakrgenesis38m_stage1_step800.pt` | 38M | 800 | 429.29 MB | `1c0d2401f827ec3d90f23d88ca298f02931a7b458cbbd52e1f2ab7bc6e589e60` |
| `chakr_chakrgenesis38m_stage1_step1000.pt` | 38M | 1000 | 429.29 MB | `9238d509a632c0274cba0285a73e6a928e185c89895186b5c7efec17b960b76a` |
| `chakr_chakrgenesis38m_stage1_step1200.pt` | 38M | 1200 | 429.29 MB | `6c2f9598317926b4845ea2e6669c5e317bbbf1189196b0ca71b690d796690ea5` |
| `chakr_chakrgenesis38m_stage1_step1305.pt` | 38M | 1305 | 429.29 MB | `9ef8aaf629532588f8d951805aa31fe51a2ec7fb2a188be4523d42b9fa969f68` |
| **`chakr_chakrgenesis38m_stage1_final.pt`** | **38M** | **1305** | **429.29 MB** | **`92f2aa7487361ec112f1e40a8760993f167733a1943a32e50078338e77ecfcea`** |

---

## 6. Recommendations for Stage 2

1. **Architecture Recommendation:**
   - **ChakrGenesis38M** is the empirically superior architecture for ChakrView. Its scaling profile demonstrates faster loss reduction, lower perplexity across all domains, and superior multilingual retention, while remaining comfortably within the 6GB VRAM budget (using only 34% of available memory).
2. **Next Gated Steps (Awaiting User Authorization):**
   - Stage 2 authorization should consider:
     - Multi-epoch training (e.g., 3 to 5 epochs) with learning rate decay schedule tailored for 4,000–6,500 steps.
     - Integration of mathematics data into the validation split for comprehensive reasoning tracking.
     - Held-out evaluation on sealed benchmarks (Benchmark V3) only after Stage 2 completion.
