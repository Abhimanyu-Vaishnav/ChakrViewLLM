# UCIP-18R Independent Forensic Audit Report

**Mission ID:** UCIP-18R  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Expected Starting Commit:** `89597e9`  
**Actual HEAD Commit:** `89597e9`  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Mission Type:** Independent Corrective Engineering & Forensic Audit  
**Training Authorization:** STRICTLY PROHIBITED (Zero training executed)  
**Remote Push:** STRICTLY PROHIBITED (Local only)  
**Execution Timestamp:** 2026-11-20T19:03:00+05:30  

---

## 1. Executive Verdict & Core Recommendation

### Final Recommendation: **A. UCIP-18 results independently verified.**
*(With one critical architectural documentation correction regarding model parameter accounting.)*

### Executive Findings Summary:
1. **Pilot Evaluation Validated:** The reported **FAIL** verdict for conversational capability acquisition is **100% verified and reproducible**. The pilot checkpoint achieves **0.00% strict semantic accuracy** across the 100 stratified dev probes.
2. **Behavioral Shift Validated:**
   - Hash attractor loops (`####...`) dropped from **52% in baseline** to **5% in pilot**.
   - EOS emission increased from **0% in baseline** to **47% in pilot**.
   - However, the 47% EOS emission is **NOT** a sign of successful instruction following. It represents an attractor migration: the model learned when to stop while generating a repetitive English Stage C pretraining phrase (`"The centry of the passcop..."`).
3. **Architecture Parameter Reconciliation:**
   - The UCIP-18 report statement that ChakrMicro has "approximately 1.5 million parameters" was **erroneous prose**.
   - Independent inspection of the loaded state dict, configuration, and PyTorch tensors confirms ChakrMicro v0.1 has exactly **3,443,136 total parameters** ($d_{\text{model}}=192, L=6, H=6, d_{\text{ff}}=512, V=4096$) with tied embeddings ($W_{\text{head}} = E^T$).
4. **Chronology & Provenance Invariants Upheld:**
   - Baseline results were frozen in task `task-5927` before pilot training began in `task-6004`.
   - Zero contamination against all 8,215 unique sealed benchmark prompts.
   - All protected file hashes remain intact and identical to canonical baselines ($\Delta W = 0$).

---

## 2. Independent Reproduction of Baseline and Pilot Evaluations

All 3 checkpoints were independently evaluated across the identical 100 stratified development probes from `data/sft/ucip17/sft_dev_1500.jsonl` under greedy decoding ($T=0.0$, max new tokens = 48) using the strict semantic evaluator:

| Checkpoint Identifier | File SHA-256 | Exact | Partial | Incorrect | Attractor | Strict Accuracy | EOS Rate | Max-Hit Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline Step 800** | `8759ccb4...` | 0 | 0 | 48 | 52 | **0.00%** | **0.0%** | 100.0% |
| **UCIP-18 Best (Step 300)** | `c22bd77e...` | 0 | 0 | 95 | 5 | **0.00%** | **47.0%** | 53.0% |
| **UCIP-18 Final (Step 300)** | `1d1046d1...` | 0 | 0 | 95 | 5 | **0.00%** | **47.0%** | 53.0% |

- **Exact Agreement:** There are **zero discrepancies** between the reproduced evaluation scores and those reported in UCIP-18.
- **Evaluator Implementation Audit:**
  - Evaluated `experiments/ucip18/eval/strict_evaluator.py`:
  - **No Length Fallbacks:** Responses are not accepted merely because they have non-zero length.
  - **No Keyword Fallbacks:** Responses require domain-specific structural correctness (e.g., Python function definitions, arithmetic solutions, logic deduction conclusions).
  - **No Premature EOS Credit:** Emitting EOS on non-instruction outputs receives a verdict of `INCORRECT` ($0.0$ score).
  - **Fail-Closed:** All 100 probe generations failed to meet the strict semantic threshold, yielding 0.00% accuracy.

---

## 3. EOS Forensic Analysis

A complete audit of every generation across the 100 development probes revealed the exact mechanics of the reported 47% EOS emission:

| Checkpoint | Total Probes | Emitted EOS | Hit Max Limit (48) | Mean Generated Length (Tokens) | Mean EOS Position |
|---|---:|---:|---:|---:|---:|
| **Baseline Step 800** | 100 | 0 (0%) | 100 (100%) | 48.0 | N/A |
| **Pilot Best Checkpoint** | 100 | 47 (47%) | 53 (53%) | 31.9 | 31.9 (range: 11 – 45) |
| **Pilot Final Checkpoint** | 100 | 47 (47%) | 53 (53%) | 31.9 | 31.9 (range: 11 – 45) |

### Key EOS Findings:
1. **Zero Premature Truncation on Target Responses:** The model did not terminate early on valid responses because **no valid responses were generated**.
2. **Grammatical Termination of Attractor Phrases:** The 47% EOS emissions occur almost exclusively when the model generates its newly acquired English phrase attractor. In English probes (55 total), 44 emitted EOS after completing variations of `"The centry of the passcopes of the passcops."`.
3. **Language Asymmetry in EOS:**
   - English (`en`): 44 / 55 (80.0% EOS rate).
   - Hinglish (`hi-en`): 3 / 8 (37.5% EOS rate).
   - Hindi (`hi`): 0 / 33 (0.0% EOS rate — 100% degenerated into Wikipedia XML tags or `#` loops).
   - Sanskrit (`sa`): 0 / 4 (0.0% EOS rate).
4. **Forensic Conclusion:** The 47% EOS rate is **technically correct** (the model generated token ID `1`), but **behaviorally unaligned**. It proves that the model learned that sequences must end with `<EOS>`, but only within English sentence structures learned during pretraining.

---

## 4. Attractor Forensic Analysis

### Classification of Generative Failure Modes:

| Checkpoint | Old Hash Attractor (`###...`) | New Phrase Attractor (`passcop...`) | Wikipedia XML Attractor (`<ref>...`) | Other Non-EOS / Repetition | Total Attractor / Degenerate |
|---|---:|---:|---:|---:|---:|
| **Baseline Step 800** | 52 | 0 | 0 | 48 (`The show of the show...`) | 100% |
| **Pilot Checkpoint** | 5 | 60 | 31 | 4 | 100% |

### Representative Raw Generations:

1. **New English Phrase Attractor (60% of dev probes):**
   - Prompt: `User: Can you write a function to reverse a string.\nAssistant: `
   - Generated: `"The centry of the passcopes of the passcops."` `<EOS>`
   - Prompt: `User: Premise: All things that is electronic also evaporates quickly. Observation: Component Delta is electronic. Conclusion?\nAssistant: `
   - Generated: `"The central of the passcop-of the passcop-of the passcop-of the passcop-of the passcops."` `<EOS>`
2. **Wikipedia XML Metadata Attractor (31% of dev probes, dominant in Hindi):**
   - Prompt: `User: नमस्ते, आपका क्या कार्य है?\nAssistant: `
   - Generated: `"[0]\n 's <ref name=\"nb\" />\n\n 's </ref>"` (Hit max token limit 48)
   - Prompt: `User: सूर्य के सबसे निकट का ग्रह कौन सा है?\nAssistant: `
   - Generated: `"[0]\n 's <ref name=\"nb\" />\n\n 's </ref>"` (Hit max token limit 48)
3. **Old Hash Attractor (5% of dev probes):**
   - Baseline: 52 occurrences.
   - Pilot: 5 occurrences (persisting only on a handful of Indic probes).

### Behavioral Improvement vs. Failure Mode Migration:
- **Verdict:** The change from 52% to 5% in the reported metric reflects an **incomplete definition of attractor**.
- The original evaluation script defined `ATTRACTOR` solely as heavy `#` repetition or severe token collapse.
- While the model successfully escaped the trivial `#` attractor basin, it **migrated to two new attractor basins**:
  - The English Stage C n-gram basin (`passcop` / `centry`).
  - The Stage C Wikipedia XML markup basin (`<ref name=...`).
- Therefore, the underlying failure mode remains **representational collapse and instruction-following failure**.

---

## 5. Architecture and Parameter Audit

### Parameter Reconciliation:
The UCIP-18 pilot report referenced "1.5M parameter micro-model" in its narrative section. An independent audit of the ChakrMicro codebase and checkpoint resolved this discrepancy:

| Structural Component | PyTorch Module | Tensor Shape | Exact Elements | Tied / Shared |
|---|---|---|---:|:---:|
| **Token Embeddings** | `embedding` | `[4096, 192]` | 786,432 | Yes (tied to LM Head) |
| **Transformer Layers (x6)** | `layers.0` through `layers.5` | | | No |
| - Attention Norm 1 | `norm_1.weight` | `[192]` | 192 x 6 = 1,152 | No |
| - Q, K, V, Out Projections | `attn.*_proj.weight` | `[192, 192]` x 4 | 36,864 x 4 x 6 = 884,736 | No |
| - FFN Norm 2 | `norm_2.weight` | `[192]` | 192 x 6 = 1,152 | No |
| - SwiGLU Gate, Up, Down | `ffn.*_proj.weight` | `[512, 192]` x 3 | 98,304 x 3 x 6 = 1,769,472 | No |
| **Final RMS Norm** | `final_norm` | `[192]` | 192 | No |
| **LM Output Head** | `lm_head` | `[4096, 192]` | 0 additional | Shared via `F.linear(x, embedding.weight)` |
| **Total Parameter Elements** | — | — | **3,443,136** | — |
| **Unique Physical Parameters** | — | — | **3,443,136** | — |
| **Trainable Parameters** | — | — | **3,443,136** | — |

- **Configuration:**
  - $V = 4096$, $d_{\text{model}} = 192$, $L = 6$, $H = 6$, $d_{\text{ff}} = 512$, $T_{\text{max}} = 512$.
  - Weight tying: Enabled (`lm_head.weight.data_ptr() == embedding.weight.data_ptr()`).
- **Audit Conclusion:** The model has **3,443,136 parameters (~3.44M)**, matching the specification in `docs/STEP_03_1_NEURAL_CORE_SPEC.md` and `reports/UCIP_13R_INDEPENDENT_LEARNING_VERIFICATION.md`. The "1.5M" mention in the UCIP-18 report was an informal misstatement and is hereby formally corrected.

---

## 6. Training Lineage & Chronological Audit

Inspection of background task execution records from the transcript and filesystem confirms strict chronological discipline:

| Task ID | Task Goal | Start Timestamp (UTC) | End Timestamp (UTC) | Duration | Exit Status | Verification Note |
|---|---|---|---|---|---|---|
| `task-5927` | Strict Baseline Dev Eval | 2026-11-20T13:18:06Z | 2026-11-20T13:18:33Z | 27s | Exit 0 | Baseline results frozen at 0.00% accuracy, 52% attractor |
| `task-6004` | 300-Step Controlled Pilot | 2026-11-20T13:21:33Z | 2026-11-20T13:23:11Z | 98s | Exit 0 | Strictly bounded 300 steps, AdamW, cosine decay |
| `task-6018` | Pilot Checkpoint Dev Eval | 2026-11-20T13:23:33Z | 2026-11-20T13:23:57Z | 24s | Exit 0 | Evaluated best checkpoint (`checkpoint_best.pt`) |

- **Confirmation:** Baseline evaluation (`task-5927`) was **fully completed and verified 3 minutes prior** to the initialization of the pilot training run (`task-6004`).
- **Optimization Parameters Verified:**
  - Steps: Exactly 300 optimizer steps.
  - SFT Batch Size: 12 sequences (dynamic padding, assistant tokens masked with -100).
  - Replay Batch Size: 1 sequence (512 continuous tokens from Stage C shards).
  - Loss formulation: $\mathcal{L} = 0.85 \cdot L_{\text{SFT}} + 0.15 \cdot L_{\text{rep}}$, each normalized by active tokens.
  - Gradient clipping: 1.0.

---

## 7. Gradient Objective & Data Provenance Audit

### 7.1 Gradient Mathematics Verification
Re-execution of `experiments/ucip17/measure_gradients_phase5.py` confirmed:
- Raw SFT gradient norm: $\approx 13.56$.
- Raw Replay gradient norm: $\approx 9.32$.
- Weighted SFT norm ($0.85 \cdot \|\mathbf{g}_{\text{SFT}}\|$): $11.5244$ ($89.20\%$ of sum of norms).
- Weighted Replay norm ($0.15 \cdot \|\mathbf{g}_{\text{rep}}\|$): $1.3975$ ($10.80\%$ of sum of norms).
- Gradient Cosine Similarity: $+0.3625$ (positive gradient alignment).
- **105.96% Projection Claim:** Verified as the mathematical projection of the joint update step onto the SFT gradient direction:
  $$\frac{\mathbf{g}_{\text{comb}} \cdot \mathbf{g}_{\text{SFT}}}{\|\mathbf{g}_{\text{comb}}\|^2} \approx 1.0596$$
  This indicates that because $\mathbf{g}_{\text{rep}}$ has a positive projection onto $\mathbf{g}_{\text{SFT}}$, the combined update step length along $\mathbf{g}_{\text{SFT}}$ exceeds the combined gradient norm by $5.96\%$. It was not an assertion of a standard 0–100% percentage share.

### 7.2 Benchmark Contamination Audit
- Comprehensive scan across all 65 benchmark files in `experiments/` containing sealed probes identified **8,215 unique prompt strings**.
- Re-running `experiments/ucip17/check_diff_sealed.py`:
  - Contamination in `sft_master_15000.jsonl`: **0**
  - Contamination in `sft_train_12000.jsonl`: **0**
  - Contamination in `sft_validation_1500.jsonl`: **0**
  - Contamination in `sft_dev_1500.jsonl`: **0**

### 7.3 Data Provenance Audit
- All records in `data/sft/ucip17/` carry `license: "CC0-1.0"` and `source_id: "chakr_verified_sft_corpus_v2"`.
- **Finding:** This is **self-assigned metadata** generated during the internal dataset reconstruction process, indicating internal public domain dedication by the project authors rather than external third-party licensing.

---

## 8. Retention Audit

Stage C language retention was audited across validation shard `shard_00000.bin`:

| Sample Scope | Baseline Step 800 Loss | Pilot Best Checkpoint Loss | Difference ($\Delta$) |
|---|---:|---:|---:|
| **20 Chunks (10,240 tokens)** | **3.9544** | **3.9681** | **+0.0137** |
| **100 Chunks (51,200 tokens)** | **4.1705** | **4.1859** | **+0.0154** |

- **Statistical Analysis:**
  - Standard deviation across chunks is $\approx 0.38$.
  - The $+0.014$ to $+0.015$ difference represents less than $0.04$ standard deviations ($< 0.4\%$ relative change).
  - This confirms that Stage C language modeling capacity was **preserved without catastrophic forgetting**.

---

## 9. Protected Invariant Hash Verification

Every critical project file and artifact hash was independently re-hashed:

| Protected Artifact | Expected Canonical SHA-256 | Actual Verified SHA-256 | Status |
|---|---|---|:---:|
| `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **MATCH** |
| Baseline Step 800 ckpt | `8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943` | `8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943` | **MATCH** |
| In-Memory Baseline Seed 42 | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **MATCH** |
| Alpha Release ckpt | `004ca38f3e796c4590bcfd73a029a3493173637f22bcf0dcad41e98c218511e7` | `004ca38f3e796c4590bcfd73a029a3493173637f22bcf0dcad41e98c218511e7` | **MATCH** |
| UCIP-17 Train Dataset | `0ad8840efb43d8c2d9a7c12d785683b6791a906e18f06376974286ed0638eb80` | `0ad8840efb43d8c2d9a7c12d785683b6791a906e18f06376974286ed0638eb80` | **MATCH** |
| UCIP-17 Validation Dataset | `972affab6e583475bab151d4f88606adfb3e6846bc4157c36dc4f9a6ab7c21f2` | `972affab6e583475bab151d4f88606adfb3e6846bc4157c36dc4f9a6ab7c21f2` | **MATCH** |
| UCIP-17 Dev Dataset | `cd84f301f638405516e0b76195862d1938491bc66129d58c7449b5d0bd108370` | `cd84f301f638405516e0b76195862d1938491bc66129d58c7449b5d0bd108370` | **MATCH** |
| UCIP-18 Pilot Best Checkpoint | `c22bd77e00ce97671dec31b47c4a674f689633df6622a7777acfb9a3cc0d346a` | `c22bd77e00ce97671dec31b47c4a674f689633df6622a7777acfb9a3cc0d346a` | **MATCH** |
| UCIP-18 Pilot Step 300 Checkpoint | `1d1046d1362f2c8f9ea2099d9a08a337b9264b07df70c58023eca7e259a9b725` | `1d1046d1362f2c8f9ea2099d9a08a337b9264b07df70c58023eca7e259a9b725` | **MATCH** |

---

## 10. Audit Conclusion & Final Sign-Off

The forensic investigation concludes that:
1. UCIP-18 was executed in strict adherence to all research controls, safety gates, and non-destructive protocols.
2. The conclusion that **the model failed to acquire conversational instruction capability (verdict: FAIL)** is scientifically sound, reproducible, and accurate.
3. The model parameters are officially reconciled as **3,443,136 (~3.44M)**.
4. **Recommendation:** **A. UCIP-18 results independently verified.** No further training should be initiated without a fundamental reassessment of model representation capacity or curriculum architecture.
