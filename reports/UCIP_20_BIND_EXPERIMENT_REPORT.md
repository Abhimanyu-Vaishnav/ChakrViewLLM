# UCIP-20 — Isolated Neural Entity-Binding Learning Experiment Report

**Mission ID:** UCIP-20  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Starting HEAD:** `cee596c`  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Base Checkpoint:** `experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt`  
- SHA-256: `8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943`  
**Architecture:** ChakrMicro v0.1 (3,443,136 parameters, tied embeddings)  
**Execution Timestamp:** 2026-11-21T11:56:00+05:30  
**Authorization:** Controlled BIND-only training pilot, maximum five complete training exposures (Zero Replay).  

---

## 1. Executive Summary & Mission Verdict

### Final Mission Verdict: **FAIL** (Entity Binding Not Acquired; Generalization Remains 0.0%)

### Core Recommendation: **3. BIND learning failed; investigate the specific bottleneck.**

### Key Findings Summary:
1. **Zero Generalization to Unseen Entities:**
   - Training loss dropped smoothly from **5.9922 to 2.5411** across the 5 complete passes, and validation loss improved from **5.8757 to 4.4208** ($-1.4549$ points).
   - However, on the 200 strictly held-out development probes (where entity values are 100% disjoint from training), exact entity accuracy was **0.00% (0 / 200)**.
2. **Attractor Collapse into Numerical Token Sinks:**
   - Rather than learning cross-attention pointer binding from the context entity to the question, the model learned a high-entropy template sink: generating numerical and bracket repetitions (e.g., `[2000/200/200/2/2/2/2...` and `AAAAAAAAAAAAAAAA...`).
   - Old `#` and `passcop` attractors were reduced (attractor rate dropped to **3.0%**), and terminal EOS emission reached **61.0%**, but the generated output failed to extract or bind any entity value.
3. **Catastrophic Forgetting Without Replay:**
   - Because Stage C pretraining replay was disabled as instructed, the model experienced severe language regression:
     - Stage C validation loss degraded from **3.9544 to 4.4462** ($+0.4918$ points, $> 1.25$ standard deviations).
     - Non-BIND capabilities collapsed (e.g. general English continuation degraded to single digits `'0'`, and arithmetic prompts generated repeated `'A'`s).
4. **Epistemic Conclusion:**
   - Even when optimization starvation is eliminated (granting 5 full epochs with 100% record exposure), standard causal self-attention in a 3.44M-parameter architecture ($d_{\text{model}}=192, L=6$) **fails to induce inductive pointer/binding mechanisms** without an explicit architectural routing inductive bias (such as pointer networks or span cross-attention).

---

## 2. Phase 0 — Repository and Checkpoint Integrity

All protected artifacts and base checkpoints were verified bit-exact prior to dataset generation and training:

| Protected Artifact | Path | SHA-256 Checksum | Invariant Status |
|---|---|---|:---:|
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **MATCH ($\Delta W=0$)** |
| **Original Step 800 ckpt** | `experiments/ucip13/.../checkpoint_step_800.pt` | `8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943` | **MATCH** |
| **In-Memory Baseline** | Seed 42 frozen instance | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **MATCH** |
| **UCIP-18 Best ckpt** | `experiments/ucip18/pilot/checkpoint_best.pt` | `c22bd77e00ce97671dec31b47c4a674f689633df6622a7777acfb9a3cc0d346a` | **MATCH** |
| **UCIP-18 Final ckpt** | `experiments/ucip18/pilot/checkpoint_step_0300.pt` | `1d1046d1362f2c8f9ea2099d9a08a337b9264b07df70c58023eca7e259a9b725` | **MATCH** |

---

## 3. Phase 1 — Isolated BIND Dataset Construction & Audit

A dedicated, isolated entity-binding corpus was constructed under `data/sft/ucip20_bind/` with 10 diverse relation categories (`project`, `person_role`, `city`, `organization`, `product`, `identifier`, `date`, `numerical_value`, `ownership`, `property`) across 3 languages (`en`, `hi`, `hi-en`).

### Split Manifest & Hash Inventory:

| Partition | Record Count | Unique Entities | Unique Prompts | Unique Responses | SHA-256 Checksum |
|---|---:|---:|---:|---:|---|
| **Train Split** | 1,600 (80%) | 90 | 1,402 | 93 | `4b376b0fbf46eac6060ec7d881f6de73be19087c9717847e91eb5877b4972cd6` |
| **Validation Split** | 200 (10%) | 18 | 112 | 21 | `3de0e86fb34b0a1b814b8485432574b6df9760a485e59d53f2c09e9e433cc432` |
| **Development Split** | 200 (10%) | 18 | 109 | 21 | `698b21208f9f1484e2112eee08f1c9c6ead3d7e9f004abb9b3c6d1d01c52fa62` |
| **Master Corpus** | 2,000 (100%) | 126 | 1,623 | 126 | `ec7cfab1262158891e8704a7f11740e5f9a8bb81362dba9bc9f734b04bc7e1e9` |

### Leakage & Contamination Audit:
- **Train vs. Val Entity Overlap:** Exactly **0** (Disjoint entity pools).
- **Train vs. Dev Entity Overlap:** Exactly **0** (Disjoint entity pools).
- **Val vs. Dev Entity Overlap:** Exactly **0** (Disjoint entity pools).
- **Sealed Benchmark Contamination:** Exactly **0** out of 8,215 unique historical sealed prompts.

---

## 4. Phase 2 & Phase 5 — Baseline vs. Trained Checkpoint Evaluation

Evaluated across the frozen 200 held-out development probes from `bind_dev_200.jsonl` under greedy decoding ($T=0.0$, max new tokens = 32):

| Metric | Baseline (Step 800) | UCIP-20 Trained Best Checkpoint | Delta / Behavioral Shift |
|---|---:|---:|:---:|
| **Exact Entity Accuracy** | **0.00% (0 / 200)** | **0.00% (0 / 200)** | Unchanged (No generalization) |
| **Attractor Rate** | **75.00% (150 / 200)** | **3.00% (6 / 200)** | **-72.00% (Reduced hash loops)** |
| **Terminal EOS Emission Rate** | **0.00% (0 / 200)** | **61.00% (122 / 200)** | **+61.00% (Learned termination)** |
| **Hallucinated Distractor Rate** | 0.00% | 0.00% | Model did not pick distractor |
| **Incorrect Responses** | 50 (25.0%) | 194 (97.0%) | Replaced hash loops with garbage |

### Sample Qualitative Generations (UCIP-20 Best Checkpoint):

1. **Relation:** `person_role` (Unseen entity: `Tanvi`)  
   - **Context:** `Tanvi serves as the Astrophysics Fellow for the initiative. Furthermore, Sameer serves as the Quantum Cryptographer...`  
   - **Question:** `What position does Tanvi hold?`  
   - **Target:** `Astrophysics Fellow`  
   - **Baseline:** `################################` (Attractor)  
   - **UCIP-20:** `[2000/200/200/2/2/2/2/2/2/2/2/2/2/2` (Non-EOS numerical repetition)

2. **Relation:** `city` (Unseen entity: `Indore`)  
   - **Context:** `The central headquarters is established in Indore, located in Madhya Pradesh...`  
   - **Question:** `In which state is Indore located?`  
   - **Target:** `Madhya Pradesh`  
   - **Baseline:** `################################` (Attractor)  
   - **UCIP-20:** `[2000/200/200/2/2/2/2/2/2/2/2/2/2/2`

3. **Relation:** `organization` (Unseen entity: `Daksha Defense`)  
   - **Context:** `Enterprise Daksha Defense specializes in radar signal processors...`  
   - **Question:** `What does Daksha Defense specialize in?`  
   - **Target:** `radar signal processors`  
   - **Baseline:** `The show of the show of the show...`  
   - **UCIP-20:** `[000/200/200/200/200/2/2/2/2/2/2/2/2`

4. **Abstention Query (Absent entity):**  
   - **Prompt:** `What is the status of QuantumAlpha?`  
   - **Target:** `Information not provided in context.`  
   - **Baseline:** `################################`  
   - **UCIP-20:** `[00/200dnes.` (Emitted EOS)

---

## 5. Phase 4 — Five Complete Training Exposures (Accounting)

- **Total Dataset Records:** 1,600
- **Total Passes (Epochs):** 5 complete passes
- **Total Sequence Exposures:** $5 \times 1,600 = 8,000$ (every record seen exactly 5 times)
- **Batch Size:** 16 (100 steps per pass)
- **Total Optimizer Steps:** **500 steps** (Total time: 229.9s)
- **Total Active Target Tokens Processed:** **108,125 tokens**

### Loss Trajectory Across Complete Passes:

| Complete Pass | Step Range | Active Target Tokens | Mean Train Loss | Validation Loss | Learning Rate | Pass Time | Checkpoint Saved |
|---|---|---:|---:|---:|---:|---:|:---:|
| **Init** | Step 0 | — | — | **5.8757** | $0.00$ | — | — |
| **Pass 1** | Steps 1 – 100 | 21,625 | 5.0254 | 4.7234 | $2.43 \times 10^{-5}$ | 47.3s | `checkpoint_pass_1.pt` |
| **Pass 2** | Steps 101 – 200 | 21,625 | 3.6720 | 4.5079 | $1.94 \times 10^{-5}$ | 45.9s | `checkpoint_pass_2.pt` |
| **Pass 3** | Steps 201 – 300 | 21,625 | 3.0145 | 4.4390 | $1.18 \times 10^{-5}$ | 46.2s | `checkpoint_pass_3.pt` |
| **Pass 4** | Steps 301 – 400 | 21,625 | 2.6828 | 4.4241 | $5.13 \times 10^{-6}$ | 45.3s | `checkpoint_pass_4.pt` |
| **Pass 5** | Steps 401 – 500 | 21,625 | **2.5411** | **4.4208** | $2.50 \times 10^{-6}$ | 45.2s | `checkpoint_pass_5.pt` |

- **Best Validation Checkpoint:** `checkpoint_best.pt` (Pass 5 / Step 500, Val Loss = **4.4208**).
- **Checkpoints Saved:**
  - `experiments/ucip20/checkpoints/checkpoint_best.pt` (SHA-256: `0b4b745e923188c6320c8c7290031bff665786c4f4c1c71a74484bcaec6b1a27`)
  - `experiments/ucip20/checkpoints/checkpoint_pass_5.pt` (SHA-256: `3519ae5602b7c177b6250271f75fe3586212b96f64f70a6ff8f771b5e5bf7cff`)

---

## 6. Phase 6 — Retention & Regression Audit

Because Stage C pretraining replay was disabled as per mission protocol, the checkpoint was audited for catastrophic forgetting against the frozen Stage C validation partition (20 chunks, 10,240 tokens):

| Evaluation Suite | Baseline Step 800 | UCIP-20 Best Checkpoint | Degradation ($\Delta$) | Regression Verdict |
|---|---:|---:|---:|:---:|
| **Stage C Language Modeling Loss** | **3.9544** | **4.4462** | **+0.4918 points** | **Severe Catastrophic Forgetting** |
| **English General Continuation** | Fluent continuation | `'0'` (single digit collapse) | Complete collapse | **REGRESSION** |
| **Hindi General Continuation** | Hindi sentence | `'\n\nStegal-pecium alye'` | Nonce English words | **REGRESSION** |
| **Arithmetic Prompt** | Numbers / Math tokens | `'AAAAAAAAAAAAAAAAAAAAAAAA'` | Alpha token sink | **REGRESSION** |
| **Python Function Prompt** | Code block tokens | `'[2000/200=2000/200/200...'` | Bracket sink | **REGRESSION** |

- **Conclusion:** Training on isolated BIND without replay completely overwrote the pretrained representation space, destroying general language modeling capability while failing to learn entity binding.

---

## 7. Deliverables & Git Lineage

- **BIND Dataset Generator:** `experiments/ucip20/generate_bind_dataset.py`
- **BIND Training Engine:** `experiments/ucip20/train_bind.py`
- **Strict BIND Evaluator:** `experiments/ucip20/eval/bind_evaluator.py`
- **Baseline Evaluation Results:** `experiments/ucip20/eval/baseline_step800_bind_eval.json`
- **Post-Training Evaluation Results:** `experiments/ucip20/eval/best_bind_eval.json`
- **Training Summary:** `experiments/ucip20/checkpoints/training_summary.json`
- **Dataset Manifest:** `data/sft/ucip20_bind/dataset_manifest.json`

---

## 8. Final Recommendation

### **Recommendation: 3. BIND learning failed; investigate the specific bottleneck.**

### Scientific Summary for Project Leadership:
1. **The Capacity vs. Architecture Reality:**
   ChakrMicro v0.1 has 3.44M parameters ($d_{\text{model}}=192, L=6$). Even when exposed to 5 complete training epochs on a single isolated task, standard causal self-attention without specialized cross-attention or pointer heads cannot perform dynamic variable binding across arbitrary entity strings.
2. **Path Forward:**
   To enable context extraction and factual question answering, ChakrView must utilize its designated runtime neural architecture components (such as `chakrview/runtime/cross_attention_pointer.py` or span pointer mechanisms) rather than relying on standard autoregressive next-token prediction to synthesize bound entities from raw causal weights.
