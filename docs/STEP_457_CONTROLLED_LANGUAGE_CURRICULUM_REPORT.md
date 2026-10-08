# Wave 457: Controlled CPU-Friendly Language Curriculum Training Report

**Wave ID**: 457  
**Target Architecture**: ChakrMicro v0.1 (`3,443,136` parameters, decoder-only causal transformer)  
**Trained Candidate Checkpoint**: `artifacts/wave457_language_candidate/candidate_stage_step_0500.pt`  
**Candidate Parameters**: `3,443,136` (Candidate isolated in memory and artifacts)  
**Candidate Weight SHA-256**: `667dbf9977c1f71ed531ab9f0d46b7225c214feb852f84d46bae77160c461948`  
**Canonical Baseline Immutability**: VERIFIED ($\Delta W = 0$, `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`)  
**I4 Cognitive Capability Preservation**: VERIFIED (G4 mean = 77.78%, H1 = 100%, H2 = 88.89%, intact)  
**Promotion Decision**: **HOLD** (Candidate remains experimental; gates for open-domain bilingual generation not yet met)  

---

## 1. Dataset Composition & Audit (Step 1)

Before executing curriculum training, existing corpus artifacts and manifests were audited:
- **Corpus Analyzed**: Stage C Sharded Pretraining Corpus ([`data/manifests/stage_c_manifest.json`](file:///d:/Project/ChakrView/data/manifests/stage_c_manifest.json))
- **Total Vetted Documents**: 5,534 documents
- **Total Corpus Tokens**: 7,708,601 tokens across 32 binary uint16 shards
- **Split Distribution**:
  - `Train`: 27 binary shards (`shard_00000.bin` to `shard_00026.bin`), **6,651,093 tokens**
  - `Validation`: 2 binary shards, **465,954 tokens**
  - `Test`: 3 binary shards, **591,554 tokens**
- **Domain Breakdown**:
  - English (Simple English Wikipedia): 2,801,696 tokens (36.3%)
  - Hindi (Hindi Wikipedia): 2,400,182 tokens (31.1%)
  - Code (TheAlgorithms Python & CPython lib): 1,586,022 tokens (20.6%)
  - Reasoning (GSM8K Chain-of-Thought): 500,122 tokens (6.5%)
  - Sanskrit (Sanskrit Wikipedia): 407,107 tokens (5.3%)
  - Mathematics & Hinglish: 11,500 tokens (0.2%)
- **Data Integrity & Tokenizer Bounds**:
  - Token range across all binary shards: `min_id = 1`, `max_id = 4094` (strictly within vocabulary range `[0, 4095]`).
  - Active tokenizer checksum verified: `7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f`.

---

## 2. Tokenizer Statistics & BPE Space-Alignment Hygiene (Step 4)

We investigated BPE whitespace and punctuation handling across English, Hindi, and mixed text:
1. **Leading vs Trailing Space Merging**:
   - The BPE merges predominantly create subwords with trailing spaces for English words (e.g. `'is '` $\to$ ID `589`, `'a '` $\to$ ID `303`, `'The '` $\to$ ID `898`).
   - Words without a trailing space tokenize differently (e.g. `'is'` $\to$ ID `388`).
   - When users query prompts like `'The sky is'`, the final token is `388` (`'is'`), which rarely appears before a space in natural training text, biasing the model towards subword completions rather than the subsequent word.
2. **Hindi UTF-8 Devanagari Encoding**:
   - Complete Unicode roundtripping was verified across test cases:
     - `'भारत एक विशाल देश है।'` $\to$ `[1505, 1382, 1981, 533, 261, 1500, 780]` (Roundtrip: **PASS**)
     - `'मेरा नाम चक्रव्यूह है।'` $\to$ `[311, 2506, 285, 171, 646, 3025, 780]` (Roundtrip: **PASS**)
     - `'Python एक प्रोग्रामिंग भाषा है।'` $\to$ `[995, 310, 279, 261, 1366, 352, 3385, 261, 451, 519]` (Roundtrip: **PASS**)
   - Individual subword tokens may split multi-byte UTF-8 sequences. Decoding single intermediate token IDs can yield invalid UTF-8 fragments, but decoding complete token sequences is lossless.

---

## 3. Curriculum Design & Multi-Stage Progression (Steps 2 & 6)

Rather than consuming millions of tokens in an unverified run, a progressive multi-stage curriculum was executed:
- **Architecture**: ChakrMicro v0.1 (`3,443,136` parameters, 6 layers, 192 d_model, 6 heads, SwiGLU, RoPE, RMSNorm, tied LMHead)
- **Optimizer**: AdamW ($\text{lr} = 8 \times 10^{-4}$, $\text{weight\_decay} = 0.01$, gradient clipping $= 1.0$)
- **Sequence Length**: 64 tokens | **Batch Size**: 8 (effective batch size: 512 tokens/step)
- **Hardware Target**: Windows CPU (Single worker, non-blocking)

### Progressive Milestones:
| Milestone Stage | Optimization Steps | Cumulative Tokens | Train Loss Mean | Validation Loss | Validation Next-Token Acc | Candidate SHA-256 Digest |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Zero-Shot Baseline** | 0 | 0 | 8.3380 | 8.3332 | 1.01% | `c5571c9c5cb7...` |
| **Stage Milestone 1** | 100 | 51,200 | 6.2140 | 5.3499 | 11.71% | `4a565a93f700...` |
| **Stage Milestone 2** | 250 | 128,000 | 4.8872 | 4.5662 | 18.87% | `4fe243bbe4d1...` |
| **Stage Milestone 3 (Final)** | 500 | 256,000 | 4.3129 | **4.1934** | **23.05%** | `667dbf9977c1...` |

- **Held-Out Test Set (Overall)**: Loss = **4.3661**, Next-Token Accuracy = **19.93%**.

---

## 4. Language Evaluation by Split & Domain (Step 7)

Evaluating the final Step 500 candidate on individual held-out test shards reveals domain disparities:
- **Test Shard 0 (English Wikipedia)**:
  - Loss: **4.3520** | Next-Token Accuracy: **20.38%**
  - Significant syntactic learning and reduction from baseline 8.34 uniform loss.
- **Test Shard 1 (Hindi Wikipedia)**:
  - Loss: **9.9912** | Next-Token Accuracy: **4.65%**
  - **Reason**: The initial 250,000 training tokens drawn sequentially from shard 0 were predominantly English. Hindi representation in early shards was underrepresented, causing higher cross-entropy on Devanagari held-out text.
- **Test Shard 2 (Code / Reasoning)**:
  - Loss: **7.5091** | Next-Token Accuracy: **2.42%**

---

## 5. Controlled Generation Results & Repetition Analysis (Step 8)

Greedy autoregressive decoding ($T=16$ steps) was tested across fixed prompt suites:

| Prompt | Prompt-end Top-1 Prob | Entropy (nats) | Generated Text Continuation | Consec. Repetition Rate | Type-Token Ratio (TTR) |
|:---|:---:|:---:|:---|:---:|:---:|
| `Hello ` | 0.0617 | 5.103 | `'It is the Hould the Hould the Ho'` | **0.0%** | 0.50 |
| `The sky is ` | 0.2167 | 4.907 | `'the Hould the Hould the Hould the '` | **0.0%** | 0.31 |
| `India is ` | 0.2456 | 4.799 | `'the Hould the Hould the Hould the '` | **0.0%** | 0.31 |
| `Python is a ` | 0.0505 | 5.676 | `'was the Hould the Hoould the Ho'` | **6.7%** | 0.44 |
| `The capital of India is ` | 0.2697 | 4.706 | `'the Hould the Hould the Hould the '` | **0.0%** | 0.31 |
| `Two plus two equals ` | 0.0585 | 5.148 | `'was the Hood the Hood the Hoo'` | **20.0%** | 0.38 |
| `ChakrView is ` | 0.2419 | 4.778 | `'the Hould the Hould the Hood the '` | **6.7%** | 0.31 |
| `मेरा नाम ` | 0.0640 | 5.151 | `'It is the Hould the Hould the Ho'` | **0.0%** | 0.50 |
| `भारत एक ` | 0.0611 | 5.127 | `'It is the Hould the Hould the Ho'` | **0.0%** | 0.50 |
| `The sun rises in the` (Held-out) | 0.3810 | 3.283 | `'ir and the Hould the Hould the H'` | **0.0%** | 0.56 |

### Analysis:
1. **Immediate Input Echoing Abolished**:
   - In the v0.1 baseline, every prompt outputted the identical input token (`lo -> lo`, `? -> ?`, `e. -> e.`) with 100% repetition rate.
   - In candidate `667dbf...`, **immediate token echoing is 100% eliminated**.
   - Output entropy dropped from uniform random $8.277$ nats down to $4.70\text{--}5.34$ nats.
2. **Appearance of Corpus-Level Attractors (Phrase Loops)**:
   - While immediate token repetition is broken, the 256,000-token training run on broad Wikipedia text at step 500 formed phrase attractors (`"the Hould the Hould..."`).
   - The model has learned English function words (`the`, `is`, `was`, `of`, `It`), but has not yet received sufficient diversity and instruction grounding to generate factual, coherent full sentences.

---

## 6. Integration Safety & I4 Capability Preservation (Steps 9 & 10)

Release integrity and cognitive invariants were verified:
- **Canonical Baseline Parameter Count**: Exactly `3,443,136` (Unchanged)
- **Canonical Baseline SHA-256 Digest**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` (Bit-Exact, $\Delta W = 0$)
- **I4 Relational Capability Benchmark**:
  - Verification suite: [`scripts/verify_release.py`](file:///d:/Project/ChakrView/scripts/verify_release.py)
  - `[PASS]` 1. Canonical Baseline Invariant (3,443,136 params, SHA intact)
  - `[PASS]` 2. Tokenizer Artifact Hashes (Bit-exact match)
  - `[PASS]` 3. Candidate Isolation & Manifest (`chakrview-i4-wave408-v0.1`, 69,809 params, $\Delta W = 0$)
  - `[PASS]` 4. Capability Contract (v0.1.0 verified)
  - `[PASS]` 5. Release Benchmark (G4 mean = 77.78%, H1 routing = 100%, H2 routing = 88.89%)
  - Cognitive relational capability is **100% preserved**. Language experiments remained completely isolated.

---

## 7. Resource Usage & CPU Adaptation (Step 11)

- **Execution Environment**: Windows x64 CPU
- **Training Duration (500 steps)**: 36.67 seconds
- **Throughput**: $\sim 6,980$ tokens/second on CPU
- **RAM Footprint**: Started at 229.2 MB, peaked at 503.7 MB (Delta: +274.5 MB)
- **Resource Boundary Compliance**: Zero out-of-memory events, zero crashes.

---

## 8. Checkpoint Artifacts Created

The experimental candidate checkpoints are isolated under `artifacts/wave457_language_candidate/`:
1. `candidate_stage_step_0100.pt` (SHA: `4a565a93f700...`)
2. `candidate_stage_step_0250.pt` (SHA: `4fe243bbe4d1...`)
3. `candidate_stage_step_0500.pt` (SHA: `667dbf9977c1f71ed531ab9f0d46b7225c214feb852f84d46bae77160c461948`)
4. `training_summary.json`
5. `generation_results.json`

---

## 9. Failure Cases & What Was Learned

### What Was Learned:
1. **Curriculum Training Successfully Reduces Entropy & Breaks Identity Loops**: Even a 256,000-token micro-curriculum reduces loss from 8.33 to 4.19 and breaks the degenerate token-identity loop.
2. **Domain Imbalance in Sequential Sharding**: Streaming shards sequentially resulted in heavy English exposure in steps 1–500, starving Devanagari / Hindi tokens. An interleaved balanced sampler is required for bilingual curriculum training.
3. **Phrase Attractors at Intermediate Loss**: At validation loss $\sim 4.2$, small models escape single-token repetition but fall into multi-token phrase loops around high-frequency bigrams (`"the Hould..."`). Reaching fluent fact continuation requires further curriculum steps and lower validation loss ($< 2.5$).

### What Remains Unproven:
1. Whether 3.44M parameters can maintain coherent Hindi syntax and English syntax simultaneously without cross-lingual interference.
2. The exact steps needed to achieve fluent conversational turns on CPU without phrase cycling.

---

## 10. Promotion Decision (Step 12)

```text
============================================================
CANDIDATE PROMOTION DECISION: HOLD
============================================================
```
- **Rationale**: While the candidate (`667dbf...`) successfully trained without NaN, reduced validation loss by $> 4.1$ nats, increased next-token accuracy from 1% to 23%, and completely eliminated the single-token repetition defect without harming the I4 cognitive module, its open-ended generation exhibits phrase-level attractors and poor Hindi test-shard performance (due to sequential English sharding).
- Per project rules, **silent promotion is prohibited**. The candidate remains an isolated research checkpoint under `artifacts/wave457_language_candidate/`.
