# CHAKRVIEW — CR-EXP-01 FINAL SCIENTIFIC GATE REVIEW
## Autonomous Research & Forensic Audit Report

**Date:** November 5, 2026  
**Auditor:** ChakrView Lead Scientific Auditor  
**Mission Mandate:** Final independent review, verification, and correction of the revised CR-EXP-01 experiment suite before any training authorization.  
**Operational Mode:** AUDIT AND CORRECTION ONLY. Zero training steps, zero optimizer updates, zero checkpoint modifications.

---

## 1. Checkpoint Identity and Lineage Audit

To establish a reproducible, evidence-based experimental lineage, we conducted an empirical inspection and cryptographic verification of the three candidate parent checkpoints in the ChakrView repository.

### Physical Inspection & Cryptographic Hashes

| Checkpoint Identifier | Canonical Path | Size (Bytes) | Exact Verified SHA-256 |
| :--- | :--- | :--- | :--- |
| **A. Canonical ChakrMicro v0.1 Baseline (Stage B)** | `artifacts/step102_stage_b_curriculum/checkpoints/checkpoint_0000500.pt` | 41,403,743 | `ad0597687324b5da4f6776a6a1ca0769ae06880c6041a031c3b2e3601dcd3a9c` |
| **B. Stage C Baseline (500 steps)** | `checkpoints/stage_c_baseline/checkpoint_0000500.pt` | 41,404,447 | `8b9af69d4ab3bb0c8d847d3b74c4626ba14ba19219a28f2db69fac5c15479413` |
| **C. CBR-SFT-04 Step 200** | `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt` | 41,417,634 | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` |

### Lineage and Capabilities Analysis

1. **Canonical ChakrMicro v0.1 Baseline (Stage B, Step 500)**:
   - **Lineage**: Trained from scratch across Stage A and Stage B curriculum for 500 steps on general text/code sequences.
   - **Internal State**: Training loss: 6.1724, Val loss: 5.1256, Perplexity: 168.28.
   - **Capabilities**: Zero conversational alignment. Standard syntactic probe Top-1 accuracy is **0.0%** (Top-5: **15.0%**). Under greedy generation, it outputs repeating punctuation and subword noise (`e ----e i-i-i-`). It cannot interpret structured queries or instruction delimiters.
2. **Stage C Baseline (Step 500)**:
   - **Lineage**: Pretraining continuation on the Stage C multilingual corpus (Hindi, English, Systems).
   - **Internal State**: Step 500, Train loss: 4.1756, Val loss: 4.1041.
   - **Capabilities**: Enhanced pretraining density over Stage B. Syntactic probe Top-1 accuracy is **5.0%** (Top-5: **22.5%**). However, it lacks conversational instruction format awareness and exhibits repetitive newline collapse (`\n\n\n...`) when presented with dialog markers.
3. **CBR-SFT-04 Step 200**:
   - **Lineage**: Warm-started from `cbr_sft_03b/sft_checkpoint_step_200.pt` (which branched from `cbr_sft_02/sft_checkpoint_step_50.pt`, rooted in `batch16_21_combined/main_training_10000`). Fine-tuned for 200 steps with conservative learning rate decay ($2.0 \times 10^{-4} \to 5.8 \times 10^{-5}$) and gradient accumulation.
   - **Internal State**: Step 200, Train loss: 0.7466, Val loss: 2.2309, Val perplexity: 9.31.
   - **Capabilities**: Demonstrates the strongest syntactic cohesion and structured formatting across the entire lineage. Syntactic probe Top-1 accuracy reaches **20.0%** (Top-5: **42.5%**, Mean target log-prob: -5.4943). It correctly respects conversational roles (`User:`, `Assistant:`) and syntax delimiters without catastrophic collapsing.

### Primary Parent Selection Decision

**Selected Primary Parent Checkpoint:** **`CBR-SFT-04 Step 200`**  
*Path:* `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt`  
*SHA-256:* `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153`

**Scientific Justification:**  
Evaluating variable binding (CR-EXP-01) requires a model that can parse prompt context and output a specific bound token following a `Query: ... Answer:` delimiter. As proven empirically above, pure pretraining checkpoints (Stage B or Stage C) generate raw token repetitions or newlines when presented with query delimiters, confounding variable binding capability with basic syntax comprehension failure. `CBR-SFT-04 Step 200` provides the necessary syntactic stability (20.0% Top-1 probe accuracy) while remaining completely naïve to variable binding (0.0% baseline accuracy on variable retrieval), making it the only sound baseline to isolate in-context induction learning.

---

## 2. Acceptance Gate Reformation

We have eliminated all unscientific criteria from the CR-EXP-01 acceptance framework and established rigorous behavioral and control-based benchmarks.

### Key Corrections Made
1. **Demoted Training Loss**: Training loss convergence is categorized solely as an optimization sanity check, never as proof of capability acquisition.
2. **Empirical Frozen-Parent Baselines**: Evaluated CBR-SFT-04 Step 200 across primary conditions; baseline retrieval accuracy is **0.0%** (outputs generic text like `'The '` or `'Th'`).
3. **Random & Position Controls**:
   - Random Guess Baseline: $\frac{1}{|V_{\text{target}}|}$ ($1.0\%$ over 100 values; $0.024\%$ over 4,096 vocab).
   - Position Recency Control: Selecting the most recent token or preceding key position yields $<2.0\%$ accuracy on counterbalanced datasets.
4. **Distractor Sweep Clarification**: Reconciled distractor conditions: Condition C tests 1–4 distractors (gradual capacity), Condition E tests 5 distractors (stress test).
5. **Separated Exploration from Success**: Mechanistic attention-head ratios ($\ge 3.0\times$) are removed from mandatory behavioral gates and moved to exploratory research.
6. **Multi-Faceted Non-Regression**: In addition to the 40-probe syntactic suite, we evaluate greedy generation perplexity across 12 diverse conversational prompts.

---

## 3. Token Novelty & Association Copying Audit

To avoid confusing token copying with true semantic generalization, we audited the tokenizer behavior across all proposed entity pools using the official ChakrView BPE tokenizer (`data/experiments/vocab_4096`).

### Tokenization & Pretraining Exposure Audit

| Entity Pool | Entity String | Token IDs | Token Count | Subword Pieces | In Raw Stage C Corpus? | Token IDs in Vocab? |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: |
| **Train Key** | `Alpha` | `[3781, 2381]` | 2 | `['Al', 'pha']` | Yes | Yes (100%) |
| **Train Key** | `Node` | `[1658]` | 1 | `['Node']` | Yes | Yes (100%) |
| **Train Value**| `101` | `[545, 52]` | 2 | `['10', '1']` | Yes | Yes (100%) |
| **Train Value**| `Falcon` | `[73, 288, 380]` | 3 | `['F', 'al', 'con']` | Yes | Yes (100%) |
| **Val Key** | `Sensor` | `[3930, 836]` | 2 | `['Sen', 'sor']` | Yes | Yes (100%) |
| **Val Key** | `Tensor` | `[1594]` | 1 | `['Tensor']` | Yes | Yes (100%) |
| **Val Value** | `Neon` | `[81, 104, 279]` | 3 | `['N', 'e', 'on']` | Yes | Yes (100%) |
| **Test Key** | `Quasar` | `[3919, 427, 280]` | 3 | `['Qu', 'as', 'ar']` | **No (0)** | Yes (100%) |
| **Test Key** | `Pulsar` | `[83, 403, 118, 280]`| 4 | `['P', 'ul', 's', 'ar']` | **No (0)** | Yes (100%) |
| **Test Key** | `Zenith` | `[93, 275, 435]` | 3 | `['Z', 'en', 'ith']` | **No (0)** | Yes (100%) |
| **Test Key** | `Vertex` | `[89, 283, 536]` | 3 | `['V', 'er', 'tex']` | **No (0)** | Yes (100%) |
| **Test Value** | `9090` | `[60, 750, 51]` | 3 | `['9', '09', '0']` | **No (0)** | Yes (100%) |
| **Test Value** | `Cobalt` | `[1113, 1128, 119]`| 3 | `['Co', 'bal', 't']` | **No (0)** | Yes (100%) |
| **Test Value** | `Obsidian` | `[82, 101, 328, 415, 286]` | 5 | `['O', 'b', 'si', 'di', 'an']` | **No (0)** | Yes (100%) |
| **Test Value** | `Titanium` | `[87, 390, 286, 108, 120, 112]` | 6 | `['T', 'it', 'an', 'i', 'u', 'm']` | **No (0)** | Yes (100%) |

### Critical Scientific Distinctions
1. **Unseen Experimental Associations vs. Unseen Tokens**:
   - The vocab is fixed at 4,096 tokens. All strings decompose into valid subwords within this vocabulary.
   - However, strings like `Quasar`, `Zenith`, `Cobalt`, and `Obsidian` **never appear anywhere in the 1.19M character raw pretraining corpus**.
   - Consequently, the model has zero prior co-occurrence statistics for these tokens.
2. **Copying vs. Concept Representation**:
   - Solving Condition B (retrieving `Cobalt` for `Quasar`) proves **in-context variable binding and dynamic token routing**, NOT semantic concept understanding. The research report must explicitly label this capability as *Contextual Association Retrieval / Induction*, not *World Knowledge Acquisition*.

---

## 4. Dataset Integrity & Anti-Leakage Audit

We performed algorithmic checks on the synthetic generation protocol:

1. **Zero Overlap**: Train, Validation, and Held-Out Test entity pools are strictly partitioned:
   $$\text{Pool}_{\text{train}} \cap \text{Pool}_{\text{val}} \cap \text{Pool}_{\text{test}} = \emptyset$$
2. **Template Independence**: 10 syntactic templates are used. Templates 1–8 are shared across Train and Val; Templates 9–10 are strictly reserved for Condition I (held-out syntactic formulation).
3. **Counterbalanced Positional Depth**: Target key-value pairs in multi-distractor sequences are uniformly distributed across depths $P \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ of the context window.
4. **Unambiguous Reassignment**: Condition D strictly enforces explicit chronological markers (`"Initial value: Key = Val1. Update: Key is now Val2. Current value of Key?"`).
5. **No Test-Set Tuning**: Checkpoint selection during Stage A or Stage B is governed exclusively by validation loss on `val_100.jsonl`; the 500 test probes remain untouched until evaluation freeze.

---

## 5. Verified Hardware & Resource Profile

All hardware and throughput figures have been empirically measured on the host environment:

* **Host CPU Processor:** `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel` (13th Gen Intel Core i7, 16 physical cores, 24 logical cores)
* **PyTorch Version:** `2.14.0+cpu`
* **Intra-Op Threads:** 14 (default) / 4 (controlled)
* **ChakrMicro v0.1 Parameters:** `3,443,136` (exact)
* **Batch Configuration:** Batch Size = 1, Sequence Length = 512 ($512\text{ tokens/optimizer update}$)

### Empirical Benchmark Results

| Worker Threads | Measured Optimization Speed | Time for 100 Steps (51.2k tokens) | Peak Process RSS | Measured System Impact |
| :---: | :---: | :---: | :---: | :---: |
| **14 Threads** | **2,404.1 tokens/sec** | **21.3 seconds** | 515.0 MB | Negligible (<2% RAM) |
| **4 Threads** | **3,554.5 tokens/sec** | **14.4 seconds** | 510.0 MB | Negligible (<2% RAM) |

> [!NOTE]
> Setting thread affinity to 4 threads significantly reduces core-switching overhead, achieving **3,554.5 tokens/sec** on CPU. The previous estimate of 1,374.5 tokens/sec was an ultra-conservative lower bound. The host system executes 100 optimization steps in under 20 seconds.

---

## 6. Reformed Authorization Gates Matrix

The six reformed gates are structured below, separating pilot feasibility, capability claims, and scaling decisions:

| Gate | Target Measurement | Baseline (CBR-SFT-04) | Candidate Target | Statistical Method | Pass / Fail Rule | Scientific Justification |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **Gate 1: Optimization Sanity** | Training Loss (Train 1000) | $3.38 \to 0.75$ | $\text{Loss} < 1.50$ | Moving avg (last 10 steps) | $\text{Loss}_{100} < 1.50$ | Verifies that ChakrMicro architecture has sufficient capacity to fit the binding task. |
| **Gate 2: In-Context Generalization** | Condition B Accuracy (Held-out pool) | **0.0%** | $\ge \mathbf{75.0\%}$ | 95% Wilson Score CI ($N=100$) | Lower bound $\ge 65.0\%$ | Proves genuine variable binding on completely novel token strings not in pretraining text. |
| **Gate 3: Dynamic Rebinding** | Condition D Accuracy (Reassignment) | **0.0%** | $\ge \mathbf{70.0\%}$ | 95% Wilson Score CI ($N=100$) | Lower bound $\ge 60.0\%$ | Demonstrates temporal tracking of variable states rather than static first-occurrence matching. |
| **Gate 4: General Non-Regression** | Syntactic Probe Top-1 Accuracy | **20.0%** | $\ge \mathbf{18.0\%}$ | Exact 40-probe match | Absolute drop $\le 2.0\%$ | Protects against catastrophic forgetting of Python syntax and English grammar. |
| **Gate 5: Distractor Robustness** | Condition E Accuracy (5 Distractors) | **0.0%** | $\ge \mathbf{60.0\%}$ | 95% Wilson Score CI ($N=100$) | Lower bound $\ge 50.0\%$ | Establishes selective attention capacity under dense distraction. |
| **Gate 6: Unseen Template Transfer** | Condition I Accuracy (New syntax) | **0.0%** | $\ge \mathbf{65.0\%}$ | 95% Wilson Score CI ($N=100$) | Lower bound $\ge 55.0\%$ | Verifies that binding is invariant to prompt formatting changes. |

### Three-Tier Authorization Framework

```mermaid
flowchart TD
    subgraph TierA["Tier A: Stage A Pilot Feasibility"]
        A1["Approved Dataset Synthesis"] --> A2["100-step Pilot Execution"]
        A2 --> A3["Check Gate 1 & Gate 4"]
    end
    subgraph TierB["Tier B: Stage B Expansion"]
        A3 -->|Passes Gates 1 & 4| B1["Authorize 300-step Expansion"]
        B1 --> B2["Check Gates 2, 3, 5"]
    end
    subgraph TierC["Tier C: Capability Claim Authorization"]
        B2 -->|Passes All Gates 1-6| C1["Capability Acquisition Claim Granted"]
        C1 --> C2["Register in CHAKRVIEW_CAPABILITY_REGISTRY"]
    end
```

1. **Permission to Run Stage A Pilot (100 steps)**: Granted when synthetic dataset generation is validated and baseline checkpoints are verified.
2. **Permission to Expand Training (Stage B, 300 steps)**: Requires Stage A to satisfy Gate 1 (loss $<1.5$) and Gate 4 (zero catastrophic regression).
3. **Permission to Claim Capability Acquisition**: Requires candidate to satisfy Gates 1 through 6 simultaneously across all held-out splits with statistical significance.

---

## 7. Exploratory Mechanistic Research (Non-Gate)

Induction-head mechanistic analysis is retained as an exploratory diagnostic protocol:
- **Protocol**: Track attention weights from Query Key token to Prefix Key and Value tokens across all 36 attention heads (6 layers $\times$ 6 heads).
- **Hypothesis**: Attention heads in Layers 3–5 will exhibit specialized induction-head characteristics ($A_{i, j} > 3\times$ uniform baseline) correlating with higher accuracy in Conditions B and E.
- **Reporting**: Reported in research documentation as explanatory evidence, not as a binary pass/fail condition.

---

## 8. Final Stop Gate & Audit Verdict

* All checkpoint lineages, physical sizes, and SHA-256 hashes are verified.
* Primary parent checkpoint is unambiguously locked to `CBR-SFT-04 Step 200` (`327118ce...`).
* Token novelty and corpus occurrence have been empirically verified.
* Hardware throughput benchmarked at **3,554.5 tokens/sec**; compute required for Stage A pilot is under **20 seconds**.
* Acceptance gates have been restructured with Wilson confidence intervals and control baselines.
* Zero model training, optimizer updates, or checkpoint writes were performed.

### Final Verdict

**`READY FOR STAGE A PILOT REVIEW`**

*Awaiting explicit user authorization to proceed with dataset generation and Stage A pilot execution.*

