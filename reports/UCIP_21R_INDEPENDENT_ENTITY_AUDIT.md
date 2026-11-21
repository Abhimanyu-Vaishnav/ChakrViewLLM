# UCIP-21R — Independent Entity Representation & Causal Conditioning Audit Report

**Mission ID:** UCIP-21R  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Starting HEAD:** `9928679`  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Mission Type:** Independent Forensic Verification (Zero Training, Zero Model/Runtime Modification)  
**Execution Timestamp:** 2026-11-21T13:05:00+05:30  
**Target Checkpoint Audited:**
- Base Checkpoint: `experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt` (SHA-256: `8759ccb4...`)
- UCIP-20 Best Checkpoint: `experiments/ucip20/checkpoints/checkpoint_best.pt` (SHA-256: `0b4b745e...`)
- Protected Runtime: `chakrview/runtime/session.py` (SHA-256: `9080ff51...`)

---

## 1. Executive Summary & Core Scientific Verdict

### Primary Objective
Independently verify whether ChakrMicro actually represents entity identity, relation information, and contextual bindings, and whether these representations influence generation-boundary logits.

### Final Scientific Verdict: **VERDICT A: Entity identity and relation information are both accessible, but generation-boundary conditioning is weak (coexisting with strong unconditional response prior dominance).**

### Core Findings & Quantitative Discoveries:
1. **Entity Identity Representation is Empirically Established (Phase 1):**
   - **In-Distribution Closed-Set Entity Probing:** On the 90 training entities, a linear classifier achieves **49.31% validation accuracy** (Chance baseline: $1.11\%$, a **44x improvement** over chance).
   - **Held-Out Unseen Entity Clustering (Open-Set Evaluation):** On the 18 held-out, completely unseen development entities (180 dev probes, 100% disjoint from training), nearest-neighbor cosine retrieval achieves **90.00% entity discrimination accuracy** (Chance baseline: $5.56\%$, Random permutation control: $8.33\%$). Representations of distinct queries referring to the same unseen entity cluster tightly in Layer 5 representation space ($T_{\text{last}}$).
   - **Binding and Relation Probing:** Relation identity is decoded with **99.50%** accuracy and language with **100.00%** accuracy.
   - **Conclusion:** ChakrMicro *does* represent both entity identity and relation structure internally.
2. **Entity-Specific Logit Sensitivity is Virtually Zero (Phase 3):**
   - In paired context-selection margin tests:
     $$\text{Margin}(A) = \log P(\text{tok}_A \mid \text{context}_A) - \log P(\text{tok}_A \mid \text{context}_B)$$
     Across 400 evaluated entity-pair comparisons, the mean margin is **$-0.0464$** (median: $-0.0020$, standard deviation: $0.3483$).
   - The positive margin fraction is only **48.50%** (essentially a coin toss).
   - Context manipulation does not systematically increase the probability of the correct entity token over an alternative entity token at the generation boundary.
3. **Factorial Causal Interventions (Phase 2):**
   - Swapping context entities produces a tiny output distribution shift of $D_{\text{JS}} = 0.020051$ and logit cosine similarity of $0.9961$.
   - The generation-boundary distribution is dominated by invariant pre-training priors (`A` and `[`), which capture over $75\%$ of argmax decisions regardless of which entity is present in the context.
4. **Independent Reproduction of UCIP-21 (Phase 4):**
   - 100% of reported UCIP-21 metrics (relation probe accuracy: $99.5\%$, language probe: $100\%$, entity swap JS divergence: $0.0178$, free-generation accuracy: $0.00\%$) were bit-exact reproduced and verified against raw machine-readable JSON files.

---

## 2. Phase 0 — Artifact Integrity & Lineage Verification

All physical checkpoint files, datasets, and runtime code were verified bit-exact:

| Artifact Name | Expected Location | SHA-256 Checksum | Invariant Status |
|---|---|---|:---:|
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **MATCH ($\Delta W=0$)** |
| **Baseline Step 800 ckpt** | `experiments/ucip13/.../checkpoint_step_800.pt` | `8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943` | **MATCH** |
| **UCIP-20 Best ckpt** | `experiments/ucip20/checkpoints/checkpoint_best.pt` | `0b4b745e923188c6320c8c7290031bff665786c4f4c1c71a74484bcaec6b1a27` | **MATCH** |
| **BIND Train Split** | `data/sft/ucip20_bind/bind_train_1600.jsonl` | `4b376b0fbf46eac6060ec7d881f6de73be19087c9717847e91eb5877b4972cd6` | **MATCH** |
| **BIND Val Split** | `data/sft/ucip20_bind/bind_val_200.jsonl` | `3de0e86fb34b0a1b814b8485432574b6df9760a485e59d53f2c09e9e433cc432` | **MATCH** |
| **BIND Dev Split** | `data/sft/ucip20_bind/bind_dev_200.jsonl` | `698b21208f9f1484e2112eee08f1c9c6ead3d7e9f004abb9b3c6d1d01c52fa62` | **MATCH** |

An immutable manifest was recorded in `experiments/ucip21r/manifests/input_manifest.json`.

---

## 3. Phase 1 — Entity Identity & Binding Representation Probing

### Leakage Safety Audit
- Train Entities: 90 unique entities.
- Validation Entities: 18 unique entities.
- Development Entities: 18 unique entities.
- Train-Dev Entity Overlap: **0 (0.00%)**.
- Val-Dev Entity Overlap: **0 (0.00%)**.
- Strict zero-leakage verified.

### Probe Results

| Probing Methodology | Target Evaluated | Number of Classes | Chance Baseline | Empirical Accuracy | Control Performance | Statistical Significance |
|---|---|---:|---:|---:|---:|:---:|
| **Closed-Set Linear Probe** | 90 Train Entities | 90 | 1.11% | **49.31%** (Val) | — | $p < 10^{-15}$ |
| **Open-Set Nearest Neighbor** | 18 Unseen Dev Entities | 18 | 5.56% | **90.00%** | 8.33% (Permutation) | $p < 10^{-15}$ |
| **Relation Linear Probe** | 10 Relations | 10 | 10.00% | **99.50%** | 10.00% (Majority) | $p < 10^{-15}$ |
| **Language Linear Probe** | 3 Languages | 3 | 33.33% | **100.00%** | 33.33% | $p < 10^{-15}$ |

### Empirical Takeaway
The hypothesis that ChakrMicro fails to represent entity identity is **DISPROVEN**. Distinct prompts that query the same held-out entity project to tightly clustered representations in Layer 5 ($90.00\%$ nearest-neighbor accuracy). The network distinctly encodes *both* the query relation and the specific entity identity.

---

## 4. Phase 2 — Factorial Causal Intervention

Five factorial conditions were evaluated across paired probes:

| Factorial Condition | Description | Mean $D_{\text{JS}}$ | Layer 5 Cosine | Logit Cosine | Top-1 Stability | Top-5 Overlap |
|---|---|---:|---:|---:|---:|---:|
| **Cond A (Unchanged)** | Same entity, same relation | 0.000000 | 1.0000 | 1.0000 | 100.0% | 100.0% |
| **Cond B (Entity Swap)** | Different entity, same relation | **0.020051** | **0.9324** | **0.9961** | **55.56%** | **77.33%** |
| **Cond C (Relation Swap)** | Same entity, different relation | 0.000067 | 0.9996 | 1.0000 | 100.0% | 99.56% |
| **Cond D (Combined Swap)** | Different entity, different relation | 0.176663 | 0.6718 | 0.9755 | 28.89% | 39.11% |
| **Cond E (Nuisance Control)** | Same entity/relation, added noise | 0.579282 | -0.0005 | 0.9003 | 0.00% | 0.00% |

### Factorial Isolation
- Changing the entity while holding relation constant (Condition B) shifts Layer 5 hidden similarity to $0.9324$, but this representation divergence is compressed into a tiny $0.020051$ JS divergence at the vocabulary level.
- Logit cosine similarity remains $0.9961$, confirming that entity-specific representation changes fail to propagate effectively through the unembedding projection.

---

## 5. Phase 3 — Entity-Specific Logit Sensitivity & Margin Analysis

To evaluate whether the output distribution provides a selective advantage for the correct entity's first answer token, paired context-selection margins were evaluated across 400 comparisons:

$$\text{Margin}(A) = \log P(\text{tok}_A \mid \text{context}_A) - \log P(\text{tok}_A \mid \text{context}_B)$$

| Metric | Measured Value | Theoretical Expectation for Responsive Logits | Status |
|---|---:|---:|:---:|
| **Total Evaluated Margins** | 400 | — | Complete |
| **Mean Context-Selection Margin** | **-0.0464** | $> +1.0$ | **Decoupled** |
| **Median Margin** | **-0.0020** | $> +1.0$ | **Decoupled** |
| **Standard Deviation** | 0.3483 | — | — |
| **Positive Margin Fraction** | **48.50%** | $> 90.0\%$ | **Chance Level (~50%)** |
| **Mean Rank Shift under Correct Context** | **-2.1 positions** | Large positive rank gain | Negligible shift |

### Definitive Conclusion on Logits:
A positive margin rate of only **48.50%** proves that presenting Context A versus Context B does not reliably increase the log-probability of Entity A's answer token. The output logits at position $T_{\text{last}}$ are effectively **uncoupled** from the entity identity represented in the hidden state.

---

## 6. Phase 4 — Independent Reproduction of UCIP-21 Findings

All metrics reported in UCIP-21 were audited against raw machine-readable JSON records:

| Audited Metric | UCIP-21 Reported | UCIP-21R Re-measured | Audit Status |
|---|---:|---:|:---:|
| **Relation Probe Dev Accuracy** | 99.50% | 99.50% | **EXACT_MATCH** |
| **Language Probe Dev Accuracy** | 100.00% | 100.00% | **EXACT_MATCH** |
| **Abstention Probe Dev Accuracy** | 93.50% | 93.50% | **EXACT_MATCH** |
| **Entity Swap JS Divergence** | 0.017824 | 0.017824 | **EXACT_MATCH** |
| **Altered-Prefix NLL Increase** | +0.2490 | +0.2490 | **EXACT_MATCH** |
| **Free-Generation Accuracy** | 0.00% | 0.00% | **EXACT_MATCH** |
| **Cross-Pair Top-1 Agreement** | 48.00% | 48.00% | **EXACT_MATCH** |

---

## 7. Phase 5 — Causal Continuation Analysis

When forced with an alternative entity's initial token on Context A, rollout was evaluated for 50 cases:
- **Switched to Alternative Entity:** **0.0% (0 / 50)**
- **Retained Context A Entity:** **0.0% (0 / 50)**
- **Collapsed into Attractor Fragment:** **100.0% (50 / 50)** (e.g. `Mangala`, `subframe`, `Aeroology`)

### Finding:
The model does not perform semantic self-correction. Once an incorrect or corrupted token is placed in the prefix, autoregressive decoding immediately locks into high-frequency unconditioned subword attractors.

---

## 8. Final Scientific Classification Matrix

| Hypothesis | Evaluation | Empirical Evidence |
|---|:---:|---|
| **A. Entity identity and relation are accessible, but generation-boundary conditioning is weak** | **FULLY SUPPORTED** | $90\%$ open-set entity retrieval; $99.5\%$ relation probe; paired logit margin is $-0.0464$ (48.5% positive rate); entity swap JS is $0.020$. |
| **B. Relation accessible, but entity identity not established** | **DISPROVEN** | Nearest-neighbor retrieval achieves $90.00\%$ accuracy on held-out unseen entities. |
| **C. Entity-specific logits respond to context** | **DISPROVEN** | Paired logit margin is $-0.0464$ with $48.5\%$ positive rate (chance level). |
| **D. Suppressed solely by unconditional response priors** | **PARTIALLY TRUE** | Priors dominate ($>75\%$ argmax), but logits also show near-zero entity sensitivity even below argmax. |

---

## 9. Deliverables and Invariant Audit Record

- Primary Diagnostic Report: [UCIP_21R_INDEPENDENT_ENTITY_AUDIT.md](file:///D:/Project/ChakrView/reports/UCIP_21R_INDEPENDENT_ENTITY_AUDIT.md)
- Suite Script: [run_ucip21r_suite.py](file:///D:/Project/ChakrView/experiments/ucip21r/scripts/run_ucip21r_suite.py)
- Manifests:
  - Input Manifest: [input_manifest.json](file:///D:/Project/ChakrView/experiments/ucip21r/manifests/input_manifest.json)
  - Output Manifest: [output_manifest.json](file:///D:/Project/ChakrView/experiments/ucip21r/manifests/output_manifest.json)
- Raw Result JSONs:
  - [phase_1_entity_identity_probing.json](file:///D:/Project/ChakrView/experiments/ucip21r/results/phase_1_entity_identity_probing.json)
  - [phase_2_factorial_intervention.json](file:///D:/Project/ChakrView/experiments/ucip21r/results/phase_2_factorial_intervention.json)
  - [phase_3_logit_sensitivity_margins.json](file:///D:/Project/ChakrView/experiments/ucip21r/results/phase_3_logit_sensitivity_margins.json)
  - [phase_4_reproduce_ucip21.json](file:///D:/Project/ChakrView/experiments/ucip21r/results/phase_4_reproduce_ucip21.json)
  - [phase_5_causal_continuation.json](file:///D:/Project/ChakrView/experiments/ucip21r/results/phase_5_causal_continuation.json)
  - [phase_6_scientific_classification.json](file:///D:/Project/ChakrView/experiments/ucip21r/results/phase_6_scientific_classification.json)
- Invariants Checked:
  - Training steps: **0**
  - Checkpoint modifications: **0**
  - Dataset modifications: **0**
  - Runtime hash `9080ff51...`: **Bit-exact match**
  - Baseline Step 800 hash `8759ccb4...`: **Bit-exact match**
  - UCIP-20 Best hash `0b4b745e...`: **Bit-exact match**
