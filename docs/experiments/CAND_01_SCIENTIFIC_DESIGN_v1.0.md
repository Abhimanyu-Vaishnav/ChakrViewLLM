# CHAKRVIEW — CAND_01 SCIENTIFIC DESIGN SPECIFICATION v1.0
## Controlled Entity-Copying & Associative Binding Under Zero-Noise Conditions

**Document ID:** `DOC-EXP-CAND01-SCI-001`  
**Location:** `docs/experiments/CAND_01_SCIENTIFIC_DESIGN_v1.0.md`  
**Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`  
**Track:** Track 1 (Entity Association and Copying)  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md` (v1.0.0)  
**Research Priority:** Priority 1 (Blocker for All Extractive Downstream Tasks)  
**Target Architecture:** ChakrMicro Baseline (3,443,136 parameters, 6L / 192D / 6H, RoPE, tied input/output embeddings)  
**Target Checkpoint:** Canonical Step 10,000 (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`, SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)  
**Status:** `READY_FOR_SCIENTIFIC_REVIEW` (Specification Only — No Execution Authorized)  
**Author:** ML Research Science & Experimental Design Specialist  
**Date:** 2026-10-31  

---

## 1. Executive Summary & Problem Context

During the Milestone 5 (M5 / Exp35) evaluation of dynamic property-conditioned extraction (DPCE), the fine-tuned ChakrMicro model achieved an extraction accuracy of exactly **0/200 (0.00%)**, despite training loss decreasing smoothly from 5.08 to 1.098. Subsequent Milestone 5.1 (M5.1) forensic investigations diagnosed four compounding systemic failures:
1. **Syntactic Template Domination:** $78.96\%$ of supervised loss tokens were static boilerplate words (`"At station"`, `"resides in"`). The model optimized syntactic sequence probabilities while leaving entity representation virtually unconstrained.
2. **Attention Sensitivity Absence:** Prompt perturbation probes revealed a mean next-token probability shift ($\Delta p$) of only $0.00014$ when switching entity slot values. The model operated entirely decoupled from dynamic context entities.
3. **Subword Fragmentation:** Natural language entity names fragmented into 3 to 6 subword tokens under the 4,096-token BPE tokenizer, compounding copying error probability.
4. **Confounded Task Framing:** Natural language conversational framing conflated three distinct capabilities: (a) syntactic template fluency, (b) context-entity retrieval/copying, and (c) key-value relational binding.

Candidate 1 (`CAND_01_CONTROLLED_ENTITY_COPYING`) is engineered to resolve this confound. By stripping away all prose framing, natural language syntax, conversational boilerplate, and multi-token entity names, CAND_01 isolates the foundational mathematical question: **Can the frozen ChakrMicro transformer architecture learn to route causal attention and reproduce an arbitrary key-value binding under clean, synthetic, low-noise conditions?**

---

## 2. Core Research Questions & Hypotheses

### 2.1 Primary Research Question
Under minimal synthetic prompt framing and 100% target-focused loss masking, can the ChakrMicro baseline checkpoint learn to associate an arbitrary key token with an arbitrary value token and reproduce the exact value token when queried with the key?

### 2.2 Formal Hypotheses
- **Alternative Hypothesis ($H_1$):**
  When trained on synthetic key-value pairs with zero conversational boilerplate and 100% loss applied strictly to the target answer token, ChakrMicro will achieve $\ge 75.0\%$ exact-match accuracy on held-out unseen key-value bindings, demonstrating that its 6-layer attention architecture is capable of forming functional induction circuits.
- **Null Hypothesis ($H_0$):**
  ChakrMicro achieves $< 5.0\%$ exact-match accuracy on held-out unseen key-value bindings (no better than random vocabulary guessing), indicating an architectural incapacity (e.g., parameter bottleneck, rotary embedding saturation, or representation collapse) to execute associative retrieval.

### 2.3 Secondary Research Questions
1. **Direct Copying vs. Associative Binding:** Does ChakrMicro succeed at copying an immediately visible token ($A \to A$) while failing at relational key-value retrieval ($A \to B$)?
2. **Distractor Degradation:** How does exact-match accuracy decay as the number of distractor key-value pairs increases from 0 to 1, 3, and 7?
3. **Positional Decay:** Does copy accuracy degrade as the distance between the definition and the query increases from 10 to 50, 100, and 200 tokens?
4. **Key Discrimination:** Can the model discriminate between keys that differ by a single character or token prefix?

---

## 3. Disentanglement of Capability Dimensions

CAND_01 decomposes associative extraction into five orthogonal, measurable behavioral axes:

```
                      [ CAND_01 CAPABILITY SPECTRUM ]
                                     |
    +--------------------------------+--------------------------------+
    |                                |                                |
[Direct Copying]         [Associative Binding]             [Interference Resistance]
• Task 1: Identity       • Task 2: Single Pair             • Task 4: Distractor Scaling
  Input: X -> X            Input: K1=V1; Q: K1 -> V1         (1 to 7 irrelevant pairs)
• Zero relational map    • 1-to-1 arbitrary binding        • Task 5: Similar-Key Disambig
                                                              (Levenshtein dist = 1)
                                     |
    +--------------------------------+--------------------------------+
    |                                                                 |
[Generalization & Unseen Binds]                          [Positional Stability]
• Task 6: Unseen Key-Value Rebinding                     • Task 7: Query Offset Scaling
  (Seen keys + Seen vals in novel pairings)                (10, 50, 100, 200 tokens)
• Task 8: Held-out Token Generalization                  • Task 8: Context Length Scaling
  (Completely disjoint token IDs)
```

1. **Memorization vs. Mechanism:** If the model only memorizes static associations, it will achieve high accuracy on seen pairs but $0\%$ on recombined pairs. Task 6 and Task 8 strictly test mechanism.
2. **Direct Copying vs. Key-Value Lookup:** Task 1 tests pure token reproduction. If Task 1 succeeds but Task 2 fails, the bottleneck is key-value relational binding, not the output softmax or token emission mechanism.
3. **Attention Routing vs. Positional Decay:** Tasks 4, 5, and 7 isolate whether attention heads selectively focus on relevant key positions or suffer from positional diffusion under RoPE.

---

## 4. The Eight Core Task Families

Each task family uses a standardized, parseable, zero-fluff format:

| Family ID | Task Name | Canonical Prompt Structure | Expected Completion | Capability Tested |
|---|---|---|---|---|
| **TF-01** | **Direct Token Copy** | `COPY: X3M1 ->` | `X3M1` | Direct contextual replication; baseline sanity check. |
| **TF-02** | **Single Key-Value Association** | `PAIR: K7P9=X3M1. QUERY: K7P9 ->` | `X3M1` | Single-step key-value relational binding without distractors. |
| **TF-03** | **Multi-Pair Association** | `PAIRS: K7=X3, M2=L8. QUERY: M2 ->` | `L8` | Selection among multiple valid bindings in local context. |
| **TF-04** | **Distractor Resistance** | `PAIRS: D1=N4, K7=X3, D2=R9. QUERY: K7 ->` | `X3` | Resistance to irrelevant context distractors (1, 3, 7 pairs). |
| **TF-05** | **Similar-Key Discrimination** | `PAIRS: K7P1=X3, K7P2=M8. QUERY: K7P1 ->` | `X3` | Prefix collision disambiguation (Levenshtein distance = 1). |
| **TF-06** | **Unseen Combination Rebinding** | `PAIRS: K1=V2, K2=V1. QUERY: K1 ->` | `V2` | Dynamic binding of familiar keys to familiar values in novel pairs. |
| **TF-07** | **Context-Position Variation** | `PAIRS: K1=V1 [PAD_TOKENS x N] QUERY: K1 ->` | `V1` | Positional robustness across token offsets (10, 50, 100, 200 tokens). |
| **TF-08** | **Context-Length Variation** | `[Dense context with N pairs] QUERY: K_target ->`| `V_target` | Capacity scaling under context saturation (total tokens: 64, 128, 256, 400). |

---

## 5. Experimental Arms & Factorial Matrix

To determine the exact causal drivers of learning, CAND_01 implements a controlled $2 \times 2$ factorial matrix across two independent variables:
- **Factor 1: Context Framing** (Minimal Key-Value Syntax vs. Natural Language Conversational Framing).
- **Factor 2: Loss Masking** (Target-Only Masked Loss vs. Unmasked Autoregressive Loss).

### 5.1 Formal Factorial Arms

| Arm ID | Context Framing | Loss Masking Policy | Supervision Focus | Primary Hypothesis Tested |
|---|---|---|---|---|
| **Arm A (Core Intervention)** | **Minimal Syntax** (`K=V`) | **Target Only** (`mask_prompt=True`, ignore template) | 100% on target token | Optimal induction circuit formation without interference. |
| **Arm B (Masking Control)** | **Minimal Syntax** (`K=V`) | **Full Sequence** (`mask_prompt=False`, prompt supervised) | ~15% on target, 85% on syntax | Tests if template gradient dominates even with minimal syntax. |
| **Arm C (Framing Control)** | **Natural Language** (Conversational) | **Target Only** (`mask_prompt=True`, entity supervised) | 100% on target entity token | Tests if natural language syntactic fluff impairs attention routing. |
| **Arm D (Historical Baseline)** | **Natural Language** (Conversational) | **Full Sequence** (Standard SFT unmasked) | ~16% entity, 84% syntax | Replicates M5/Exp35 conditions to establish exact baseline parity. |

### 5.2 Controlled Variables Across All Arms
- **Model Architecture:** ChakrMicro (6 layers, 192 hidden dim, 6 heads, 3.44M params, RoPE, tied weights).
- **Base Checkpoint:** Step 10,000 canonical checkpoint (`96357705a683482e...`).
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}$, weight decay = $0.01$).
- **Learning Rate Schedule:** Peak LR $1.0 \times 10^{-4}$, linear warmup 50 steps, cosine decay to $1.0 \times 10^{-5}$.
- **Compute Budget:** Exactly 500 optimization steps.
- **Hardware Target:** Single-core CPU deterministic execution.

---

## 6. Pre-Registered Decision Criteria

In accordance with Experiment Governance Protocol v1.0 Section 10, the outcomes are mapped strictly to three non-overlapping operational actions:

```
                  [ CAND_01 DECISION THRESHOLDS ]
                                 |
    +----------------------------+----------------------------+
    |                            |                            |
[ PROMOTE ]               [ ITERATE ]                    [ ABANDON ]
• Test Acc >= 75.0%        • Test Acc: 25.0% - 74.9%      • Test Acc < 25.0%
• Distractor (3) >= 60.0%  • Distractor (3) >= 20.0%      • Attn entropy = uniform
• Delta p >= 0.20          • Delta p: 0.05 - 0.19         • Delta p < 0.05
• Rank <= 5 / 4096         • Rank: 6 - 50 / 4096          • Rank > 50 / 4096
-> Advance to Track 2      -> Refine LR / Curricula       -> Flag Architectural Defect
```

1. **PROMOTE (Success):**
   - Held-out test exact-match copy accuracy $\ge 75.0\%$.
   - Distractor resistance (TF-04, 3 pairs) $\ge 60.0\%$.
   - Prompt sensitivity delta $\Delta p \ge 0.20$.
   - Target token average rank $\le 5$ out of 4,096.
   *Action:* Validates that ChakrMicro architecture supports associative copying; proceed to Track 2 (Curriculum & Multihop Binding).

2. **ITERATE (Partial Capability Detected):**
   - Held-out test exact-match copy accuracy between $25.0\%$ and $74.9\%$.
   - Target token average rank between 6 and 50 out of 4,096.
   *Action:* Investigate learning rate, warmup length, curriculum ordering (starting with TF-01 before TF-02), or RoPE base frequency.

3. **ABANDON / DEFECT (Architectural Bottleneck Confirmed):**
   - Held-out test exact-match copy accuracy $< 25.0\%$.
   - Prompt sensitivity delta $\Delta p < 0.05$.
   - Attention patterns show complete dispersion or persistent attractor collapse across all heads.
   *Action:* File formal architectural limitation notice. ChakrMicro (6L/192D) cannot perform induction copying without auxiliary mechanisms (e.g., explicit pointer networks, recurrent memory, or increased head capacity).
