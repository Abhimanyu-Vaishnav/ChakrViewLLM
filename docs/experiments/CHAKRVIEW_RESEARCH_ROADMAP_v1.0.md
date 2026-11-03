# ChakrView Research Roadmap v1.0
## Targeted Research Tracks, Empirical Capability Development & Grounded Learning

**Repository:** `ChakrView`  
**Location:** `docs/experiments/CHAKRVIEW_RESEARCH_ROADMAP_v1.0.md`  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md`  
**Baseline Architecture:** ChakrMicro (6 layers, 192 dim, 6 heads, 4096 vocab, 3.44M params, CPU-first)  
**Target Window:** Q4 2026 – Q1 2027  
**Version:** 1.0.0  
**Effective Date:** 2026-10-31  
**Status:** Canonical Research Roadmap  

---

## 1. Executive Summary and Strategic Principles

ChakrView is an indigenous neural language model engineered for low-resource CPU execution without external pre-trained checkpoints or hosted API crutches. Historical milestones (M2 through M5.1) demonstrated steady optimization progress (achieving stable EOS generation, repetition reduction, and 33.3% Python syntax validity), yet conclusively revealed that **lower validation loss does not produce factual comprehension, instruction following, or multi-turn conversational capability**.

This roadmap defines the minimal, scientifically disciplined sequence of controlled research tracks required to resolve ChakrMicro's demonstrated learning failure modes. It rejects broad, uncontrolled fine-tuning in favor of modular, causal investigations that isolate representation, attention routing, supervision balance, and context retention.

---

## 2. Empirical Grounding: The Baseline Evidence (M3 – M5.1)

| Milestone / Artifact | Verified Empirical Findings | Established Limitations | Primary Unresolved Hypotheses |
| :--- | :--- | :--- | :--- |
| **M3 Conversational** (`DS-M3-*`) | Step 250 reduced repetition loops; achieved 33.3% coding syntax validity; stabilized EOS stopping. | Conversational benchmark: 1/60 (1.7%); Intelligence benchmark: 2/32 (6.3%). | Model collapses into low-entropy conversational attractor basins (e.g. repetitive pleasantries). |
| **M4 Capability Audit** | Comprehensive forensic mapping across 10 capability families. | Systematic failure in entity grounding, arithmetic calculation, and structured format adherence. | Attributed failure to lack of induction circuits and capacity bottlenecks (unproven hypothesis). |
| **M5 Exp35 DPCE** (`DS-EXP35-*`) | 1,000 unique synthetic response strings; loss converged from 5.82 to 1.0980 (Arm B). | Grounded extraction benchmark: **0/200 (0.0%)** across both Arm A (scratch) and Arm B (canonical). | Hypothesis of prompt-loss masking failed to trigger dynamic copying; model over-fitted template words. |
| **M5.1 Forensic Diagnostic** | Measured token-level probabilities: 78.96% supervised tokens were static template words; only 16.53% were entity tokens; target token mean rank was 205.5/4096. | Mean probability sensitivity delta between correct and incorrect entity contexts was only **0.00014**. | Competing hypotheses: static template dominance vs. tokenizer subword fragmentation vs. lack of copy mechanism. |

---

## 3. Five Focused Research Tracks

```
                        [ CHAKRVIEW RESEARCH ROADMAP ]
                                       │
     ┌──────────────────┬──────────────┴──────────────┬──────────────────┐
     ▼                  ▼                             ▼                  ▼
[ TRACK 1 ]        [ TRACK 2 ]                   [ TRACK 3 ]        [ TRACK 4 ]
Entity Association  Sensitivity &                 Multi-Turn State   Generalization
& Copying           Causal Diagnosis              Retention          & OOD Transfer
     │                  │                             │                  │
     └──────────────────┴──────────────┬──────────────┴──────────────────┘
                                       ▼
                                  [ TRACK 5 ]
                             Capability Validation
```

### Track 1: Entity Association and Grounded Copying
- **Core Research Question:** Can a 3.44M-parameter architecture reliably learn to route attention to a variable entity token in its context window and reproduce that exact token in its completion under controlled, minimal-template conditions?
- **Scientific Objectives:**
  1. Determine the minimal supervision ratio required for ChakrMicro to perform 1-hop associative copying (e.g. `Key: X -> Target: X`).
  2. Evaluate copy fidelity across increasing context distances (10 tokens, 50 tokens, 150 tokens).
  3. Measure the impact of distractor entities introduced into the context window.
  4. Separate true associative copying from rote token memorization by enforcing complete disjointness between training entity pools and test entity pools.
- **Key Methodological Constraint:** Do not combine entity copying with complex prose conversational formatting; isolate the copy operation from stylistic template noise.

### Track 2: Sensitivity and Causal Attribution
- **Core Research Question:** What is the primary causal driver of ChakrMicro's near-zero prompt sensitivity ($\Delta p = 0.00014$ in M5.1)?
- **Competing Hypotheses to Investigate:**
  - *Hypothesis 2A (Supervision Imbalance):* 78.96% template-token loss gradient overwhelms the 16.53% entity gradient, forcing the optimizer to treat the entity as stochastic noise.
  - *Hypothesis 2B (Tokenizer Subword Fragmentation):* Unseen or multi-token entity segmentations break token-to-token attention alignment in early layers.
  - *Hypothesis 2C (Positional / Architectural Limitation):* Rotary embeddings at width 192 with 6 heads lack sufficient dimension per head (32 dim) to resolve relative offset binding between distant key-value pairs.
- **Key Methodological Constraint:** Test hypotheses via controlled ablations and factorial comparisons; do not accept any single explanation without experimental isolation.

### Track 3: Multi-Turn State Retention and Context Tracking
- **Core Research Question:** Can ChakrMicro preserve and update a discrete conversational state across sequential dialogue turns without suffering recency collapse or catastrophic attention decay?
- **Scientific Objectives:**
  1. Test state retention when an entity introduced in Turn 1 is queried in Turn 3.
  2. Evaluate the effect of intervening conversational dialogue turns (dialogue fillers vs. task-relevant content).
  3. Measure performance under contradictory state updates (e.g. `Turn 1: Box is red` $\rightarrow$ `Turn 3: Box is now green` $\rightarrow$ `Turn 4: What color is the box?`).
- **Key Methodological Constraint:** Track 3 experiments must branch only from checkpoints that have first demonstrated verified single-turn entity association in Track 1.

### Track 4: Out-of-Distribution (OOD) Generalization
- **Core Research Question:** Does associative capability acquired on simple synthetic structures generalize to novel entity types, diverse syntactic framings, and natural language passages?
- **Scientific Objectives:**
  1. Measure transfer from synthetic entity slots to natural language prose (e.g. raw text from `data/raw/*`).
  2. Quantify the performance delta between in-distribution (ID) entity combinations and completely novel (OOD) entity pairs.
  3. Ensure that generalization benchmarks do not share lexical templates with training splits.

### Track 5: Holistic Capability Validation
- **Core Research Question:** How do we empirically verify that algorithmic and associative learning translates into genuine, user-facing language model intelligence?
- **Scientific Objectives:**
  1. Replace loose substring evaluators with strict word-boundary and semantic parsers.
  2. Implement composite capability scorecards evaluating generation stability, format adherence, instruction compliance, and coding syntax validity simultaneously.
  3. Establish verified non-inferiority gates ensuring domain training produces zero regressions on canonical Step 10,000 baseline tasks.

---

## 4. Sequential Dependency Structure

Research must progress along strict dependency gates. Advanced multi-turn or conversational fine-tuning must not be attempted while low-level associative copying remains unproven:

```
[ Step 1: Track 2 Sensitivity Diagnosis & Track 1 Controlled Copying Pilot ]
                                    │
                         (Gate: Copy Accuracy > 50%)
                                    │
                                    ▼
         [ Step 2: Track 2 Tokenizer & Supervision Balance Experiments ]
                                    │
                   (Gate: OOD Generalization Verified)
                                    │
                                    ▼
            [ Step 3: Track 3 Multi-Turn State Retention Pilots ]
                                    │
                   (Gate: State Tracking Validated)
                                    │
                                    ▼
       [ Step 4: Track 4 & 5 Grounded Conversational SFT Milestone ]
```

