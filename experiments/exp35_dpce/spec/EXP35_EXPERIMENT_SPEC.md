# CHAKRVIEW — EXPERIMENT SPECIFICATION: EXP35-DPCE
## Dynamic Prompt-Conditioned Extraction (DPCE) Controlled Learning Experiment

**Experiment ID:** Exp35-DPCE  
**Milestone:** M5  
**Project:** ChakrView  
**Status:** FROZEN BEFORE TRAINING  
**Date:** 2026-10-30  
**Hardware:** CPU-first execution  
**Target Architecture:** ChakrMicro (3,443,136 parameters, frozen)  

---

## 1. RESEARCH HYPOTHESIS & OBJECTIVE

### Primary Hypothesis
A small neural language model (ChakrMicro, 3.44M parameters) trained on sufficiently diverse, dynamically conditioned extraction examples with 100% unique completions can learn to map information explicitly supplied in the prompt context to the correct response span, generalizing to unseen entity combinations and novel template families.

### Null Hypothesis
The model primarily memorizes response templates and high-frequency target values; when evaluated on unseen entity combinations or template families, extraction accuracy will remain indistinguishable from zero or random fallback to training attractors.

### Core Research Question
Can ChakrMicro learn to extract a variable value from a prompt based on supplied context rather than falling back to static memorized attractor completions?

---

## 2. DATASET DESIGN SPECIFICATION

### Total Records: 1,000
- **Training Split:** 900 records (90.0%)
- **Validation Split:** 100 records (10.0%)
- **Format:** Unified JSONL records with strict fields:
  `id`, `category`, `template_family`, `prompt`, `response`, `target_value`, `entity_pool`, `difficulty`, `split`.

### Category Distribution (Target 1,000 Records)
1. `single_entity_extraction`: 150 records
2. `attribute_extraction`: 150 records
3. `multi_entity_disambiguation`: 150 records
4. `multi_attribute_selection`: 100 records
5. `slot_filling`: 100 records
6. `closed_world_question_answering`: 100 records
7. `context_dependent_extraction`: 100 records
8. `simple_multi_turn_fact_retention`: 75 records
9. `changed_fact_tracking`: 75 records

### Response Diversity Rule
- **100% unique response strings:** Every example must have a distinct, prompt-conditioned answer.
- Zero static-answer duplication.
- Every response must be computable and extractable directly from the prompt context.

### Synthetic Entity Pools (Controlled & Non-Personal)
- **Entities / Agents:** Synthetic fictional names (`Aarav`, `Meera`, `Kabir`, `Nisha`, `Rohan`, `Tara`, `Dev`, `Ananya`, `Vikram`, `Priya`, `Siddharth`, `Ishaan`, `Kavya`, `Aditya`, `Tanvi`).
- **Locations / Cities:** Fictional synthetic cities (`Rivapur`, `Nandipur`, `Devnagar`, `Somgarh`, `Chandrapur`, `Neelgiri`, `Surajgarh`, `Vayuputra`, `Indrapuri`, `Ratnagiri`).
- **Artifacts / Systems:** Synthetic project/device code names (`Orion`, `Vela`, `Nivox`, `Astra`, `Zephyr`, `Krypton`, `Apex`, `Zenith`, `Solas`, `Titan`).
- **Identifiers / Codes:** Distinct alphanumeric codes (`KX-104`, `MR-208`, `DL-502`, `TX-883`, `ZQ-411`, `VN-907`, `BW-305`, `FL-612`).
- **Attributes:** Color, role, department, integer quantity, status code, fruit, animal.

---

## 3. LEAKAGE-RESISTANT SPLIT METHODOLOGY

To prevent memorization masquerading as generalization, the 900 train / 100 val split enforces three independent barriers:
1. **Entity Combination Separation:** Specific entity pairs (e.g. `Dev` + `Surajgarh`) and specific code combinations are held out exclusively in validation.
2. **Template Family Separation:** Validation reserves distinct syntactic phrasings not present in training.
3. **Response Disjointness:** Zero target response strings in validation appear identically in training.

---

## 4. INDEPENDENT BENCHMARK SPECIFICATION (200 TASKS)

A frozen 200-task benchmark (`exp35_benchmark_200.json`) is constructed independently from the training set.
- 9 test families:
  - Single-entity extraction: 30 tasks
  - Multi-attribute extraction: 30 tasks
  - Multi-entity disambiguation: 30 tasks
  - Unseen entity combinations: 30 tasks
  - Slot filling: 20 tasks
  - Context-dependent QA: 20 tasks
  - Multi-turn retention: 20 tasks
  - Changed-fact tracking: 10 tasks
  - Distractor resistance: 10 tasks
- Strict evaluation rule: Exact word-boundary match (`\b<TARGET>\b`). No substring-only false passes.

---

## 5. CONTROLLED TRAINING CONFIGURATION

Two training arms evaluated under strictly identical hyperparameter budgets:
- **Arm A:** Initialized from `sft_checkpoint_step_250.pt` (M3 Candidate, SHA-256 `91b0bae2...`)
- **Arm B:** Initialized from `sft_checkpoint_step_50.pt` (M2 Arm A Masked Base, SHA-256 `150f6214...`)

### Training Hyperparameters
- Max Steps: 500
- Micro-batch Size: 2
- Gradient Accumulation: 2
- Effective Batch Size: 4
- Initial Learning Rate: $5.0 \times 10^{-5}$
- Min Learning Rate: $5.0 \times 10^{-6}$ (Cosine decay)
- Weight Decay: 0.01
- Gradient Clipping: 1.0
- Loss Function: Causal Cross-Entropy with response-only supervision (`mask_prompt=True`, `ignore_index=-100`)
- Validation Frequency: Every 50 steps (evaluated on `exp35_validation.jsonl`)
- Checkpoint Intervals: Steps 100, 250, 500
- Deterministic Seed: 42
- Hardware: Local CPU (float32)

---

## 6. ACCEPTANCE GATES & SUCCESS CRITERIA

| Gate | Criterion | Target Threshold |
|---|---|---|
| **Gate A: Basic Extraction** | Overall exact extraction on benchmark | $\ge 80.0\%$ |
| **Gate B: Generalization** | Accuracy on unseen entity combinations & templates | $\ge 60.0\%$ |
| **Gate C: Context & Tracking** | Multi-turn retention & changed-fact tracking | $\ge 60.0\%$ |
| **Gate D: Generation Stability** | Genuine EOS termination rate $\ge 85\%$; Repetition ratio $< 0.05$ | PASS |
| **Gate E: Experimental Integrity**| 100% hash verification, zero data leakage, local CPU | PASS |

### Classification Rubric
- **SUCCESS:** All primary gates passed.
- **PARTIAL SUCCESS:** Significant extraction improvement over baseline ($\ge 25\%$), but generalization or multi-turn gates unmet.
- **NO SIGNIFICANT IMPROVEMENT:** Benchmark extraction remains $< 10\%$.
- **FAILED:** Training instability, loss explosion, or safety protocol violation.

