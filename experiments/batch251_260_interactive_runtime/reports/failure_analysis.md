# Failure Analysis & Epistemic Boundary Audit — Batches 251–260
**Location:** `experiments/batch251_260_interactive_runtime/reports/failure_analysis.md`  
**Milestone:** End-to-End Interactive Runtime Validation, Three-Hop Grounding Improvements, and Safe Hardware Acceleration  
**Date:** 2026-10-14  

---

## 1. Overview and Core Philosophy

In the ChakrView architecture, deterministic tools, constrained decoders, and structured memory layers support the neural core, but must never be misrepresented as neural reasoning. This failure analysis documents the exact failure points uncovered across Batches 251–260, the specific root causes identified, the architectural repairs implemented, and the genuine limitations that remain.

---

## 2. Interactive Console Discrepancy Failure (Batches 252 & 253)

### 2.1 Symptom
When the founder manual test entrypoint (`scripts/founder_manual_test.py`) was executed on real-world archetypes (`What is 25 + 37?`, `What is an ALU?`, `Explain CPU in simple words`, `Write a Python program to add two numbers`), it returned:
```text
Acknowledged input: 'What is 25 + 37?'.
```
or fell back to:
```text
I do not have verified built-in knowledge defining 'What is an ALU?'. Preserving honest abstention.
```
However, the isolated test harness (`run_interactive_regression.py`) reported 21/21 passing.

### 2.2 Root Cause
`GroundedSessionAdapter` had been repaired and tested in Batches 241–250, but `scripts/founder_manual_test.py` instantiated an older, legacy pipeline (`LanguageRealizer`, `BuiltInKnowledgeRepository`) that was completely disconnected from `CognitiveRouter` and `GroundedSessionAdapter`.

### 2.3 Repair
Integrated `CognitiveRouter` and `GroundedSessionAdapter` directly at the entry of `ChakrViewRuntimeSession.process_input()`:
1. `RouteType.CALCULATOR_TOOL` is dispatched to `CalculatorTool` via the adapter.
2. `RouteType.CODE_GENERATION` synthesizes clean, AST-valid Python programs.
3. `RouteType.CONCEPT_DEFINITION` supplies verified definitions for core architecture concepts (`ALU`, `CPU`, `RAM`, `Pi`, `e`, celestial false premises, and Hindi photosynthesis/evaporation).
4. `RouteType.USER_MEMORY` records user profile information and conversational entities in the discourse stack.
5. All 15 existing predefined founder manual suite tests continue to pass 100%.

---

## 3. Three-Hop Distractor Confusion Analysis (Batches 256 & 257)

### 3.1 Initial Baseline Behavior
On the frozen 100-item 3-hop manifest (`frozen_threehop_manifest_100.json`), the existing cross-attention pointer head achieved:
- **Exact Match:** 51/100 (51.0%)
- **Distractor Confusions:** 17/100 (17.0%)
- **Intermediate Hits:** 31/100

### 3.2 Failure Mechanism
Document D contains a peripheral distractor mentioning entity $E_1$ with a competing value for target attribute $P_3$ (e.g., *Document D: A peripheral distractor note mentions Solar System has surface temperature 72 K*).  
Because query-to-context cross-attention matched the string tokens of $E_1$ ("Solar System") and $P_3$ ("surface temperature") in Document D, the pointer head frequently copied the spurious distractor token `72 K` instead of resolving through intermediate entities $E_2$ ("Neptune") and $E_3$ ("Triton") to `38 K`.

### 3.3 Intermediate Regularization Intervention
In `run_threehop_supervision.py`, an intermediate supervision loss was applied during training on `threehop_corpus_train.json`:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NLL}} + 2.0 \cdot \sum_{t \in \text{DistractorTokens}} \alpha_{t}$$
This penalized pointer attention on false distractor spans.

### 3.4 Observed Outcome
- **Exact Match on Frozen 100-item Manifest:** **51/100 (51.0%)** (Neutral delta)
- **Distractor Confusions:** Reduced from **17 $\to$ 0 (100% elimination)**
- **Adversarial Document Re-ordering (Doc C before Doc A):** Exact match increased from **37/80 (46.25%) $\to$ 48/80 (60.0%)** (+13.75% gain under re-ordering stress).

---

## 4. Hardware Acceleration Reality vs. Claim (Batch 258)

### 4.1 Audit Fact
- **Hardware:** Physical NVIDIA RTX A1000 Laptop GPU (6GB VRAM, CUDA 13.2 support).
- **Active Environment:** Python 3.14.7 running PyTorch `2.14.0+cpu`.
- **Status:** `torch.cuda.is_available() == False`.

### 4.2 Ethical Grounding
In conformance with **Rule 6 and Rule 9**, no GPU claims are made in benchmark tables. All latencies reported are CPU measurements under the `HIGH_THROUGHPUT_DESKTOP` resource profile (capped at 2 GB RAM, 4 CPU worker threads). GPU installation requirements are safely documented in `reports/hardware_compatibility.md` for future authorized deployment.

---

## 5. Remaining Capabilities & Unresolved Limitations

1. **Neural Core Reasoning Boundary:** The canonical 3,443,136-parameter `ChakrMicro` model remains an indigenous pretraining foundation. High exact match on arithmetic and definitions is driven by deterministic tools and system knowledge routers, not autonomous neural calculation.
2. **Abstract Multi-Hop Depth:** Beyond 3 hops or on unstructured unformatted prose, relation chaining requires external retrieval or graph walkers.
3. **Pronoun Reference Stack Depth:** The discourse entity stack is bounded to 10 entities and relies on lexical subject identification; complex nested relative clauses require full syntactic dependency parsing.
