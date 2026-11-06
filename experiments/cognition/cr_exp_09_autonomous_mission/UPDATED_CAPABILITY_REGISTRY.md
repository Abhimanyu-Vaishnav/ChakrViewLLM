# CHAKRVIEW CAPABILITY REGISTRY
**Status:** Canonical Living Document  
**Last Updated:** November 6, 2026 (Post CR-EXP-09 Autonomous R&D Mission)  
**Evaluator Architecture:** ChakrMicro v0.1 (3,443,136 parameters, d=192, l=6, h=6, vocab=4096)  
**Standardized Evaluation Harness:** Frozen 600-Probe Multi-Dimensional Capability Suite (`benchmark_suite.json`)

---

## 1. OFFICIAL MODEL LINEAGE & BENCHMARK SCORECARD

The table below catalogs all benchmarked candidate models against the frozen 600-probe multi-dimensional cognition suite:

| Capability Dimension | Baseline Arm B (`cr_exp_08_arm_b`) | Candidate 1 (`candidate_1_best_val`) | Candidate 2 (`candidate_2_best_val`) | Candidate 3 Champion (`candidate_3_best_val`) |
|---|---|---|---|---|
| **Conversational Persona** | **82.67%** (62/75) | 77.33% (58/75) | 77.33% (58/75) | 77.33% (58/75) |
| **Single-Hop & Counterfactual** | 52.00% (39/75) | 12.00% (9/75) | 57.33% (43/75) | **64.00%** (48/75) |
| **Distractor Disambiguation** | 28.00% (21/75) | 30.67% (23/75) | 33.33% (25/75) | **42.67%** (32/75) |
| **Epistemic Abstention** | 0.00% (0/75) | **98.67%** (74/75) | 40.00% (30/75) | 46.67% (35/75) |
| **Multi-Turn Working Memory** | 4.00% (3/75) | 53.33% (40/75) | 64.00% (48/75) | **78.67%** (59/75) |
| **Elementary State Arithmetic** | 0.00% (0/75) | 30.67% (23/75) | 36.00% (27/75) | **44.00%** (33/75) |
| **Conditional Logic Deduction** | 0.00% (0/75) | 60.00% (45/75) | 9.33% (7/75) | **37.33%** (28/75) |
| **Constraint Satisfaction** | 42.67% (32/75) | 0.00% (0/75) | 40.00% (30/75) | 16.00% (12/75) |
| **GRAND TOTAL ACCURACY** | **26.17% (157/600)** | **45.33% (272/600)** | **44.67% (268/600)** | **50.83% (305/600)** |

---

## 2. CAPABILITY REGISTRY SUMMARY

### 1. Conversational Understanding & Persona Preservation: **CONFIRMED CAPABILITY**
- **Status**: Stable across all iterations ($77.33\%$ to $82.67\%$).
- **Evidence**: Retains cordial greetings, persona identification as "ChakrView", helpful sign-offs, and refusal to claim external embodiment without collapse.

### 2. Single-Hop & Counterfactual Variable Binding: **CONFIRMED CAPABILITY**
- **Status**: Advanced from $52.0\%$ (Baseline) to $64.0\%$ (Candidate 3).
- **Evidence**: Model accurately reads non-canonical assignments (e.g., `"Ravi lives in Jaipur." -> "Jaipur"`) rather than defaulting to memorized training priors.

### 3. Multi-Turn Working Memory & Latest-Fact Updating: **CONFIRMED BREAKTHROUGH**
- **Status**: Grew from $4.0\%$ (Baseline) to **$78.67\%$** (Candidate 3).
- **Evidence**: In conversational sequences where an entity changes locations across turns (`"Aarav was in Delhi" -> "Aarav moved to Pune" -> "Where is Aarav now?"`), the model tracks state progression and outputs the updated location without recency forgetting.

### 4. Epistemic Abstention on Missing Context: **CONFIRMED CAPABILITY**
- **Status**: Grew from $0.0\%$ (Baseline) to $46.67\%$ (Candidate 3 calibrated) / $98.67\%$ (Candidate 1 uncalibrated).
- **Evidence**: When asked about entities not present in context, the model explicitly abstains with `"The context does not state..."` instead of hallucinating random entities.

### 5. Elementary Arithmetic & State Transformations: **EMERGING CAPABILITY**
- **Status**: Grew from $0.0\%$ (Baseline) to **$44.0\%$** (Candidate 3).
- **Evidence**: Solves small structured increment/decrement balance updates ($n \pm k \le 20$).

### 6. Conditional Logic & Modus Ponens: **EMERGING CAPABILITY**
- **Status**: Grew from $0.0\%$ (Baseline) to **$37.33\%$** (Candidate 3).
- **Evidence**: Evaluates `If P then Q; P; Therefore Q` logic templates with binary `"Yes"` / `"No"` determination.

### 7. Multi-Entity Distractor Resistance: **EMERGING CAPABILITY**
- **Status**: Grew from $28.0\%$ (Baseline) to **$42.67\%$** (Candidate 3).
- **Evidence**: When contexts contain 3 or more named entities, correctly isolates the queried subject and suppresses neighbouring entity tokens.

### 8. Strict Negative Constraint Filtering: **FRONTIER DEFICIT**
- **Status**: Regressed to $16.0\%$ in Candidate 3 as multi-task capacity squeezed single-token negative set negation. Marked as primary target for future scaling.
