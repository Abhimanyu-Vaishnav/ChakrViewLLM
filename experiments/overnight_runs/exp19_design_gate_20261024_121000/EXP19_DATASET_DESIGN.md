# ChakrView Exp19 — Dataset Diversification Design Specification

**Target Experiment**: `exp19_debiased_sft_cpu_600`  
**Location**: `experiments/overnight_runs/exp19_design_gate_20261024_121000/`  
**Timestamp**: 2026-10-24T12:12:00  
**Status**: `VALIDATED & FROZEN FOR EXPERIMENTATION`  

---

## 1. Problem Diagnosis: The 77.5% Prefix Monoculture

In `exp18_aligned_sft_cpu_300`, **77.5% (124/160)** of generated test outputs began with the identical two-word phrase `"The difference..."`.

### Root Cause in Exp18 Training Distribution
The template audit ([RESPONSE_TEMPLATE_AUDIT.json](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_design_gate_20261024_121000/RESPONSE_TEMPLATE_AUDIT.json)) revealed severe opening-template concentration across training targets:
- **`reference_resolution`**: 150 / 150 (100.0%) began with `"The pronoun 'it' designates..."`
- **`basic_arithmetic`**: 150 / 150 (100.0%) began with `"The <operation> of/when..."`
  - `"The sum..."`: 38
  - `"The difference..."`: 38
  - `"The product..."`: 37
  - `"The quotient..."`: 37
- **`uncertainty_abstention`**: 113 / 150 (75.3%) began with `"The <source> does not..."`
- **`factual_qa`**: 72 / 150 (48.0%) began with `"The chemical..."` or `"The <entity>..."`

Because 41.0% (492/1200) of training examples in Exp18 began with `"The "`, and `"The difference..."` overlapped strongly with arithmetic and base pre-training tokens, gradient descent minimized training cross-entropy by projecting almost all instructional prompts into a single memorized opening template basin.

---

## 2. Category-Specific Diversification Strategy

To prevent prefix mode collapse, training responses must be balanced across diverse, syntactically varied sentence openers while preserving 100% of the factual content, numerical values, and named entities.

### A. Basic Arithmetic (150 train records)
Instead of forcing 100% of arithmetic answers into `"The <op> of X and Y is Z"`, cycle through 4 deterministic variants based on record index:
1. **Canonical Formulation**: `"The sum of 1002 and 1007 is 2009."`
2. **Direct Operator**: `"Adding these values yields 2009."` / `"Subtracting gives 50."`
3. **Explicit Label**: `"Result: 2009."`
4. **Short Equational Syntax**: `"Sum equals 2009."` / `"Difference equals 50."`

### B. Reference Resolution (150 train records)
Instead of 100% `"The pronoun 'it' designates X"`, cycle through:
1. **Canonical Formulation**: `"The pronoun 'it' designates the antique brass compass."`
2. **Direct Pronoun**: `"'It' refers to the antique brass compass."`
3. **Linguistic Antecedent**: `"The antecedent of 'it' is the antique brass compass."`
4. **Structured Identification**: `"Designated entity: the antique brass compass."`

### C. Factual QA (Chemical Symbols & Units)
Instead of 100% `"The chemical symbol for X is Y"`, cycle through:
1. **Canonical Formulation**: `"The chemical symbol for iron is Fe."`
2. **Inverted Subject**: `"Fe is the chemical symbol for iron."`
3. **Direct Label**: `"Chemical symbol for iron: Fe."`
4. **Element-First**: `"The element iron has the symbol Fe."`

### D. Uncertainty Abstention (150 train records)
Instead of 75% `"The <document> does not record..."`, incorporate:
1. **Canonical Formulation**: `"The flight log does not state the total fuel consumption."`
2. **Direct Statement**: `"The total fuel consumption is not specified in the flight log."`
3. **Unrecorded Status**: `"This information is unrecorded in the provided log."`
4. **Source Limitation**: `"The record contains no data regarding total fuel consumption."`

---

## 3. Preservation of Invariant Constraints

1. **Semantic Fidelity**: All numbers, units, chemical symbols, entity names, and abstention truths are strictly identical.
2. **Prompt Conditioning Invariance**: Input prompts, delimiters (`Instruction:\n`, `Response:\n`), and mode wrappers remain 100% untouched.
3. **Loss-Mask Invariance**: All diversified tokens remain within the response region ($Y \neq -100$), preserving prompt loss masking.
4. **Zero Test Contamination**: Test records remain frozen. Diversification is applied exclusively to training examples in memory.
