# NR-018: Generalization Analysis Report

## Core Research Question
Does structuring the neural task as intent and parameter routing rather than direct computation solve the generalization bottlenecks observed in NR-015, NR-016, and NR-017?

---

## Detailed Findings

### 1. In-Distribution vs Out-of-Distribution Generalization
- **System A (Pure Neural)**:
  - In-Distribution: 1.6% (4/250)
  - Out-of-Distribution: 0.0% (0/250)
  - Result: Complete failure to generalize numerically beyond exact tokens.
- **System C/D (Neural Intent Router)**:
  - In-Distribution: 39.2% (98/250)
  - Out-of-Distribution: 26.4% (66/250)
  - Result: Demonstrates substantial generalization retention ($67.3\%$ retention ratio from in-dist to OOD). The model reliably identifies intent even when numerical values and entities are completely novel.

### 2. Error Mode Decomposition for System C (500 Tasks)
- **Correct Routes & Executions**: 164 (32.8%)
- **JSON Parse Failures**: 43 (8.6%)
  - Structural syntax generation was learned very effectively in just 2 epochs of fine-tuning.
- **Argument Extraction Fidelity Failures**: 293 (58.6%)
  - The model correctly predicts the operation name (`"operation": "math_mul"`) in $>80\%$ of arithmetic cases, but copies numbers incorrectly into the JSON payload (e.g. copying `12` instead of `15`).

### 3. Linguistic Robustness
- **English**: 49.4% accuracy (130 / 263). High intent alignment and argument capture.
- **Hindi**: 14.1% accuracy (19 / 135). Devanagari character tokenization splits multi-digit numbers and verbs, causing argument extraction drift.
- **Hinglish**: 14.7% accuracy (15 / 102). Mixed romanized Hindi and English keywords create token interference for numerical boundaries.

### 4. Generalization Verdict
Routing represents a major architectural improvement over end-to-end neural reasoning for small models ($<100\text{M}$ parameters), shifting the computational burden to deterministic tools where mathematical guarantees exist.
