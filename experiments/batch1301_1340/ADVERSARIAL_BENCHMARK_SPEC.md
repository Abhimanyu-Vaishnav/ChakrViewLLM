# Adversarial Benchmark Specification — Batches 1301–1340
## 100-Item Decision Benchmark for Evidence-Grounded Refutation and Safety

### 1. Specification Overview
- **File**: `experiments/batch1301_1340/data/adversarial_decision_100.json`
- **Manifest**: `experiments/batch1301_1340/data/ADVERSARIAL_BENCHMARK_MANIFEST.json`
- **Total Test Items**: 100
- **SHA-256**: `c87654b1ade2aeb0619f1c20e644f582e7a1afb7f4dd1eb154086ec1442308a7`
- **Decision Distribution**:
  - `ANSWER`: 58 items
  - `REFUTE`: 22 items
  - `ABSTAIN`: 20 items

### 2. Category Distribution & Design Rationale

| Category | Count | Primary Expected Decision | Design Rationale & Challenge |
| :--- | :---: | :---: | :--- |
| **`numerical_premise`** | 20 | 10 `REFUTE`, 10 `ANSWER` | Paired pairs: False asserted numbers vs true numbers on identical attributes. |
| **`permission_prohibition`**| 10 | 5 `REFUTE`, 5 `ANSWER` | Tests action/subject alignment (e.g. sudo forbidden vs unprivileged execution allowed). |
| **`why_valid_with_context_num`**| 10 | 10 `ANSWER` | Valid "Why" questions containing contextual numbers (e.g. "on 1 machine", "for 1 user"). Must NOT falsely refute. |
| **`attribute_unit_discrimination`**| 10 | 5 `REFUTE`, 5 `ANSWER` | Tests attributing numbers to the wrong property (e.g. feed-forward 1536 vs attention heads). |
| **`version_conflicts`** | 10 | 4 `REFUTE`, 6 `ANSWER` | V0.6 (10.8M parameters, 6 layers) vs V0.7 (24.5M parameters, GQA). |
| **`honest_abstention_unsupported`**| 20 | 20 `ABSTAIN` | Out-of-domain and unindexed topics (including database tokens). Must trigger `ABSTAIN`. |
| **`grounded_retrieval_and_hinglish`**| 20 | 20 `ANSWER` | Grounded facts, technical synonyms, paraphrases, and Hinglish queries. |
