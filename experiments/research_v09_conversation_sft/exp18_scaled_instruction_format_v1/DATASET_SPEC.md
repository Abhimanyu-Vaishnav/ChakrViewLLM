# ChakrView Exp18 — Scaled Dataset Specification

**Experiment**: `exp18_scaled_instruction_format_v1`  
**Location**: `experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/DATASET_SPEC.md`  
**Version**: 1.0 (Frozen Design)  
**Total Target Size**: 1,520 records (Train: 1,200 | Validation: 160 | Test: 160)  
**Split Partitioning**: 78.9% Train / 10.5% Val / 10.5% Test  

---

## 1. Category Distribution & Sampling Strategy

The dataset spans eight orthogonal functional categories, evenly balanced across all splits:

| Category | Category Description | Train | Val | Test | Total | Deterministic Scoring Method |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`basic_arithmetic`** | Integer addition, subtraction, multiplication, and integer division | 150 | 20 | 20 | 190 | Arithmetic expression parsing & integer comparison |
| **`structured_formatting`** | Valid RFC 8259 JSON objects with specific keys and primitive types | 150 | 20 | 20 | 190 | `json.loads` parsing + key/type schema validation |
| **`context_extraction`** | Extracting entities/values explicitly present in the provided context | 150 | 20 | 20 | 190 | Normalized exact entity substring match |
| **`reference_resolution`** | Resolving pronouns/referents (`the first`, `the latter`, `it`) | 150 | 20 | 20 | 190 | Normalized referent entity match |
| **`instruction_following`** | Strict string transformations (casing, reversing, token emission) | 150 | 20 | 20 | 190 | Exact string equivalence check |
| **`factual_qa`** | Closed-domain factual questions verified against gazetteers/standards | 150 | 20 | 20 | 190 | Normalized focal fact / entity match |
| **`uncertainty_abstention`** | Abstaining when context lacks answer (75%) vs answering controls (25%) | 150 | 20 | 20 | 190 | Proof-of-absence refusal or proof-of-grounding answer |
| **`short_explanation`** | 1-2 sentence physical/causal explanations | 150 | 20 | 20 | 190 | Held out for semantic review (`NEEDS_REVIEW`) |
| **TOTAL** | | **1,200** | **160** | **160** | **1,520** | **87.5% Deterministically Scorable** |

---

## 2. Dataset Quality Audit of Exp17 Prior Art

Before scaling, the 400 Exp17 records were analyzed to identify defects and design constraints:
1. **Synthetic Token Artifacts**:
   - Exp17 relied heavily on generated identifiers: `Apollo_train_0`, `alpha_test_1`, `metric_train_0`.
   - *Impact*: The model learned to predict code-like variable names (`self._2 = self._3`) rather than real-world English words or syntax.
   - *Exp18 Remediation*: Replace synthetic identifiers with naturalistic entities, diverse JSON keys (`host`, `status`, `port`, `temperature`), and standard vocab words.
2. **Short Explanations & Semantic Ambiguity**:
   - In Exp17, short explanations (e.g. *"Water molecules absorb red light, reflecting blue light outward."*) could not be scored deterministically without risking false positives.
   - *Exp18 Policy*: Explicitly separate short explanations from scorable accuracy metrics. Label them `NEEDS_REVIEW` in Task Evaluator v0.1; evaluate them via human audit packet rather than heuristic string matching.
3. **Control Balancing in Abstention**:
   - Exp17 included 38 unanswerable queries and 12 answerable controls.
   - *Exp18 Policy*: Maintain a 3:1 ratio of unanswerable queries (requiring explicit abstention) to answerable controls (requiring grounded factual responses) to prevent the model from collapsing into indiscriminate refusal.

---

## 3. Contamination Prevention, Leakage Checks & Integrity

### 3.1 Split Isolation Rules
- **Disjoint Entity Sets**: Entities, numerical operand pairs, and JSON key structures used in the Test split MUST NOT appear in the Train or Validation splits.
- **N-Gram Deduplication**: Exact prompt string matching across splits is prohibited. Any overlap of 5-gram token sequences between Train and Test prompts triggers an automatic rejection.
- **Answer Validation**: Every target response in the scorable categories must be independently verified by an automated oracle script before split serialization.

### 3.2 Data Serialization Schema
Every dataset record adheres strictly to the canonical JSONL schema:
```json
{
  "id": "EXP18_TRAIN_BASIC_ARITHMETIC_0001",
  "split": "train",
  "category": "basic_arithmetic",
  "prompt_mode": "plain_instruction",
  "prompt": "Calculate the sum of 45 and 28.",
  "context": "",
  "formatted_prompt": "Instruction: Calculate the sum of 45 and 28.\nResponse:",
  "response": "The sum of 45 and 28 is 73.",
  "expected_target": "73",
  "metadata": {
    "scorable": true,
    "requires_abstention": false,
    "domain": "exp18_basic_arithmetic",
    "verification_method": "deterministic_python_calc"
  }
}
```

### 3.3 Dataset Hash Locking
Upon dataset generation, official SHA-256 hashes are computed and recorded in `ARTIFACT_MANIFEST.json`. These hashes become immutable protected assets for the duration of the experiment.
