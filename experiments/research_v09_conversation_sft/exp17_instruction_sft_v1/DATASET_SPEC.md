# ChakrView Exp17 Instruction SFT v1 — Dataset Specification

**Experiment ID**: `exp17_instruction_sft_v1`  
**Parent Directory**: `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/`  
**Dataset Version**: `1.0.0`  
**Status**: DESIGN SPECIFICATION ONLY

---

## 1. Dataset Design Principles & Objectives

To directly remediate the bottlenecks observed in **Normal Chat Gate v0.1**, the Exp17 dataset is engineered with four core principles:
1. **Compact & High-Density**: A total volume of **400 curated examples** (320 Train / 40 Validation / 40 Held-Out Test).
2. **Explicit Turn Termination Targets**: 100% of target sequences end cleanly with a single, unambiguous `<EOS>` token. Responses are concise (5–35 tokens) to discourage repetitive runaway generation.
3. **Suppression of Arithmetic Formula Bias**: Synthetic math headers (`Step-by-Step Solution:`) are explicitly discouraged; basic arithmetic prompts map to direct numerical responses (e.g., `17 + 25 = 42.`).
4. **Balanced Functional Diversity**: Stratified across 8 distinct conversational categories covering factual QA, instruction following, context extraction, abstention, and formatting.

---

## 2. Category Distribution & Sampling Strategy

The dataset comprises exactly 400 examples allocated across 8 categories:

| Category ID | Category Name | Description & Capability Tested | Train Count | Val Count | Test Count | Total Count |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `CAT_01` | **Factual QA** | Direct, factual knowledge (astronomy, geography, science). | 40 | 5 | 5 | 50 |
| `CAT_02` | **Instruction & Constraints** | Strict negative constraints, exact case, reversal, single-word outputs. | 40 | 5 | 5 | 50 |
| `CAT_03` | **Context Extraction** | Extractive question answering strictly bounded by context text. | 40 | 5 | 5 | 50 |
| `CAT_04` | **Reference Resolution** | Multi-entity context resolving pronouns and anaphora. | 40 | 5 | 5 | 50 |
| `CAT_05` | **Explanation & Reasoning** | Short (1–2 sentence) causal explanations without math headers. | 40 | 5 | 5 | 50 |
| `CAT_06` | **Basic Arithmetic** | Elementary addition, subtraction, and multiplication with direct answers. | 40 | 5 | 5 | 50 |
| `CAT_07` | **Uncertainty & Abstention** | Appropriate refusal or uncertainty when context is missing/unanswerable. | 40 | 5 | 5 | 50 |
| `CAT_08` | **Formatting & Output Structure**| Numbered lists, JSON key extraction, bullet points. | 40 | 5 | 5 | 50 |
| **TOTAL** | **8 Categories** | **Balanced Stratified Dataset** | **320** | **40** | **40** | **400** |

---

## 3. JSONL Schema Specification

Every record in `train_exp17_v1.jsonl`, `val_exp17_v1.jsonl`, and `test_exp17_v1.jsonl` adheres strictly to the following unified JSON schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Exp17InstructionRecord",
  "type": "object",
  "required": [
    "id",
    "split",
    "category",
    "prompt_mode",
    "prompt",
    "context",
    "response",
    "expected_intent",
    "metadata"
  ],
  "properties": {
    "id": {
      "type": "string",
      "pattern": "^EXP17_(TRAIN|VAL|TEST)_[A-Z0-9_]+$"
    },
    "split": {
      "type": "string",
      "enum": ["train", "val", "test"]
    },
    "category": {
      "type": "string",
      "enum": [
        "factual_qa",
        "instruction_following",
        "context_extraction",
        "reference_resolution",
        "short_explanation",
        "basic_arithmetic",
        "uncertainty_abstention",
        "structured_formatting"
      ]
    },
    "prompt_mode": {
      "type": "string",
      "enum": ["plain_instruction", "context_qa"]
    },
    "prompt": {
      "type": "string",
      "minLength": 5
    },
    "context": {
      "type": "string"
    },
    "response": {
      "type": "string",
      "minLength": 1
    },
    "expected_intent": {
      "type": "string"
    },
    "metadata": {
      "type": "object",
      "required": ["requires_abstention", "target_token_count"],
      "properties": {
        "requires_abstention": { "type": "boolean" },
        "target_token_count": { "type": "integer", "minimum": 1, "maximum": 64 },
        "domain": { "type": "string" },
        "source": { "type": "string", "enum": ["curated_synthetic", "manual_ground_truth"] }
      }
    }
  }
}
```

---

## 4. Prompt & Response Formatting Rules

### 4.1 Plain Instruction (`prompt_mode == "plain_instruction"`)
Used for general conversational, factual, reasoning, and instruction tasks without a separate reference document:
- **Prompt String**:
  ```text
  Instruction: {prompt}
  Response: 
  ```
- **Response String**:
  ```text
  {response}
  ```
- *Example*:
  ```text
  Instruction: What planet is known as the Red Planet?
  Response: Mars is known as the Red Planet.
  ```

### 4.2 Context QA (`prompt_mode == "context_qa"`)
Used for extractive QA, reference resolution, and context-dependent abstention:
- **Prompt String**:
  ```text
  Context: {context}
  Question: {prompt}
  Answer: 
  ```
- **Response String**:
  ```text
  {response}
  ```
- *Example*:
  ```text
  Context: Special Agent 007 uses vehicle model Delta-X which operates on liquid nitrogen.
  Question: Which vehicle model does Agent 007 use?
  Answer: Agent 007 uses vehicle model Delta-X.
  ```

---

## 5. Leakage Prevention & Trust Gate Protections

### 5.1 Normal Chat Gate Benchmark Protection (Absolute Isolation)
- **Zero Prompt Contamination**: None of the 20 prompts from `experiments/normal_chat_gate_v01/prompts.jsonl` may appear verbatim in `train_exp17_v1.jsonl` or `val_exp17_v1.jsonl`.
- **Pre-Training Gate Verification**: An automated n-gram overlap and Levenshtein similarity scan must verify:
  $$\text{Similarity}(P_{\text{train}}, P_{\text{gate}}) < 0.80$$
  Any prompt sharing $\ge 80\%$ surface similarity or identical entity pairings with Gate v0.1 prompts is flagged and rejected.

### 5.2 Split Independence & Cross-Split Leakage Checks
- **Entity & Query Disjointness**:
  - Validation (40 examples) and Test (40 examples) use distinct entities, numerical values, and phrasing templates not present in Train (320 examples).
- **Exact ID & Sequence Uniqueness**:
  - $ID_{\text{train}} \cap ID_{\text{val}} \cap ID_{\text{test}} = \emptyset$.
  - Exact prompt string overlap between splits: **0%**.

### 5.3 Synthetic Data Limitations
- All synthetic records must be generated with deterministic pseudo-random seeds.
- Records must not use template repetition that would train alternative looping attractors (e.g., repeating the same closing phrase across multiple examples).
- Abstention phrasing must vary across examples (e.g., `"The provided context does not mention..."`, `"Information not available."`, `"Cannot be determined from the text."`).
