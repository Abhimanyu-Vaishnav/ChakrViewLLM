# Evaluation Metrics Specification: Rigorous Conversational Grounding
**Directory**: `experiments/research_v09_conversation_sft/exp02_context_grounding/`  
**Date**: October 22, 2026

---

## 1. Principles of Measurement

1. **Denominators are Inviolate**: Every evaluated example ($N=60$) is included in every denominator. Never filter out failures, unparsed outputs, or syntax errors.
2. **Deterministic & Self-Contained**: Every metric is computed via pure Python logic, regex matching, or standard library AST parsing. Zero reliance on external models or subjective lexical overlap.
3. **Failure Classification**: Invalid outputs, repetitive loops, and empty generations are scored as 0 across all competence metrics.

---

## 2. Metric Specifications

### Metric 1: Exact Entity Binding (EEB)
- **Definition**: The proportion of examples where the model's generated output contains the exact target entity name specified in the user prompt or context, without emitting memorized training entity names.
- **Formula**:
  $$\text{EEB} = \frac{\sum_{i=1}^N \mathbf{1}[\text{target\_entity}_i \in \text{output}_i \land \text{no\_memorized\_leakage}]}{N}$$
- **Failures Included**: Repetitive loops, empty outputs, memorized training entity names (e.g. `is_positive_train_0`).
- **What it does NOT prove**: Does not prove the model understands the semantic role of the entity, only that the token sequence was successfully selected and emitted.

### Metric 2: Context Reference & Anaphora Resolution (CRAR)
- **Definition**: When a prompt asks "What was the reading for that one?" following an ambiguous mention of two components, the model must explicitly identify the ambiguity and ask for clarification, rather than guessing.
- **Formula**:
  $$\text{CRAR} = \frac{\sum_{i=1}^N \mathbf{1}[\text{contains\_clarification\_request}(\text{output}_i)]}{N}$$
- **Numerator**: Outputs containing `"clarify"`, `"specify"`, `"which"`, or naming both candidate units.

### Metric 3: User Correction Adherence (UCA)
- **Definition**: When the user issues a correction ("Wait, cancel that. I meant Unit B instead"), the model must bind `Unit B` and must NOT bind `Unit A`.
- **Formula**:
  $$\text{UCA} = \frac{\sum_{i=1}^N \mathbf{1}[\text{new\_target}_i \in \text{output}_i \land \text{superseded\_target}_i \notin \text{output}_i]}{N}$$
- **Numerator**: Output contains the new entity AND excludes the cancelled entity.

### Metric 4: Distractor Rejection Accuracy (DRA)
- **Definition**: In multi-entity contexts with 2 distractors, the model must output the target state/value while omitting the distractor states/values.
- **Formula**:
  $$\text{DRA} = \frac{\sum_{i=1}^N \mathbf{1}[\text{target\_state}_i \in \text{output}_i \land \forall d \in \text{distractors}_i, d \notin \text{output}_i]}{N}$$

### Metric 5: Explicit Abstention & Missing Context Rate (EAMR)
- **Definition**: When an answer cannot be deduced from context, the model must explicitly state that the information is missing.
- **Formula**:
  $$\text{EAMR} = \frac{\sum_{i=1}^N \mathbf{1}[\text{contains\_abstention\_statement}(\text{output}_i)]}{N}$$
- **Keywords**: `"does not state"`, `"not provided"`, `"cannot be predicted"`, `"unknown"`, `"none"`.

### Metric 6: Constrained Code Synthesis Accuracy (CCSA)
- **Definition**: The model must generate Python code that parses as valid syntax via `ast.parse()` AND defines the exact function name requested in the prompt.
- **Formula**:
  $$\text{CCSA} = \frac{\sum_{i=1}^N \mathbf{1}[\text{ast\_valid}(\text{output}_i) \land \text{func\_name}_i \in \text{output}_i]}{N}$$

### Metric 7: Attractor Repetition Rate (ARR)
- **Definition**: Percentage of outputs suffering from repetitive n-gram collapse ($N \ge 3$ repeated).
- **Formula**:
  $$\text{ARR} = \frac{\sum_{i=1}^N \mathbf{1}[\text{is\_repetitive}(\text{output}_i)]}{N}$$

---

## 3. Human Review Rubric for Open-Ended Conversational Responses

For nuanced dialogue where string matching is insufficient, responses are evaluated on a 1–5 scale:
- **5 (Exemplary)**: Accurately grounds in context, concise, appropriate tone, zero hallucination.
- **4 (Sufficient)**: Correct answer, minor stylistic verbosity.
- **3 (Partially Grounded)**: Identifies general domain, but misses precise numerical reading or entity.
- **2 (Attractor / Hallucinated)**: Generates plausible language from pretraining, but ignores the prompt.
- **1 (Gibberish / Collapse)**: Repetitive loops or immediate empty termination.
