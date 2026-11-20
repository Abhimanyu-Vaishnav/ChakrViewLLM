# UCIP-14: Instruction Dataset Specification
## Taxonomy, Schema, Quality Control & Provenance Framework

**Document ID:** UCIP-14-DATA-01  
**Project:** ChakrView  
**Status:** DESIGN ONLY — TRAINING NOT AUTHORIZED  

---

## 1. Dataset Taxonomy & Capability Dimensions

The instruction-alignment dataset is structured across 18 distinct capability categories, ensuring comprehensive alignment across reasoning, dialogue, coding, and multilingual interaction:

| Category ID | Category Name | Target Capability | Example Focus | Failure Risk | Target % | Target Records | Min Records | Max Records |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `CONV` | Conversation | Natural greeting, dialogue turns, friendly interaction | "Hello, who are you?", greetings, courteous closures | Sycophancy, repetitive loop | 6.0% | **900** | 720 | 1,125 |
| `INST` | Instruction Following | Explicit task constraints (length, bullet points, format) | "List 3 items", "Summarize in 20 words" | Ignoring constraints | 8.0% | **1,200** | 960 | 1,500 |
| `QA` | Question Answering | Direct, factual, non-hallucinatory answering | Factual queries on geography, science, history | Hallucinated facts | 7.0% | **1,050** | 840 | 1,312 |
| `CTX` | Context Retention | Multi-turn reference resolution, state tracking | Resolving pronouns ("What was his name again?") | Catastrophic context drift | 6.0% | **900** | 720 | 1,125 |
| `COPY` | Exact Copying | High-fidelity character and string reproduction | Replicating IDs, tokens, exact text blocks | Character substitution | 4.0% | **600** | 480 | 750 |
| `BIND` | Entity Binding | Consistent attribute-entity assignment | "John lives in Delhi, Priya in Pune. Where is John?" | Entity swapped values | 5.0% | **750** | 600 | 938 |
| `ARITH` | Arithmetic | Step-by-step arithmetic ($+,-,\times,\div$), small numbers | "15 + 27 = 42", subtraction with word problems | Arithmetic hallucination | 6.0% | **900** | 720 | 1,125 |
| `LOGIC` | Logic & Deduction | Modus ponens, syllogisms, basic comparative logic | "All squares are rectangles. A is a square..." | Fallacious leaps | 5.0% | **750** | 600 | 938 |
| `EXPL` | Explanations | Clear conceptual explanations of mechanisms | "How does photosynthesis work?" | Rambling word salad | 7.0% | **1,050** | 840 | 1,312 |
| `PYTHON` | Python Programming | Syntactically valid Python code, docstrings, bugs | "Write a function to check if number is prime" | Syntax errors, broken indentation | 8.0% | **1,200** | 960 | 1,500 |
| `ENG` | English Linguistic | English grammar, vocabulary, sentence completion | Stylistic variation, English nuances | Lexical degradation | 6.0% | **900** | 720 | 1,125 |
| `HIN` | Hindi (Devanagari) | Devanagari Hindi instructions and cultural responses | "भारत की राजधानी क्या है?", व्याकरण | Subword scrambling | 8.0% | **1,200** | 960 | 1,500 |
| `HING` | Hinglish | Romanized Hindi-English colloquial dialogue | "Mujhe programming seekhni hai, kahan se shuru karun?" | Code-switching confusion | 4.0% | **600** | 480 | 750 |
| `SAN` | Sanskrit | Classical Sanskrit shlokas, grammar, word meaning | Subhashitani meanings, sandhi rules | Script corruption | 3.0% | **450** | 360 | 562 |
| `UNCERT` | Uncertainty & Abstention | Explicit refusal to guess when information is missing | "What is the secret code inside box X?", missing facts | Confident hallucination | 5.0% | **750** | 600 | 938 |
| `FORMAT` | Formatting Constraints | JSON output, markdown tables, numbered lists | "Output as valid JSON: {'key': 'value'}" | Syntax delimiter breaks | 4.0% | **600** | 480 | 750 |
| `CORR` | Error Correction | Acknowledging and repairing mistakes in dialogue | "Actually, 12 * 4 is not 46. Correct yourself." | Stubborn persistence | 4.0% | **600** | 480 | 750 |
| `CLARIF` | Clarification Request | Asking clarifying questions when prompt is ambiguous | "Sort it." -> "Could you specify what to sort?" | Blind assumption | 4.0% | **600** | 480 | 750 |
| **TOTAL** | — | — | — | — | **100.0%** | **15,000** | **12,000** | **18,747** |

### Clarification on Dataset Units:
The 15,000 target count refers strictly to **unique supervised training examples**, where each record contains exactly one prompt and one gold assistant response turn. For multi-turn conversations, each turn of user-prompt and assistant-response constitutes an individual record with preceding context serialized into the prompt.

### Language Intersection Matrix (Non-Overlapping):
To avoid double-counting across task categories and languages, language metadata (`language`) is an independent orthogonal attribute for each record:
- **English (`en`):** 8,250 records (55.0%) across all tasks.
- **Hindi Devanagari (`hi`):** 4,500 records (30.0%) across all tasks.
- **Hinglish (`hi-en`):** 1,500 records (10.0%) primarily conversational, QA, and instructional.
- **Sanskrit (`sa`):** 750 records (5.0%) shlokas, grammar, and translations.
- **Total Languages:** Exactly 15,000 records (100.0%).

### Dataset Partitioning Splits:
- **Train Split (80%):** 12,000 records.
- **Validation Split (10%):** 1,500 records.
- **Dev Evaluation Split (10%):** 1,500 records.
- **Quarantined Sealed Final Benchmark:** 500 separately authored probes (strictly excluded from the 15,000 pool).

---

## 2. Dataset Record Schema (JSONL Specification)

Each instruction record in the SFT dataset must strictly follow this versioned JSON specification:

```json
{
  "$schema": "https://chakrview.ai/schemas/sft_record_v1.json",
  "id": "UCIP14-ARITH-00042",
  "version": "1.0.0",
  "category": "ARITH",
  "difficulty": "EASY",
  "language": "en",
  "source_id": "chakr_verified_arithmetic_v1",
  "provenance": "Hand-verified mathematical word problem created by ChakrView engineering team",
  "license": "CC0-1.0",
  "split": "TRAIN",
  "data_role": "TRAIN",
  "review_status": "VERIFIED",
  "contamination_group_id": "CONTAM-ARITH-ADD-01",
  "duplicate_group_id": "DUP-ARITH-ADD-01",
  "prompt": "User: If a train has 15 carriages and 7 more are attached, how many carriages does it have in total?\nAssistant: ",
  "response": "The train has 15 + 7 = 22 carriages in total.",
  "messages": [
    {
      "role": "user",
      "content": "If a train has 15 carriages and 7 more are attached, how many carriages does it have in total?"
    },
    {
      "role": "assistant",
      "content": "The train has 15 + 7 = 22 carriages in total."
    }
  ],
  "metadata": {
    "prompt_token_count": 27,
    "response_token_count": 14,
    "total_token_count": 41,
    "verifier": "forensic_audit_suite",
    "verified_timestamp_utc": "2026-11-20T12:00:00Z"
  }
}
```

---

## 3. Strict Quality-Control Pipeline

Before any dataset file is admitted into the SFT corpus, it must pass a 7-stage automated forensic validation pipeline:

1. **Schema & Integrity Gate:** Strict validation against `sft_record_v1.json`. All fields non-empty; `prompt` and `response` present; `review_status == "VERIFIED"`.
2. **Exact & Normalized Duplicate Elimination:** Case-insensitive, whitespace-normalized hash checking. Rejection of identical prompts across records.
3. **Template Diversity Check:** Regex template extraction to prevent synthetic repetition (e.g., maximum 5 variations per template family).
4. **Factual & Code Validation:** 
   - Arithmetic solutions verified with Python `ast.literal_eval`.
   - Python code parsed with `ast.parse` to guarantee syntactically valid code.
5. **Unicode & Language Integrity:** UTF-8 byte validation. Devanagari range check (`U+0900` to `U+097F`) for Hindi and Sanskrit.
6. **Token Horizon Compliance:** Sequence length check: $1 + \text{len}(\text{prompt}) + \text{len}(\text{response}) \le 512$.
7. **Anti-Contamination Quarantine:** Cryptographic cross-check against all sealed evaluation benchmarks (`benchmark_r2_sealed.json`, `ucip11_sealed.json`, and the proposed UCIP-14 sealed set). Any record matching a sealed prompt is rejected immediately.
