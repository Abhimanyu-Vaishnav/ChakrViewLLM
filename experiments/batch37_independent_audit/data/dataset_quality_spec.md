# ChakrView SFT Dataset Quality Specification (v1.0)

## 1. Principles
Every SFT dataset record must satisfy:
1. **Source Grounding**: Factual statements in the assistant response must be directly verifiable in the cited source text. Abstract claims or architectural properties not present in the source cannot cite that source.
2. **Document-Level Disjointness**: A source file utilized in the training split cannot appear in the validation split under any circumstances.
3. **No Near-Duplicate Overlap**: Distinct records must have genuinely distinct prompts and responses, avoiding repetitive templates.
4. **Executable Code Verification**: Any code provided in the assistant response must be syntactically valid and pass local unit execution.
5. **Format Alignment**: Conversations must follow the versioned `ConversationTemplate` (ChatML format) and support multi-turn history without dropping prior context.

## 2. Schema Specification
```json
{
  "id": "SFT-DOMAIN-001",
  "domain": "english_systems | hindi_literature | hinglish_tech | code | mathematics",
  "source_file": "relative/path/to/source.txt",
  "source_sha256": "64_char_hex_digest",
  "source_passage_line_range": [12, 18],
  "provenance_status": "verified_source_grounded | codebase_grounded | synthetic_diagnostic",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "review_status": "automated_verified | human_reviewed | unverified"
}
```

## 3. Disjoint Source Allocation Strategy
- **Training Documents (15 files)**:
  - `english/stage_b_systems.txt`, `english/english_foundation.txt`
  - `hindi/stage_b_literature.txt`, `hindi/hindi_foundation.txt`, `hindi/hindi_dialogue.txt`
  - `hinglish/stage_b_dialogues.txt`, `hinglish/hinglish_foundation.txt`
  - `code/stage_b_algorithms.txt`, `code/code_foundation.txt`
  - `mathematics/stage_b_derivations.txt`, `mathematics/math_foundation.txt`
  - `reasoning/stage_b_procedures.txt`
  - `sanskrit/stage_b_classics.txt`, `sanskrit/sanskrit_classics.txt`
  - `numbers/stage_b_tables.txt`
- **Validation Documents (12 files)**:
  - `english/english_computer_science.txt`, `english/english_systems.txt`
  - `hindi/hindi_computing.txt`, `hindi/hindi_science_culture.txt`
  - `hinglish/hinglish_tech.txt`
  - `code/code_python.txt`, `code/code_configs.txt`
  - `mathematics/math_formulas.txt`
  - `mixed/mixed_foundation.txt`, `mixed/mixed_multilingual.txt`
  - `numbers/numbers_foundation.txt`, `numbers/numbers_dataset.txt`
