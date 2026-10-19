# Evaluation Protocol Audit: Batch 2901–2940
## Audit of Held-Out SFT Test Set (24 Cases) & Generation Benchmark (28 Cases)

### 1. Objective Evaluation of Scoring Criteria

We audited all evaluation logic applied across the 24-case SFT test set and the 28-case generation benchmark:

#### A. Objective vs. Qualitative Metrics
- **Exact Match (`exact_match`)**: Strictly objective. Evaluates case-insensitive substring inclusion (`target.lower() in generated.lower()`).
- **Structured JSON Validity (`json_valid`)**: Strictly objective. Uses Python's standard `json.loads(text.strip())` parser.
- **Uncertainty Refusal (`abstention_detected`)**: Objective keyword match against explicit refusal phrases (`"not contain"`, `"unavailable"`, `"does not state"`, `"cannot"`, `"no information"`).
- **Bigram Repetition Ratio (`repetition_ratio`)**: Strictly objective mathematical calculation ($1 - \frac{|U|}{|B|}$).
- **Stop Reason (`stop_reason`)**: Objective runtime condition (`MAX_TOKENS`, `STOP_TOKEN`, `CONTEXT_LIMIT`).

#### B. Audit of Exact Match Strictness
- **Finding**: Substring exact match is appropriate for factual target entities (e.g. `28.6%`, `Sarah Lin`, `token-zeta-77`, `4022`).
- **Risk**: For open-ended explanatory instructions (e.g., `INST_028`: *"State the law of conservation of energy"*), exact substring matching of the full 20-word reference target (`"The law of conservation of energy states that energy cannot be created or destroyed..."`) is excessively strict and flags semantically valid paraphrases as incorrect.
- **Recommendation**: For future benchmarks, decouple **Extractive / Entity Tasks** (which require exact token match) from **Generative / Explanatory Tasks** (which require semantic overlap or human review).

#### C. Empty Output & EOS Identification
- The reported empty-output rate strictly and accurately identifies instances where the very first generated token was `<EOS>` (token ID 1).
- No silent failures, crashes, or unhandled exceptions were conflated with empty generations.

#### D. Split Leakage & Contamination Check
- Verified that all 24 test cases were completely excluded from the 192 training examples.
- Neither test prompt strings nor test target responses appeared anywhere in `train.jsonl`.
- Data split isolation remained 100% clean.
