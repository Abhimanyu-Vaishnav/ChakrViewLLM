# SFT Evaluation Protocol: Batch 2821–2860
## Evaluation Methodology, Quality Gates & Regression Protection

### 1. Objective
Ensure that candidate SFT checkpoints trained under Batch 2821–2860 guidelines achieve genuine instruction and context-conditioning capability without degrading baseline safety, deterministic routes, or pretrained knowledge.

### 2. Evaluation Suites

#### Suite 1: Batch 2821 SFT Held-Out Test Set (24 Cases)
Location: `experiments/batch2821_2860/SFT_EVALUATION_SET.json`
Comprises 3 examples from each of the 8 curated categories:
1. `simple_instruction`: Evaluates direct answers to conceptual, technical, or factual questions.
2. `context_extraction`: Evaluates exact match of extracted tokens (e.g. `28.6%`, `Sarah Lin`).
3. `context_grounded_qa`: Evaluates accuracy of responses grounded strictly in supplied context.
4. `short_summarization`: Assesses whether summaries capture key concepts concisely.
5. `structured_output`: Validates syntax and schema validity of generated JSON.
6. `uncertainty_insufficient_info`: Measures honesty rate (must state information is unavailable).
7. `conversational_acknowledgement`: Evaluates appropriate conversational assistant tone.
8. `bounded_text_transformation`: Evaluates deterministic string transforms (e.g. uppercase, reverse).

#### Suite 2: Batch 2781 Generation Quality Benchmark (28 Cases)
Location: `experiments/batch2781_2820/GENERATION_BENCHMARK.json`
Maintained as an immutable regression check:
- Measures raw continuation performance.
- Evaluates bigram repetition ratio under greedy decoding.
- Validates context ceiling and boundary probes.

### 3. Quantitative Scoring Metrics
- **Extraction Exact Match Rate**: Percentage of extraction targets found verbatim in model response.
- **JSON Parse Validity Rate**: Percentage of structured output answers that successfully parse via `json.loads`.
- **Honest Abstention Rate**: Percentage of insufficient-information probes where the model declines to hallucinate.
- **Bigram Repetition Ratio**: Ratio of duplicate bigrams to total bigrams (target $< 20\%$).
- **Perplexity on Validation Split**: Cross-entropy validation loss on response tokens.
- **Runtime Attribution**: All responses must maintain `[EXPERIMENTAL CHAKRVIEW GENERATION - NOT VERIFIED]` banner and report valid parameter verification.

### 4. Qualitative Judgment & Labeling
- Every automated score must be derived deterministically.
- Semantic ratings (e.g. summarization quality) must be explicitly flagged as `human-reviewed` or `unreviewed`; never fabricated.
