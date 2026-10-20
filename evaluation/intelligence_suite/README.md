# ChakrView Native Intelligence Benchmark Suite (v0.8)

## Purpose & Scope
This evaluation suite measures the **actual native language model capabilities** of ChakrView's canonical core (`ChakrMicro` Step-10,000, 3.44M parameters) across 32 standardized tasks in 10 categories:

1. `conversational_greetings`: Basic conversational openings and identities.
2. `direct_instruction_following`: Exact format and single-word instruction adherence.
3. `context_copying`: Context-dependent copying of synthetic IDs, codes, and names.
4. `grounded_qa`: Context-grounded extraction of facts from short passages.
5. `summarization`: Abstractive distillation of brief paragraphs.
6. `simple_reasoning_arithmetic`: Elementary word problem arithmetic in natural language.
7. `format_adherence`: Generation of JSON, XML, or structured lists.
8. `coding_continuation`: Python function signature completion and assertion emission.
9. `repetition_collapse_probe`: Probing susceptibility to attractor loop degradation.
10. `multi_turn_context`: Tracking conversational state across multi-turn exchanges.

## Demarcation of Tools vs. Model Intelligence
- **Native Model Output**: Raw autoregressive token generations from the model.
- **External Tools**: Completely excluded from this suite. Safe AST calculators, subprocess runners, and mock generators are forbidden here.

## How to Run

### Weight-Independent Smoke Verification
```powershell
python evaluation/intelligence_suite/run_intelligence_benchmark.py --smoke --output-path evaluation/intelligence_suite/smoke_results.json
```

### Full Neural Evaluation (Requires Checkpoint)
When the canonical checkpoint is available:
```powershell
python evaluation/intelligence_suite/run_intelligence_benchmark.py --checkpoint-key step_10000 --output-path evaluation/intelligence_suite/intelligence_baseline_results.json
```