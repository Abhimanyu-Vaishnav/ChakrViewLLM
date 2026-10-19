# Generation Protocol: Batch 2781–2820
## Prompt-Conditioned Neural Generation, Decoding Controls & Repeatable Quality Benchmark

### 1. Objective
Establish an empirical, reproducible generation quality benchmark comparing:
1. `raw_completion`: unconditioned raw token continuation from prompt tokens.
2. `plain_instruction`: instruction prefix conditioning (`Instruction: {input}\nResponse:`).

Both modes are evaluated on the canonical Step-10,000 checkpoint (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`, SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`) under identical, deterministic decoding parameters:
- `decoding`: `greedy`
- `max_new_tokens`: 32
- `stop_on_eos`: True
- Hardware: CPU (`torch.device("cpu")`)

In addition, targeted probes are conducted to measure the effect of decoding controls (`sampling` with temperature and repetition penalty).

### 2. Test Dataset
Dataset location: `experiments/batch2781_2820/GENERATION_BENCHMARK.json`
Total cases: 28 cases across 7 categories:
1. `raw_continuation` (4 cases): Standard sentence and code completions.
2. `simple_instruction` (4 cases): Common direct questions or requests.
3. `context_extraction` (4 cases): Supplies factual data within the prompt and queries it.
4. `short_summarization` (4 cases): Supplies 1-2 sentences and asks for a short summary.
5. `repetition_probe` (4 cases): Repetitive prefix designed to probe cyclical attractor states.
6. `unknown_fact_probe` (4 cases): Probes for fictional/private facts to observe ungrounded hallucination.
7. `context_boundary_probe` (4 cases): Validates context handling, medium token lengths, and hard limits.

### 3. Quantitative & Qualitative Metrics
For every test case:
- `prompt_tokens`: Number of tokens in conditioned prompt.
- `generated_tokens`: Number of newly generated tokens.
- `latency_ms`: Total execution time on CPU in milliseconds.
- `stop_reason`: Reason for stopping (`max_tokens`, `eos`, `context_limit`).
- `empty_output`: Whether generated text contains zero non-whitespace characters.
- `repetition_ratio`: Fraction of repeated bigrams in the generated sequence (measure of degeneration).
- `context_fact_preserved`: For extraction cases, whether the ground truth entity appears in the output.
- `hallucinated_answer`: For unknown-fact cases, whether the model generated confident claims rather than abstaining.

### 4. Non-Negotiable Controls
- No parameter weights are modified ($\Delta W = 0$ parameter hash verification post-generation).
- Context ceiling: 512 tokens.
- Generation hard cap: $\le 64$ new tokens.
- Model opt-in: Assistant runtime defaults to neural disabled; neural model loads only upon explicit caller request.
