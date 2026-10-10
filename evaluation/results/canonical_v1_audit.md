# Benchmark Report: RANDOM_INITIALIZATION
- **Suite Version**: `1.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- **Checkpoint Path**: `None (Random Init)`
- **Checkpoint SHA-256**: `N/A`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 10
- **Total Generated Tokens**: 320
- **Mean Latency**: 187.99 ms (Median: 210.81 ms)
- **Mean TTFT**: 20.46 ms
- **Mean Throughput**: 181.07 tok/s
- **Mean Unique-Token Ratio**: 0.031
- **Mean 4-Gram Repetition Ratio**: 0.966
- **Termination**: 0 EOS / 10 Token Limit
- **Deterministic Task Accuracy**: 0/0 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 50 chars) | Correct? | Failure Category | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `FACT-01` | factual_knowledge | What is the primary funct... | `????????????????????????????????` | N/A | `degeneration / character_loop` | 241.0 | 0.03 | 0.97 |
| `FACT-02` | factual_knowledge | What does a translation l... | `????????????????????????????????` | N/A | `degeneration / character_loop` | 267.6 | 0.03 | 0.97 |
| `EXPL-01` | short_explanation | Explain the difference be... | `ence.ence.ence.ence.ence.ence.ence.ence.ence.ence.` | N/A | `irrelevant_response` | 251.3 | 0.03 | 0.97 |
| `EXPL-02` | short_explanation | How does multi-level cach... | `????????????????????????????????` | N/A | `degeneration / character_loop` | 170.7 | 0.03 | 0.97 |
| `ARITH-01` | arithmetic_logic | Calculate: If a cache has... | `????????????????????????????????` | N/A | `degeneration / character_loop` | 151.8 | 0.03 | 0.97 |
| `ARITH-02` | arithmetic_logic | A program executes 100 in... | `????????????????????????????????` | N/A | `degeneration / character_loop` | 145.0 | 0.03 | 0.97 |
| `CODE-01` | coding_algorithmic | Write a python function t... | `::::::::::::::::::::::::::::::::` | N/A | `degeneration / character_loop` | 161.9 | 0.03 | 0.97 |
| `CODE-02` | coding_algorithmic | In C, write a struct defi... | `s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:` | N/A | `irrelevant_response` | 141.6 | 0.03 | 0.97 |
| `CONT-01` | open_continuation | Computer architecture con... | `sesesesesesesesesesesesesesesesesesesesesesesesese` | N/A | `human_judgment_required` | 139.3 | 0.03 | 0.97 |
| `CONT-02` | open_continuation | The development of indige... | `resresresresresresresresresresresresresresresresre` | N/A | `human_judgment_required` | 140.3 | 0.03 | 0.97 |