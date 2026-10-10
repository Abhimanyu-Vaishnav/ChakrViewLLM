# Benchmark Report: RANDOM_INITIALIZATION_BASELINE
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- **Checkpoint Path**: `None (Random Init)`
- **Checkpoint SHA-256**: `N/A`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 10
- **Total Generated Tokens**: 320
- **Mean Latency**: 107.71 ms (Median: 107.72 ms)
- **Mean TTFT**: 15.42 ms
- **Mean Throughput**: 300.14 tok/s
- **Mean Unique-Token Ratio**: 0.031
- **Mean 4-Gram Repetition Ratio**: 0.966
- **Termination**: 0 EOS / 10 Token Limit

## Prompt Generation Samples

| ID | Category | Prompt | Generated Output (First 60 chars) | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `FACT-01` | factual_knowledge | What is the primary function o... | `????????????????????????????????` | 307.2 | 0.03 | 0.97 |
| `FACT-02` | factual_knowledge | What does a translation lookas... | `????????????????????????????????` | 297.0 | 0.03 | 0.97 |
| `EXPL-01` | short_explanation | Explain the difference between... | `ence.ence.ence.ence.ence.ence.ence.ence.ence.ence.ence.ence.` | 284.4 | 0.03 | 0.97 |
| `EXPL-02` | short_explanation | How does multi-level caching i... | `????????????????????????????????` | 321.5 | 0.03 | 0.97 |
| `ARITH-01` | arithmetic_logic | Calculate: If a cache has 4 se... | `????????????????????????????????` | 303.5 | 0.03 | 0.97 |
| `ARITH-02` | arithmetic_logic | A program executes 100 instruc... | `????????????????????????????????` | 297.1 | 0.03 | 0.97 |
| `CODE-01` | coding_algorithmic | Write a python function to che... | `::::::::::::::::::::::::::::::::` | 287.2 | 0.03 | 0.97 |
| `CODE-02` | coding_algorithmic | In C, write a struct definitio... | `s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:` | 324.8 | 0.03 | 0.97 |
| `CONT-01` | open_continuation | Computer architecture continue... | `sesesesesesesesesesesesesesesesesesesesesesesesesesesesesese` | 346.2 | 0.03 | 0.97 |
| `CONT-02` | open_continuation | The development of indigenous ... | `resresresresresresresresresresresresresresresresresresresres` | 232.5 | 0.03 | 0.97 |