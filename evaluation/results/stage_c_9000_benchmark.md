# Benchmark Report: STAGE_C_9000_STEP_PRETRAINED
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `36711d25831740840dc9d62d60087a898aa4a19746322ef3cb0e04974a2770e3`
- **Checkpoint Path**: `checkpoints\experiments\neural_learning_v02\run_10k\checkpoint_0009000.pt`
- **Checkpoint SHA-256**: `3e03ae0a134da40439038d965278f4da2996b1ac2e8f959d6059859600b0bb23`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 10
- **Total Generated Tokens**: 266
- **Mean Latency**: 122.83 ms (Median: 137.59 ms)
- **Mean TTFT**: 16.74 ms
- **Mean Throughput**: 211.18 tok/s
- **Mean Unique-Token Ratio**: 0.537
- **Mean 4-Gram Repetition Ratio**: 0.254
- **Termination**: 3 EOS / 7 Token Limit

## Prompt Generation Samples

| ID | Category | Prompt | Generated Output (First 60 chars) | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `FACT-01` | factual_knowledge | What is the primary function o... | `  The first of the group of the government of the government` | 232.6 | 0.47 | 0.24 |
| `FACT-02` | factual_knowledge | What does a translation lookas... | `&lt;ref&gt;&lt;/ref&gt;  References` | 216.1 | 0.50 | 0.16 |
| `EXPL-01` | short_explanation | Explain the difference between... | ` The most of the group of the government of the government o` | 221.1 | 0.38 | 0.45 |
| `EXPL-02` | short_explanation | How does multi-level caching i... | `&quot; &lt;ref&gt;&lt;/ref&gt;  References` | 290.9 | 0.50 | 0.14 |
| `ARITH-01` | arithmetic_logic | Calculate: If a cache has 4 se... | `  The first of the group of the group of the group of the go` | 300.8 | 0.44 | 0.41 |
| `ARITH-02` | arithmetic_logic | A program executes 100 instruc... | `  References` | 140.7 | 0.86 | 0.00 |
| `CODE-01` | coding_algorithmic | Write a python function to che... | `  * [http://www.non-gooday.com/inda/in` | 183.9 | 0.66 | 0.00 |
| `CODE-02` | coding_algorithmic | In C, write a struct definitio... | `  References` | 120.0 | 0.86 | 0.00 |
| `CONT-01` | open_continuation | Computer architecture continue... | `. The century is the group of the group of the group of the ` | 220.6 | 0.50 | 0.38 |
| `CONT-02` | open_continuation | The development of indigenous ... | `s in the first of the first of the first of the first of the` | 185.2 | 0.22 | 0.76 |