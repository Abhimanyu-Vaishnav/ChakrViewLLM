# Benchmark Report: STAGE_D_500_STEP_PILOT
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `5f3712d80940fb9345d01b5434e8bb64936d01e4d3aaff096ab5268c282d506c`
- **Checkpoint Path**: `checkpoints\experiments\neural_learning_v02\stage_d_pilot_run500\checkpoints\checkpoint_0000500.pt`
- **Checkpoint SHA-256**: `8ab1ec36b5c5a09929b8510332f18fecc0183eb0e9332ab99375995a8cc1a449`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 10
- **Total Generated Tokens**: 266
- **Mean Latency**: 131.64 ms (Median: 146.64 ms)
- **Mean TTFT**: 17.87 ms
- **Mean Throughput**: 188.37 tok/s
- **Mean Unique-Token Ratio**: 0.637
- **Mean 4-Gram Repetition Ratio**: 0.155
- **Termination**: 2 EOS / 8 Token Limit

## Prompt Generation Samples

| ID | Category | Prompt | Generated Output (First 60 chars) | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `FACT-01` | factual_knowledge | What is the primary function o... | ` The number of the number of the number of the number of the` | 212.0 | 0.19 | 0.79 |
| `FACT-02` | factual_knowledge | What does a translation lookas... | ` The answer is 4.` | 151.5 | 1.00 | 0.00 |
| `EXPL-01` | short_explanation | Explain the difference between... | ` There are 5 of the number of pay is 4*2=<<2*2=6>>60 c` | 226.3 | 0.75 | 0.00 |
| `EXPL-02` | short_explanation | How does multi-level caching i... | `2=200>>120 There are 50*2=<<200*2=60>>400 ` | 236.2 | 0.50 | 0.00 |
| `ARITH-01` | arithmetic_logic | Calculate: If a cache has 4 se... | ` The number of pay is 60/2=<<200/2=6>>12 years old.` | 191.0 | 0.78 | 0.00 |
| `ARITH-02` | arithmetic_logic | A program executes 100 instruc... | `` | 31.5 | 1.00 | 0.00 |
| `CODE-01` | coding_algorithmic | Write a python function to che... | `So =<<2*2=200>>120 So he has 60/2=<<200*2` | 172.1 | 0.50 | 0.03 |
| `CODE-02` | coding_algorithmic | In C, write a struct definitio... | `40=<<40/2=4>>36. The number of paid is $40 = <<4` | 262.1 | 0.66 | 0.00 |
| `CONT-01` | open_continuation | Computer architecture continue... | `. There are 5 of the number of pakes is 4*2=<<2*2=60>>` | 182.6 | 0.78 | 0.00 |
| `CONT-02` | open_continuation | The development of indigenous ... | `, the total number of the number of the number of the number` | 218.2 | 0.22 | 0.72 |