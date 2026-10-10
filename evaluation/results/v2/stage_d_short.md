# Benchmark Report: STAGE_D_500_STEP_PILOT
- **Suite Version**: `2.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `5f3712d80940fb9345d01b5434e8bb64936d01e4d3aaff096ab5268c282d506c`
- **Checkpoint Path**: `checkpoints\experiments\neural_learning_v02\stage_d_pilot_run500\checkpoints\checkpoint_0000500.pt`
- **Checkpoint SHA-256**: `8ab1ec36b5c5a09929b8510332f18fecc0183eb0e9332ab99375995a8cc1a449`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 30
- **Total Generated Tokens**: 878
- **Mean Latency**: 137.45 ms (Median: 136.37 ms)
- **Mean TTFT**: 21.53 ms
- **Mean Throughput**: 213.13 tok/s
- **Mean Unique-Token Ratio**: 0.546
- **Mean 4-Gram Repetition Ratio**: 0.160
- **Termination**: 4 EOS / 26 Token Limit
- **Deterministic Task Accuracy**: 0/20 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 60 chars) | Correct? | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley site ... | ` The answer is 200.` | FAIL | 177.2 | 1.00 | 0.00 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was laun... | ` The answer is 200.` | FAIL | 143.1 | 1.00 | 0.00 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata satellit... | ` The answer is 200.` | FAIL | 131.8 | 1.00 | 0.00 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis occurs ... | ` So he has 200*2=<<2*2=200>>12 years old. So ` | FAIL | 207.6 | 0.59 | 0.00 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line size of... | ` The number of the number of the number of the number of the` | FAIL | 249.0 | 0.19 | 0.79 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides virtua... | `40=<<2*2=200>>60 So he has 200*2=<<200*2` | N/A | 161.8 | 0.47 | 0.07 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled language c... | `So he has $200 = <<2*2=200>>120 hours The answer is 200.` | N/A | 234.9 | 0.72 | 0.00 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous commun... | `50=<<2*2=200>>12 years old. So he has 200*2=` | N/A | 234.7 | 0.66 | 0.00 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through caches ... | `2=<<2*2=200>>120 So he has 200*2=<<2*2=2` | N/A | 169.4 | 0.44 | 0.21 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing checks ... | `2=<<2*2=200>>120 So he has 200*2=<<2*2=2` | N/A | 224.9 | 0.44 | 0.21 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `240>>40 There are 50*2=<<200*2=60>>40 Th` | FAIL | 276.0 | 0.47 | 0.14 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `<<240=240>>400 There are 500/2=<<200*2=6` | FAIL | 263.5 | 0.50 | 0.00 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `<<2*2=200>>120 There are 50*2=<<200*2=60>` | FAIL | 247.8 | 0.47 | 0.00 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `<<200*2=200>>60 There are 50/2=<<200*2=60` | FAIL | 236.7 | 0.47 | 0.14 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 apples. ... | `2=<<2*2=2>>12 she has 24*2=<<2*2=2>>6` | FAIL | 251.6 | 0.41 | 0.28 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are mamm... | `200>=<<2*2=200>>120 cans So he has 200*2=<<` | FAIL | 229.2 | 0.53 | 0.00 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number sequ... | `40 There are 50*2=<<200*2=60>>60 So he has 6` | FAIL | 223.2 | 0.62 | 0.00 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean express... | `2=<<2*2=200>>120 So he has 200*2=<<200*2=` | FAIL | 199.2 | 0.44 | 0.14 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data structu... | ` So he has $50*2=<<2*2=200>>120 - 200 = $<<2` | FAIL | 254.6 | 0.59 | 0.00 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-letter... | `24=<<2*2=200>>120 So he has 200*2=<<200*2` | FAIL | 270.8 | 0.47 | 0.07 |
| `CODE-01` | safe_coding_structure | Write a Python function named ... | `2=<<2*2=200>>120 So he has 200*2=<<2*2=2` | FAIL | 194.7 | 0.44 | 0.21 |
| `CODE-02` | safe_coding_structure | Write a Python function named ... | `So he has $5 = <<2*2=200>>120 x 200 = <<200*2=2` | FAIL | 249.4 | 0.53 | 0.07 |
| `CODE-03` | safe_coding_structure | In C, declare a variable named... | ` the total number of paks. There are 5 of the number of pake` | FAIL | 250.4 | 0.72 | 0.10 |
| `CODE-04` | safe_coding_structure | Write the SQL query to select ... | `2=<<2*2=200>>120 So he has 200*2=<<2*2=2` | FAIL | 188.2 | 0.44 | 0.21 |
| `CODE-05` | safe_coding_structure | In Python, how do you import t... | `2=200>>12 canswer is 4.` | FAIL | 158.5 | 0.88 | 0.00 |
| `CONT-01` | open_continuation | In modern operating systems, v... | ` There are the number of the number of the number of the num` | N/A | 189.5 | 0.28 | 0.69 |
| `CONT-02` | open_continuation | The primary advantage of pipel... | `our to pay the total number of the number of the number of t` | N/A | 197.4 | 0.38 | 0.55 |
| `CONT-03` | open_continuation | When designing reliable distri... | `h of the number of the number of the number of the number of` | N/A | 194.8 | 0.16 | 0.83 |
| `CONT-04` | open_continuation | Deep learning models require b... | `. There are $<<2*2=40>>40 x 200 = <<40*2=60` | N/A | 183.0 | 0.59 | 0.00 |
| `CONT-05` | open_continuation | An indigenous sovereign artifi... | ` So he has 60/2=<<2*2=200>>36 So he has 200*` | N/A | 201.4 | 0.50 | 0.10 |