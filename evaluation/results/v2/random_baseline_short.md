# Benchmark Report: RANDOM_BASELINE_V2
- **Suite Version**: `2.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- **Checkpoint Path**: `None (Random Init)`
- **Checkpoint SHA-256**: `N/A`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 30
- **Total Generated Tokens**: 960
- **Mean Latency**: 149.72 ms (Median: 144.42 ms)
- **Mean TTFT**: 20.64 ms
- **Mean Throughput**: 223.81 tok/s
- **Mean Unique-Token Ratio**: 0.031
- **Mean 4-Gram Repetition Ratio**: 0.966
- **Termination**: 0 EOS / 30 Token Limit
- **Deterministic Task Accuracy**: 0/20 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 60 chars) | Correct? | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley site ... | `????????????????????????????????` | FAIL | 262.0 | 0.03 | 0.97 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was laun... | `????????????????????????????????` | FAIL | 301.9 | 0.03 | 0.97 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata satellit... | `????????????????????????????????` | FAIL | 304.2 | 0.03 | 0.97 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis occurs ... | `????????????????????????????????` | FAIL | 308.9 | 0.03 | 0.97 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line size of... | `????????????????????????????????` | FAIL | 298.0 | 0.03 | 0.97 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides virtua... | `e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:` | N/A | 282.2 | 0.03 | 0.97 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled language c... | `::::::::::::::::::::::::::::::::` | N/A | 304.4 | 0.03 | 0.97 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous commun... | `s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:` | N/A | 256.1 | 0.03 | 0.97 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through caches ... | `::::::::::::::::::::::::::::::::` | N/A | 233.5 | 0.03 | 0.97 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing checks ... | `::::::::::::::::::::::::::::::::` | N/A | 164.4 | 0.03 | 0.97 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `================================` | FAIL | 144.8 | 0.03 | 0.97 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `================================` | FAIL | 157.4 | 0.03 | 0.97 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `================================` | FAIL | 189.6 | 0.03 | 0.97 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `================================` | FAIL | 194.9 | 0.03 | 0.97 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 apples. ... | `::::::::::::::::::::::::::::::::` | FAIL | 196.3 | 0.03 | 0.97 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are mamm... | `::::::::::::::::::::::::::::::::` | FAIL | 206.5 | 0.03 | 0.97 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number sequ... | `,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,` | FAIL | 235.4 | 0.03 | 0.97 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean express... | `::::::::::::::::::::::::::::::::` | FAIL | 249.1 | 0.03 | 0.97 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data structu... | `????????????????????????????????` | FAIL | 232.0 | 0.03 | 0.97 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-letter... | `::::::::::::::::::::::::::::::::` | FAIL | 195.3 | 0.03 | 0.97 |
| `CODE-01` | safe_coding_structure | Write a Python function named ... | `::::::::::::::::::::::::::::::::` | FAIL | 222.9 | 0.03 | 0.97 |
| `CODE-02` | safe_coding_structure | Write a Python function named ... | `::::::::::::::::::::::::::::::::` | FAIL | 230.6 | 0.03 | 0.97 |
| `CODE-03` | safe_coding_structure | In C, declare a variable named... | `0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;` | FAIL | 174.0 | 0.03 | 0.97 |
| `CODE-04` | safe_coding_structure | Write the SQL query to select ... | `::::::::::::::::::::::::::::::::` | FAIL | 159.8 | 0.03 | 0.97 |
| `CODE-05` | safe_coding_structure | In Python, how do you import t... | `????????????????????????????????` | FAIL | 176.9 | 0.03 | 0.97 |
| `CONT-01` | open_continuation | In modern operating systems, v... | `ssssssssssssssssssssssssssssssss` | N/A | 201.4 | 0.03 | 0.97 |
| `CONT-02` | open_continuation | The primary advantage of pipel... | `isisisisisisisisisisisisisisisisisisisisisisisisisisisisisis` | N/A | 241.3 | 0.03 | 0.97 |
| `CONT-03` | open_continuation | When designing reliable distri... | `ssssssssssssssssssssssssssssssss` | N/A | 177.9 | 0.03 | 0.97 |
| `CONT-04` | open_continuation | Deep learning models require b... | `sesesesesesesesesesesesesesesesesesesesesesesesesesesesesese` | N/A | 190.8 | 0.03 | 0.97 |
| `CONT-05` | open_continuation | An indigenous sovereign artifi... | `ssssssssssssssssssssssssssssssss` | N/A | 221.6 | 0.03 | 0.97 |