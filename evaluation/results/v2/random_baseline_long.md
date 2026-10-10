# Benchmark Report: RANDOM_BASELINE_V2
- **Suite Version**: `2.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- **Checkpoint Path**: `None (Random Init)`
- **Checkpoint SHA-256**: `N/A`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 30
- **Total Generated Tokens**: 2880
- **Mean Latency**: 377.85 ms (Median: 370.31 ms)
- **Mean TTFT**: 20.23 ms
- **Mean Throughput**: 257.36 tok/s
- **Mean Unique-Token Ratio**: 0.010
- **Mean 4-Gram Repetition Ratio**: 0.989
- **Termination**: 0 EOS / 30 Token Limit
- **Deterministic Task Accuracy**: 0/20 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 50 chars) | Correct? | Failure Category | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley ... | `??????????????????????????????????????????????????` | FAIL | `degeneration / character_loop` | 259.6 | 0.01 | 0.99 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was... | `??????????????????????????????????????????????????` | FAIL | `degeneration / character_loop` | 341.2 | 0.01 | 0.99 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata sat... | `??????????????????????????????????????????????????` | FAIL | `degeneration / character_loop` | 298.6 | 0.01 | 0.99 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis oc... | `??????????????????????????????????????????????????` | FAIL | `degeneration / character_loop` | 197.9 | 0.01 | 0.99 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line si... | `??????????????????????????????????????????????????` | FAIL | `degeneration / character_loop` | 266.6 | 0.01 | 0.99 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides v... | `e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:` | N/A | `unsupported_claim / non_grounded` | 236.6 | 0.01 | 0.99 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled langu... | `::::::::::::::::::::::::::::::::::::::::::::::::::` | N/A | `degeneration / character_loop` | 284.0 | 0.01 | 0.99 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous c... | `s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:` | N/A | `unsupported_claim / non_grounded` | 287.8 | 0.01 | 0.99 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through ca... | `::::::::::::::::::::::::::::::::::::::::::::::::::` | N/A | `degeneration / character_loop` | 276.1 | 0.01 | 0.99 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing ch... | `::::::::::::::::::::::::::::::::::::::::::::::::::` | N/A | `degeneration / character_loop` | 303.7 | 0.01 | 0.99 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `==================================================` | FAIL | `degeneration / character_loop` | 265.4 | 0.01 | 0.99 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `==================================================` | FAIL | `degeneration / character_loop` | 282.5 | 0.01 | 0.99 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `==================================================` | FAIL | `degeneration / character_loop` | 259.8 | 0.01 | 0.99 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `==================================================` | FAIL | `degeneration / character_loop` | 227.6 | 0.01 | 0.99 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 app... | `::::::::::::::::::::::::::::::::::::::::::::::::::` | FAIL | `degeneration / character_loop` | 216.8 | 0.01 | 0.99 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are... | `::::::::::::::::::::::::::::::::::::::::::::::::::` | FAIL | `degeneration / character_loop` | 258.4 | 0.01 | 0.99 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number... | `,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,` | FAIL | `instruction_noncompliance` | 256.8 | 0.01 | 0.99 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean ex... | `::::::::::::::::::::::::::::::::::::::::::::::::::` | FAIL | `degeneration / character_loop` | 220.4 | 0.01 | 0.99 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data st... | `??????????????????????????????????????????????????` | FAIL | `degeneration / character_loop` | 224.7 | 0.01 | 0.99 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-l... | `::::::::::::::::::::::::::::::::::::::::::::::::::` | FAIL | `degeneration / character_loop` | 251.5 | 0.01 | 0.99 |
| `CODE-01` | safe_coding_structure | Write a Python function n... | `::::::::::::::::::::::::::::::::::::::::::::::::::` | FAIL | `degeneration / character_loop` | 258.0 | 0.01 | 0.99 |
| `CODE-02` | safe_coding_structure | Write a Python function n... | `::::::::::::::::::::::::::::::::::::::::::::::::::` | FAIL | `degeneration / character_loop` | 229.2 | 0.01 | 0.99 |
| `CODE-03` | safe_coding_structure | In C, declare a variable ... | `0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;` | FAIL | `syntax_error / non_code` | 222.1 | 0.01 | 0.99 |
| `CODE-04` | safe_coding_structure | Write the SQL query to se... | `::::::::::::::::::::::::::::::::::::::::::::::::::` | FAIL | `degeneration / character_loop` | 264.1 | 0.01 | 0.99 |
| `CODE-05` | safe_coding_structure | In Python, how do you imp... | `??????????????????????????????????????????????????` | FAIL | `degeneration / character_loop` | 243.3 | 0.01 | 0.99 |
| `CONT-01` | open_continuation | In modern operating syste... | `ssssssssssssssssssssssssssssssssssssssssssssssssss` | N/A | `human_judgment_required` | 229.8 | 0.01 | 0.99 |
| `CONT-02` | open_continuation | The primary advantage of ... | `isisisisisisisisisisisisisisisisisisisisisisisisis` | N/A | `human_judgment_required` | 264.1 | 0.01 | 0.99 |
| `CONT-03` | open_continuation | When designing reliable d... | `ssssssssssssssssssssssssssssssssssssssssssssssssss` | N/A | `human_judgment_required` | 270.8 | 0.01 | 0.99 |
| `CONT-04` | open_continuation | Deep learning models requ... | `sesesesesesesesesesesesesesesesesesesesesesesesese` | N/A | `human_judgment_required` | 259.2 | 0.01 | 0.99 |
| `CONT-05` | open_continuation | An indigenous sovereign a... | `ssssssssssssssssssssssssssssssssssssssssssssssssss` | N/A | `human_judgment_required` | 264.1 | 0.01 | 0.99 |