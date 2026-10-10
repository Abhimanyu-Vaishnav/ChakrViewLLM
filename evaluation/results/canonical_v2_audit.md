# Benchmark Report: RANDOM_INITIALIZATION
- **Suite Version**: `2.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- **Checkpoint Path**: `None (Random Init)`
- **Checkpoint SHA-256**: `N/A`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 30
- **Total Generated Tokens**: 960
- **Mean Latency**: 202.74 ms (Median: 217.26 ms)
- **Mean TTFT**: 25.31 ms
- **Mean Throughput**: 162.26 tok/s
- **Mean Unique-Token Ratio**: 0.031
- **Mean 4-Gram Repetition Ratio**: 0.966
- **Termination**: 0 EOS / 30 Token Limit
- **Deterministic Task Accuracy**: 0/20 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 50 chars) | Correct? | Failure Category | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley ... | `????????????????????????????????` | FAIL | `degeneration / character_loop` | 195.4 | 0.03 | 0.97 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was... | `????????????????????????????????` | FAIL | `degeneration / character_loop` | 164.1 | 0.03 | 0.97 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata sat... | `????????????????????????????????` | FAIL | `degeneration / character_loop` | 170.1 | 0.03 | 0.97 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis oc... | `????????????????????????????????` | FAIL | `degeneration / character_loop` | 192.4 | 0.03 | 0.97 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line si... | `????????????????????????????????` | FAIL | `degeneration / character_loop` | 186.6 | 0.03 | 0.97 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides v... | `e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:e:` | N/A | `unsupported_claim / non_grounded` | 130.3 | 0.03 | 0.97 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled langu... | `::::::::::::::::::::::::::::::::` | N/A | `degeneration / character_loop` | 136.2 | 0.03 | 0.97 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous c... | `s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:s:` | N/A | `unsupported_claim / non_grounded` | 145.2 | 0.03 | 0.97 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through ca... | `::::::::::::::::::::::::::::::::` | N/A | `degeneration / character_loop` | 197.8 | 0.03 | 0.97 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing ch... | `::::::::::::::::::::::::::::::::` | N/A | `degeneration / character_loop` | 174.3 | 0.03 | 0.97 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `================================` | FAIL | `degeneration / character_loop` | 209.1 | 0.03 | 0.97 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `================================` | FAIL | `degeneration / character_loop` | 192.2 | 0.03 | 0.97 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `================================` | FAIL | `degeneration / character_loop` | 144.4 | 0.03 | 0.97 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `================================` | FAIL | `degeneration / character_loop` | 142.3 | 0.03 | 0.97 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 app... | `::::::::::::::::::::::::::::::::` | FAIL | `degeneration / character_loop` | 165.0 | 0.03 | 0.97 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are... | `::::::::::::::::::::::::::::::::` | FAIL | `degeneration / character_loop` | 153.8 | 0.03 | 0.97 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number... | `,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,` | FAIL | `instruction_noncompliance` | 128.7 | 0.03 | 0.97 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean ex... | `::::::::::::::::::::::::::::::::` | FAIL | `degeneration / character_loop` | 138.5 | 0.03 | 0.97 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data st... | `????????????????????????????????` | FAIL | `degeneration / character_loop` | 137.8 | 0.03 | 0.97 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-l... | `::::::::::::::::::::::::::::::::` | FAIL | `degeneration / character_loop` | 136.2 | 0.03 | 0.97 |
| `CODE-01` | safe_coding_structure | Write a Python function n... | `::::::::::::::::::::::::::::::::` | FAIL | `degeneration / character_loop` | 129.0 | 0.03 | 0.97 |
| `CODE-02` | safe_coding_structure | Write a Python function n... | `::::::::::::::::::::::::::::::::` | FAIL | `degeneration / character_loop` | 191.4 | 0.03 | 0.97 |
| `CODE-03` | safe_coding_structure | In C, declare a variable ... | `0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;` | FAIL | `syntax_error / non_code` | 147.1 | 0.03 | 0.97 |
| `CODE-04` | safe_coding_structure | Write the SQL query to se... | `::::::::::::::::::::::::::::::::` | FAIL | `degeneration / character_loop` | 134.8 | 0.03 | 0.97 |
| `CODE-05` | safe_coding_structure | In Python, how do you imp... | `????????????????????????????????` | FAIL | `degeneration / character_loop` | 139.0 | 0.03 | 0.97 |
| `CONT-01` | open_continuation | In modern operating syste... | `ssssssssssssssssssssssssssssssss` | N/A | `human_judgment_required` | 146.3 | 0.03 | 0.97 |
| `CONT-02` | open_continuation | The primary advantage of ... | `isisisisisisisisisisisisisisisisisisisisisisisisis` | N/A | `human_judgment_required` | 185.0 | 0.03 | 0.97 |
| `CONT-03` | open_continuation | When designing reliable d... | `ssssssssssssssssssssssssssssssss` | N/A | `human_judgment_required` | 147.3 | 0.03 | 0.97 |
| `CONT-04` | open_continuation | Deep learning models requ... | `sesesesesesesesesesesesesesesesesesesesesesesesese` | N/A | `human_judgment_required` | 244.1 | 0.03 | 0.97 |
| `CONT-05` | open_continuation | An indigenous sovereign a... | `ssssssssssssssssssssssssssssssss` | N/A | `human_judgment_required` | 163.3 | 0.03 | 0.97 |