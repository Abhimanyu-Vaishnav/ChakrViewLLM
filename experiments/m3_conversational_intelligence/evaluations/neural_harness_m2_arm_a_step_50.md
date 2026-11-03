# Benchmark Report: M2 Arm A Step 50 Checkpoint
- **Suite Version**: `2.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `45557fd8d2106d7d5d7a04b89da145e91e3d9884f4df4c6af70a9c58f583bec8`
- **Checkpoint Path**: `D:\Project\ChakrView\experiments\exp34_attractor_suppression\arm_a_masked\sft_checkpoint_step_50.pt`
- **Checkpoint SHA-256**: `150f6214862eefe0baa4f33fbf9f075594e7f00c61b09834446cc25b0c54cecf`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 30
- **Total Generated Tokens**: 940
- **Mean Latency**: 865.17 ms (Median: 1029.40 ms)
- **Mean TTFT**: 75.59 ms
- **Mean Throughput**: 51.63 tok/s
- **Mean Unique-Token Ratio**: 0.566
- **Mean 4-Gram Repetition Ratio**: 0.186
- **Termination**: 2 EOS / 28 Token Limit
- **Deterministic Task Accuracy**: 0/20 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 50 chars) | Correct? | Failure Category | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley ... | `  Step-by-Step Solution: The first number of the f` | FAIL | `unsupported_claim / non_grounded` | 27.8 | 0.62 | 0.17 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was... | `  Step-by-Step Solution: The first of the first of` | FAIL | `repetition / loop_collapse` | 28.8 | 0.56 | 0.24 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata sat... | `  Step-by-Step Solution:  The first of the first o` | FAIL | `repetition / loop_collapse` | 28.5 | 0.56 | 0.21 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis oc... | `   Stemperature is a pectivels to be used to be a ` | FAIL | `unsupported_claim / non_grounded` | 27.8 | 0.78 | 0.00 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line si... | `  Step-by-Step Solution: The total number of the n` | FAIL | `repetition / loop_collapse` | 30.2 | 0.59 | 0.10 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides v... | ` * [https://web.archive.org/web/20050/20050/` | N/A | `repetition / loop_collapse` | 32.5 | 0.62 | 0.10 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled langu... | ` * [https://en.wikipedia.org/wiki/Book.co` | N/A | `repetition / loop_collapse` | 30.2 | 0.69 | 0.00 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous c... | `  from the first resulting the word of the word of` | N/A | `unsupported_claim / non_grounded` | 27.8 | 0.44 | 0.45 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through ca... | ` * [https://en.wikipedia.org/wiki/200/d/d` | N/A | `repetition / loop_collapse` | 28.0 | 0.66 | 0.00 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing ch... | `   The read of the word of the rection of the file` | N/A | `unsupported_claim / non_grounded` | 27.4 | 0.38 | 0.31 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `700+10 = <<240+24=40>>50 century. #### 200` | FAIL | `template_leakage / memorization` | 29.6 | 0.62 | 0.00 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `<<40+2=4>>4 hours. #### 2` | FAIL | `template_leakage / memorization` | 29.1 | 0.68 | 0.00 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `<<3*2=4>>4 hours in the total number of the total ` | FAIL | `template_leakage / memorization` | 29.8 | 0.75 | 0.10 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `<<40+4=4>>4 hours. #### 2` | FAIL | `template_leakage / memorization` | 28.6 | 0.68 | 0.00 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 app... | `  Step-by-Step Solution: The total number of the t` | FAIL | `arithmetic_error / ungrounded` | 32.4 | 0.59 | 0.21 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are... | ` * The first number of the word of the world of th` | FAIL | `instruction_noncompliance` | 31.5 | 0.41 | 0.45 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number... | `     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,` | FAIL | `instruction_noncompliance` | 30.5 | 0.12 | 0.86 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean ex... | `         if not isinstance("__main__":         if ` | FAIL | `instruction_noncompliance` | 30.3 | 0.62 | 0.10 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data st... | ` * [https://web.archive.org/web/2012060606/h` | FAIL | `repetition / loop_collapse` | 30.2 | 0.62 | 0.03 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-l... | ` * [http://www.gov.co.com/books/pdf/` | FAIL | `repetition / loop_collapse` | 31.1 | 0.62 | 0.00 |
| `CODE-01` | safe_coding_structure | Write a Python function n... | `         if not self._subset("__main__":          ` | FAIL | `syntax_error / non_code` | 32.6 | 0.66 | 0.10 |
| `CODE-02` | safe_coding_structure | Write a Python function n... | `         if not isinstance("__main__":         if ` | FAIL | `syntax_error / non_code` | 33.1 | 0.56 | 0.10 |
| `CODE-03` | safe_coding_structure | In C, declare a variable ... | ` the first common is the first common is the world` | FAIL | `syntax_error / non_code` | 31.1 | 0.53 | 0.24 |
| `CODE-04` | safe_coding_structure | Write the SQL query to se... | `      # Test the number of the number of the numbe` | FAIL | `repetition / loop_collapse` | 62.5 | 0.34 | 0.59 |
| `CODE-05` | safe_coding_structure | In Python, how do you imp... | `      # Test the result of the number of the numbe` | FAIL | `repetition / loop_collapse` | 161.2 | 0.44 | 0.41 |
| `CONT-01` | open_continuation | In modern operating syste... | `] is a community of the world of the world. The re` | N/A | `human_judgment_required` | 110.9 | 0.59 | 0.14 |
| `CONT-02` | open_continuation | The primary advantage of ... | `h is a list of the world. The first common is a li` | N/A | `human_judgment_required` | 152.3 | 0.53 | 0.28 |
| `CONT-03` | open_continuation | When designing reliable d... | ` * [https://web.archive.org/web/20120120606/h` | N/A | `repetition / loop_collapse` | 143.1 | 0.62 | 0.00 |
| `CONT-04` | open_continuation | Deep learning models requ... | `.      >>> linked_list.insert("A", "A", "A", "A", ` | N/A | `human_judgment_required` | 103.0 | 0.44 | 0.38 |
| `CONT-05` | open_continuation | An indigenous sovereign a... | ` * [https://web.archive.org/web/20120120606/h` | N/A | `repetition / loop_collapse` | 126.9 | 0.62 | 0.00 |