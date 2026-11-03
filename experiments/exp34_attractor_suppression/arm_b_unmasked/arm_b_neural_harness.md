# Benchmark Report: Arm B Unmasked (Step 50)
- **Suite Version**: `2.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `9b2f8030bf9ed0702d265717ec562ab178b753f156b8635d9b3ad0e938a491f0`
- **Checkpoint Path**: `experiments\exp34_attractor_suppression\arm_b_unmasked\sft_checkpoint_step_50.pt`
- **Checkpoint SHA-256**: `ab4e61e1ba95c5705ff009e2b14962e3f8f103d1b303e13bff92da449aecfb99`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 30
- **Total Generated Tokens**: 931
- **Mean Latency**: 120.59 ms (Median: 105.44 ms)
- **Mean TTFT**: 17.63 ms
- **Mean Throughput**: 283.27 tok/s
- **Mean Unique-Token Ratio**: 0.568
- **Mean 4-Gram Repetition Ratio**: 0.233
- **Termination**: 4 EOS / 26 Token Limit
- **Deterministic Task Accuracy**: 0/20 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 50 chars) | Correct? | Failure Category | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley ... | `  Step-by-Step Solution: The total number of the f` | FAIL | `unsupported_claim / non_grounded` | 316.4 | 0.69 | 0.10 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was... | `  Step-by-Step Solution: The first of the first of` | FAIL | `repetition / loop_collapse` | 367.5 | 0.56 | 0.24 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata sat... | `  Step-by-Step Solution: * [https://wsons.get_` | FAIL | `repetition / loop_collapse` | 402.0 | 0.78 | 0.00 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis oc... | `   Stemperature is a pecial number of the word of ` | FAIL | `unsupported_claim / non_grounded` | 368.5 | 0.56 | 0.31 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line si... | `  Step-by-Step Solution: The total number of the t` | FAIL | `unsupported_claim / non_grounded` | 382.7 | 0.59 | 0.21 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides v... | ` * [https://web.archive.org/web/20050/20050/` | N/A | `repetition / loop_collapse` | 350.3 | 0.62 | 0.10 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled langu... | ` * [https://en.wikipedia.org/wiki/Book and the ` | N/A | `repetition / loop_collapse` | 392.1 | 0.72 | 0.00 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous c... | `  from the first resulting the word of the word of` | N/A | `unsupported_claim / non_grounded` | 338.9 | 0.44 | 0.45 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through ca... | ` * [https://en.wikipedia.org/wiki/Booketc` | N/A | `repetition / loop_collapse` | 407.7 | 0.72 | 0.00 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing ch... | `   The result is a list of the result of the word ` | N/A | `unsupported_claim / non_grounded` | 366.7 | 0.44 | 0.28 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `10.50 = <<240+24=40>>50 century. #### 200` | FAIL | `template_leakage / memorization` | 412.0 | 0.66 | 0.00 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `<<40+2=4>>4 hours. #### 2` | FAIL | `template_leakage / memorization` | 222.7 | 0.68 | 0.00 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `<<3*2=4>>4 hours. #### 2` | FAIL | `template_leakage / memorization` | 138.5 | 0.71 | 0.00 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `<<40/2=4>>6 candles. #### 2` | FAIL | `template_leakage / memorization` | 184.8 | 0.71 | 0.00 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 app... | `  Step-by-Step Solution: The total number of the t` | FAIL | `arithmetic_error / ungrounded` | 214.5 | 0.59 | 0.21 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are... | ` * The first number of the word of the word of the` | FAIL | `instruction_noncompliance` | 301.7 | 0.38 | 0.59 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number... | `     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,` | FAIL | `instruction_noncompliance` | 303.5 | 0.12 | 0.86 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean ex... | `         raise ValueError("IncrementalEncoder")   ` | FAIL | `instruction_noncompliance` | 299.2 | 0.88 | 0.00 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data st... | `  Step-by-Step Solution: The total number of the t` | FAIL | `instruction_noncompliance` | 307.8 | 0.59 | 0.21 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-l... | ` * [http://www.gov.co.com/books?v=200` | FAIL | `repetition / loop_collapse` | 186.1 | 0.69 | 0.00 |
| `CODE-01` | safe_coding_structure | Write a Python function n... | `         if not self._subset_node:             sel` | FAIL | `syntax_error / non_code` | 181.8 | 0.53 | 0.17 |
| `CODE-02` | safe_coding_structure | Write a Python function n... | `         if not self._subset("__main__":          ` | FAIL | `syntax_error / non_code` | 198.5 | 0.66 | 0.10 |
| `CODE-03` | safe_coding_structure | In C, declare a variable ... | ` the first number of the number of the number of t` | FAIL | `repetition / loop_collapse` | 176.6 | 0.25 | 0.69 |
| `CODE-04` | safe_coding_structure | Write the SQL query to se... | `         if not self._subset("__main__":          ` | FAIL | `syntax_error / non_code` | 156.8 | 0.66 | 0.10 |
| `CODE-05` | safe_coding_structure | In Python, how do you imp... | `"      # This is the result of the number of the n` | FAIL | `repetition / loop_collapse` | 280.5 | 0.47 | 0.41 |
| `CONT-01` | open_continuation | In modern operating syste... | `] and the word of the word of the word of the word` | N/A | `human_judgment_required` | 308.9 | 0.19 | 0.79 |
| `CONT-02` | open_continuation | The primary advantage of ... | `ins to be a community of the word of the world. Th` | N/A | `human_judgment_required` | 176.8 | 0.66 | 0.14 |
| `CONT-03` | open_continuation | When designing reliable d... | `?  Step-by-Step Solution: The total number of the ` | N/A | `degeneration / character_loop` | 152.1 | 0.62 | 0.17 |
| `CONT-04` | open_continuation | Deep learning models requ... | `s.      >>> linked_list.insert("A", "A", "A", "A",` | N/A | `human_judgment_required` | 283.3 | 0.47 | 0.38 |
| `CONT-05` | open_continuation | An indigenous sovereign a... | `      # If the first number of the number of the n` | N/A | `repetition / loop_collapse` | 318.9 | 0.41 | 0.48 |