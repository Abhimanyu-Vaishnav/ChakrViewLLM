# Benchmark Report: Canonical Pretrained (Step 10,000)
- **Suite Version**: `2.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `c5bf628bd059910dd89d1240c8df2bbfc102e131fa5123c56813401c28e3be71`
- **Checkpoint Path**: `experiments\batch16_21_combined\main_training_10000\training_checkpoint_step_10000.pt`
- **Checkpoint SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 30
- **Total Generated Tokens**: 942
- **Mean Latency**: 142.88 ms (Median: 133.65 ms)
- **Mean TTFT**: 18.89 ms
- **Mean Throughput**: 239.27 tok/s
- **Mean Unique-Token Ratio**: 0.531
- **Mean 4-Gram Repetition Ratio**: 0.254
- **Termination**: 2 EOS / 28 Token Limit
- **Deterministic Task Accuracy**: 0/20 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 50 chars) | Correct? | Failure Category | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley ... | `  Step-by-Step Solution: The Andrince of the Chris` | FAIL | `unsupported_claim / non_grounded` | 329.7 | 0.84 | 0.00 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was... | `  Step-by-Step Solution: The first of the first of` | FAIL | `repetition / loop_collapse` | 329.0 | 0.56 | 0.24 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata sat... | `  Step-by-Step Solution:   Specia is the file, whi` | FAIL | `unsupported_claim / non_grounded` | 315.1 | 0.72 | 0.00 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis oc... | `   Steleft of the city of the son is the world of ` | FAIL | `unsupported_claim / non_grounded` | 349.0 | 0.66 | 0.00 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line si... | `  Step-by-Step Solution: The total number of the c` | FAIL | `unsupported_claim / non_grounded` | 340.7 | 0.66 | 0.00 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides v... | `  * [https://web.archive.orlcanday of the word of ` | N/A | `repetition / loop_collapse` | 307.2 | 0.78 | 0.00 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled langu... | `   The resulting of the word of the city of the wo` | N/A | `unsupported_claim / non_grounded` | 337.6 | 0.41 | 0.38 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous c... | `                                ` | N/A | `empty_output / early_eos` | 330.3 | 0.03 | 0.97 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through ca... | `   Sumber 2000 and the city of the city of the cit` | N/A | `unsupported_claim / non_grounded` | 323.3 | 0.47 | 0.34 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing ch... | `                                ` | N/A | `empty_output / early_eos` | 316.8 | 0.03 | 0.97 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `105 / 200 = <<240+24=24>>50 centuries. ####` | FAIL | `template_leakage / memorization` | 173.0 | 0.66 | 0.00 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `<<2*2=2>>6 centuries. #### 2` | FAIL | `template_leakage / memorization` | 194.3 | 0.65 | 0.00 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `<<2*2=2>>6 hours in the total of the total of the ` | FAIL | `template_leakage / memorization` | 273.6 | 0.53 | 0.17 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `<<40+2=4>>6 century. #### 2` | FAIL | `template_leakage / memorization` | 208.8 | 0.70 | 0.00 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 app... | `  Step-by-Step Solution: The total of the city of ` | FAIL | `arithmetic_error / ungrounded` | 171.0 | 0.75 | 0.00 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are... | ` * John Book is a shows the city of the shows the ` | FAIL | `instruction_noncompliance` | 219.0 | 0.56 | 0.21 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number... | `     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,` | FAIL | `instruction_noncompliance` | 190.3 | 0.12 | 0.86 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean ex... | `         raise ValueError("The number of the numbe` | FAIL | `repetition / loop_collapse` | 246.6 | 0.53 | 0.38 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data st... | `  Step-by-Step Solution: The total of the first of` | FAIL | `repetition / loop_collapse` | 168.2 | 0.62 | 0.17 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-l... | ` * [http://www.gov.co.com/books?id=200` | FAIL | `repetition / loop_collapse` | 272.3 | 0.72 | 0.00 |
| `CODE-01` | safe_coding_structure | Write a Python function n... | `         ...         >>> h.insert(")         >>> h` | FAIL | `syntax_error / non_code` | 239.4 | 0.31 | 0.45 |
| `CODE-02` | safe_coding_structure | Write a Python function n... | `         if not isinstance(")         if not self.` | FAIL | `syntax_error / non_code` | 149.4 | 0.66 | 0.14 |
| `CODE-03` | safe_coding_structure | In C, declare a variable ... | ` the first of the world, which is the world of the` | FAIL | `syntax_error / non_code` | 211.6 | 0.56 | 0.10 |
| `CODE-04` | safe_coding_structure | Write the SQL query to se... | `      # This is the number of the number of the nu` | FAIL | `repetition / loop_collapse` | 207.1 | 0.31 | 0.62 |
| `CODE-05` | safe_coding_structure | In Python, how do you imp... | `   Step-by-Step Solution: The total number of the ` | FAIL | `syntax_error / non_code` | 156.0 | 0.66 | 0.00 |
| `CONT-01` | open_continuation | In modern operating syste... | ` * [http://www.gov.co.com/index.html Gre` | N/A | `repetition / loop_collapse` | 150.4 | 0.66 | 0.00 |
| `CONT-02` | open_continuation | The primary advantage of ... | `land, which is the world of the world. The first o` | N/A | `human_judgment_required` | 155.3 | 0.56 | 0.28 |
| `CONT-03` | open_continuation | When designing reliable d... | `      >>> linked_list.insert(')     >>> linked_lis` | N/A | `human_judgment_required` | 165.8 | 0.41 | 0.45 |
| `CONT-04` | open_continuation | Deep learning models requ... | `.      >>> linked_list.insert(')     >>> linked_li` | N/A | `human_judgment_required` | 165.9 | 0.41 | 0.45 |
| `CONT-05` | open_continuation | An indigenous sovereign a... | `      >>> linked_list.insert(')     >>> linked_lis` | N/A | `human_judgment_required` | 181.4 | 0.41 | 0.45 |