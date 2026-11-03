# Benchmark Report: M3 Step 500 Checkpoint
- **Suite Version**: `2.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `646f45d416bbdc7908caf39c72f124121fd655dc54b09354f01e9a4a6a1c6082`
- **Checkpoint Path**: `D:\Project\ChakrView\experiments\m3_conversational_intelligence\runs\m3_sft_500steps\sft_checkpoint_step_500.pt`
- **Checkpoint SHA-256**: `36d5aa5ca98fb995b0b7b86d12ecb53558b97e4c4c6a5769f3eb9a9ee82bffe6`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 30
- **Total Generated Tokens**: 296
- **Mean Latency**: 77.00 ms (Median: 53.95 ms)
- **Mean TTFT**: 26.84 ms
- **Mean Throughput**: 93.31 tok/s
- **Mean Unique-Token Ratio**: 0.878
- **Mean 4-Gram Repetition Ratio**: 0.028
- **Termination**: 25 EOS / 5 Token Limit
- **Deterministic Task Accuracy**: 0/20 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 50 chars) | Correct? | Failure Category | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley ... | `` | FAIL | `empty_output / early_eos` | 20.0 | 1.00 | 0.00 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was... | `` | FAIL | `empty_output / early_eos` | 19.6 | 1.00 | 0.00 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata sat... | `` | FAIL | `empty_output / early_eos` | 20.4 | 1.00 | 0.00 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis oc... | `` | FAIL | `empty_output / early_eos` | 19.0 | 1.00 | 0.00 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line si... | `` | FAIL | `empty_output / early_eos` | 20.0 | 1.00 | 0.00 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides v... | `` | N/A | `empty_output / early_eos` | 18.1 | 1.00 | 0.00 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled langu... | `The nucleus.` | N/A | `unsupported_claim / non_grounded` | 102.6 | 1.00 | 0.00 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous c... | `` | N/A | `empty_output / early_eos` | 19.3 | 1.00 | 0.00 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through ca... | `The nucleus.` | N/A | `unsupported_claim / non_grounded` | 110.6 | 1.00 | 0.00 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing ch... | `The green.` | N/A | `unsupported_claim / non_grounded` | 86.2 | 1.00 | 0.00 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `10.` | FAIL | `arithmetic_error / ungrounded` | 70.5 | 1.00 | 0.00 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `10.` | FAIL | `arithmetic_error / ungrounded` | 69.5 | 1.00 | 0.00 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `<<3*2=6>>10 hours.` | FAIL | `template_leakage / memorization` | 147.7 | 0.86 | 0.00 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `<<60+2=6>>10 hours.` | FAIL | `template_leakage / memorization` | 161.3 | 0.80 | 0.00 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 app... | `The sentral is the largest largest amount of plane` | FAIL | `arithmetic_error / ungrounded` | 222.9 | 0.59 | 0.21 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are... | `Linux was many square she has 60 + 2 = $<<2*2=2>>3` | FAIL | `template_leakage / memorization` | 196.7 | 0.81 | 0.00 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number... | ` (2, 4)(4)(42)2)` | FAIL | `instruction_noncompliance` | 163.8 | 0.47 | 0.08 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean ex... | `         raise ValueError: natural cannot be not h` | FAIL | `instruction_noncompliance` | 197.0 | 0.66 | 0.17 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data st... | `` | FAIL | `empty_output / early_eos` | 23.4 | 1.00 | 0.00 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-l... | `&quot; OS&quot; \|  \|- \| What \|\| Month \|\|` | FAIL | `repetition / loop_collapse` | 197.8 | 0.50 | 0.07 |
| `CODE-01` | safe_coding_structure | Write a Python function n... | `         ...` | FAIL | `syntax_error / non_code` | 111.2 | 0.67 | 0.00 |
| `CODE-02` | safe_coding_structure | Write a Python function n... | `         ..` | FAIL | `syntax_error / non_code` | 93.0 | 0.80 | 0.00 |
| `CODE-03` | safe_coding_structure | In C, declare a variable ... | ` What is not be reflecting reflecting the nucleus.` | FAIL | `syntax_error / non_code` | 191.3 | 0.76 | 0.09 |
| `CODE-04` | safe_coding_structure | Write the SQL query to se... | `         >>> linked_list.add()` | FAIL | `syntax_error / non_code` | 135.9 | 0.93 | 0.00 |
| `CODE-05` | safe_coding_structure | In Python, how do you imp... | `` | FAIL | `empty_output / early_eos` | 25.9 | 1.00 | 0.00 |
| `CONT-01` | open_continuation | In modern operating syste... | `] is a gressity of the largest common is the large` | N/A | `human_judgment_required` | 188.8 | 0.50 | 0.21 |
| `CONT-02` | open_continuation | The primary advantage of ... | `land.` | N/A | `human_judgment_required` | 64.3 | 1.00 | 0.00 |
| `CONT-03` | open_continuation | When designing reliable d... | `` | N/A | `empty_output / early_eos` | 27.3 | 1.00 | 0.00 |
| `CONT-04` | open_continuation | Deep learning models requ... | `.` | N/A | `human_judgment_required` | 47.7 | 1.00 | 0.00 |
| `CONT-05` | open_continuation | An indigenous sovereign a... | `` | N/A | `empty_output / early_eos` | 27.6 | 1.00 | 0.00 |