# Benchmark Report: M3 Step 250 Checkpoint
- **Suite Version**: `2.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `7c5708f7e9d6f6b5533c04a9e60a7620cc0cfdcf190af1dcc8058a3952e9f4ee`
- **Checkpoint Path**: `D:\Project\ChakrView\experiments\m3_conversational_intelligence\runs\m3_sft_500steps\sft_checkpoint_step_250.pt`
- **Checkpoint SHA-256**: `91b0bae2a8c737c8db4079c8ac3645f8ff8436b12501f687660f4f9ab7b83433`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 30
- **Total Generated Tokens**: 279
- **Mean Latency**: 85.24 ms (Median: 67.10 ms)
- **Mean TTFT**: 29.04 ms
- **Mean Throughput**: 84.19 tok/s
- **Mean Unique-Token Ratio**: 0.904
- **Mean 4-Gram Repetition Ratio**: 0.022
- **Termination**: 28 EOS / 2 Token Limit
- **Deterministic Task Accuracy**: 0/20 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 50 chars) | Correct? | Failure Category | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley ... | `` | FAIL | `empty_output / early_eos` | 19.9 | 1.00 | 0.00 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was... | `` | FAIL | `empty_output / early_eos` | 21.5 | 1.00 | 0.00 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata sat... | `` | FAIL | `empty_output / early_eos` | 20.6 | 1.00 | 0.00 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis oc... | `` | FAIL | `empty_output / early_eos` | 19.9 | 1.00 | 0.00 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line si... | `` | FAIL | `empty_output / early_eos` | 19.6 | 1.00 | 0.00 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides v... | `` | N/A | `empty_output / early_eos` | 14.9 | 1.00 | 0.00 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled langu... | ` * The light is a facea is the word of the nucleus` | N/A | `unsupported_claim / non_grounded` | 134.5 | 0.87 | 0.00 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous c... | `` | N/A | `empty_output / early_eos` | 17.6 | 1.00 | 0.00 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through ca... | `The nucleus.` | N/A | `unsupported_claim / non_grounded` | 83.6 | 1.00 | 0.00 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing ch... | `The green.` | N/A | `unsupported_claim / non_grounded` | 80.0 | 1.00 | 0.00 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `10.` | FAIL | `arithmetic_error / ungrounded` | 64.5 | 1.00 | 0.00 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `70>>9.` | FAIL | `arithmetic_error / ungrounded` | 109.6 | 0.86 | 0.00 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `<<3*2=6>>10 hours.` | FAIL | `template_leakage / memorization` | 142.2 | 0.86 | 0.00 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `<<40+4=6>>10 hours.` | FAIL | `template_leakage / memorization` | 157.4 | 0.80 | 0.00 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 app... | `The total number of chemical number.` | FAIL | `arithmetic_error / ungrounded` | 133.3 | 0.86 | 0.00 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are... | `Linux was many square she has 60 + 2 = $<<2*2=2>>6` | FAIL | `template_leakage / memorization` | 132.4 | 0.81 | 0.00 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number... | `777.` | FAIL | `instruction_noncompliance` | 68.6 | 0.60 | 0.00 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean ex... | `         raise ValueError: Caller is the nucleus.` | FAIL | `instruction_noncompliance` | 124.9 | 0.96 | 0.00 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data st... | `` | FAIL | `empty_output / early_eos` | 19.5 | 1.00 | 0.00 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-l... | `&quot; OS&quot; is a nucleus.` | FAIL | `instruction_noncompliance` | 128.2 | 0.75 | 0.12 |
| `CODE-01` | safe_coding_structure | Write a Python function n... | `         ...` | FAIL | `syntax_error / non_code` | 96.3 | 0.67 | 0.00 |
| `CODE-02` | safe_coding_structure | Write a Python function n... | `         if not _result in in range(len(self._stac` | FAIL | `syntax_error / non_code` | 155.2 | 0.86 | 0.00 |
| `CODE-03` | safe_coding_structure | In C, declare a variable ... | ` What is the nucleus.` | FAIL | `syntax_error / non_code` | 141.4 | 1.00 | 0.00 |
| `CODE-04` | safe_coding_structure | Write the SQL query to se... | `         if not _result in range(len("index")` | FAIL | `syntax_error / non_code` | 196.8 | 0.90 | 0.00 |
| `CODE-05` | safe_coding_structure | In Python, how do you imp... | `` | FAIL | `empty_output / early_eos` | 26.1 | 1.00 | 0.00 |
| `CONT-01` | open_continuation | In modern operating syste... | `] is a person is a person of the word of the word ` | N/A | `human_judgment_required` | 222.7 | 0.31 | 0.55 |
| `CONT-02` | open_continuation | The primary advantage of ... | `land.` | N/A | `human_judgment_required` | 71.9 | 1.00 | 0.00 |
| `CONT-03` | open_continuation | When designing reliable d... | `` | N/A | `empty_output / early_eos` | 27.9 | 1.00 | 0.00 |
| `CONT-04` | open_continuation | Deep learning models requ... | `.` | N/A | `human_judgment_required` | 47.3 | 1.00 | 0.00 |
| `CONT-05` | open_continuation | An indigenous sovereign a... | `` | N/A | `empty_output / early_eos` | 27.5 | 1.00 | 0.00 |