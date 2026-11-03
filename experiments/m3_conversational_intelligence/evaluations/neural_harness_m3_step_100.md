# Benchmark Report: M3 Step 100 Checkpoint
- **Suite Version**: `2.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `1ca749b0b9754a98f2f49c5dd5efc7e864406a08fa31b4b185120dd10ce1628e`
- **Checkpoint Path**: `D:\Project\ChakrView\experiments\m3_conversational_intelligence\runs\m3_sft_500steps\sft_checkpoint_step_100.pt`
- **Checkpoint SHA-256**: `8cd534e1d13a17672026420cded4e4cbf2e2b6ec11c30fadeb3c245879859454`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 30
- **Total Generated Tokens**: 440
- **Mean Latency**: 112.67 ms (Median: 88.41 ms)
- **Mean TTFT**: 29.69 ms
- **Mean Throughput**: 97.93 tok/s
- **Mean Unique-Token Ratio**: 0.794
- **Mean 4-Gram Repetition Ratio**: 0.124
- **Termination**: 20 EOS / 10 Token Limit
- **Deterministic Task Accuracy**: 0/20 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 50 chars) | Correct? | Failure Category | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley ... | `` | FAIL | `empty_output / early_eos` | 22.7 | 1.00 | 0.00 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was... | `` | FAIL | `empty_output / early_eos` | 22.7 | 1.00 | 0.00 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata sat... | `` | FAIL | `empty_output / early_eos` | 22.1 | 1.00 | 0.00 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis oc... | `` | FAIL | `empty_output / early_eos` | 19.7 | 1.00 | 0.00 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line si... | `` | FAIL | `empty_output / early_eos` | 17.5 | 1.00 | 0.00 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides v... | `` | N/A | `empty_output / early_eos` | 20.9 | 1.00 | 0.00 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled langu... | ` * The refers is a list of the word of the word of` | N/A | `unsupported_claim / non_grounded` | 177.2 | 0.44 | 0.52 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous c... | `` | N/A | `empty_output / early_eos` | 19.7 | 1.00 | 0.00 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through ca... | `The refers is the non-green.` | N/A | `unsupported_claim / non_grounded` | 133.3 | 0.93 | 0.00 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing ch... | `The green polished to mainying the word of the wor` | N/A | `unsupported_claim / non_grounded` | 172.9 | 0.69 | 0.21 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `10.` | FAIL | `arithmetic_error / ungrounded` | 68.7 | 1.00 | 0.00 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `<<60/2=6>>10 hours. #### 5` | FAIL | `template_leakage / memorization` | 187.5 | 0.73 | 0.00 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `<<3*2=4>>10 hours.` | FAIL | `template_leakage / memorization` | 158.2 | 0.86 | 0.00 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `<<40+4=6>>10 hours.` | FAIL | `template_leakage / memorization` | 169.7 | 0.80 | 0.00 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 app... | `The total of the cannot have a do of years are mad` | FAIL | `arithmetic_error / ungrounded` | 197.8 | 0.78 | 0.10 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are... | `Lake is the non-green?` | FAIL | `instruction_noncompliance` | 150.9 | 1.00 | 0.00 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number... | `777.` | FAIL | `instruction_noncompliance` | 66.7 | 0.60 | 0.00 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean ex... | `         raise ValueError: The function is the non` | FAIL | `instruction_noncompliance` | 136.2 | 0.72 | 0.10 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data st... | `` | FAIL | `empty_output / early_eos` | 21.3 | 1.00 | 0.00 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-l... | `&quot; is a non-group is the non-greenside is the ` | FAIL | `instruction_noncompliance` | 141.8 | 0.66 | 0.10 |
| `CODE-01` | safe_coding_structure | Write a Python function n... | `         if not _conversion.__main__.__main__.__ma` | FAIL | `syntax_error / non_code` | 115.3 | 0.41 | 0.45 |
| `CODE-02` | safe_coding_structure | Write a Python function n... | `         if not isinstance("__main__":         if ` | FAIL | `syntax_error / non_code` | 154.0 | 0.56 | 0.10 |
| `CODE-03` | safe_coding_structure | In C, declare a variable ... | ` the non-section of the word of the word of the wo` | FAIL | `syntax_error / non_code` | 150.2 | 0.34 | 0.59 |
| `CODE-04` | safe_coding_structure | Write the SQL query to se... | `         if not _result in range(len("index")` | FAIL | `syntax_error / non_code` | 136.3 | 0.90 | 0.00 |
| `CODE-05` | safe_coding_structure | In Python, how do you imp... | `` | FAIL | `empty_output / early_eos` | 24.4 | 1.00 | 0.00 |
| `CONT-01` | open_continuation | In modern operating syste... | `] is the word of the word of the word of the word ` | N/A | `human_judgment_required` | 181.3 | 0.19 | 0.79 |
| `CONT-02` | open_continuation | The primary advantage of ... | `lands in the word of the word of the word of the w` | N/A | `human_judgment_required` | 155.2 | 0.22 | 0.76 |
| `CONT-03` | open_continuation | When designing reliable d... | `` | N/A | `empty_output / early_eos` | 25.0 | 1.00 | 0.00 |
| `CONT-04` | open_continuation | Deep learning models requ... | `.` | N/A | `human_judgment_required` | 42.6 | 1.00 | 0.00 |
| `CONT-05` | open_continuation | An indigenous sovereign a... | `` | N/A | `empty_output / early_eos` | 26.2 | 1.00 | 0.00 |