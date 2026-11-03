# Benchmark Report: M3 Step 50 Checkpoint
- **Suite Version**: `2.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `750851844741d8e09c6152e40d3eaba3c18456923183e39aaddf91188f9c1086`
- **Checkpoint Path**: `D:\Project\ChakrView\experiments\m3_conversational_intelligence\runs\m3_sft_500steps\sft_checkpoint_step_50.pt`
- **Checkpoint SHA-256**: `df3bc1c5fee3a3b2e4d1c975c40426660eef4625597e493bbdc88835bb16b5bb`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 30
- **Total Generated Tokens**: 480
- **Mean Latency**: 119.95 ms (Median: 132.10 ms)
- **Mean TTFT**: 31.20 ms
- **Mean Throughput**: 100.65 tok/s
- **Mean Unique-Token Ratio**: 0.766
- **Mean 4-Gram Repetition Ratio**: 0.125
- **Termination**: 19 EOS / 11 Token Limit
- **Deterministic Task Accuracy**: 0/20 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 50 chars) | Correct? | Failure Category | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley ... | `` | FAIL | `empty_output / early_eos` | 19.0 | 1.00 | 0.00 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was... | `` | FAIL | `empty_output / early_eos` | 20.7 | 1.00 | 0.00 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata sat... | `` | FAIL | `empty_output / early_eos` | 19.2 | 1.00 | 0.00 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis oc... | `` | FAIL | `empty_output / early_eos` | 18.5 | 1.00 | 0.00 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line si... | `` | FAIL | `empty_output / early_eos` | 22.6 | 1.00 | 0.00 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides v... | `` | N/A | `empty_output / early_eos` | 15.8 | 1.00 | 0.00 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled langu... | ` * The result is a list of the word of the word of` | N/A | `unsupported_claim / non_grounded` | 169.8 | 0.47 | 0.48 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous c... | `` | N/A | `empty_output / early_eos` | 16.4 | 1.00 | 0.00 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through ca... | ` * [https://en.wikipedia.` | N/A | `repetition / loop_collapse` | 118.2 | 0.85 | 0.00 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing ch... | ` * The reference is a list of the receiving is the` | N/A | `unsupported_claim / non_grounded` | 109.8 | 0.59 | 0.21 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `700.` | FAIL | `arithmetic_error / ungrounded` | 59.6 | 1.00 | 0.00 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `<<4*2=4>>6 hours. #### 2` | FAIL | `template_leakage / memorization` | 159.0 | 0.67 | 0.00 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `<<3*2=4>>10 hours.` | FAIL | `template_leakage / memorization` | 116.9 | 0.86 | 0.00 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `<<4*2=4>>6 hours. #### 2` | FAIL | `template_leakage / memorization` | 106.2 | 0.67 | 0.00 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 app... | `A should be a day, but not be a famous of the show` | FAIL | `arithmetic_error / ungrounded` | 189.4 | 0.81 | 0.00 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are... | `Lake is a famous of the city of the city of the ci` | FAIL | `instruction_noncompliance` | 183.3 | 0.56 | 0.24 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number... | `73,777.` | FAIL | `instruction_noncompliance` | 122.5 | 0.62 | 0.00 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean ex... | ` * [https://web.archive.org/web/2012050505/h` | FAIL | `repetition / loop_collapse` | 185.4 | 0.62 | 0.03 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data st... | `` | FAIL | `empty_output / early_eos` | 22.7 | 1.00 | 0.00 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-l... | `&quot; is a non-group of the word of the word of t` | FAIL | `instruction_noncompliance` | 199.7 | 0.53 | 0.38 |
| `CODE-01` | safe_coding_structure | Write a Python function n... | `         if not isinstance("__main__":         if ` | FAIL | `syntax_error / non_code` | 182.0 | 0.47 | 0.38 |
| `CODE-02` | safe_coding_structure | Write a Python function n... | `         if not isinstance("__main__":         if ` | FAIL | `syntax_error / non_code` | 180.6 | 0.56 | 0.10 |
| `CODE-03` | safe_coding_structure | In C, declare a variable ... | ` the non-set is the reference of the word of the w` | FAIL | `syntax_error / non_code` | 184.9 | 0.44 | 0.45 |
| `CODE-04` | safe_coding_structure | Write the SQL query to se... | `      # Check is the non-each other than the non-s` | FAIL | `syntax_error / non_code` | 170.7 | 0.75 | 0.03 |
| `CODE-05` | safe_coding_structure | In Python, how do you imp... | `` | FAIL | `empty_output / early_eos` | 25.3 | 1.00 | 0.00 |
| `CONT-01` | open_continuation | In modern operating syste... | `] is a list of the word of the word of the word of` | N/A | `human_judgment_required` | 161.8 | 0.28 | 0.69 |
| `CONT-02` | open_continuation | The primary advantage of ... | `lands in the word of the word of the word of the w` | N/A | `human_judgment_required` | 147.3 | 0.22 | 0.76 |
| `CONT-03` | open_continuation | When designing reliable d... | `` | N/A | `empty_output / early_eos` | 22.9 | 1.00 | 0.00 |
| `CONT-04` | open_continuation | Deep learning models requ... | `.` | N/A | `human_judgment_required` | 45.4 | 1.00 | 0.00 |
| `CONT-05` | open_continuation | An indigenous sovereign a... | `` | N/A | `empty_output / early_eos` | 23.9 | 1.00 | 0.00 |