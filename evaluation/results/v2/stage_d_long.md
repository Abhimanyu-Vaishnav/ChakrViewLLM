# Benchmark Report: STAGE_D_500_STEP_PILOT
- **Suite Version**: `2.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `5f3712d80940fb9345d01b5434e8bb64936d01e4d3aaff096ab5268c282d506c`
- **Checkpoint Path**: `checkpoints\experiments\neural_learning_v02\stage_d_pilot_run500\checkpoints\checkpoint_0000500.pt`
- **Checkpoint SHA-256**: `8ab1ec36b5c5a09929b8510332f18fecc0183eb0e9332ab99375995a8cc1a449`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 30
- **Total Generated Tokens**: 2358
- **Mean Latency**: 340.11 ms (Median: 387.04 ms)
- **Mean TTFT**: 21.72 ms
- **Mean Throughput**: 231.00 tok/s
- **Mean Unique-Token Ratio**: 0.345
- **Mean 4-Gram Repetition Ratio**: 0.463
- **Termination**: 9 EOS / 21 Token Limit
- **Deterministic Task Accuracy**: 0/20 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 50 chars) | Correct? | Failure Category | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley ... | ` The answer is 200.` | FAIL | `template_leakage / memorization` | 159.7 | 1.00 | 0.00 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was... | ` The answer is 200.` | FAIL | `template_leakage / memorization` | 163.5 | 1.00 | 0.00 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata sat... | ` The answer is 200.` | FAIL | `template_leakage / memorization` | 183.5 | 1.00 | 0.00 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis oc... | ` So he has 200*2=<<2*2=200>>12 years old. So he ha` | FAIL | `template_leakage / memorization` | 318.5 | 0.38 | 0.40 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line si... | ` The number of the number of the number of the num` | FAIL | `repetition / loop_collapse` | 319.5 | 0.06 | 0.94 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides v... | `40=<<2*2=200>>60 So he has 200*2=<<200*2=200>>60 S` | N/A | `template_leakage / memorization` | 322.1 | 0.16 | 0.71 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled langu... | `So he has $200 = <<2*2=200>>120 hours The answer i` | N/A | `template_leakage / memorization` | 301.7 | 0.73 | 0.00 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous c... | `50=<<2*2=200>>12 years old. So he has 200*2=<<2*2=` | N/A | `template_leakage / memorization` | 200.4 | 0.47 | 0.28 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through ca... | `2=<<2*2=200>>120 So he has 200*2=<<2*2=200>>120 So` | N/A | `template_leakage / memorization` | 247.5 | 0.17 | 0.68 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing ch... | `2=<<2*2=200>>120 So he has 200*2=<<2*2=200>>60 So ` | N/A | `template_leakage / memorization` | 184.1 | 0.17 | 0.65 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `240>>40 There are 50*2=<<200*2=60>>40 There are 50` | FAIL | `template_leakage / memorization` | 248.0 | 0.17 | 0.56 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `<<240=240>>400 There are 500/2=<<200*2=60>>400 The` | FAIL | `template_leakage / memorization` | 235.2 | 0.24 | 0.45 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `<<2*2=200>>120 There are 50*2=<<200*2=60>>400 Ther` | FAIL | `template_leakage / memorization` | 266.8 | 0.18 | 0.48 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `<<200*2=200>>60 There are 50/2=<<200*2=60>>60 Ther` | FAIL | `template_leakage / memorization` | 247.8 | 0.17 | 0.62 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 app... | `2=<<2*2=2>>12 she has 24*2=<<2*2=2>>60 The answer ` | FAIL | `template_leakage / memorization` | 197.8 | 0.50 | 0.21 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are... | `200>=<<2*2=200>>120 cans So he has 200*2=<<2*2=200` | FAIL | `template_leakage / memorization` | 254.9 | 0.20 | 0.61 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number... | `40 There are 50*2=<<200*2=60>>60 So he has 60*2=<<` | FAIL | `template_leakage / memorization` | 194.5 | 0.23 | 0.49 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean ex... | `2=<<2*2=200>>120 So he has 200*2=<<200*2=200>>120 ` | FAIL | `template_leakage / memorization` | 236.8 | 0.27 | 0.42 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data st... | ` So he has $50*2=<<2*2=200>>120 - 200 = $<<2*2=200` | FAIL | `template_leakage / memorization` | 207.1 | 0.22 | 0.55 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-l... | `24=<<2*2=200>>120 So he has 200*2=<<200*2=<<200*2=` | FAIL | `template_leakage / memorization` | 286.7 | 0.38 | 0.39 |
| `CODE-01` | safe_coding_structure | Write a Python function n... | `2=<<2*2=200>>120 So he has 200*2=<<2*2=200>>120 So` | FAIL | `template_leakage / memorization` | 249.5 | 0.24 | 0.45 |
| `CODE-02` | safe_coding_structure | Write a Python function n... | `So he has $5 = <<2*2=200>>120 x 200 = <<200*2=200>` | FAIL | `template_leakage / memorization` | 225.1 | 0.25 | 0.49 |
| `CODE-03` | safe_coding_structure | In C, declare a variable ... | ` the total number of paks. There are 5 of the numb` | FAIL | `template_leakage / memorization` | 247.4 | 0.39 | 0.31 |
| `CODE-04` | safe_coding_structure | Write the SQL query to se... | `2=<<2*2=200>>120 So he has 200*2=<<2*2=200>>36 So ` | FAIL | `template_leakage / memorization` | 198.8 | 0.22 | 0.56 |
| `CODE-05` | safe_coding_structure | In Python, how do you imp... | `2=200>>12 canswer is 4.` | FAIL | `syntax_error / non_code` | 162.4 | 0.88 | 0.00 |
| `CONT-01` | open_continuation | In modern operating syste... | ` There are the number of the number of the number ` | N/A | `repetition / loop_collapse` | 172.1 | 0.09 | 0.90 |
| `CONT-02` | open_continuation | The primary advantage of ... | `our to pay the total number of the number of the n` | N/A | `repetition / loop_collapse` | 255.8 | 0.12 | 0.86 |
| `CONT-03` | open_continuation | When designing reliable d... | `h of the number of the number of the number of the` | N/A | `repetition / loop_collapse` | 229.3 | 0.05 | 0.95 |
| `CONT-04` | open_continuation | Deep learning models requ... | `. There are $<<2*2=40>>40 x 200 = <<40*2=60>>400 T` | N/A | `template_leakage / memorization` | 180.3 | 0.22 | 0.44 |
| `CONT-05` | open_continuation | An indigenous sovereign a... | ` So he has 60/2=<<2*2=200>>36 So he has 200*2=<<20` | N/A | `template_leakage / memorization` | 233.2 | 0.23 | 0.49 |