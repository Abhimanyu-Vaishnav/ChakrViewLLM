# Benchmark Report: STAGE_C_9K
- **Suite Version**: `2.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `36711d25831740840dc9d62d60087a898aa4a19746322ef3cb0e04974a2770e3`
- **Checkpoint Path**: `checkpoints\experiments\neural_learning_v02\run_10k\checkpoint_0009000.pt`
- **Checkpoint SHA-256**: `3e03ae0a134da40439038d965278f4da2996b1ac2e8f959d6059859600b0bb23`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 30
- **Total Generated Tokens**: 906
- **Mean Latency**: 189.95 ms (Median: 200.55 ms)
- **Mean TTFT**: 24.99 ms
- **Mean Throughput**: 159.37 tok/s
- **Mean Unique-Token Ratio**: 0.428
- **Mean 4-Gram Repetition Ratio**: 0.356
- **Termination**: 3 EOS / 27 Token Limit
- **Deterministic Task Accuracy**: 0/20 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 50 chars) | Correct? | Failure Category | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley ... | `  The Great of the Great of the Great of the Great` | FAIL | `repetition / loop_collapse` | 202.3 | 0.22 | 0.72 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was... | `  The Great of the Great of the Great of the Great` | FAIL | `repetition / loop_collapse` | 220.0 | 0.22 | 0.72 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata sat... | `  The first of the first of the first of the first` | FAIL | `repetition / loop_collapse` | 152.7 | 0.22 | 0.72 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis oc... | `  The first of the group of the group of the group` | FAIL | `repetition / loop_collapse` | 144.5 | 0.34 | 0.59 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line si... | `  The famous is the place of the group of the grou` | FAIL | `repetition / loop_collapse` | 191.1 | 0.59 | 0.21 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides v... | `  The first of the group of the government of the ` | N/A | `unsupported_claim / non_grounded` | 130.1 | 0.44 | 0.38 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled langu... | `  References` | N/A | `unsupported_claim / non_grounded` | 83.5 | 0.86 | 0.00 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous c... | `  The first of the first of the first of the first` | N/A | `repetition / loop_collapse` | 161.5 | 0.22 | 0.72 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through ca... | `  The first of the place of the place of the group` | N/A | `unsupported_claim / non_grounded` | 128.6 | 0.53 | 0.17 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing ch... | `  The first of the group of the group of the group` | N/A | `repetition / loop_collapse` | 159.0 | 0.34 | 0.59 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `&quot;&quot; \|  \|  \|  \|  \|  \|  \|  \| ` | FAIL | `repetition / loop_collapse` | 143.8 | 0.22 | 0.62 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `&quot;&quot; \|  \|  \|  \|  \|  \|  \|  \| ` | FAIL | `repetition / loop_collapse` | 155.2 | 0.22 | 0.62 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `&quot;row&quot; \|  * &lt;ref&gt;&lt;/ref` | FAIL | `arithmetic_error / ungrounded` | 198.9 | 0.50 | 0.07 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `&quot;&quot; \|  \|  \|  \|  \|  \|  \|  \| ` | FAIL | `repetition / loop_collapse` | 201.0 | 0.22 | 0.62 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 app... | `  * [http://www.goood.org/good/good` | FAIL | `repetition / loop_collapse` | 145.5 | 0.47 | 0.07 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are... | `  * [http://www.natook.com/wi/wi/wi` | FAIL | `repetition / loop_collapse` | 150.8 | 0.53 | 0.10 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number... | `&lt;ref&gt;&lt;/ref&gt;  References` | FAIL | `instruction_noncompliance` | 164.0 | 0.50 | 0.16 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean ex... | `  * [http://www.goood.org/good/good` | FAIL | `repetition / loop_collapse` | 128.0 | 0.47 | 0.07 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data st... | `  References` | FAIL | `instruction_noncompliance` | 85.4 | 0.86 | 0.00 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-l... | ` * [http://www.natook.com/wi/wi/ww/` | FAIL | `repetition / loop_collapse` | 175.1 | 0.53 | 0.07 |
| `CODE-01` | safe_coding_structure | Write a Python function n... | `  * [http://www.goood.org/good/good` | FAIL | `repetition / loop_collapse` | 162.0 | 0.47 | 0.07 |
| `CODE-02` | safe_coding_structure | Write a Python function n... | `  * [http://www.natook.com/wi/wi/wi` | FAIL | `repetition / loop_collapse` | 147.5 | 0.53 | 0.10 |
| `CODE-03` | safe_coding_structure | In C, declare a variable ... | ` the group of the group of the group of the group ` | FAIL | `repetition / loop_collapse` | 141.3 | 0.19 | 0.76 |
| `CODE-04` | safe_coding_structure | Write the SQL query to se... | ` * [http://www.natook.com/wi/wi/inda` | FAIL | `repetition / loop_collapse` | 155.8 | 0.62 | 0.03 |
| `CODE-05` | safe_coding_structure | In Python, how do you imp... | `, the group of the group of the group of the group` | FAIL | `repetition / loop_collapse` | 191.3 | 0.22 | 0.76 |
| `CONT-01` | open_continuation | In modern operating syste... | `titure. The first of the group of the government o` | N/A | `human_judgment_required` | 194.4 | 0.53 | 0.17 |
| `CONT-02` | open_continuation | The primary advantage of ... | `. The first of the first of the first of the first` | N/A | `repetition / loop_collapse` | 171.5 | 0.25 | 0.72 |
| `CONT-03` | open_continuation | When designing reliable d... | `h and the first of the century of the century. The` | N/A | `human_judgment_required` | 190.3 | 0.47 | 0.31 |
| `CONT-04` | open_continuation | Deep learning models requ... | `. The famous is the group of the group of the grou` | N/A | `repetition / loop_collapse` | 159.6 | 0.41 | 0.52 |
| `CONT-05` | open_continuation | An indigenous sovereign a... | ` * [http://www.non-gooday.com/inda/ind` | N/A | `repetition / loop_collapse` | 146.8 | 0.66 | 0.00 |