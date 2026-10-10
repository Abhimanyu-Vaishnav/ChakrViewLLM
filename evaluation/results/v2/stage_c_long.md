# Benchmark Report: STAGE_C_9000_STEP
- **Suite Version**: `2.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `36711d25831740840dc9d62d60087a898aa4a19746322ef3cb0e04974a2770e3`
- **Checkpoint Path**: `checkpoints\experiments\neural_learning_v02\run_10k\checkpoint_0009000.pt`
- **Checkpoint SHA-256**: `3e03ae0a134da40439038d965278f4da2996b1ac2e8f959d6059859600b0bb23`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 30
- **Total Generated Tokens**: 2634
- **Mean Latency**: 439.73 ms (Median: 488.29 ms)
- **Mean TTFT**: 23.98 ms
- **Mean Throughput**: 196.80 tok/s
- **Mean Unique-Token Ratio**: 0.229
- **Mean 4-Gram Repetition Ratio**: 0.628
- **Termination**: 3 EOS / 27 Token Limit
- **Deterministic Task Accuracy**: 0/20 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 50 chars) | Correct? | Failure Category | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley ... | `  The Great of the Great of the Great of the Great` | FAIL | `repetition / loop_collapse` | 185.0 | 0.18 | 0.75 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was... | `  The Great of the Great of the Great of the Great` | FAIL | `repetition / loop_collapse` | 218.0 | 0.07 | 0.88 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata sat... | `  The first of the first of the first of the first` | FAIL | `repetition / loop_collapse` | 193.3 | 0.09 | 0.83 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis oc... | `  The first of the group of the group of the group` | FAIL | `repetition / loop_collapse` | 193.5 | 0.22 | 0.60 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line si... | `  The famous is the place of the group of the grou` | FAIL | `repetition / loop_collapse` | 250.7 | 0.23 | 0.68 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides v... | `  The first of the group of the government of the ` | N/A | `repetition / loop_collapse` | 175.2 | 0.15 | 0.76 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled langu... | `  References` | N/A | `unsupported_claim / non_grounded` | 85.0 | 0.86 | 0.00 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous c... | `  The first of the first of the first of the first` | N/A | `repetition / loop_collapse` | 178.6 | 0.08 | 0.87 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through ca... | `  The first of the place of the place of the group` | N/A | `repetition / loop_collapse` | 186.0 | 0.18 | 0.70 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing ch... | `  The first of the group of the group of the group` | N/A | `repetition / loop_collapse` | 166.8 | 0.11 | 0.80 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `&quot;&quot; \|  \|  \|  \|  \|  \|  \|  \|  \|  \` | FAIL | `repetition / loop_collapse` | 200.2 | 0.07 | 0.88 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `&quot;&quot; \|  \|  \|  \|  \|  \|  \|  \|  \|  \` | FAIL | `repetition / loop_collapse` | 179.8 | 0.07 | 0.88 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `&quot;row&quot; \|  * &lt;ref&gt;&lt;/ref&gt; * &l` | FAIL | `repetition / loop_collapse` | 187.0 | 0.30 | 0.35 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `&quot;&quot; \|  \|  \|  \|  \|  \|  \|  \|  \|  \` | FAIL | `repetition / loop_collapse` | 182.3 | 0.07 | 0.88 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 app... | `  * [http://www.goood.org/good/good/good/good/good` | FAIL | `repetition / loop_collapse` | 169.9 | 0.16 | 0.70 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are... | `  * [http://www.natook.com/wi/wi/wi/inda/infa-grou` | FAIL | `repetition / loop_collapse` | 185.1 | 0.34 | 0.31 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number... | `&lt;ref&gt;&lt;/ref&gt;  References` | FAIL | `instruction_noncompliance` | 178.2 | 0.50 | 0.16 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean ex... | `  * [http://www.goood.org/good/good/good/good/good` | FAIL | `repetition / loop_collapse` | 196.2 | 0.16 | 0.71 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data st... | `  References` | FAIL | `instruction_noncompliance` | 101.9 | 0.86 | 0.00 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-l... | ` * [http://www.natook.com/wi/wi/ww/www/wide/wide/w` | FAIL | `repetition / loop_collapse` | 201.8 | 0.19 | 0.58 |
| `CODE-01` | safe_coding_structure | Write a Python function n... | `  * [http://www.goood.org/good/good/good/good/good` | FAIL | `repetition / loop_collapse` | 196.6 | 0.16 | 0.68 |
| `CODE-02` | safe_coding_structure | Write a Python function n... | `  * [http://www.natook.com/wi/wi/wi/inda/infa/infa` | FAIL | `repetition / loop_collapse` | 221.7 | 0.29 | 0.38 |
| `CODE-03` | safe_coding_structure | In C, declare a variable ... | ` the group of the group of the group of the group ` | FAIL | `repetition / loop_collapse` | 174.3 | 0.16 | 0.69 |
| `CODE-04` | safe_coding_structure | Write the SQL query to se... | ` * [http://www.natook.com/wi/wi/inda/infa-group/ht` | FAIL | `repetition / loop_collapse` | 220.7 | 0.35 | 0.37 |
| `CODE-05` | safe_coding_structure | In Python, how do you imp... | `, the group of the group of the group of the group` | FAIL | `repetition / loop_collapse` | 221.1 | 0.15 | 0.80 |
| `CONT-01` | open_continuation | In modern operating syste... | `titure. The first of the group of the government o` | N/A | `repetition / loop_collapse` | 224.3 | 0.18 | 0.73 |
| `CONT-02` | open_continuation | The primary advantage of ... | `. The first of the first of the first of the first` | N/A | `repetition / loop_collapse` | 230.8 | 0.14 | 0.83 |
| `CONT-03` | open_continuation | When designing reliable d... | `h and the first of the century of the century. The` | N/A | `repetition / loop_collapse` | 245.5 | 0.16 | 0.71 |
| `CONT-04` | open_continuation | Deep learning models requ... | `. The famous is the group of the group of the grou` | N/A | `repetition / loop_collapse` | 287.2 | 0.20 | 0.65 |
| `CONT-05` | open_continuation | An indigenous sovereign a... | ` * [http://www.non-gooday.com/inda/inda/inda/inda/` | N/A | `repetition / loop_collapse` | 267.2 | 0.22 | 0.69 |