# Benchmark Report: STAGE_C_9000_STEP
- **Suite Version**: `2.0.0`
- **Model**: ChakrMicro (3,443,136 parameters)
- **Weight Hash**: `36711d25831740840dc9d62d60087a898aa4a19746322ef3cb0e04974a2770e3`
- **Checkpoint Path**: `checkpoints\experiments\neural_learning_v02\run_10k\checkpoint_0009000.pt`
- **Checkpoint SHA-256**: `3e03ae0a134da40439038d965278f4da2996b1ac2e8f959d6059859600b0bb23`
- **Device / Threads**: CPU (2 threads)
- **Prompts Evaluated**: 30
- **Total Generated Tokens**: 906
- **Mean Latency**: 131.96 ms (Median: 135.48 ms)
- **Mean TTFT**: 20.42 ms
- **Mean Throughput**: 230.94 tok/s
- **Mean Unique-Token Ratio**: 0.428
- **Mean 4-Gram Repetition Ratio**: 0.356
- **Termination**: 3 EOS / 27 Token Limit
- **Deterministic Task Accuracy**: 0/20 (0.0%)

## Prompt Generation Samples

| ID | Domain | Prompt | Generated Output (First 60 chars) | Correct? | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley site ... | `  The Great of the Great of the Great of the Great of the Gr` | FAIL | 179.3 | 0.22 | 0.72 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was laun... | `  The Great of the Great of the Great of the Great of the Gr` | FAIL | 235.9 | 0.22 | 0.72 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata satellit... | `  The first of the first of the first of the first of the fi` | FAIL | 261.5 | 0.22 | 0.72 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis occurs ... | `  The first of the group of the group of the group of the gr` | FAIL | 196.9 | 0.34 | 0.59 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line size of... | `  The famous is the place of the group of the group of the g` | FAIL | 214.3 | 0.59 | 0.21 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides virtua... | `  The first of the group of the government of the government` | N/A | 188.0 | 0.44 | 0.38 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled language c... | `  References` | N/A | 128.9 | 0.86 | 0.00 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous commun... | `  The first of the first of the first of the first of the fi` | N/A | 250.6 | 0.22 | 0.72 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through caches ... | `  The first of the place of the place of the group of the go` | N/A | 269.0 | 0.53 | 0.17 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing checks ... | `  The first of the group of the group of the group of the gr` | N/A | 281.2 | 0.34 | 0.59 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `&quot;&quot; \|  \|  \|  \|  \|  \|  \|  \| ` | FAIL | 294.1 | 0.22 | 0.62 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `&quot;&quot; \|  \|  \|  \|  \|  \|  \|  \| ` | FAIL | 263.6 | 0.22 | 0.62 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `&quot;row&quot; \|  * &lt;ref&gt;&lt;/ref` | FAIL | 233.6 | 0.50 | 0.07 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `&quot;&quot; \|  \|  \|  \|  \|  \|  \|  \| ` | FAIL | 265.0 | 0.22 | 0.62 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 apples. ... | `  * [http://www.goood.org/good/good` | FAIL | 306.0 | 0.47 | 0.07 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are mamm... | `  * [http://www.natook.com/wi/wi/wi` | FAIL | 269.1 | 0.53 | 0.10 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number sequ... | `&lt;ref&gt;&lt;/ref&gt;  References` | FAIL | 265.1 | 0.50 | 0.16 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean express... | `  * [http://www.goood.org/good/good` | FAIL | 268.1 | 0.47 | 0.07 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data structu... | `  References` | FAIL | 149.1 | 0.86 | 0.00 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-letter... | ` * [http://www.natook.com/wi/wi/ww/` | FAIL | 280.4 | 0.53 | 0.07 |
| `CODE-01` | safe_coding_structure | Write a Python function named ... | `  * [http://www.goood.org/good/good` | FAIL | 265.1 | 0.47 | 0.07 |
| `CODE-02` | safe_coding_structure | Write a Python function named ... | `  * [http://www.natook.com/wi/wi/wi` | FAIL | 162.7 | 0.53 | 0.10 |
| `CODE-03` | safe_coding_structure | In C, declare a variable named... | ` the group of the group of the group of the group of the gro` | FAIL | 205.6 | 0.19 | 0.76 |
| `CODE-04` | safe_coding_structure | Write the SQL query to select ... | ` * [http://www.natook.com/wi/wi/inda` | FAIL | 236.2 | 0.62 | 0.03 |
| `CODE-05` | safe_coding_structure | In Python, how do you import t... | `, the group of the group of the group of the group of the gr` | FAIL | 205.6 | 0.22 | 0.76 |
| `CONT-01` | open_continuation | In modern operating systems, v... | `titure. The first of the group of the government of the gove` | N/A | 213.2 | 0.53 | 0.17 |
| `CONT-02` | open_continuation | The primary advantage of pipel... | `. The first of the first of the first of the first of the fi` | N/A | 218.2 | 0.25 | 0.72 |
| `CONT-03` | open_continuation | When designing reliable distri... | `h and the first of the century of the century. The century o` | N/A | 214.6 | 0.47 | 0.31 |
| `CONT-04` | open_continuation | Deep learning models require b... | `. The famous is the group of the group of the group of the g` | N/A | 226.2 | 0.41 | 0.52 |
| `CONT-05` | open_continuation | An indigenous sovereign artifi... | ` * [http://www.non-gooday.com/inda/ind` | N/A | 181.3 | 0.66 | 0.00 |