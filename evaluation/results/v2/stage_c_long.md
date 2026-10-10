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

| ID | Domain | Prompt | Generated Output (First 60 chars) | Correct? | TPS | Unique Ratio | 4-Gram Rep |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `PASS-FACT-01` | passage_factual_recall | Source: The Indus Valley site ... | `  The Great of the Great of the Great of the Great of the Gr` | FAIL | 185.0 | 0.18 | 0.75 |
| `PASS-FACT-02` | passage_factual_recall | Source: Project Tiger was laun... | `  The Great of the Great of the Great of the Great of the Gr` | FAIL | 218.0 | 0.07 | 0.88 |
| `PASS-FACT-03` | passage_factual_recall | Source: The Aryabhata satellit... | `  The first of the first of the first of the first of the fi` | FAIL | 193.3 | 0.09 | 0.83 |
| `PASS-FACT-04` | passage_factual_recall | Source: Photosynthesis occurs ... | `  The first of the group of the group of the group of the gr` | FAIL | 193.5 | 0.22 | 0.60 |
| `PASS-FACT-05` | passage_factual_recall | Source: The cache line size of... | `  The famous is the place of the group of the group of the g` | FAIL | 250.7 | 0.23 | 0.68 |
| `PASS-EXPL-01` | passage_explanation_summarization | Passage: Paging divides virtua... | `  The first of the group of the government of the government` | N/A | 175.2 | 0.15 | 0.76 |
| `PASS-EXPL-02` | passage_explanation_summarization | Passage: A compiled language c... | `  References` | N/A | 85.0 | 0.86 | 0.00 |
| `PASS-EXPL-03` | passage_explanation_summarization | Passage: In synchronous commun... | `  The first of the first of the first of the first of the fi` | N/A | 178.6 | 0.08 | 0.87 |
| `PASS-EXPL-04` | passage_explanation_summarization | Passage: Write-through caches ... | `  The first of the place of the place of the group of the go` | N/A | 186.0 | 0.18 | 0.70 |
| `PASS-EXPL-05` | passage_explanation_summarization | Passage: Static typing checks ... | `  The first of the group of the group of the group of the gr` | N/A | 166.8 | 0.11 | 0.80 |
| `ARITH-01` | arithmetic_logic | Calculate: 12 + 15 = ... | `&quot;&quot; \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \| ` | FAIL | 200.2 | 0.07 | 0.88 |
| `ARITH-02` | arithmetic_logic | Calculate: 50 - 18 = ... | `&quot;&quot; \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \| ` | FAIL | 179.8 | 0.07 | 0.88 |
| `ARITH-03` | arithmetic_logic | Calculate: 8 * 7 = ... | `&quot;row&quot; \|  * &lt;ref&gt;&lt;/ref&gt; * &lt;ref&gt;&` | FAIL | 187.0 | 0.30 | 0.35 |
| `ARITH-04` | arithmetic_logic | Calculate: 72 / 9 = ... | `&quot;&quot; \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \|  \| ` | FAIL | 182.3 | 0.07 | 0.88 |
| `ARITH-05` | arithmetic_logic | Problem: A box has 24 apples. ... | `  * [http://www.goood.org/good/good/good/good/good/good/good` | FAIL | 169.9 | 0.16 | 0.70 |
| `LOGIC-01` | algorithmic_reasoning | Question: If all cats are mamm... | `  * [http://www.natook.com/wi/wi/wi/inda/infa-group/html /ww` | FAIL | 185.1 | 0.34 | 0.31 |
| `LOGIC-02` | algorithmic_reasoning | Task: Complete the number sequ... | `&lt;ref&gt;&lt;/ref&gt;  References` | FAIL | 178.2 | 0.50 | 0.16 |
| `LOGIC-03` | algorithmic_reasoning | Question: In a boolean express... | `  * [http://www.goood.org/good/good/good/good/good/good/good` | FAIL | 196.2 | 0.16 | 0.71 |
| `LOGIC-04` | algorithmic_reasoning | Question: A stack data structu... | `  References` | FAIL | 101.9 | 0.86 | 0.00 |
| `LOGIC-05` | algorithmic_reasoning | Task: Reverse the three-letter... | ` * [http://www.natook.com/wi/wi/ww/www/wide/wide/wide/wide/w` | FAIL | 201.8 | 0.19 | 0.58 |
| `CODE-01` | safe_coding_structure | Write a Python function named ... | `  * [http://www.goood.org/good/good/good/good/good/good/good` | FAIL | 196.6 | 0.16 | 0.68 |
| `CODE-02` | safe_coding_structure | Write a Python function named ... | `  * [http://www.natook.com/wi/wi/wi/inda/infa/infa/infa/infa` | FAIL | 221.7 | 0.29 | 0.38 |
| `CODE-03` | safe_coding_structure | In C, declare a variable named... | ` the group of the group of the group of the group of the gro` | FAIL | 174.3 | 0.16 | 0.69 |
| `CODE-04` | safe_coding_structure | Write the SQL query to select ... | ` * [http://www.natook.com/wi/wi/inda/infa-group/html Austric` | FAIL | 220.7 | 0.35 | 0.37 |
| `CODE-05` | safe_coding_structure | In Python, how do you import t... | `, the group of the group of the group of the group of the gr` | FAIL | 221.1 | 0.15 | 0.80 |
| `CONT-01` | open_continuation | In modern operating systems, v... | `titure. The first of the group of the government of the gove` | N/A | 224.3 | 0.18 | 0.73 |
| `CONT-02` | open_continuation | The primary advantage of pipel... | `. The first of the first of the first of the first of the fi` | N/A | 230.8 | 0.14 | 0.83 |
| `CONT-03` | open_continuation | When designing reliable distri... | `h and the first of the century of the century. The century o` | N/A | 245.5 | 0.16 | 0.71 |
| `CONT-04` | open_continuation | Deep learning models require b... | `. The famous is the group of the group of the group of the g` | N/A | 287.2 | 0.20 | 0.65 |
| `CONT-05` | open_continuation | An indigenous sovereign artifi... | ` * [http://www.non-gooday.com/inda/inda/inda/inda/inda/inda/` | N/A | 267.2 | 0.22 | 0.69 |