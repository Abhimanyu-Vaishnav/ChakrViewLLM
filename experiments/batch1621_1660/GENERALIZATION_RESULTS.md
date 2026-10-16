# GENERALIZATION RESULTS: HELD-OUT TASK EVALUATION
## Batches 1621–1660 Neural Generalization Audit

### 1. The Core Scientific Question
Does an 86% reduction in validation perplexity (from 175.74 to 11.30) indicate that ChakrMicro has learned generalizable arithmetic, code synthesis, or compositional reasoning?

**Answer:** No. Despite the dramatic drop in perplexity, **the exact-match task performance on unseen compositions remained at 0.0%**. This provides critical, irrefutable empirical evidence separating statistical next-token prediction from true neural task generalization.

---

### 2. Multi-Candidate Generalization Comparison

| Capability Track | Canonical Baseline | Adapted Baseline | Candidate (Step 100) | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Track 1: Language Continuation** | 30/30 (100%) | 30/30 (100%) | 30/30 (100%) | Fluid token progression preserved; surface phrasing adapted to new corpus tokens (`packets`, `limits`). |
| **Track 2: Arithmetic ($2 \times N$ for $N \in [25, 54]$)** | 0/30 (0%) | 0/30 (0%) | 0/30 (0%) | **Zero arithmetic generalization**. Model memorized token fragments from $[1, 20]$ but fails on $[25, 54]$. |
| **Track 3: Short Code Completion** | 0/30 (0%) | 0/30 (0%) | 0/30 (0%) | Generates code-like punctuation (`>>> linked_list.insert(')`) without syntactic or semantic correctness. |
| **Track 4: Compositional Generalization** | 0/30 (0%) | 0/30 (0%) | 0/30 (0%) | Model fails to link Hop 1 ($X \rightarrow Y$) and Hop 2 ($Y \rightarrow Z$) into $X \rightarrow Z$. |
| **Track 5: Factual Recall & Controls** | 15/30 (50%) | 15/30 (50%) | 15/30 (50%) | Retained correct rejection on 15 negative controls; failed exact port recall on 15 trained facts. |
| **Overall Held-out Accuracy** | **30.0%** (45/150) | **30.0%** (45/150) | **30.0%** (45/150) | **No task-level accuracy gain**. |

---

### 3. Detailed Inspection by Capability Surface

#### 3.1 Arithmetic Generalization Gap
- **Training Distribution:** The model saw prompts like `"The double of 3 is 6."` up to `"The double of 20 is 40."`.
- **Held-Out Test Prompts:** `"The double of 25 is"`, `"The double of 30 is"`, etc.
- **Candidate Model Output:**
  - Prompt: `"The double of 25 is"` $\rightarrow$ Generated: `"lands of 5 is 36. The double of 28 is 12."`
  - Prompt: `"The double of 30 is"` $\rightarrow$ Generated: `"lands of 5 sland 3. The total of 5 is 12"`
- **Diagnostic Finding:** The model acquired the surface pattern words (`"double"`, `"is"`, numbers) but learned no underlying arithmetic function. It outputs superficial tokens without functional calculation.

#### 3.2 Compositional Generalization Gap
- **Training Distribution:** Hop 1 (`"bridge bronze connects cluster south to router beta"`) and Hop 2 (`"router beta routes directly to archive limit of 18 packets"`).
- **Held-Out Test Prompts:** `"Through bridge bronze, cluster south relays to archive"` (Expects `"limit of 18 packets"`).
- **Candidate Model Output:** `"mber websites. It is 36 packets. I"`
- **Diagnostic Finding:** Multi-hop compositional transfer did not occur. The model did not associate intermediate routing entity `router beta` with the destination endpoint.

#### 3.3 Code Completion Gap
- **Prompt:** `"class TokenRing:\n    def __init__(self, size):\n        self.size = size\n        # Initialize ring buffer"`
- **Candidate Output:** Repetitive token sequences containing method delimiters and parenthesis fragments, but failing Python lexing and exact token matching.

---

### 4. Conclusion
The experimental candidate acquired vocabulary and stylistic associations from the training data, sharply dropping validation loss. However, it completely failed to generalize across unseen arithmetic values and multi-hop compositional rules. Deterministic tools (SafeCalculator, Sandbox, StructuredVerifier) remain strictly mandatory for arithmetic, execution, and relational deduction.
