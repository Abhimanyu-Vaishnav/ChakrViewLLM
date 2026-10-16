# BASELINE RESULTS: CHAKRMICRO CANONICAL & ADAPTED MODELS
## Batches 1621–1660 Neural Learning & Capability Gate

### 1. Executive Summary

In Batches 1621–1660, we conducted an unmediated, strictly neural evaluation of the canonical ChakrMicro model (10,000 steps) and the previous domain-adapted experimental model (10,080 steps). No deterministic tools, no SafeCalculator, no StructuredVerifier, and no BM25 retrieval augmentations were permitted. The evaluation was run against a frozen, 150-task multi-track benchmark designed to probe 5 distinct capability surfaces under rigorous leakage control.

### 2. Evaluated Checkpoint Reference Table

| Metric / Attribute | Canonical Reference | Experimental Adapted |
| :--- | :--- | :--- |
| **Checkpoint Path** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt` |
| **Exact SHA-256** | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139` |
| **Total Parameters** | 1,586,816 | 1,586,816 |
| **Architecture** | 4-layer Transformer Decoder (`d_model=256, n_head=4, d_ff=1024`) | 4-layer Transformer Decoder (`d_model=256, n_head=4, d_ff=1024`) |
| **Pre-experiment PPL (Val)** | 175.74 | 143.74 |
| **Cross-Entropy (Val)** | 5.1690 | 4.9680 |
| **Training Loss (Base)** | 5.3148 (PPL: 203.33) | 5.1737 (PPL: 176.57) |

### 3. Track-by-Track Capability Matrix (150 Held-Out Tasks)

| Evaluation Track | Canonical Checkpoint (Step 10,000) | Adapted Checkpoint (Step 10,080) | Baseline Random / Deterministic |
| :--- | :--- | :--- | :--- |
| **Track 1: Language Continuation** (30 tasks) | **100.0%** (30/30) | **100.0%** (30/30) | Random: ~0.1% |
| **Track 2: Arithmetic Continuation** (30 tasks) | **0.0%** (0/30) | **0.0%** (0/30) | SafeCalculator: 100.0% |
| **Track 3: Code Completion** (30 tasks) | **0.0%** (0/30) | **0.0%** (0/30) | Python Lexer: 100.0% |
| **Track 4: Compositional Generalization** (30 tasks) | **0.0%** (0/30) | **0.0%** (0/30) | Symbolic Graph: 100.0% |
| **Track 5: Factual Recall & Rejection** (30 tasks) | **50.0%** (15/30) | **50.0%** (15/30) | BM25 Extractor: 100.0% |
| **Overall Exact Match / Accuracy** | **30.0%** (45/150) | **30.0%** (45/150) | Mixed |

### 4. Qualitative Behavioral Analysis

#### 4.1 Track 1: Language Continuation
Both models generated coherent English sentences following distributed systems prompts, repeatedly producing phrases such as:
`"in the world, which is the world of the world."` or `"in the first of the world, which is the world of..."`
Because the acceptance criterion for language continuation checks non-degeneration, vocabulary validity, and absence of NaN/crashes, all 30 passed. However, repetitive looping is prominent.

#### 4.2 Track 2: Arithmetic
When prompted with `"The double of 25 is"`, the canonical model completed:
`"lands of the city of the city of the city of the city of the"`
The adapted model produced similar lexical drift. Neither baseline has any intrinsic arithmetic understanding without external tools.

#### 4.3 Track 3: Code Completion
When prompted with `"def get_worker_status(node_id):\n    # Return worker dictionary"`, both models generated repetitive English prose instead of valid Python statements. Exact match against expected target tokens was 0.0%.

#### 4.4 Track 4: Multi-Hop Composition
When prompted with `"Through bridge bronze, cluster south relays to archive"`, the canonical baseline outputted generic language loops, completely failing to perform the 2-hop deduction.

#### 4.5 Track 5: Factual Recall and Hallucination Control
- **Known Facts (15 prompts):** Models failed to retrieve the newly defined service port mappings, answering with generic words.
- **Negative Controls / Unknown Facts (15 prompts):** Models did not hallucinate port numbers for ungrounded services like `trino`, `superset`, or `airflow` (scored 15/15 passed on the negative control criteria).

### 5. Conclusion
Without external deterministic augmentation, both canonical and adapted ChakrMicro checkpoints operate as basic statistical language smoothers. They possess zero zero-shot arithmetic reasoning, zero executable code synthesis, and zero compositional graph inference.
