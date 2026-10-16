# PER-TASK FAILURE TAXONOMY
## Batches 1621–1660 Neural Diagnostics

### 1. Classification of Model Failure Modes

The 150-task evaluation exposed several fundamental failure categories inherent to the 1.58M-parameter ChakrMicro neural core. Below is the systematic failure taxonomy with representative examples and causal analyses.

```
Total Held-Out Tasks: 150
  ├── Passed (Non-Degenerate Text / Negative Controls): 45 (30.0%)
  └── Failed (Semantic / Syntactic / Generalization Failure): 105 (70.0%)
        ├── FM-1: Failure of Extrapolative Arithmetic (30 tasks)
        ├── FM-2: Syntactic Code Synthesis Failure (30 tasks)
        ├── FM-3: Compositional Multi-Hop Reasoning Failure (30 tasks)
        └── FM-4: Factual Association Interference (15 tasks)
```

---

### 2. Failure Mode Details and Concrete Raw Outputs

#### FM-1: Extrapolative Arithmetic Failure (30 tasks)
- **Description:** Failure to compute $2 \times N$ for numbers $N \in [25, 54]$ after training on $N \in [1, 20]$.
- **Task ID:** `arith_031`
- **Prompt:** `"The double of 25 is"`
- **Expected Completion:** `" 50."`
- **Candidate Output:** `"lands of 5 is 36. The double of 28 is 12."`
- **Failure Root Cause:** The neural core treats numbers as discrete BPE token strings rather than mathematical magnitudes. It has no inductive bias or internal computational mechanism for addition or multiplication. Without a calculator tool, autoregressive language modeling cannot perform arithmetic.

#### FM-2: Syntactic Code Synthesis Failure (30 tasks)
- **Description:** Inability to generate syntactically valid and semantically correct Python statements given function signatures and docstrings.
- **Task ID:** `code_061`
- **Prompt:** `"def get_worker_status(node_id):\n    # Return worker dictionary"`
- **Expected Completion:** `"\n    return {'status': 'active', 'id': node_id}"`
- **Candidate Output:** `"packets.\n    >>> linked_list.insert(')..."`
- **Failure Root Cause:** The model lacks sufficient capacity (only 4 layers, 256 embedding dimension) to maintain long-range syntax stacks (indentation, matching brackets, variable scoping). It outputs locally plausible token fragments without executable structure.

#### FM-3: Compositional Multi-Hop Reasoning Failure (30 tasks)
- **Description:** Inability to compose two independently learned transitive relations ($A \rightarrow B$ and $B \rightarrow C$) into $A \rightarrow C$.
- **Task ID:** `comp_116`
- **Prompt:** `"Through bridge bronze, cluster south relays to archive"`
- **Expected Completion:** `"limit of 18 packets"`
- **Candidate Output:** `"mber websites. It is 36 packets. I"`
- **Failure Root Cause:** Multi-hop deduction requires latent state routing across attention heads that 4 attention heads cannot coordinate without step-by-step chain-of-thought prompting or symbolic graph traversal.

#### FM-4: Factual Association Interference (15 tasks)
- **Description:** Inability to recall exact discrete attributes (port numbers) for newly taught entities.
- **Task ID:** `fact_121`
- **Prompt:** `"Service auth operates on TCP port"`
- **Expected Completion:** `" 7001."`
- **Candidate Output:** `"ing to archive limit of 27 packets. The relayer"`
- **Failure Root Cause:** The dense parameter space suffers from catastrophic superposition and interference when trained on a small number of updates. High-frequency corpus terms overshadow specific entity bindings.

---

### 3. Engineering Recommendations by Failure Category

| Failure Mode | Ineffective Solution | Effective Engineering Solution |
| :--- | :--- | :--- |
| **FM-1 (Arithmetic)** | More training epochs or more math text | **Deterministic Tool Routing** (`SafeCalculator`) |
| **FM-2 (Code Generation)** | Prompt engineering | **Sandboxed Python Runtime** + Syntax Checkers |
| **FM-3 (Multi-Hop)** | Larger unigram datasets | **Knowledge Graph / Structured Retrieval** |
| **FM-4 (Factual Precision)**| Weight-memorization training | **BM25 / Bi-Encoder Extractive Retrieval** |
