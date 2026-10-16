# DATASET & LEAKAGE AUDIT: DIRECT CLASSIFICATION & SCALING
## Batches 1701–1740 Data Provenance, Split Integrity & Multi-Budget Partitions

### 1. Dataset Architecture & Cryptographic Manifest

The Batches 1701–1740 benchmark introduces an expanded, multi-budget evaluation dataset designed to rigorously test direct neural classification heads, representations, and data scaling laws without cross-partition contamination.

- **Suite Name:** `batch1701_1740_direct_classification_and_scaling_v1`
- **Sealed Manifest:** `experiments/batch1701_1740/data/DATASET_MANIFEST.json`
- **Classes:** Uniform 6-way balance across all partitions (`CALCULATE`, `VERIFY_CLAIM`, `KNOWLEDGE_QUERY`, `CODE_TASK`, `CLARIFY`, `OUT_OF_SCOPE`).

| Partition File | Purpose | Sample Count | Verified SHA-256 Digest |
| :--- | :--- | :---: | :--- |
| **`train_full.jsonl`** | 100% budget training partition | 240 (40 / class) | `8e529bdb07c95f3dd7bfec931770326670013b94a9216909f8ea40dbe61b6768` |
| **`train_50pct.jsonl`**| 50% budget training partition | 120 (20 / class) | `38f9b639ec36cbc13cabc021ebfeace02ce239585168a140ee769a8e7dfddccb` |
| **`train_25pct.jsonl`**| 25% budget training partition | 60 (10 / class) | `2b06f66fe6513b61db29f8b65e8adb6b4d6a3d0b923662af59328e3304dee63c` |
| **`val.jsonl`** | Hyperparameter & checkpoint tracking | 60 (10 / class) | `63aabf05256fc5b8886a9d02ff15073e9d7f81a6342633d5913ecf9361b5a91f` |
| **`test_held_out.json`** | Frozen held-out generalization test | 180 (30 / class) | `63bc6ca288683e74b048e8eda88b6d0033f403722c550b3e03410a46dbe57d6f` |

---

### 2. Leakage Prevention and Semantic Family Splits

1. **Strict Family Isolation:**
   - Training partitions strictly employ structural prefixes and standard command templates (`"Evaluate math expression: ..."`, `"Verify whether ..."`, `"Search documentation for ..."`).
   - The held-out test partition strictly forbids these phrasing patterns. It features conversational phrasings (`"Could you work out what 96 divided by 8 is?"`, `"I heard that database connections terminate after 60s. Is this true?"`), distinct arithmetic formulations (`"quotient of 144 over 12"`, `"twenty times fifteen"`), and indirect requests.
2. **Adversarial & Negative Control Inclusion:**
   - The held-out set includes 10 negative control queries mentioning tools without invoking action (e.g. `"The calculator is a useful deterministic tool"`, `"Python is my favorite programming language"` $\rightarrow$ `OUT_OF_SCOPE`).
   - The held-out set includes 5 malicious payloads (SQL injection `"; DROP TABLE"`, command injection `"; cat /etc/passwd"`, code execution `eval('__import__...')` $\rightarrow$ `OUT_OF_SCOPE`).
3. **Missing Argument / Clarification Partition:**
   - 30 distinct held-out queries test missing parameters (`"Calculate it for me please"`, `"Can you verify this?"`, `"Execute the Python program"` $\rightarrow$ `CLARIFY`), testing whether models differentiate complete requests from incomplete requests.

### 3. Conclusion
The multi-budget datasets are cryptographically sealed, balanced across all 6 intents, and guaranteed leak-free across template and semantic families.
