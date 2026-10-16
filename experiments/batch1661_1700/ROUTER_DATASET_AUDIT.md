# ROUTER DATASET AUDIT & LEAKAGE AUDIT
## Batches 1661–1700 Data Provenance, Class Distribution & Partitioning Verification

### 1. Dataset Provenance & Generation Specifications

The neural router benchmark dataset was constructed specifically to probe whether ChakrMicro can learn to classify natural language requests into six canonical tool intents and extract structured arguments, without relying on regex shortcuts or leaking training templates into test partitions.

- **Suite Identifier:** `batch1661_1700_neural_router_benchmark_v1`
- **Frozen Manifest:** `experiments/batch1661_1700/data/DATASET_MANIFEST.json`
- **Class Balance:** Perfectly uniform 6-way intent distribution across all partitions.

| Partition | Total Samples | Purpose | Verified SHA-256 Digest |
| :--- | :---: | :--- | :--- |
| **`train.jsonl`** | 150 | Supervised adaptation of neural router | `c34cb16f493b0811abca0df2d1b31d24dc8c06c31683aeb0eb8672b88b15b655` |
| **`val.jsonl`** | 30 | Hyperparameter & checkpoint tracking | `54202cb6d8fd9145bb25fb3ac201569963fc3f81dd87e9d9fcf4cc7556c791c3` |
| **`test_held_out.json`** | 120 | Frozen, unmediated generalization evaluation | `6d56d955d953c2bcd343e88a1954f4eaaae63e4363313a8527c2289bd73c64b0` |

---

### 2. Class Distribution & Intent Coverage

Each partition features an exact equal balance across the 6 supported intents:

```
Intent Class Distribution (Held-Out Test Set: 120 Samples)
  ├── CALCULATE: 20 samples (16.67%)
  ├── VERIFY_CLAIM: 20 samples (16.67%)
  ├── KNOWLEDGE_QUERY: 20 samples (16.67%)
  ├── CODE_TASK: 20 samples (16.67%)
  ├── CLARIFY: 20 samples (16.67%)
  └── OUT_OF_SCOPE: 20 samples (16.67%)
```

---

### 3. Leakage Control and Split Strategy

To prevent superficial template memorization from passing as semantic generalization:
1. **Syntactic Disjointness:** The training partition uses explicit prefix templates (e.g., `"Evaluate the math expression {expr}"`, `"Check the claim: {claim}"`, `"Search the docs for {topic}"`). The held-out test partition strictly forbids these templates, using conversational phrasing, idioms, and indirect constructions (e.g., `"I need to find the sum of 512 and 1024"`, `"Could you work out what 96 divided by 8 is?"`, `"I heard that database connections terminate after 60s. Is this true?"`).
2. **Negative Controls & Keyword Traps:** The test partition includes inputs that contain tool-associated keywords without requesting tool action (e.g., `"We should calculate the risk, but do not invoke any tools"` $\rightarrow$ `OUT_OF_SCOPE`). A naive keyword matcher triggers false executions; a semantic router must recognize absence of intent.
3. **Adversarial Injections:** The test partition includes SQL injection strings (`"; DROP TABLE"`), shell metacharacters (`"; cat /etc/passwd"`), and malicious Python imports (`__import__('os').system(...)`) to test whether candidate routers reject hostile payloads.
4. **Underspecified / Missing Arguments:** 20 held-out tasks omit essential parameters (e.g., `"Calculate it for me please"`), testing whether the router accurately identifies `CLARIFY` and flags the missing field.

### 4. Conclusion
The dataset partitions are cryptographically sealed, balanced, and free of cross-split prompt leakage.
