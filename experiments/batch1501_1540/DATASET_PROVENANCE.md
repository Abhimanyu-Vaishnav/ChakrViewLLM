# Dataset Provenance

## Batches 1501–1540: 300-Task Frozen Evaluation Suite

### 1. Dataset Overview

To objectively assess the newly implemented neuro-symbolic routing, safe arithmetic evaluation, structured claim verification, and end-to-end assistant capabilities, an independently authored frozen benchmark suite was generated under:
`experiments/batch1501_1540/data/evaluation_suite_300.json`

- **Total Tasks:** 300
- **Cryptographic Hash (SHA-256):** `8d0339a76e416c7c2559c0d2043737e757efc724c529c3294f8612384c750c2d`
- **Manifest File:** `experiments/batch1501_1540/data/EVALUATION_SUITE_MANIFEST.json`

---

### 2. Track Composition & Provenance

| Track | Task Count | Target Scope | Provenance & Source Documents |
|---|---|---|---|
| **Track 1: Arithmetic** | 100 tasks | Add/Sub (20), Mul/Div (20), Mod/Pow (20), Nested Parens (20), Security & Domain Limits (20). | Independently authored mathematical expressions and adversarial security injections (e.g. `eval`, `__import__`, overflow limits). |
| **Track 2: Structured Verification** | 100 tasks | Supported (25), Contradicted (25), Insufficient Evidence (25), Conflicting / Invalid (25). | Grounded in ChakrView technical documentation (`storage_bm25_index.md`, `billing_subscription_tiers.md`, `network_consensus_raft.md`, `storage_blob_store.md`, etc.). |
| **Track 3: Tool Routing** | 50 tasks | Calculator (10), Verification (10), Direct Chat (10), Knowledge Retrieval (10), Honest Abstention (10). | Balanced operational intents with natural language formatting, conversational phrasing, and out-of-domain knowledge queries. |
| **Track 4: End-to-End Assistant**| 50 tasks | Quantitative math (15), Factual QA (15), False-premise refutation (10), Out-of-domain abstention (10). | Real-world conversational queries combining Hinglish, vernacular phrasing, and multi-tier verification. |

---

### 3. Separation of Development and Test Sets
- None of the 300 tasks were used in prior training runs.
- The evaluation dataset was created, verified, and hashed before running benchmark evaluations.
