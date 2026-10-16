# Dataset Provenance: Batches 1541–1580

## 300-Task Frozen Multi-Turn, Chaining & Sandbox Evaluation Suite

### 1. Dataset Overview

The Batches 1541–1580 evaluation suite tests multi-turn tool chaining, state isolation, sandboxed code execution, structured proposition extraction, and router recovery:

- **Path:** `experiments/batch1541_1580/data/evaluation_suite_300.json`
- **Manifest:** `experiments/batch1541_1580/data/EVALUATION_SUITE_MANIFEST.json`
- **Total Tasks:** 300
- **Cryptographic Hash (SHA-256):** `078aa3e59f474ccf7ac7aede5884e4fdac9847659998d86bf1351f9425ac7b26`

---

### 2. Track Composition & Provenance

| Track | Task Count | Scope & Coverage | Ground Truth / Provenance Source |
|---|---|---|---|
| **Track 1: Multi-Turn Reference** | 80 tasks | Valid anaphora resolution (40), missing/reset antecedents (20), cross-tenant isolation (20). | Synthetic multi-turn dialogues testing `"that"`, `"it"`, `"the result"`, `"the answer"`. |
| **Track 2: Arithmetic Chaining** | 60 tasks | Multi-step 3-operation chains (30), chained division by zero traps (30). | Exact mathematical ground truth across signed, decimal, and fractional operations. |
| **Track 3: Sandbox Execution** | 60 tasks | Safe data scripts (20), forbidden import attacks (20), timeout enforcement (10), output buffer clamping (10). | Isolated Python test snippets; adversarial network/process injection attempts. |
| **Track 4: Proposition Verification** | 60 tasks | Multi-entity pricing (15), adjacent parameter lines (15), unindexed entities (15), conflicting specs (15). | Grounded in ChakrView technical documentation corpus (`billing_subscription_tiers.md`, `storage_bm25_index.md`, `compute_arch_core.md`, etc.). |
| **Track 5: Routing & Recovery** | 40 tasks | Calculator (10), Structured Verification (10), Direct Greetings (10), Honest Abstentions (10). | Balanced operational intents with natural language formulations and failure recovery. |
