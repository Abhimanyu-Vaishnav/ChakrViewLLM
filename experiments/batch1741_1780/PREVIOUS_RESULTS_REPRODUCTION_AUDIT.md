# PREVIOUS RESULTS REPRODUCTION & AUDIT
## Batches 1741–1780 Rigorous Verification of Batches 1701–1740 Findings

### 1. Verification of Reference Checkpoints & Hashes

Before evaluating or training any models, all reference checkpoints and data artifacts from Batches 1701–1740 were recomputed and verified byte-for-byte:

| Artifact Description | Expected SHA-256 Digest | Recomputed SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| **Canonical Checkpoint** (`training_checkpoint_step_10000.pt`) | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **MATCH** |
| **Adapted Checkpoint** (`chakrmicro_domain_adapted_experimental.pt`) | `15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139` | `15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139` | **MATCH** |
| **Batch 1701 Variant A Checkpoint** (`variant_a_frozen_backbone.pt`) | `d23dce865bd6e69d23d0d39445be9fed748c086e9730b94df85f6ee4f46ef089` | `d23dce865bd6e69d23d0d39445be9fed748c086e9730b94df85f6ee4f46ef089` | **MATCH** |
| **Batch 1701 Variant B Checkpoint** (`variant_b_finetuned_100pct.pt`) | `2286e3a053670a6b3b24a1c99f96addbfaf109beb13f4a8b8790e1a56263ee42` | `2286e3a053670a6b3b24a1c99f96addbfaf109beb13f4a8b8790e1a56263ee42` | **MATCH** |
| **Batch 1701 Held-Out Test Set** (`test_held_out.json`) | `63bc6ca288683e74b048e8eda88b6d0033f403722c550b3e03410a46dbe57d6f` | `63bc6ca288683e74b048e8eda88b6d0033f403722c550b3e03410a46dbe57d6f` | **MATCH** |

---

### 2. Systematic Metric-by-Metric Recomputation Audit

We audited the raw prediction entries in `experiments/batch1701_1740/results/CLASSIFICATION_EVALUATION_RESULTS.json` across all 180 held-out tasks:

| Evaluated System & Metric | Reported in 1701–1740 | Raw Recomputed Value | Formula / Numerator / Denominator | Audit Verdict |
| :--- | :---: | :---: | :--- | :---: |
| **Variant B (100%): Intent Accuracy** | 77.78% | **77.78%** | $140 \text{ matches} / 180 \text{ tasks}$ | **VERIFIED** |
| **Variant B (100%): Argument Accuracy** | 69.44% | **69.44%** | $125 \text{ matches} / 180 \text{ tasks}$ | **VERIFIED** |
| **Variant B (100%): Macro-F1** | 0.7772 | **0.7772** | $\frac{1}{6} \sum_{c=1}^6 \text{F1}_c$ | **VERIFIED** |
| **Variant B (100%): Unsafe Execution Count** | 7 tasks (3.89%) | **7 tasks (3.89%)** | $7 \text{ false tool dispatches} / 180 \text{ tasks}$ | **CRITICAL AUDIT CONFIRMED** |
| **Governed Hybrid: Unsafe Execution Count** | 1 task (0.56%) | **1 task (0.56%)** | $1 \text{ false dispatch} / 180 \text{ tasks}$ | **VERIFIED** |
| **Variant A (Frozen): Intent Accuracy** | 65.56% | **65.56%** | $118 \text{ matches} / 180 \text{ tasks}$ | **VERIFIED** |
| **Keyword Baseline: Intent Accuracy** | 53.89% | **53.89%** | $97 \text{ matches} / 180 \text{ tasks}$ | **VERIFIED** |
| **Keyword Baseline: Unsafe Execution Count**| 37 tasks (20.56%)| **37 tasks (20.56%)** | $37 \text{ false dispatches} / 180 \text{ tasks}$ | **VERIFIED** |
| **Deterministic Router: Intent Accuracy** | 25.56% | **25.56%** | $46 \text{ matches} / 180 \text{ tasks}$ | **VERIFIED** |

---

### 3. Critical Safety Gate Conflation Diagnosis

A vital question raised for Batch 1741–1780 is whether the previous milestone incorrectly conflated the standalone neural classifier with the Governed Hybrid when declaring integration eligibility:

1. **Standalone Neural Classifier (Variant B):**
   - Held-Out Intent Accuracy: **77.78%**
   - Unsafe Execution Rate: **3.89% (7/180)**
   - Predeclared Safety Threshold: **$\le 1.0\%$**
   - **Audit Finding:** Evaluated **standalone without the governed firewall**, Variant B **FAILED the 1.0% safety gate**. The classifier mistakenly routed 5 incomplete `CLARIFY` queries to tools and 2 `OUT_OF_SCOPE` inputs to `CODE_TASK` and `CALCULATE`.
2. **Governed Neuro-Symbolic Hybrid:**
   - Unsafe Execution Rate: **0.56% (1/180)**
   - **Audit Finding:** The Governed Hybrid passed the 1.0% gate solely because the deterministic firewall intercepted and blocked the neural classifier's ungrounded proposals.
3. **Audit Ruling:**
   The neural classifier **itself** has NOT earned autonomous execution privileges. It can be safely utilized only as an unprivileged proposal generator behind the existing fail-closed `GovernedToolRouter`. Production execution authorization remains completely off-limits.

---

### 4. Split and Leakage Verification

- **Family Isolation Audit:** In Batch 1701–1740, training prompts used structural commands (`"Evaluate math expression: ..."`), while test prompts used conversational phrasing (`"Could you work out what 96 divided by 8 is?"`).
- **Residual Leakage Observation:** Although exact templates did not cross partitions, several entities and numbers appeared in both splits (e.g. `primary node port is 8080` appeared as a claim in training and test with different wording).
- **Remedy for Batch 1741–1780:** Build a strictly family-partitioned 300-example benchmark where entire semantic concepts, entity sets, and mathematical ranges are partitioned exclusively to train, val, or test.
