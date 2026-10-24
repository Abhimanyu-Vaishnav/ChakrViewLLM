# ChakrView Exp16 v3.1 — Leakage and Shortcut Audit Report## 1. Methodology and Query Extraction Contract
In Exp16 v3.1, queries are extracted by parsing the final turn preceding the `Assistant:` response prefix (matching the repository's evaluation conversation contract).
- Extraction Logic: For each record prompt, find the final line preceding `Assistant:`, extract text after the speaker colon (e.g. `Analyst: <query>` -> <query>).

---

## 2. In-Sample vs. Held-Out Evaluation Baselines

| Baseline Family | Split Evaluated | Metric / Decision Rule | Performance |
| :--- | :--- | :--- | :--: |
| **Global Label Prior (Train)** | Train (In-Sample) | Always predict Majority Label (`False` / Answerable) | 222 / 300 (74.00%) |
| **Global Label Prior (Val)** | Val (Held-Out) | Always predict Train Majority (`False` / Answerable) | 46 / 60 (76.67%) |
| **Global Label Prior (Test)** | Test (Held-Out) | Always predict Train Majority (`False` / Answerable) | 92 / 120 (76.67%) |
| **Exact-Query Majority** | Train (In-Sample) | Predict query-specific majority label within Train | 222 / 300 (74.00%) |
| **Exact-Query Majority** | Val (In-Sample Ceiling) | Predict query-specific majority label within Val | 49 / 60 (81.67%) |
| **Train-to-Val Generalization** | Val (Held-Out) | Apply Train Query Prior; fallback to Global Prior if unseen | 46 / 60 (76.67%) |
| **Train-to-Test Generalization** | Test (Held-Out) | Apply Train Query Prior; fallback to Global Prior if unseen | 92 / 120 (76.67%) |

### Key Distinctions:
1. **Train In-Sample ('74.00%')**: Every single one of the 25 distinct queries in Train appears with **both** answerable and abstention labels (100% label overlap). No query in Train is 100% biased towards one label.
2. **Train-to-Val Generalization ('76.67%')**: 42/60 validation queries appear in the training pool; 18/60 use unseen domain entities. Applying training query priors yields 76.67%, identical to the global prior.


---

## 3. Potential Shortcuts and Residual Leakage Risks
1. **Context Length Disparity in Certain Categories**:
   - In Category 1 (`turn_dependent_retrieval`), answerable records contain three sequential calibration stages, whereas abstention records contain only one stage before an abort notice. A model attending solely to context length could infer abstention without semantic reasoning.
2. **Filler Sentence Distributions**:
   - Synthetic fillers are drawn from a shared pool of 8 domain statements in Train, 6 in Val, and 6 in Test. Distinct domain lexicons prevent token overlap across splits but preserve synthetic template regularities.
3. **Identifier Suffix Regularities**:
   - Category 9 identifiers use deterministic prefixes (`sync_packet` in Train, `parse_frame` in Val). A pointer network attends to in-context identifiers rather than generative synthesis.
