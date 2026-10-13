# Batch 162 — Benchmark Contract & Leakage-Proof Data Splits
## Data Integrity, Partition Manifests, Evaluation Protocol & Non-Leakage Guarantees

### 1. Benchmark Dimensions & Contracts
To independently evaluate each layer of the cognitive architecture, the benchmark defines five orthogonal evaluation tracks:
1. **Track A (Retrieval & Semantic Ranking)**:
   - Measures Top-1 Recall, Mean Reciprocal Rank (MRR), and Positive-Negative Score Separation.
   - Evaluates on paraphrased queries, cross-lingual queries (Hindi), and hard lexical distractors.
2. **Track B (Neural In-Context Copy Induction)**:
   - Evaluates whether the neural core outputs the exact target span conditioned on `Context: ... Question: ...`.
   - Metrics: Exact Match (EM) over target value, answer-token cross-entropy loss, and termination token generation rate.
3. **Track C (Grounded Extraction & Conflict Detection)**:
   - Evaluates deterministic slot extraction against ambiguous and conflicting evidence.
4. **Track D (Safe Abstention & Negative Controls)**:
   - Tests ungrounded, out-of-domain, or missing queries to ensure 100% clean abstention without hallucination.
5. **Track E (Memory-Guided Task Execution & Recovery)**:
   - Multi-step workflow state transitions, idempotency under repeat requests, and crash recovery.

### 2. Leakage Prevention Protocol
- **Entity Partitioning**: Fact entities (e.g. elements, celestial objects, geographical regions, historical items) are strictly grouped into disjoint partitions:
  - **Train Split (60%)**: Used for contrastive representation training and copy-induction SFT.
  - **Validation Split (20%)**: Used for hyperparameter tuning and early stopping.
  - **Held-Out Test Split (20%)**: Purely unseen entities and key-value relations never seen during any training phase.
- **Template Diversity**: Evaluation uses distinct question templates and prompt orderings never seen in the training split.
- **No Test Set Feedback**: Hyperparameters and thresholds are tuned strictly on validation splits.
