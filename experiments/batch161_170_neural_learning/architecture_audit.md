# Batch 161 — Architecture and Failure Audit
## Learned Representations, Neural Copy Induction, Memory-Guided Tasks & Honest Evaluation

### 1. Executive Summary & Repository Context
This audit reviews the state of ChakrView following the completion of Batches 151–160 (`cb49b6b`).
The runtime environment consists of:
- **Canonical Model**: ChakrMicro v0.1 (`3,443,136` parameters, `4,096` vocabulary, tied embeddings, RoPE, RMSNorm, SwiGLU).
- **Canonical Step 10,000 Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (Verified SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Cognitive Routing & Session Architecture**: `CognitiveRouter`, `RetrievalAdapter`, `EvidenceExtractor`, `PersistentSessionMemory`, and `GroundedSessionAdapter`.
- **Hardware Profile**: 20 CPU logical cores, CUDA unavailable (`torch.cuda.is_available() == False`). Execution is CPU-optimized with multi-threaded PyTorch primitives.

### 2. Component Analysis & Responsibility Separation
In Batches 151–160, all 15 benchmark test cases passed. However, an honest architectural audit reveals the exact breakdown of responsibilities:
1. **Conversational Greeting / Continuity**: Solved by the **neural core** (`neural_core` origin).
2. **Factual Grounded QA (Atomic numbers, desert location, crater names)**: Solved by the **deterministic extractor** (`deterministic_extractor` origin), NOT by neural comprehension.
3. **Out-of-Domain Absent Evidence**: Solved by the **retrieval score threshold / fallback policy** (`fallback_policy` origin).
4. **Multilingual (Hindi) Extraction**: Solved by the deterministic Unicode regex pattern matcher.
5. **Session Continuity & Isolation**: Solved by atomic file storage in `PersistentSessionMemory`.

### 3. Identified Deficiencies & Failure Boundaries
1. **Weak Neural In-Context Copying**:
   - The base causal language model `step_10000.pt` was trained on standard next-token prediction across text corpora. When presented with `Context: <fact>\nQuestion: <query>\nAnswer:`, it does not reliably copy the fact tokens to the output position, instead emitting generic or repetitive tokens.
   - SFT attempts without targeted loss masking dilute the training signal across prompt and context tokens.
2. **Heuristic Hash Embeddings in Retrieval**:
   - `DeterministicHashEmbeddingProvider` computes 3-gram feature hashing over SHA-256 buckets. It contains zero learned semantic representations.
   - True semantic retrieval (e.g., matching "copper" to "conductive metal" or "lunar landing" to "spacecraft") cannot be achieved via hash buckets.
3. **Absence of Memory-Guided Multi-Step Task Tracking**:
   - While `PersistentSessionMemory` records dialogue turns, it lacks structured task state tracking (current step, pending steps, verification outcomes, retry budgets, and idempotent recovery).

### 4. Experimental Design for Batches 161–170
To solve these limitations without compromising repository invariants:
1. **Learned Representation Prototype (Batches 163–164)**:
   - Implement an indigenous, lightweight bi-encoder representation model (`ChakrBiEncoder` / `LearnedEmbeddingProvider`) operating on ChakrView's 4096-vocabulary tokens.
   - Train on contrastive pairs (query $\to$ positive document vs. hard negatives) with normalized cosine loss.
   - Compare strictly against the BM25 and hash baselines on held-out facts.
2. **Neural Copy-Induction Loss Contract & Training (Batches 165–166)**:
   - Construct a rigorous causal loss mask where loss is computed **exclusively** over answer target tokens and the turn-termination token (`<|im_end|>` or `EOS`).
   - Execute controlled fine-tuning from `step_10000.pt` on a diverse, leakage-proof dataset of key-value and factual contexts.
3. **Learned Retrieval Bridge (Batch 167)**:
   - Integrate `LearnedEmbeddingProvider` into `RetrievalAdapter` as a configurable option alongside the deterministic hash and BM25 fallbacks.
4. **Memory-Guided Multi-Step Task Engine (Batch 168)**:
   - Implement `CognitiveTaskState` and `TaskManager` supporting multi-step execution (Read $\to$ Retrieve $\to$ Extract $\to$ Verify $\to$ Record) with state recovery and retry boundaries.
5. **Adversarial Evaluation & Synthesis (Batches 169–170)**:
   - Measure actual neural copy accuracy, retrieval recall/MRR, and task recovery on unseen held-out splits.
