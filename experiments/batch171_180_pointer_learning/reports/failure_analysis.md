# Failure Analysis & Architectural Diagnostics: Batches 171–180

## 1. Executive Summary & Problem Formulation
In previous milestones (Batches 161–170), Supervised Fine-Tuning (SFT) over standard vocabulary next-token cross-entropy failed completely on unseen copy induction (0.0% Exact Match). The model memorized high-frequency vocabulary tokens or collapsed on formatting, unable to route context tokens dynamically into output predictions.

In Batches 171–180, we designed, implemented, and empirically validated an indigenous, non-invasive **Pointer-Generator Mechanism** (`ChakrPointerHead`) alongside **Contrastive Bi-Encoder Representation Calibration** and **Cognitive Task Engine Verified Workers**.

---

## 2. Root Cause Analysis: The 0% SFT Copy Failure
### 2.1 The Inductive Bias Gap in Fixed-Vocab Causal Transformers
Standard causal transformers with tied token embeddings predict tokens via:
$$P_{\text{vocab}}(y_t) = \text{softmax}(W_{\text{embed}} h_t)$$
Where $W_{\text{embed}} \in \mathbb{R}^{V \times d}$ is fixed during inference. For rare or unseen entities (e.g., novel atomic numbers, proper nouns, arbitrary keys), the model can only select tokens if the projection $W_{\text{embed}} h_t$ serendipitously favors them. Without an explicit pointing mechanism, the network is incentivized to minimize training loss by memorizing static prior frequencies across training facts rather than learning an input-relative copy operation.

### 2.2 SFT Checkpoint Observations
- Base Canonical Model (Step 10,000): Zero-shot factual extraction EM = **20.0%** (accidental lexical fragments, heavy template repetition).
- SFT on Token-Vocabulary: 0.0% EM on held-out facts, with severe bias toward common numeric tokens like "2" or "3".

---

## 3. Pointer Architecture Evaluation (Arm C)
### 3.1 Auxiliary ChakrPointerHead Design
To preserve canonical model invariants (SHA-256 `96357705...` and 3,443,136 base parameters), `ChakrPointerHead` was engineered as a separate auxiliary module ($24,961$ trainable parameters):
$$P_{\text{final}}(y_t) = (1 - g_t) P_{\text{vocab}}(y_t) + g_t \sum_{i: x_i = y_t, M_i = 1} P_{\text{copy}}(i)$$
- **Context Masking ($M_i$):** Restricts copying strictly to valid context tokens ($M_i = 1$), excluding prompt templates, system instructions, queries, and future generation positions.
- **Dynamic Mixture Gate ($g_t$):** Sigmoid projection over concatenated current hidden state $h_t$ and context summary $c_t$.
- **Base Model Immutability:** Base weights are completely frozen during pointer training.

### 3.2 Empirical Results Across Arms (Held-Out Test Facts)
| Arm | Checkpoint Lineage | Trainable Parameters | Held-Out Test EM | Mean Gate Activation |
| :--- | :--- | :--- | :--- | :--- |
| **Arm A (Base)** | Step 10,000 Canonical | 0 (Zero-shot) | **20.0%** | N/A |
| **Arm B (SFT)** | Vocab Next-Token SFT | 3,443,136 | **0.0%** | N/A |
| **Arm C (Pointer)** | Base + `ChakrPointerHead` | 24,961 (Auxiliary) | **40.0%** | 0.3353 |

**Key Diagnostic Finding:**
Arm C doubled the held-out exact match performance over base zero-shot and completely eliminated the 0% SFT failure mode. When tested with forced pointer mode ($g=1.0$), accuracy reached 40.0%, with the learned gate operating dynamically ($g \approx 0.18 - 0.72$) depending on whether factual copying or syntax token generation was needed.

---

## 4. Learned Retrieval Diagnosis: Contrastive Margin Calibration
### 4.1 Previous Triplet Margin Collapse
Batch 164 reported 50% ranking accuracy with a narrow 0.0415 separation margin. The fixed triplet margin loss allowed the encoder to overfit shared template syntax without distinguishing query-document relevance.

### 4.2 InfoNCE Contrastive Calibration
Switching to symmetric InfoNCE in-batch contrastive loss expanded the separation margin from **0.0415 to 0.2521** (a 6x improvement in embedding space discriminability).

| Provider | Ranking Accuracy | Pos Sim | Neg Sim | Separation Margin | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25 Lexical** | **100.0%** | N/A | N/A | N/A | 0.03 |
| **Deterministic Hash** | **100.0%** | 0.4045 | 0.2097 | 0.1949 | 0.50 |
| **Bi-Encoder (B164)** | 50.0% | 0.6653 | 0.6238 | 0.0415 | 7.60 |
| **Bi-Encoder (B176)** | 50.0% | 0.5692 | 0.3171 | **0.2521** | 9.21 |

**Policy Decision:**
While embedding margin improved substantially (0.2521), deterministic BM25 and Hash baselines remain 100% accurate on factual queries with orders-of-magnitude lower latency. Therefore, deterministic providers remain the primary safe fallback in cognitive routing.

---

## 5. Cognitive Task Engine & Verified Workflows
- Integrated `EvidenceVerificationWorker` and `MathVerifierWorker` for multi-source consensus validation.
- Demonstrated end-to-end multi-step workflow execution, process restart recovery, and cross-session isolation.
- Integrated adversarial evaluation battery passed **20/20 test cases (100.0%)**.
