# Embedding Space Analysis — Batches 1221–1260
## Geometric Properties, Margin Analysis, and False-Positive Attractors

### 1. Vector Normalization & Geometry
- **Normalization Invariant**: Both `ChakrBiEncoder` models (old physics checkpoint and domain-adaptive checkpoint) employ strict $L_2$ normalization (`F.normalize(x, p=2, dim=-1)`). All vector norms are exactly $1.0 \pm 10^{-7}$.
- **Dimensionality**: $d_{\text{embed}} = 64$.
- **Duplicate Vectors**: Zero bitwise duplicate vectors observed across the 116 corpus chunks.

### 2. Metric Margin Comparison on Contrastive Test Split (68 Triples)

| Metric | Untrained/Physics Bi-Encoder | Domain-Adaptive Bi-Encoder | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy ($s_{\text{pos}} > s_{\text{neg}}$)** | 70.59% (48/68) | **100.0% (68/68)** | +29.41% |
| **Mean Positive Similarity** | 0.3682 | 0.9410 | +0.5728 |
| **Mean Negative Similarity** | 0.2113 | 0.0806 | -0.1307 |
| **Mean Triplet Margin** | 0.1569 | **0.8605** | +0.7036 |
| **Min Triplet Margin** | -0.3621 | +0.5061 | +0.8682 |
| **Max Triplet Margin** | 0.8088 | 1.1688 | +0.3600 |

### 3. Corpus Embedding Space Geometry & "Attractor" Chunks
When we compute inter-chunk cosine similarities across all corpus chunks:
- **Mean Background Inter-Chunk Cosine**: **0.6350** ($\text{std} = 0.3527$).
- **The "Attractor Chunk" Phenomenon**:
  In dense vector space, chunks that contain high frequencies of frequent BPE tokens with broad semantic range (e.g., currency symbols, numeric prices, words like "per", "month", "unlimited") have embeddings that cluster near the positive center of the projection hypersphere.
  - Chunk: `billing_subscription_tiers.md:0` ("Professional tier starting at $12,000 per year with unlimited workers...")
  - Cosine with Query 101 ("What is the parameter count of ChakrMicro V0.6?"): **0.9504**
  - Cosine with Gold Chunk (`compute_arch_core.md:0`): **0.7916**
  - Result: The model ranks the billing chunk #1 with a **0.1588 margin deficit** against the true gold chunk!

### 4. Mathematical Diagnosis
Because training minimized triplet loss with *only one hard negative from the same topic family*, the model learned:
$$\text{Sim}(Q, P) > \text{Sim}(Q, N_{\text{same\_topic}}) + 0.2$$
It was **never constrained** against out-of-domain chunks:
$$\text{Sim}(Q, N_{\text{other\_topic}}) \le 0$$
Consequently, the model minimized loss by projecting query and document vectors into a dense quadrant where generic text patterns yielded high positive similarities ($>0.90$), creating catastrophic false-positive attractors across unrelated documents.
