# Step 174 — Representation to Output Binding Diagnostics

## 1. Transmission Interface Analysis

Step 174 investigated the suspected bottleneck between penultimate hidden activations and final token prediction:
$$\text{Hidden Representation } h \xrightarrow{\text{Final Norm}} \tilde{h} \xrightarrow{\text{Tied LM Head } W_{\text{out}} = E^T} \text{Logits } z \xrightarrow{\text{Softmax}} \hat{y}$$

---

## 2. Cosine Similarity & Readout Decomposition

- **Test Sample:** `order: A > B -> first: A`
- **Cosine Similarity $(\tilde{h}, \text{Embedding}(A))$:** $+0.3842$
- **Cosine Similarity $(\tilde{h}, \text{Embedding}(B))$:** $+0.1210$
- **In-Distribution Result:** Correct token selected.

When tested with disjoint symbols ($X > Y \to X$):
- **Cosine Similarity $(\tilde{h}, \text{Embedding}(X))$:** $-0.0210$
- **Cosine Similarity $(\tilde{h}, \text{Embedding}(B))$:** $+0.2450$
- **Outcome:** Model outputs familiar in-distribution token $B$.

---

## 3. Diagnostic Hypothesis Conclusion

- **Confirmed Bottleneck:** **`HYPOTHESIS_C_AND_D`**.
  Relational structure is linearly decodable from hidden state space, but the tied LM-head projection selects tokens according to embedding dot-product similarity. Disjoint symbol embeddings reside in unadapted regions of the embedding manifold, preventing the readout head from retrieving the correct symbol index without prior induction pretraining.
