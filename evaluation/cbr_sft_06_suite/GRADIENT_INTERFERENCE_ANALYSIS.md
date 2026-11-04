# Phase H: Gradient Interference & Optimization Diagnostics

**Date:** 2026-11-04  
**Evaluated Checkpoint:** CBR-SFT-04 Step 200 (`327118ce...`)  
**Methodology:** In-memory, non-destructive gradient accumulation across 5 distinct task objectives.  

---

## 1. Measured Objective Gradient Norms

| Objective | Gradient Norm ($\|g\|_2$) | Relative Magnitude | Interpretation |
| :--- | :---: | :---: | :--- |
| **Key-Value Retrieval** | **79.23** | **3.65×** | Highest optimization pressure; large backpropagation through projections |
| **Exact Technical Copy** | **57.24** | **2.64×** | Strong cross-entropy loss due to arbitrary token transitions |
| **Identity Grounding** | **41.01** | **1.89×** | Moderate gradient magnitude |
| **Conversational Greeting** | **24.34** | **1.12×** | Low gradient norm; low loss on familiar greeting tokens |
| **Abstention Refusal** | **21.65** | **1.00×** | Lowest gradient norm; consistent sequence embeddings |

---

## 2. Empirical Cosine Similarity Matrix

$$\cos(g_i, g_j) = \frac{\langle g_i, g_j \rangle}{\|g_i\|_2 \|g_j\|_2}$$

| Objective | Greeting | Identity | Exact Copy | Key-Value | Abstention |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Greeting** | 1.0000 | +0.0785 | +0.0765 | +0.0612 | +0.1464 |
| **Identity** | +0.0785 | 1.0000 | +0.0466 | +0.0101 | +0.0689 |
| **Exact Copy**| +0.0765 | +0.0466 | 1.0000 | **+0.2141** | **+0.1974** |
| **Key-Value**  | +0.0612 | +0.0101 | **+0.2141** | 1.0000 | **+0.2632** |
| **Abstention** | +0.1464 | +0.0689 | **+0.1974** | **+0.2632** | 1.0000 |

---

## 3. Key Findings & Diagnostic Verdict

1. **Near-Orthogonality to Conversational Spaces:**
   - The cosine similarity between `greeting` and `key_value` is only **+0.0612**, and between `identity` and `key_value` is only **+0.0101**.
   - These objectives are **almost completely orthogonal in parameter update space**.
   - Because they are orthogonal, training **exclusively** on key-value tasks updates weights in directions that cause random walk / destructive interference across the greeting sub-manifolds, unless constrained by explicit replay gradients!
2. **Gradient Magnitude Dominance:**
   - Key-value and exact copy gradients are **$3.65\times$ and $2.64\times$ larger** than greeting gradients.
   - Without gradient balancing or replay, the retrieval tasks overwhelm the weight updates, completely overwriting the delicate balance achieved in CBR-SFT-04.
3. **Strong Alignment Between Key-Value and Abstention:**
   - `key_value` $\leftrightarrow$ `abstention` shares the highest cosine similarity (**+0.2632**).
   - This positive alignment means updates aimed at learning key-value queries and updates aimed at learning abstentions push parameters along similar attention projection subspaces. Because the abstention objective has lower entropy and higher predictability, the optimization trajectory quickly slides along this shared direction into the pure abstention attractor!
