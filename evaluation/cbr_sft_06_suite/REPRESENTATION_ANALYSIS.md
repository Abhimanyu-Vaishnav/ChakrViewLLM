# Phase I: Model Capacity & Representation Analysis

**Date:** 2026-11-04  
**Evaluated Architecture:** ChakrMicro (4 layers, 8 heads, $d_{\text{model}}=512$, $d_{\text{head}}=64$, Context 512)  
**Evaluated Checkpoint:** CBR-SFT-04 Step 200 (`327118ce...`)  

---

## 1. Empirical Hidden State Probing

Measuring the cosine similarity of the final normalized representation at the `Assistant:` prediction position:

$$\cos(h_1, h_2) = \frac{\langle h_1, h_2 \rangle}{\|h_1\|_2 \|h_2\|_2}$$

| Prompt Pair | Final Hidden State Cosine Similarity | Interpretation |
| :--- | :---: | :--- |
| `Port 8080 Query` $\leftrightarrow$ `Port 9999 Query` | **0.9855** | Structural query template representation dominates the vector space |
| `Port 8080 Query` $\leftrightarrow$ `Greeting [08:00:00]` | **0.1266** | Completely separated conversational representation space |

---

## 2. Key Insights: The Value-Binding Bottleneck

1. **Query Template Domination:**
   - In a 4-layer transformer, the final token representation `Assistant:` in `Port 8080 Query` and `Port 9999 Query` has a **98.55% cosine similarity**.
   - The hidden vector primarily encodes "I am an assistant answering a port query".
   - The specific value (`8080` vs `9999`) represents a tiny perturbation ($< 1.5\%$) on top of this dominant template embedding.
2. **Why 4 Layers Struggle with Multi-Token Retrieval:**
   - In larger architectures (e.g., 12 to 32 layers), induction heads in middle layers copy the value to the residual stream, and later MLP layers isolate and amplify the extracted entity.
   - In ChakrMicro, with only 4 layers, the self-attention mechanism succeeds in attending to the context entity (as proven in Phase E where the first token probability ranks #1 or #2), but the residual stream has insufficient depth to both retain the query format and decode a 4-to-8 token fragmented entity without collapsing into an attractor state.
3. **Recommendation for CBR-SFT-06:**
   - Do not train on long, high-entropy arbitrary strings.
   - Ground copy-induction on **single-token key-value pairs** (e.g., single digits, single letters, short common keywords) so the model develops the linear routing circuit before attempting 51-token hex hashes.
