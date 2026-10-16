# OBJECTIVE RESEARCH: MASKED-SPAN RECONSTRUCTION
## Batches 1821–1860: Architectural Compatibility & Future Objective Feasibility

### 1. Motivation & Context

In Batches 1781–1820, it was hypothesized that pre-training the ChakrMicro transformer backbone on a masked-span reconstruction objective (similar to BERT or T5 / SpanBERT) might produce richer bidirectional representations than purely autoregressive causal language modeling.

As mandated by Work Package 7, we investigated whether masked-span reconstruction is appropriate for the current ChakrMicro architecture **without prematurely implementing incompatible changes or corrupting existing generation paths**.

---

### 2. Architectural Analysis of Current ChakrMicro Causal Attention

In `chakrview/brain/model.py`, self-attention is implemented using a strict upper-triangular causal attention mask:

```python
# Causal Attention Masking in ChakrMicro Attention Block:
mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
scores = scores + mask
attn_weights = F.softmax(scores, dim=-1)
```

#### Key Findings:
1. **Unidirectional Receptive Field**: At token position $t$, representations can attend only to tokens $\le t$. Token $t$ cannot observe future tokens $t+1, \dots, T$.
2. **Impact on Causal Generation**: This structure enables $O(1)$ KV-caching during autoregressive inference and ensures that training loss directly optimizes next-token probability $P(x_t \mid x_{<t})$.
3. **Incompatibility with Naive Bidirectional MLM**:
   - Simply setting the mask to zeros turns the network into a bidirectional encoder (like BERT).
   - However, **doing so destroys autoregressive text generation**, rendering the model incapable of generating open-ended sentences, streaming tokens, or acting as a generative language model.

---

### 3. Alternative Architectures for Masked-Span Learning

To integrate bidirectional masked-span learning into ChakrView without breaking causal generation, three potential architectures exist:

| Strategy | Architecture Description | Generative Capability | Compatibility with Current Checkpoints | Computational Overhead |
| :--- | :--- | :---: | :---: | :---: |
| **Option 1: Dual-Mode Attention (Prefix-LM / UniLM)** | Use bidirectional attention on a prompt/prefix, and causal attention on target spans. | Retained | Partial (requires position adaptation) | Low ($1.0\times$) |
| **Option 2: Encoder-Decoder (T5 / ChakrSeq2Seq)** | Separate bidirectional encoder and causal autoregressive decoder. | Retained | None (incompatible parameter layout) | High ($2.0\times$ params) |
| **Option 3: Infilling / Span-Corruption (FIM / GLM)** | Shuffle corrupted spans to the end of the sequence and predict them autoregressively using causal mask. | Retained | **100% Compatible with ChakrMicro** | **Zero architecture change** |

---

### 4. Recommended Research Recommendation: Fill-in-the-Middle (FIM)

The optimal, zero-risk pathway for future span-corruption learning in ChakrView is **Fill-in-the-Middle (FIM) Causal Pretraining**:
- **Mechanism**: Rather than altering attention masks, a document span is masked out with a special sentinel `<mask_0>`, and the excised span is appended to the end:
  $$\text{Input: } \text{"The architecture } \langle\text{mask\_0}\rangle \text{ causal attention."} \to \langle\text{mask\_0}\rangle \text{ "relies on"}$$
- **Advantages**:
  1. **Zero Architecture Changes**: Causal attention mask, RoPE positional encoding, and KV-cache inference remain 100% untouched.
  2. **Bidirectional Context Infilling**: The model learns to utilize both prefix and suffix context to predict the middle span.
  3. **Seamless Checkpoint Lineage**: Can be trained directly from the canonical Step 10,000 checkpoint without adding weights or changing tensor shapes.

**Smallest Controlled Next Experiment**: Evaluate FIM causal pre-training on 500 documents with a 15% span corruption rate against standard causal pre-training to measure downstream routing accuracy.
