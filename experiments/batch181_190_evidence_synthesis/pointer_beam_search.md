# Pointer Beam Search Specification: Batch 182
## Bounded Multi-Token Beam Decoding for ChakrPointerHead

### 1. Motivation & Mathematical Formulation
In greedy decoding ($k=1$), token selection is myopic:
$$y_t = \arg\max_{w \in V} P_{\text{final}}(w \mid x, y_{<t})$$
If the first sub-token of a multi-token entity receives slightly lower score due to vocabulary bias (e.g., proper noun prefix), greedy decoding collapses into repetitive formatting tokens.

Beam search explores $B$ parallel hypotheses:
$$\mathcal{H}_t = \text{Top-}B \left( \sum_{\tau=1}^t \log P_{\text{final}}(y_\tau \mid x, y_{<\tau}) \right)$$
With length-penalty normalization:
$$\text{Score}(\mathcal{H}) = \frac{\sum_{\tau=1}^{|\mathcal{H}|} \log P_{\text{final}}(y_\tau)}{|\mathcal{H}|^\alpha}, \quad \alpha = 0.7$$

### 2. Constraints & Causal Protection
1. **Context Mask Invariance:** At every generation step $T = T_{\text{prompt}} + t$, context mask $M_i$ remains strictly restricted to $[T_{\text{context\_start}}, T_{\text{context\_end}})$.
2. **Scatter Aggregation:** Multiple identical tokens inside context have their attention weights summed into $P_{\text{copy}}(w)$.
3. **Equivalence Invariant:** When $B=1$, `decode_pointer_beam` reduces identically to `decode_pointer_greedy`.
4. **Token Attribution:** Every emitted token preserves its origin:
   - `copy` if gate $g > 0.6$
   - `vocab` if gate $g < 0.2$
   - `mixture` if $0.2 \le g \le 0.6$
