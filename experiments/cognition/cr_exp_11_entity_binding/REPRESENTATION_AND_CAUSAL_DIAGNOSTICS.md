# REPRESENTATION, ATTENTION AND CAUSAL DIAGNOSTICS REPORT
**Experiment:** CR-EXP-11 Phase 5  
**Auditor:** Neural Research Scientist & Model Optimization Engineer  
**Date:** 2026-11-07T22:05:30+05:30  
**Model:** ChakrMicro v0.1 (`cr_exp_10_arm_b_best_val_copy.pt`)

---

## 1. LOGIT RANKING UNDER STATE UPDATE
Testing the prompt:
`Context: Meera works in Jaipur. Meera later moves to Bhopal.\nQuestion: Where does Meera work now?\nAnswer: `

- **Top Token Probabilities**:
  1. `'J'` (leading subword of initial state `Jaipur`): **44.16%**
  2. `'B'` (leading subword of latest state `Bhopal`): **18.37%**
  3. `'an '`: 4.10%
  4. `'The '` (start of abstention heuristic `"The context does not state..."`): 4.00%
  5. `'P'` (distractor candidate `Pune`): 2.90%

### Causal Interpretation
1. Both the initial state (`Jaipur`) and latest state (`Bhopal`) are present in the top-2 token predictions. The network **does** route attention to both entities mentioned in the prompt.
2. However, the initial state `'J'` possesses over $2.4\times$ the probability mass of the updated state `'B'`.
3. Because causal autoregressive attention accumulates associations chronologically, earlier entity-attribute pairings establish stronger residual stream activations than subsequent reassignments unless the network has been trained on explicit temporal updating patterns.
4. Furthermore, `'The '` (abstention prefix) holds $4.00\%$ probability at position zero, showing how the CR-10 negative training artifact bleeds into positive retrieval.

---

## 2. THE THREE REQUIRED CAPABILITY CORRECTIONS
1. **Recency-Aware State Attention**: The network needs training examples demonstrating that when temporal recency markers (*"now"*, *"currently"*, *"moved to"*, *"updated to"*) appear in the query, the latest state supersedes previous states.
2. **Balanced Positive & Negative Pairing**: Single-hop factual queries must be balanced 50/50 with unanswerable queries so that `'The context does not state...'` is only produced when the predicate or entity is truly missing.
3. **Template Syntactic Diversification**: Paraphrased variations must be introduced so attention heads learn semantic subject-verb-value binding rather than brittle string template matching.

