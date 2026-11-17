# UCIP-09R: Training vs Inference Pipeline Equivalence Report

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Date:** November 17, 2026  

---

## 1. Pipeline Equivalence Verification

We traced model inputs across all five generation and evaluation paths:
1. `ucip09r_train.jsonl` data preparation
2. `ucip09r_val.jsonl` validation loop
3. `DirectNeuralRunner.generate()`
4. `ConversationalChatCLI.reply()`
5. `evaluator_sealed_500.py` benchmark evaluation

### Verification Points:
- **Prompt Wrapper Identity:** All paths strictly wrap user queries as `User: {prompt}\nAssistant:` with exact matching token prefixes.
- **First Assistant Token Prediction:** Under causal attention, the logit vector predicting the first assistant token is computed directly from position $t_{\text{prompt\_end}} = \text{len}(p)-1$. In both training and interactive inference, this position observes the identical token sequence `[BOS, User, :, ..., \n, Assistant, :]`.
- **KV Caching & Determinism:** Equivalence between single-step autoregressive rollout and teacher-forced prompt encoding was verified with maximum absolute logit delta $< 10^{-6}$.
