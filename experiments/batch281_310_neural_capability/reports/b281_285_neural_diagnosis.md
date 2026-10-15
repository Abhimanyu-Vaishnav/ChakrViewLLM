# Batch 281–285 Audit: Repository Lineage, Invariants, and Neural Learning Bottleneck Diagnosis

**Date:** 2026-10-15  
**Milestone:** Batches 281–310  
**Branch:** `dev/v0.6-v0.7-validation`  
**Parent Commit:** `43e44b1` (`feat(batch271-280): audit benchmark, integrate compound arithmetic and governed research`)  
**Previous Milestone Parent:** `90a48f3` (Batches 261–270)  
**Execution Environment:** CPU-First, Hardware-Adaptive (AMD/Intel x86_64, Windows NT, PyTorch 2.14.0+cpu, 14 execution threads, no CUDA GPU)

---

## 1. Batch 281 — Repository and Milestone Audit

### Git State Verification
- **Branch:** `dev/v0.6-v0.7-validation`
- **HEAD Commit:** `43e44b1e32d16c4fa3158c89c565ffb48be04085`
- **Parent Commit:** `90a48f3e173d4417f1bc1d6069ec9cc97e8a168a`
- **Working Tree:** User modifications in `chakrview/runtime/session.py` and untracked sprint artifacts from batches 103–122 are strictly preserved and uncommitted.

---

## 2. Batch 282 — Canonical Checkpoint Verification

- **Path:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Verified SHA-256 Digest:**
  `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Architecture Invariants:**
  - Parameters: Exactly **3,443,136** parameters.
  - Configuration: `d_model=192`, `n_layers=6`, `n_heads=6`, `d_ff=768`, `vocab_size=4096`, `context_len=512`.
  - Tokenizer: Byte-Pair Encoding (BPE), vocabulary 4,096 tokens, located at `release/chakrview-v0.5.0/data/experiments/vocab_4096`.
  - Verification Status: **100% INTACT & FROZEN**.

---

## 3. Batch 283 & 284 — Neural Learning Path & Corpus Audit

A comprehensive trace of the neural training pipeline from raw shards to autoregressive generation revealed the architectural dynamics of `ChakrMicro`:

1. **Pretraining Stage C Corpus (`data/tokenized/stage_c`):**
   - 27 training shards (6,651,093 tokens), 2 validation shards (465,954 tokens), 3 test shards (591,554 tokens).
   - Domain distribution: ~61.6% English Wikipedia/web, 17.1% Hindi (Devanagari), 15.2% Python code, 5.5% Sanskrit, 0.3% Math.
   - At Step 10,000 (2.56M tokens trained), the canonical core achieved validation loss **3.2306** (Perplexity: 25.30) and held-out test loss **2.9489** (Perplexity: 19.08).
   - While syntactically aware of English, Hindi, and Python tokens, its greedy completions on open-ended queries frequently cycle through repetitive n-grams or Wikipedia table attractors (`| ... |`).

2. **The SFT Formatting & Delimiter Pitfall (Audited from Batch 63):**
   - In earlier fine-tuning attempts, stripping prompt delimiters (`prompt.strip()`) caused training on prompts ending in token `ant` while generation prompts ended in `\n` (token `13`).
   - SFT target loss averaged over 88% template tokens (`<|im_end|>\n`), masking high loss on actual content tokens.
   - When exact ChatML delimiters are preserved (`strip_prompt_delimiters=False`), `ChakrMicro` reliably terminates on `<|im_end|>` (100% compliance).

3. **The Parameter Capacity & Induction Head Bound (Audited from Batches 147–150 & 211–220):**
   - At 3.44M parameters, `ChakrMicro` lacks sufficient width ($d_{model}=192$, 6 heads) to form deep multi-layer induction heads that dynamically copy arbitrary unseen strings from a 500-token context into the output vocabulary without assistance.
   - In Batch 147–150, direct fine-tuning reduced test loss to 1.47 nats, but greedy decoding generated frequent training values rather than copying unseen test entities.
   - Conversely, when equipped with a learned query-aware cross-attention pointer head (`ChakrCrossAttentionPointerHead`), the system achieves **47.0% exact match on unseen relational tuples** and **100% precision on intermediate-hop entity tracking** without altering canonical weights.

---

## 4. Batch 285 — Neural Bottleneck Decision & Engineering Strategy

### The 3 Core Bottlenecks of Weak Neural Performance:
1. **Unsupervised General Pretraining Recovery Plateau:**
   - *Evidence:* Small pretraining bursts (100–250 steps, 128k tokens) on the 3.44M core cannot beat the Step 10,000 checkpoint on general language loss without risking catastrophic degradation (Batch 139–142 demonstrated $+0.12$ loss spike without warmup).
   - *Status:* Confirmed fact.
2. **Target Masking & Unweighted Template Dominance in SFT:**
   - *Evidence:* In raw SFT, the model expends 85%+ of gradient updates fitting boilerplate tokens (`<|im_start|>assistant\n`, `<|im_end|>\n`), leaving content prediction under-supervised.
   - *Status:* Confirmed fact.
3. **Induction Head Failure for Direct Vocabulary Copying:**
   - *Evidence:* In context-dependent QA, the unassisted neural core cannot reliably copy unseen entity spans directly through vocabulary logits, whereas attention-based pointer heads provide strong relational binding.
   - *Status:* Confirmed fact.

### High-Value Training Intervention:
To build genuine, measurable neural capability that improves language modeling, coherent generation, and transferable reasoning:
1. **Task Contract:** Supervised instruction & continuation learning with:
   - Proper loss masking on prompt tokens (loss calculated **only** on response tokens).
   - Target weighting on semantic content tokens.
   - Linear warmup (50 steps) + cosine annealing decay to prevent representational collapse.
2. **Dual Capability Evaluation:**
   - Measure token-level cross-entropy loss and perplexity on held-out validation and test sets.
   - Measure exact-match accuracy, clean `<|im_end|>` termination, and n-gram repetition on held-out, unseen test prompts.
3. **Hardware-Adaptive Implementation:**
   - Execute on the 14-thread CPU engine with batching and memory management, logging exact step latencies and throughput.
