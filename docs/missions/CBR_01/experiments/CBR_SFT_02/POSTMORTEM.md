# CBR-SFT-01 — POSTMORTEM & ROOT-CAUSE REPORT
## Investigation into Conversational Failure and Formatting Mechanics

**Experiment ID:** `CBR-SFT-01`  
**Parent Checkpoint:** Canonical ChakrMicro Step 10,000 (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)  
**Training Set:** Exp34 Candidate v3 (`exp34_train_split_900.jsonl`, 900 records, 100% unique responses)  
**Evaluation Set:** 60-Task M3 Conversational Suite (`m3_conversational_tasks.json`)  
**Evaluation Outcome:** 0/60 Passed (0.00%) across Steps 25, 50, 75, 100  
**Date:** 2026-11-04  

---

## 1. Confirmed Empirical Findings

Through rigorous inspection of both datasets, raw generation outputs, and tokenizer representations, the following findings are verified:

1. **Zero Scaffolding in Training Data:**
   All 900 training records in `exp34_train_split_900.jsonl` are raw single-turn factual/algorithmic queries without role delimiters (e.g. `"How does BFS find the shortest path in an unweighted graph?"`). Exactly zero records contain `User:` or `Assistant:` turn headers.
2. **Strict Scaffolding in Benchmark Probes:**
   All 60 tasks in `m3_conversational_tasks.json` enforce two-turn chat scaffolding:
   ```
   User: {prompt}\nAssistant:
   ```
3. **Severe Attention Redirection & Pretraining Attractor Triggering:**
   Because `User:` and `Assistant:` prefixes were never supervised during SFT, the model encountering `'Assistant:'` decodes colons and newlines through its dominant Stage C pretraining distribution (Wikipedia article and table markup). When prompted with `'User: Hi, good morning!\nAssistant:'`, the model generated:
   ```
   \n! scope=&quot;background-color: #ffffff&quot; | \n
   ```
   The model never received training signal that `Assistant:` signals an answer continuation.
4. **Reproducibility of the 0/60 Baseline:**
   The 0/60 baseline failure was independently re-executed on native CPU and proved 100% reproducible.

---

## 2. Hypotheses & Remaining Uncertainties

### Primary Hypothesis (Serialization Mismatch):
The primary reason CBR-SFT-01 scored 0/60 on conversational benchmarks is the lack of chat turn scaffolding in the training set. Training with explicit `User: ...\nAssistant: ...` formatting will condition attention heads to respond specifically to dialogue prompts.

### Competing / Secondary Hypotheses:
1. **Pretraining Vocabulary & Entity Fragmentation:**
   Even if chat scaffolding is learned, dynamic entity copying may remain fragile due to subword fragmentation in the 4,096 BPE vocabulary.
2. **Context Window Attention Decay:**
   In multi-turn conversations exceeding 200 tokens, a 6-layer model ($d_{model}=192$) might suffer from attention dispersion if not explicitly trained on multi-turn dialogue histories.

---

## 3. Mandatory Corrections for CBR-SFT-02

1. **Unified Chat Formatting:** Enforce `User: {prompt}\nAssistant: {response}` across 100% of training and validation examples.
2. **Multi-Turn Curriculum:** Include 100 dedicated multi-turn context retention examples where the assistant response strictly depends on entities established in Turn 1.
3. **Dynamic Entity Grounding:** Include 150 diverse single-token and multi-token entity copying examples.
4. **Dedicated Independent Suites:** Establish held-out Dynamic Entity Grounding (100 tasks) and Multi-Turn Context (50 conversations) benchmarks alongside the canonical M3 suite.

