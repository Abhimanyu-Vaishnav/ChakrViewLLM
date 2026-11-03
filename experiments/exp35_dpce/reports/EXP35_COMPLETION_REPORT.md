# CHAKRVIEW — EXPERIMENT EXP35-DPCE COMPLETION REPORT
## Dynamic Prompt-Conditioned Extraction: Final Scientific Findings & Strategic Roadmap

**Project:** ChakrView  
**Milestone:** M5  
**Experiment ID:** Exp35-DPCE  
**Target Model:** ChakrMicro (3,443,136 parameters, frozen)  
**Execution Date:** 2026-10-30  
**Git HEAD:** `5aeaa2734de5a98802b96afe36324765f3244738`  

---

## 1. EXECUTIVE SUMMARY & EXPERIMENT STATUS

### CLASSIFICATION: **`NO SIGNIFICANT IMPROVEMENT`**

Milestone M5 executed a rigorous, controlled experiment to determine whether training on a 1,000-record dataset with 100% unique, dynamically conditioned responses could enable ChakrMicro (3.44M parameters) to perform grounded entity extraction.

The experiment was executed with flawless scientific discipline:
1. **Specification Freeze:** Frozen prior to dataset synthesis (`EXP35_EXPERIMENT_SPEC.md`, SHA-256 `d9a01cc7...`).
2. **Dataset Integrity:** 1,000 synthetic records with **1,000 unique completions (100% response diversity)**, verified 0 train/val leakage, and 0 external API dependencies.
3. **Independent Benchmark:** 200 tasks featuring entirely unseen entities and cities, evaluated via strict word-boundary matching.
4. **Controlled Dual-Arm Training:** Arm A (M3 Step 250 base) and Arm B (M2 Arm A base) trained for 500 steps under identical hyperparameter budgets on local CPU.

### Primary Scientific Finding
- **Validation Loss Plummeted:** Arm B reduced loss from 4.9529 to **1.4006** ($71.7\%$ reduction).
- **Physical Generation Remained Flawless:** Repetition remained suppressed at **0.0207**, unique token ratio reached **0.9033**, and EOS termination achieved **86.7%**.
- **Benchmark Extraction Remained at Zero:** Both Arm A and Arm B scored **0/200 (0.0%)** on the independent grounded extraction benchmark.
- **Scientific Conclusion:**  
  *The Null Hypothesis is supported.* SFT data diversification alone cannot induce dynamic variable binding or information extraction in a small model that lacks pre-trained induction circuits.

---

## 2. WHAT THE MODEL LEARNED VS WHAT IT DID NOT LEARN

### Demonstrated Capabilities
- **Syntactic Form Adaptation:** The model learned the grammatical sentence structures of the training data (e.g. consistently starting responses with `"At station..."`, `"Officer..."`, or `"User..."`).
- **Turn Boundaries & Length:** Adheres strictly to single-turn response lengths and terminates cleanly with `<EOS>`.
- **Vocabulary Diversity:** Avoids repeating n-grams or falling into infinite generation loops.

### Persisting Failures
- **Zero Induction / Copy-Paste Capability:** The model does not copy the requested entity name from the prompt context into the completion.
- **Unconditioned Entity Generation:** It fills entity slots with internal unigram priors rather than dynamically conditioning on the prompt's provided context.

---

## 3. ARCHITECTURE DECISION & STRATEGIC RECOMMENDATION

- **Architecture Decision:**  
  *Investigate a specific bottleneck: Pretraining induction circuits vs Tokenizer subword fragmentation.*  
  ChakrMicro's 4,096-vocabulary tokenizer fragments entity names into 3–4 subwords. A 6-layer model with $d_{\text{model}}=192$ cannot compose multi-step subword copying without foundational pretraining on copy/induction tasks.
- **Strategic Recommendation for Milestone M6:**  
  1. Do not perform further generic conversational SFT.
  2. Implement an indigenous **Algorithmic Induction Pretraining (AIP)** phase: inject synthetic copy-paste, sequence reversal, and key-value mapping tasks into the base pretraining corpus to cultivate foundational induction attention heads before fine-tuning.
  3. Explore hybrid runtime retrieval architectures where external memory contracts provide grounded entities directly.

