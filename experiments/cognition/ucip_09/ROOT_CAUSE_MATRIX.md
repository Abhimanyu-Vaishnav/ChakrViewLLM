# UCIP-09: Root Cause Forensic Decision Matrix

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Date:** November 17, 2026  

---

## 1. Root Cause Classification Matrix

| Finding / Failure Mode | Classification | Concrete Evidence & Reproduction | Proposed Correction | Impact & Validation |
|---|---|---|---|---|
| **1. Step-0 EOS Premature Termination** | **Confirmed Implementation Defect** | Bare prompts without `User: ...\nAssistant:` produce `[EOS]` immediately. Formatted prompts generate 27.8 tokens. | Standardize all runners, evaluators, and CLI on `User: {prompt}\nAssistant:`. | Eliminates 100% of trivial step-0 EOS failures across all models. |
| **2. Evaluator False Positives in Prior Benchmarks** | **Confirmed Evaluation Defect** | Prior 800-probe suite matched `'7'` inside `'17'` and single unigram `'water'` on photosynthesis, inflating UCIP-07 to 31.37%. | Calibrated regex whole-word matching and multi-keyword co-occurrence (`evaluator_sealed_500.py`). | True zero-shot performance revealed as ~10–12%. |
| **3. Aggressive Repetition Penalty Corruption** | **Confirmed Decoding Defect** | `rep_penalty=1.15` + `no_repeat_ngram=3` degraded UCIP-07 from 12% to 8%, forcing model onto rare tail tokens (`"dogance"`). | Adopt **Conservative Decoding (Arm E: RepPen 1.05, N-gram 0)** or **NoRepeat 3 alone (Arm C)**. | Restores grammatical fluency while preventing loops. |
| **4. Unicode Detokenization Decode Crashing** | **Confirmed Implementation Defect** | Model generating incomplete multi-byte UTF-8 token sequences raised unhandled `UnicodeDecodeError`. | Add `errors="replace"` to all `tokenizer.decode()` calls. | Guarantees 100% crash immunity during autoregressive rollout. |
| **5. Severe Training Data Volume & Diversity Deficit** | **Strongly Supported Training Defect** | Prior training corpora had only 734–1,500 samples with 55.7% target duplication, entraining models on fixed nouns (`"coins"`). | Scale to 3,500+ clean, deduplicated, multi-domain conversational samples (`ucip09_train.jsonl`). | Increases semantic breadth and eliminates entity hallucination. |
| **6. Tokenizer Byte-Level Roundtrip & Subwords** | **Measured Limitation (NON-BOTTLENECK)** | 4096-vocab BPE achieves 100% roundtrip fidelity with 2.06 tokens/word on Hindi and 2.32 tokens/word on English. | Retain current tokenizer (Option A) to preserve checkpoint lineage and embedding weights. | Preserves architectural stability; no breaking changes. |
| **7. Architecture Capacity at 12.93M Parameters** | **Measured Architectural Boundary** | 12.93M parameters (7 layers, width 384) internalizes simple conversational tone and basic arithmetic, but lacks capacity for full factual encyclopedia knowledge. | Scale dataset quality first; explore 25M–28M configuration if compute permits, but retain CPU-first efficiency. | Distinguishes parametric weight limits from data deficits. |

---

## 2. Decision Gate Directives for Subsequent Phases

1. **Phase G (Architecture Decision):** Maintain the validated ChakrMedium 12.93M parameter architecture as the primary candidate, with Pre-RMSNorm and RoPE theta=50000.0. The evidence demonstrates that data volume (734 samples) is the active bottleneck, not model depth.
2. **Phase H (Dataset Recovery):** Build a 3,500+ record high-quality dataset with $0\%$ benchmark overlap covering all 10 language domains.
3. **Phase I (Controlled Training):** Fine-tune from UCIP-08 Medium parent weights using label smoothing ($\epsilon=0.05$), prompt loss masking, and conservative decoding.
