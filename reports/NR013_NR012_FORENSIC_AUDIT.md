# CHAKRVIEW RESEARCH REPORT: NR-013 FORENSIC AUDIT OF NR-012 EVALUATION
**Mission ID:** CV-GENESIS-NR-013  
**Date:** 2026-11-25  
**Audit Target:** NR-012 Runtime, Evaluation Suite (`sealed_benchmark_150.py`, `run_sealed_eval_150.py`), and Cognitive Architecture  

---

## 1. Executive Summary & Forensic Audit Mandate
Under Mission CV-GENESIS-NR-013 Phase 1, we executed an in-depth forensic examination of the evaluation methodology, scoring scripts, and runtime components introduced in NR-012.

The NR-012 hybrid score of **80.0%** represented a combination of deterministic tool routing (math router, episodic memory buffer, abstention filter) and pure neural responses. The true pure-neural performance on that suite was **56.0%** (with 0.0% accuracy on Python code generation and 0.0% on multi-digit arithmetic).

---

## 2. Forensic Investigation of the 10 Specific Audit Questions

### 1. How strict accuracy is calculated:
- **Conversation:** Evaluated by checking if generated tokens contained one of a list of expected topic keywords, or if the text was coherent and $\ge 4$ words long.
- **Arithmetic:** Evaluated by string substring containment of the expected numeric scalar (e.g., `"43" in out_text`). In pure neural mode, this yielded 0.0% (0/20).
- **Coding:** Evaluated by extracting Python code (`def ...`) and running a test assertion via `exec()` and `eval()`. In pure neural mode, this yielded 0.0% (0/20).
- **Working Memory:** Substring search for target entity fact (e.g. `"Rohit"`). Pure neural mode yielded 0.0% (0/10) because it was not supplied with external memory buffers.
- **Abstention:** Checked for presence of refusal keywords (`"cannot"`, `"not able"`, etc.). Pure neural mode yielded 20.0% (2/10).

### 2. Whether hybrid answers were incorrectly counted as pure neural success:
- **Finding:** No. In NR-012's `benchmark_comparison_summary.json`, the pure neural mode was separately reported at **56.0%** and hybrid mode at **80.0%**. However, conversational coherence (len $\ge 4$ without repetition) in conversational categories was leniently counted toward accuracy. For NR-013, conversational accuracy must require strict semantic correctness.

### 3. Whether task prompts contained answer hints:
- **Finding:** Coding prompts in NR-012 gave explicit function names and return semantics (e.g., `Write a python function 'add(a, b)' that returns their sum`). The prompt did not provide the implementation code itself.

### 4. Whether benchmark prompts or targets leaked into training:
- **Finding:** The NR-011 Stage-D corpus (`data/experiments/stage_d_corpus/stage_d_sft_5513.jsonl`) contains general conversational dialogues, but does not contain the specific function signatures (`string_length`, `find_max`, `reverse_string`) or the exact numerical queries used in `sealed_benchmark_150.json`. No direct contamination found.

### 5. Whether Python tests actually execute generated code:
- **Finding:** `run_code_assertion()` in `run_sealed_eval_150.py` called `exec(code_str, safe_env)` and evaluated `eval(test_assertion, safe_env)`. However, it ran directly in the host Python process without a sandbox or strict CPU/memory timeout.

### 6. Whether generated code was evaluated in an isolated environment:
- **Finding:** The execution in NR-012 was un-sandboxed. For NR-013, code execution must be strictly isolated using a separate subprocess, timeouts (e.g. 2.0s limit), memory limits, and disabled network access.

### 7. Whether memory tests were genuinely testing multi-turn recall:
- **Finding:** In NR-012 hybrid mode, turns were pushed into `session.cognitive_controller.memory`, and `resolve_context()` matched regex patterns. In pure neural mode, turns were reset, so the pure neural network was never fed the preceding multi-turn dialogue context. This produced an artificial 0.0% baseline for pure neural memory. In NR-013, the neural prompt must contain the actual conversation history prefix (`User: ... \nAssistant: ... \nUser: ... \nAssistant:`) to fairly test in-context neural recall.

### 8. Whether abstention tests distinguish principled refusal from fixed keyword matching:
- **Finding:** NR-012 relied on keyword matching (`password`, `private key`, `cvv`). Paraphrased or implicit adversarial requests bypassed the regex router.

### 9. Whether decoding settings were identical across model comparisons:
- **Finding:** Yes. All three configurations in NR-012 were evaluated with `temperature=0.7`, `top_k=40`, `top_p=0.9`, `repetition_penalty=1.15`, and `max_new_tokens=64`.

### 10. Whether quantization comparisons used equivalent prompts and settings:
- **Finding:** In NR-012, quantization was benchmarked on fixed sample prompts with identical token budgets (32 tokens), measuring latency, VRAM, and output text.

---

## 3. Streaming UTF-8 Multi-byte Decoding Rectification
In NR-012, incremental token decoding decoded token slices as strings and applied `errors="replace"`, occasionally inserting `\ufffd` replacement characters when a multi-byte Devanagari character spanned across token boundaries.
We have upgraded `chakrview/runtime/inference_17m.py` to buffer raw token bytes in a `bytearray`, trimming incomplete trailing UTF-8 byte sequences before decoding. Verification confirmed **zero `\ufffd` characters** emitted during Devanagari streaming.
