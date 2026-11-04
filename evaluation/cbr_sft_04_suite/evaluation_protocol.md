# CBR-SFT-04 Evaluation Protocol

## 1. Overview

The **CBR-SFT-04 Evaluation Suite** comprises 80 held-out evaluation tasks designed to independently and rigorously benchmark model capability across all four operational tracks without data leakage from training.

## 2. Benchmark Suite Composition (80 Held-Out Tasks)

| Track | Category | Count | Primary Objective |
| :--- | :--- | :---: | :--- |
| **Track A** | Temporal Greetings | 20 | Test greeting temporal accuracy (5 morning, 5 afternoon, 5 evening, 5 generic). Evaluates whether the model respects the time of day indicated in the user greeting without defaulting to a morning bias. |
| **Track B** | Identity & Negative Boundaries | 15 | Test model name identification ("ChakrView"), architecture recognition ("ChakrMicro", CPU-native), and strict negative constraints (refusal of ChatGPT/LLaMA/Claude attribution). |
| **Track C** | Dynamic Entity Grounding & Copying | 25 | Test exact copy extraction of IP addresses, ports, commit hashes, filenames, names, numbers (20 tasks) and explicit abstention on unprovided facts (5 tasks). |
| **Track D** | Multi-Turn Context Retention | 20 | Test retention of parameters, user preferences, variables, and names across a 2-turn dialogue history (`User: ...\nAssistant: ...\nUser: ...\nAssistant:`). |
| **TOTAL** | **Held-Out Tasks** | **80** | **100% disjoint from all training prompts and target responses.** |

---

## 3. Inference & Decoding Protocol

All model evaluations must be executed under standardized, reproducible decoding conditions:
- **Temperature:** `0.0` (greedy argmax decoding for determinism)
- **Top-p / Top-k:** Disabled (`top_k=0`, `top_p=1.0`)
- **Max Generation Tokens:**
  - Track A & Track B: 24 new tokens
  - Track C & Track D: 32 new tokens
- **Repetition Penalty:** None (1.0)
- **Stop Condition:** `<|eos|>` token ID or max tokens reached.

---

## 4. Dual-Scorer Architecture

Each generation is scored under two complementary rubrics:
1. **Legacy Word-Boundary Regex Scorer:**
   - Evaluates whether `\b{expected}\b` matches the response.
   - Preserved for continuous backward comparison with CBR-01, CBR-SFT-02, and CBR-SFT-03B benchmarks.
2. **Semantic / Task-Specific Scorer:**
   - Case-normalized, punctuation-stripped entity comparison.
   - Contextual temporal greeting validation (rejects mixed greetings such as "Good morning" when prompted with "Good afternoon").
   - Semantic abstention synonym matching ("That information is not provided", "does not specify", "cannot determine").

---

## 5. Execution Gate Notice

**Phase 1 concludes with dataset and evaluation suite preparation.**
No model training shall commence until Phase 1 data, audits, and evaluation suites are validated and formally approved.

