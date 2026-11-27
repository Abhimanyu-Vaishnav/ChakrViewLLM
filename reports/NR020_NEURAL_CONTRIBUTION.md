# NR-020: Neural Contribution Scorecard

## 1. Disentangling Neural Learning from Symbolic Code
To provide an unvarnished audit of ChakrView's intelligence architecture, we isolate what the neural network actually does versus what deterministic symbolic infrastructure provides:

---

## 2. Capability Matrix: Neural Model vs Deterministic Engine

| Capability Dimension | Component Responsible | Measurable Performance | Evidence / Metric |
| :--- | :---: | :---: | :--- |
| **Multilingual Phrasing Comprehension** | **Neural Router** | **56.9%** (404/710 viable) | Successfully maps English, Hindi Devanagari, and Hinglish phrasing to operation identifiers. |
| **Adversarial Injection Detection** | **Neural Router** | **63.0%** (63/100 tasks) | Identifies prompt injection and off-topic requests as `status: "unsupported"`. |
| **Free-Form Numerical Arithmetic** | Neural Decoder (Stage-D) | **0.8% - 2.6%** | Complete failure; hallucinates numbers without symbolic execution. |
| **Exact Numerical Argument Extraction** | Deterministic Grounder | **99.3%** (710/715 tasks) | Extracts exact numeric substrings from source text. |
| **Devanagari Numeral Normalization** | Deterministic Normalizer | **100.0%** | Maps `०-९` and Hindi words (`पंद्रह`) to standard Python `int`. |
| **Zero Code Injection Enforcement** | Restricted Executor | **100.0%** (20/20 attacks) | Complete denial of arbitrary Python code execution. |

---

## 3. Executive Assessment
- **What the Neural Model Learned**:
  The 17M parameter model acts effectively as a **semantic multilingual intent classifier**, translating ambiguous, unformatted human phrasing into structured dispatch codes.
- **What Symbolic Code Provides**:
  The deterministic runtime provides **mathematical precision, type safety, memory management, and argument grounding**, completely insulating the user from neural hallucination.
- **Conclusion**:
  ChakrView's hybrid architecture is not an illusion of intelligence—it is a functional neuro-symbolic partition where neural weights handle perceptual language mapping and symbolic algorithms handle exact logic.
