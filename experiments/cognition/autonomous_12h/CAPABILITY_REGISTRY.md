# CHAKRVIEW — CAPABILITY REGISTRY & EVALUATION REPORT (STAGE 5)

## 1. Capability Status Matrix

| Capability ID | Capability Name | Status | Empirical Basis & Verification |
| :--- | :--- | :---: | :--- |
| **CAP-001** | Basic Conversational Greeting & Politeness | **VALIDATED** | Accurately generates fluent, courteous greetings (`"Good morning! I am ready to assist you today?"`) without loops. |
| **CAP-002** | Indigenous Persona & Identity Retention | **VALIDATED** | Consistently responds `"I am ChakrView, an indigenous neural language model developed from first principles."` without external LLM leakage. |
| **CAP-003** | Epistemic Abstention on Missing Keys | **VALIDATED** | On held-out key-value abstention tests (Test Sample 4, Prompt D1, Prompt E2), correctly outputs: `"That information is not provided in the context."` |
| **CAP-004** | Anti-Collapse Stability under Fine-Tuning | **VALIDATED** | Successfully eliminated the Stage A mode collapse (`Cube` / `Mumbai` attractors) by using effective batch size 16, cosine decay, and weight decay 0.05. Validation loss improved monotonically from 4.25 to 2.52. |
| **CAP-005** | Exact Multi-turn Entity Variable Binding | **UNDER EXPERIMENT** | Candidate held-out test loss dropped by **$41.78\%$** ($5.23 \to 3.04$). However, exact multi-turn variable retrieval in long conversational prompts remains noisy at $3.44\text{M}$ parameter scale. |
| **CAP-006** | Complex Dynamic State Updates | **NOT DEMONSTRATED** | Fails to reliably track `"STATUS changed from PENDING to ACTIVE"` in zero-shot prompts; drifts to default syntax. |
| **CAP-007** | Persistent Cross-Session Episodic Memory | **NOT DEMONSTRATED** | In-context transient attention only. No persistent weight-based episodic memory verified. |
| **CAP-008** | Multi-lingual Fluency (Pure Hindi Devanagari)| **UNDER EXPERIMENT** | Handles Hinglish greetings coherently; struggles with pure Devanagari syntax (token fragmentation). |

---

## 2. Quantitative Benchmark Comparison

| Metric / Evaluation Split | Frozen Parent (`cbr_sft_04_step_200`) | New Candidate (`conversational_pilot_best_val`) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Validation Loss** (50 dialogue pairs) | $4.2503$ | **$2.5263$** | **$-40.56\%$** (Strong generalization) |
| **Held-out Test Loss** (54 unseen pairs) | $5.2353$ | **$3.0480$** | **$-41.78\%$** (Strong held-out fit) |
| **Mode Collapse Attractor Rate** | $0\%$ | **$0\%$** (Stage A collapsed to $100\%$) | Clean Language Maintained |
| **Persona Adherence** | $100\%$ | **$100\%$** | Preserved Indigenous Identity |
| **Epistemic Abstention Accuracy** | $50\%$ | **$100\%$** (Exact phrasing match) | $+50.0\%$ |
| **Inference Latency per Token** | $3.5\text{ ms}$ | $3.5\text{ ms}$ | Identical (Zero overhead) |

