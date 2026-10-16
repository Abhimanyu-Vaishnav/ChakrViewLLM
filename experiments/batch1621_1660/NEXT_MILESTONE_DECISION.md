# NEXT MILESTONE DECISION
## Batches 1661–1700 Strategic Roadmap

### 1. Context and Empirical Foundation

The empirical results from Batches 1621–1660 establish clear boundaries for the ChakrMicro neural core (1.58M parameters):
- Perplexity reduction (from 175.74 to 11.30) is achievable in 100 steps.
- Pure next-token prediction does **not** transfer to unseen mathematical extrapolation, syntax-bound code execution, or multi-hop compositional reasoning.
- Attempting to force a 1.58M parameter model to act as a monolithic calculator, compiler, and relational database is mathematically and computationally unfeasible.

### 2. Strategic Evaluation: What Should NOT Be Done Next

1. **Do NOT increase parameter size indiscriminately (e.g., scaling to 100M+ parameters):**
   - ChakrView is designed for indigenous edge execution on resource-constrained hardware (laptops, micro-nodes) without external GPU clusters.
   - Simply scaling parameters does not fundamentally solve the arithmetic hallucination problem without deterministic tool integration.
2. **Do NOT abandon neural core development in favor of pure regex/rules:**
   - Neural classification, intent extraction, and semantic routing are vital for handling natural human variance that brittle regex cannot accommodate.
3. **Do NOT tune benchmarks on the test set:**
   - Benchmarks must retain strict leakage control and separate held-out splits.

---

### 3. Recommended Next Milestone: Batches 1661–1700

**Milestone Title:** *Neural-Guided Intent Routing and Tool Argument Extraction Gate*

#### Core Research Question:
Can the adapted ChakrMicro neural core be trained to reliably classify user intent into deterministic tool calls (`CALCULATOR`, `CODE_SANDBOX`, `KNOWLEDGE_RETRIEVAL`, `DIRECT_ANSWER`) and extract structured JSON arguments, outperforming naive rule heuristics under natural linguistic variations?

#### Proposed Work Packages:
1. **WP-1: Intent and Slot-Filling Dataset:** Author a controlled, leakage-split dataset of complex natural queries requiring tool invocation, featuring paraphrases, typos, and negative controls (queries requiring abstention).
2. **WP-2: Fine-Tuning for Structured Intent Generation:** Train ChakrMicro on token-level tool call schemas with constrained decoding / grammar masks.
3. **WP-3: End-to-End Neuro-Symbolic Latency & Accuracy Audit:** Measure the speed and accuracy of Neural Router $\rightarrow$ Deterministic Executor $\rightarrow$ Structured Verifier compared to purely deterministic routing.

This approach leverages the neural core's true strength (handling natural linguistic diversity) while delegating exact execution to the robust deterministic tools built in Batches 1501–1620.
