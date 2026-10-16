# NEXT MILESTONE DECISION
## Batches 1741–1780 Roadmap: Governed Neural Router Runtime Integration

### 1. Empirical Foundation & Context

Batches 1701–1740 achieved a landmark empirical breakthrough for ChakrView:
- Discarding autoregressive JSON synthesis in favor of a **Direct Classification Head** raised intent accuracy from $19.17\%$ to **$77.78\%$**, increased format validity to **$100.0\%$**, and accelerated CPU latency from $259\text{ ms}$ down to **$4.56\text{ ms}$**.
- Controlled data scaling demonstrated predictable, monotonic improvement ($63.89\% \rightarrow 70.00\% \rightarrow 77.78\%$).
- The Governed Neuro-Symbolic Hybrid maintained a **$0.56\%$ unsafe execution rate**, proving that neural semantic classification can be safely paired with fail-closed deterministic tool execution.

---

### 2. Strategic Evaluation: What Should NOT Be Done Next

1. **Do NOT remove the Governed Router or Structured Verifier:**
   - Standalone neural classification alone has a 3.89% false execution rate on negative controls and adversarial prompts. The deterministic firewall remains strictly necessary.
2. **Do NOT revert to autoregressive generative tool calling:**
   - Causal string generation is too slow, brittle, and unconstrained for structured slot dispatch at 3.4M parameters.
3. **Do NOT claim broad AGI reasoning:**
   - ChakrMicro is an effective semantic intent classifier, but exact math, code execution, and database lookups must remain delegated to deterministic modules.

---

### 3. Recommended Next Milestone: Batches 1741–1780

**Milestone Title:** *Governed Neural Router Runtime Integration & Live Anaphora Tracking Gate*

#### Primary Objective:
Integrate the validated `ChakrMicroIntentClassifier` (Variant B) into `GovernedToolRouter` as the primary semantic pre-routing layer, combined with multi-turn conversation state tracking and natural number normalization.

#### Key Engineering Deliverables:
1. **WP-1: Runtime Adapter Layer (`chakrview/runtime/neural_intent_router.py`):**
   Expose an official runtime interface that loads `variant_b_finetuned_100pct.pt` and connects it directly into `GovernedToolRouter.route_request()`.
2. **WP-2: Spoken-Number Text Normalizer:**
   Implement a lightweight text preprocessor to convert natural numbers (e.g. `"twenty times fifteen"` $\rightarrow$ `"20 * 15"`), eliminating Failure Mode 2.
3. **WP-3: Live End-to-End Latency & Robustness Verification:**
   Benchmark the full pipeline (User Query $\rightarrow$ Neural Intent Classifier $\rightarrow$ SafeCalculator / StructuredVerifier $\rightarrow$ Response) across multi-turn conversational sessions to guarantee $<10$ ms latency and zero regressions.
