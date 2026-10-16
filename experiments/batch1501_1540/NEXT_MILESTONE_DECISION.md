# Next Milestone Decision: Batches 1541–1580

## Strategic Roadmap Following the Neuro-Symbolic Tool Integration

### 1. Empirical Grounding for the Next Decision

The completion of Batches 1501–1540 establishes a highly functional, reliable baseline for ChakrView:
- **Restricted Arithmetic:** 100.0% accurate, sub-millisecond, zero vulnerability.
- **Tool Routing & Classification:** 98.0% accuracy across 6 canonical operational routes.
- **Entity-Attribute Verification:** 91.0% accuracy, cleanly distinguishing `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, and `CONFLICTING_EVIDENCE`.
- **End-to-End Orchestration:** 92.0% accuracy on complex multi-tier user queries.

---

### 2. Remaining Bottlenecks and Failure Analysis

While tool dispatch and structured verification operate with high precision, the remaining failures point directly to the next critical engineering bottlenecks:

1. **Extraction Ambiguity on Dense Multi-Attribute Passages:**
   - When a single documentation line contains multiple parameters (e.g. `BM25: k1=1.5, b=0.75` or `Mesh: discovery port 5353, data port 8443`), unstructured extraction can associate a value with the wrong attribute key if the sentence structure is inverted.
2. **Multi-Turn Context Resolution for Tool Execution:**
   - In conversational sessions, users often ask follow-up questions referencing previous tool calculations (e.g., Turn 1: `"Calculate 50 * 20"`, Turn 2: `"Now divide that by 4"`).
   - Currently, follow-up anaphora resolution in `local_knowledge_assistant.py` is tailored for document retrieval queries rather than chaining tool inputs across conversational turns.
3. **Absence of Sandboxed Python Code Execution for Algorithmic Tasks:**
   - While arithmetic calculation is now 100% solved via `SafeCalculator`, algorithmic and programming tasks (Track D in Batch 1461–1500) remain unevaluable/unsupported because ChakrMicro cannot synthesize code and no sandboxed Python code runner exists in the governed runtime.

---

### 3. Recommended Roadmap for Batches 1541–1580

#### Milestone Theme:
**Multi-Turn Tool Chaining, State-Governed Context Resolution & Sandboxed Code Execution**

#### Primary Objectives:
1. **Multi-Turn Tool Memory & Chaining:**
   - Extend `ConversationContextManager` to track previous tool results (e.g. `last_tool_output`, `last_arithmetic_value`).
   - Implement anaphora resolution for arithmetic (e.g. `"divide that by 4"` resolves to `1000 / 4 = 250`).
2. **Restricted Subprocess Code Sandbox:**
   - Implement a safe, ephemeral Python code execution sandbox (`SandboxedPythonTool`) utilizing resource limits (CPU time, memory caps, network isolation) for programmatic data transformations.
3. **Structured Proposition Parser:**
   - Upgrade the natural-language claim extractor into a formal dependency-aware proposition parser, eliminating attribute cross-talk in multi-parameter sentences.
4. **Target Benchmark Metrics:**
   - Multi-turn tool chaining success rate: $\ge 85\%$.
   - Structured verification accuracy: raise from $91.0\% \to \ge 96.0\%$.
   - Zero regression across existing arithmetic (100%) and retrieval (86.7%) benchmarks.

---

### 4. Constraints Maintained
- Canonical checkpoint `training_checkpoint_step_10000.pt` remains unmodified.
- Adapted checkpoint `chakrmicro_domain_adapted_experimental.pt` remains experimental.
- No model re-training or fine-tuning will be performed until runtime tool chaining and sandboxed execution are fully validated.
