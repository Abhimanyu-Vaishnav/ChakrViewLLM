# ChakrView Step 51: Evaluation Protocol & Objective Metrics

- **Version**: 1.0.0
- **Scope**: Step 51 — Coding Language Acquisition & Project Arena Foundation
- **Target**: Separation of Model Quality, Engine Quality, and Arena Quality

---

## 1. Tripartite Quality Framework

To avoid conflating raw model statistical learning with execution infrastructure reliability, ChakrView establishes a formal three-tier separation of concerns:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TRIPARTITE EVALUATION FRAMEWORK                 │
├─────────────────────────┬──────────────────────┬───────────────────────┤
│    1. MODEL QUALITY     │  2. ENGINE QUALITY   │   3. ARENA QUALITY    │
├─────────────────────────┼──────────────────────┼───────────────────────┤
│ • Cross-Entropy Loss    │ • KV Cache Validity  │ • Workspace Isolation │
│ • Perplexity (PPL)      │ • Sampling Frequencies│ • Timeout Enforcement │
│ • Token Distribution    │ • Forward Latency    │ • Subprocess Sandbox  │
│ • Syntax AST Validity   │ • TTFT (Time to 1st) │ • Stdout/Stderr Trap  │
│ • Functional Pass Rate  │ • Memory Peak (RSS)  │ • Failure Diagnosis   │
└─────────────────────────┴──────────────────────┴───────────────────────┘
```

---

## 2. Metric Formulations & Mathematical Contracts

### Tier 1: Model Quality Metrics

1. **Validation Loss & Perplexity**:
   $$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^T \log P(x_t \mid x_{<t}), \quad \text{PPL} = \exp(\mathcal{L})$$

2. **AST Syntax Validity Rate ($S_{\text{valid}}$)**:
   $$S_{\text{valid}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{ast.parse}(c_i) \text{ succeeds})$$
   Measures the proportion of generated code completions that are syntactically valid in the target language.

3. **Functional Test Pass Rate ($R_{\text{pass}}$)**:
   $$R_{\text{pass}} = \frac{N_{\text{passed}}}{N_{\text{total\_tests}}}$$
   Measures whether code executes without exceptions and satisfies expected assertion invariants.

4. **Lexical Repetition Ratio ($r_{\text{rep}}$)**:
   $$r_{\text{rep}} = 1.0 - \frac{|\text{unique } n\text{-grams}|}{|\text{total } n\text{-grams}|}$$
   Monitors degenerative looping (e.g. repeated keywords or tokens).

---

### Tier 2: Engine Quality Metrics

1. **Time to First Token (TTFT)**:
   Latency (in ms) from input prompt handoff to the emission of the first generated token.
   Target on standard CPU: $\le 15.0\text{ ms}$.

2. **Generation Throughput**:
   Tokens generated per second during autoregressive decoding.
   Target on standard CPU: $\ge 35.0\text{ tokens/sec}$.

3. **Zero Weight Mutation ($\Delta W$)**:
   $$\Delta W = \sum_{p \in \Theta} ||W_{p,\text{post}} - W_{p,\text{pre}}||_2 \equiv 0.0$$
   Guarantees that inference never alters frozen model weights.

---

### Tier 3: Arena Quality Metrics

1. **Workspace Isolation Rate ($I_{\text{iso}}$)**:
   $100\%$ required. Asserts zero leaked temporary files, zero environment variable contamination, and complete directory containment.

2. **Timeout Enforcement Rate ($T_{\text{enforce}}$)**:
   $100\%$ compliance. Any test execution exceeding $5.0$ seconds wall-clock must be forcefully killed and classified as `TIMEOUT`.

3. **Failure Classification Accuracy**:
   Properly categorizes execution results into:
   - `SYNTAX_ERROR`: Code fails AST parsing.
   - `IMPORT_ERROR`: Missing or unauthorized import attempt.
   - `ASSERTION_FAILURE`: Test executed but an assertion was violated.
   - `RUNTIME_ERROR`: Unhandled exception during execution.
   - `TIMEOUT`: Execution exceeded allotted wall-clock seconds.
   - `SUCCESS`: All tests executed and passed cleanly.

4. **Iteration Improvement Delta ($\Delta_{\text{iter}}$)**:
   $$\Delta_{\text{iter}} = R_{\text{pass}}^{(k)} - R_{\text{pass}}^{(k-1)}$$
   Measures whether subsequent repair attempts improve test pass rates without introducing regressions.
