# Contamination and Leakage Audit: Batches 1461–1500

## Verification of Task Independence and Evaluation Isolation

### 1. Separation of Tracks and Checkpoint Independence

To guarantee objective capability measurement, the evaluation suite was subjected to strict contamination checks:

1. **Model Checkpoint Isolation:**
   - Both the canonical checkpoint (`Step 10,000`, SHA-256: `96357705a6...`) and the domain-adapted checkpoint (`Step 10,080`, SHA-256: `15d48176...`) were created in prior milestones and were frozen prior to this milestone's task generation.
   - Neither checkpoint has ever had backpropagation, gradient descent, or prompt-tuning exposure to the 200 tasks in [capability_audit_200.json](file:///D:/Project/ChakrView/experiments/batch1461_1500/data/capability_audit_200.json).

2. **Coding Task Isolation (Track D):**
   - The 40 coding unit tests were executed in a sandboxed execution harness with zero network, filesystem mutation, or cross-process state leakage.
   - Test assertions are completely withheld from prompt context: the model receives only the function header and docstring comment (e.g. `def add(a, b):\n`), while assertions are executed independently against the generated function definition.

3. **Arithmetic Task Isolation (Track C):**
   - No calculators, external Python `eval()`, or tool bindings are provided to the model during the model-only generation phase. The model must autoregressively generate the answer directly from prompt tokens.

4. **Factual Grounding Isolation (Track B):**
   - When evaluating "ChakrMicro alone" (Configuration 1), all external retrieval indices (BM25 and bi-encoder) are completely disabled. The model is presented only with the question string.
   - When evaluating "ChakrMicro + Local Knowledge Assistant" (Configuration 2), BM25 candidate retrieval and evidence verification are enabled, enabling precise component attribution.
