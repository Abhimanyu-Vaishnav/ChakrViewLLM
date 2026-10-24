# ChakrView Experiment 13: Research Report
## Native Human Conversation Understanding, Context Learning, and Self-Correction

### 1. Executive Summary

Experiment 13 investigated whether ChakrView's native 3.44M parameter causal language model (ChakrMicro v0.1) can learn multi-turn conversational turn-taking, context grounding, and error correction without external APIs, hosted LLMs, or deterministic controllers.

In the preflight baseline smoke test, the frozen canonical Step-10000 model completely failed conversational probes ($0.0\%$ relevant responses, $23.3\%$ repetitive n-gram attractor loops) because it had only been pre-trained on generic web/code text without instruction or conversational turn formatting.

In Experiment 13, an isolated multi-turn conversational dataset across 20 diverse categories was generated with strictly disjoint splits (180 Train, 60 Val, 60 Test). The candidate model was fine-tuned for 120 steps on local CPU using prompt-masked causal loss (gradients strictly on assistant responses). Additionally, a controlled self-correction loop was deployed to detect concrete failure traces, verify corrections against independent AST parsers, and quarantine unverified feedback.

### Final Experiment Status:
`TRAINING STABLE — GENERALIZATION NOT DEMONSTRATED`

---

### 2. Empirical Findings & Quantified Results

#### Held-Out Evaluation ($N=60$ Test Examples across 20 Categories):
| Metric | Frozen Baseline (Step-10000) | Trained Candidate (Step-100 Best) | Delta / Change |
| :--- | :---: | :---: | :---: |
| **Response Relevance** | 0 / 60 (0.00%) | 11 / 60 (18.33%) | **+18.33%** |
| **Repetitive Attractor Loops** | 14 / 60 (23.33%) | 10 / 60 (16.67%) | **-6.66%** |
| **Empty / Aborted Outputs** | 4 / 60 (6.67%) | 0 / 60 (0.00%) | **-6.67%** |
| **Inference Time (60 samples)** | 37.70 s | 22.11 s | **-15.59 s (Faster termination)** |

#### Detailed Failure Autopsy: Why Generalization Was Not Demonstrated
While the candidate model improved relevance from $0\%$ to $18.33\%$ and eliminated empty outputs, a forensic analysis of the raw predictions revealed:
1. **Memorized Training Attractors**:
   - In `explanation_and_coding`, the model learned the Python syntax template `def ...:\n return ...`, but it repeatedly emitted `def is_positive_train_0:\n return x_0` on test prompts that asked for `is_positive_test_0`. It memorized the training token sequence rather than abstracting the prompt variable.
2. **Context Misattribution**:
   - In `ambiguity_clarification`, the model ceased generating generic math loops and began generating conversational phrases (`"The function is not be a..."`), but failed to extract the target pod name from the user turn.
3. **Small Model Capacity Constraint**:
   - At 3.44M parameters, the model lacks the representational depth to simultaneously track multi-turn dialogue history, resolve anaphora, and perform zero-shot variable binding without explicit slot pointer heads.

---

### 3. Controlled Self-Correction Research Loop

A verified self-correction engine was implemented and audited:
- **Total Audited Candidate Predictions**: 60
- **Detected Concrete Failures**: 13
  - **Function Name Memorization (3 cases)**: Detected via prompt regex. Proposed correction replaced the memorized name with the test target. Independently verified using Python `ast.parse()`. Status: **`ACCEPTED_FOR_TRAINING` (3 traces)**.
  - **Repetitive Attractor Loops (10 cases)**: Quarantined because no deterministic AST/symbolic verifier was available to guarantee truth without oracle supervision. Status: **`QUARANTINED_UNVERIFIED` (10 traces)**.
- **Scientific Takeaway**: Verified self-correction successfully prevents ungrounded hallucinations from polluting future training data by enforcing independent symbolic verifiers.

---

### 4. Integrity and Cryptographic Audit

All canonical, runtime, and historical assets remain untouched and byte-identical:
- `canonical_checkpoint_step_10000`: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` [VERIFIED]
- `tokenizer_vocab`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` [VERIFIED]
- `tokenizer_merges`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` [VERIFIED]
- `runtime_session`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` [VERIFIED]

All 5 verification test suites in `test_conversation_experiment.py` passed with exit code 0.
Zero Git mutations, commits, merges, or pushes were performed.

---

### 5. Remaining Limitations & Recommended Next Step

**Primary Limitation**:
Native 3.44M causal LM fine-tuning alone is insufficient for open-domain variable binding and complex multi-turn conversation. The model memorizes training surface strings instead of binding dynamic conversation entities.

**Recommended Next Step**:
Combine the conversational role-formatting pipeline developed here with **Field-Conditioned Pointer Attention** (Milestone 1). By enabling the neural pointer head to bind conversation entities directly from context tokens, the model can achieve accurate context reproduction without memorization collapse.
