# Capability Evaluation Protocol: Batches 1461–1500

## Independent Capability Audit: From Token Prediction to Executable Tasks

### 1. Mission Mandate and Scope

Batches 1341–1460 established that `ChakrMicro` (3.44M unique parameters) learns valid token-level transitions, achieving a 41.5% perplexity reduction on held-out technical text after domain adaptation.

However, **lower cross-entropy loss does not imply higher-order intelligence**. The objective of Batches 1461–1500 is to execute a rigorous, independent capability audit evaluating what `ChakrMicro` can actually accomplish across six distinct tracks:

- **Track A:** Language Modeling (Held-out token prediction, cross-entropy, perplexity, top-k accuracy).
- **Track B:** Factual Recall & Evidence Grounding (Exact answers, citation validity, abstention on unanswerable queries, false-premise refutation).
- **Track C:** Arithmetic & Mathematical Reasoning (Exact numeric evaluation, multi-step word problems, failure attribution).
- **Track D:** Coding & Execution (Python function generation, syntax validity, sandboxed test pass rate).
- **Track E:** Generalization & Robustness (Paraphrased query pairs, entity-swapped prompts, perturbation sensitivity).
- **Track F:** Component Attribution (Learned neural weights vs. BM25 retrieval vs. regex heuristics vs. Python wrappers).

---

### 2. Pre-Declared Hypotheses and Pass/Fail Criteria

Before running any test or benchmark, we freeze the evaluation criteria:

| Track | Evaluation Target | Hypothesis / Expected Neural Performance | Pass / Demonstrated Criterion | Failure / Non-Demonstrated Criterion |
|---|---|---|---|---|
| **Track A** | Language Modeling | Candidate maintains superior token loss ($PPL < 135$) across held-out tokens. | PPL $< 135.0$, Top-1 $> 16.0\%$. | PPL $> 160.0$, Top-1 $< 12.0\%$. |
| **Track B** | Factual Recall (Standalone Model) | Without external retrieval, ChakrMicro will hallucinate or fail exact entity recall ($< 15\%$ accuracy). | Model alone: Accuracy $> 50\%$. | Model alone: Accuracy $< 20\%$ (Confirms reliance on retrieval). |
| **Track B** | Grounded Pipeline (Assistant) | Assistant pipeline achieves $> 85\%$ decision accuracy and $> 90\%$ citation validity. | Decision Acc $> 85\%$, Citations $> 90\%$. | Decision Acc $< 75\%$ or Citations $< 80\%$. |
| **Track C** | Mathematics & Arithmetic | Model will fail exact arithmetic ($< 10\%$ pass rate) due to absence of tool calling or chain-of-thought training. | Exact accuracy $> 30\%$. | Exact accuracy $< 10\%$ (Confirms absence of arithmetic reasoning). |
| **Track D** | Coding & Execution | Model can generate valid Python syntax for small completions ($> 40\%$), but functional test pass rate will be low ($< 10\%$). | Test pass rate $> 25\%$. | Test pass rate $< 10\%$ (Confirms no functional code synthesis). |
| **Track E** | Generalization & Paraphrases | System maintains performance on paraphrased pairs with consistency $> 80\%$. | Pair consistency $> 75\%$. | Pair consistency $< 50\%$. |
| **Track F** | Component Attribution | Precise attribution showing all fact extraction and refutation are deterministic. | Strict attribution matrix documented. | Ambiguous neural claims. |

---

### 3. Checkpoint Identities & Environment Invariants

- **Canonical Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  
  - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Candidate Checkpoint:** `experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt`  
  - SHA-256: `15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139`
- **Protected File:** `chakrview/runtime/session.py` (Must remain untouched).
- **Execution Rule:** Zero training or fine-tuning permitted in this mission.
