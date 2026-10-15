# WP 999–1004: Core Neural Capability Growth Audit

## 1. Has the Core Neural Model Itself Improved?

To answer this question truthfully, we must compare the **original Canonical Checkpoint** (`step_10000.pt`) against subsequent fine-tuned candidates on **raw neural tasks without deterministic scaffolding**:

### 1.1 Evaluated Dimensions

1. **Pretraining Loss vs Fine-Tuning Loss**:
   - Canonical step 10,000 checkpoint was trained on diverse general web and language tokens, reaching an NLL loss of **~2.18**.
   - Subsequent candidates (Batches 541, 741, 821, 861) reached lower training losses (0.10–0.40) **only because they were trained on tiny, repetitive synthetic templates** (20 to 50 examples) with restricted grammar.
2. **General Language Coherence**:
   - When prompted with natural English sentences (`"The capital of France is"`, `"Question: What is 5 multiplied by 6?"`), the **Canonical Step 10,000 checkpoint is significantly more coherent** than any planner candidate.
   - Every fine-tuned planning candidate experienced catastrophic forgetting, outputting raw DSL delimiters (`",opop=strip} | deps=[]\nST"`) even when asked standard factual questions.
3. **Multi-token Argument Extraction**:
   - The core transformer itself has not developed intrinsic operand grounding; it required peripheral classification heads (`SpanGroundingHead`) or deterministic left-to-right sorting to reliably extract operands like `600` or `1024`.

---

## 2. Definitive Finding on Core Capability Growth
**The core neural transformer has NOT grown in general capability since pretraining Step 10,000.**
- All reported accuracy gains in Batches 741–980 occurred in **external task-specific adapter heads** (intent routers, span grounding heads, topology classifiers) and **deterministic execution wrappers** (`PlanDSLValidator`, `UnifiedPlan.materialize_dsl()`, positional sorting).
- The underlying transformer backbone (`d_model=192`, 6 layers, 6 heads, ~3.8M parameters) remains small, undertrained on diverse reasoning, and prone to severe catastrophic forgetting whenever fine-tuned on narrow tasks without broad replay.
