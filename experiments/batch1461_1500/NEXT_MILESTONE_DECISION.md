# Next Milestone Decision: Batches 1501–1540

## Strategy Following the Independent Capability Audit

### 1. Empirical Grounding for Next Steps

The Batches 1461–1500 capability audit revealed two undeniable realities:

1. **The Strengths:**
   - **Language Modeling (Track A):** ChakrMicro functions effectively as a statistical language model, and domain adaptation yields measurable, generalizing perplexity reductions ($218 \to 127$).
   - **Deterministic Knowledge Scaffolding (Track B, E, Controls):** The BM25 retrieval, extractive answer verification, and threshold abstention logic provide reliable, 100% cited answers ($75\%$ accuracy) and robust unanswerable abstention ($100\%$).

2. **The Weaknesses (The Bottlenecks):**
   - **Zero Arithmetic & Coding (Tracks C & D):** The neural model cannot perform numeric calculation or write functions ($0.0\%$ accuracy on both).
   - **Zero Parametric Recall (Track B):** The model cannot answer questions from memory without prompt context ($0.0\%$).
   - **Brittle Refutation (Controls):** Refutation currently succeeds only when deterministic regexes match specific attributes ($50.0\%$), leaving unaligned premises unrefuted.

---

### 2. Strategic Options Considered

#### Option A: Large-Scale Supervised Fine-Tuning (SFT) for Math and Code
- *Hypothesis:* Fine-tune ChakrMicro on synthetic arithmetic problems and Python function pairs.
- *Risk/Downside:* At ChakrMicro's compact parameter scale (~a few million parameters), forcing multi-digit arithmetic and code synthesis without chain-of-thought or symbolic execution is known from literature to induce memorization rather than generalized reasoning, while risking catastrophic forgetting on language modeling.
- *Verdict:* **REJECTED.**

#### Option B: Continue Domain Adaptation on More Text
- *Hypothesis:* Adapt ChakrMicro for more steps on additional technical text corpora.
- *Risk/Downside:* Returns are diminishing. Perplexity is already audited and proven to transfer well (PPL 127.70). Further text adaptation will not unlock arithmetic or coding abilities.
- *Verdict:* **REJECTED.**

#### Option C (RECOMMENDED): Hybrid Neuro-Symbolic Tool Routing & Robust Evidence Refutation
- *Objective:* Accept that ChakrMicro is a statistical sequence model, and build a principled **Neuro-Symbolic Tool Gate** and **Formal Evidence-Fact Verification Pipeline**:
  1. **Deterministic Tool Dispatch for Mathematics:** When a user query requires quantitative calculation, route the computation directly to an isolated Python calculator rather than asking the neural model to predict number tokens.
  2. **Formal Attribute-Checking for Refutation:** Upgrade the refutation engine from brittle regexes to a structured proposition extractor that compares query claims against retrieved entity attributes, raising refutation accuracy from 50% to >85%.
  3. **Preserve Checkpoint Integrity:** Maintain the experimental candidate as an auxiliary language modeling head while keeping production pipelines fully deterministic and grounded.

---

### 3. Recommended Roadmap for Batches 1501–1540

1. **Milestone Theme:** *Neuro-Symbolic Tool Gating and Attribute-Grounded Refutation*.
2. **Key Deliverables:**
   - Implement query intent classification to route arithmetic queries to the Python math evaluator.
   - Replace regex-based refutation in `chakrview/knowledge/refutation.py` with structured entity-relation comparison.
   - Benchmark the upgraded assistant on the frozen 200-task audit suite.
   - Target metrics:
     - Math task pass rate: $0\% \to >95\%$ (via tool routing).
     - Refutation accuracy: $50\% \to >85\%$ (via attribute comparison).
     - Language modeling and retrieval latencies: zero regression.
