# Evidence-Based Neural Core Capability Inventory

## Batches 1301–1340: Rigorous Separation of Neural Learning vs. Deterministic Runtime Engineering

### 1. Executive Summary

ChakrView's runtime architecture integrates several subsystems: an autoregressive decoder core (`ChakrMicro`), a learned contrastive encoder (`ChakrBiEncoder`), a deterministic BM25 lexical index, regex-based claim parsers, and multi-turn context compaction.

Previous project iterations occasionally attributed system capabilities—such as grounded citation generation, exact plan synthesis, or false-premise refutation—to "neural reasoning." This audit provides an evidence-based inventory classifying every asserted capability into one of four verified categories:

- **Category A:** Demonstrated by the neural model itself (statistical/learned weights).
- **Category B:** Demonstrated by deterministic code, retrieval, rules, or extractive processing.
- **Category C:** Demonstrated only on synthetic or narrow benchmarks (memorization/overfitting risk).
- **Category D:** Not yet demonstrated or non-existent in the codebase.

---

### 2. Comprehensive Capability Matrix

| Capability Dimension | Category | Implementation / Evidence Path | Evaluation Method & Limitations | Confidence Level |
|---|---|---|---|---|
| **Contrastive Sentence Representation** | **A** | `chakrview/runtime/learned_embedding.py` (`ChakrBiEncoder`) | 128-dim dense representation trained via contrastive margin loss on 400 domain pairs. Successfully projects technical synonyms into proximity. Median latency: ~0.8 ms. | **HIGH** |
| **Document Retrieval & Ranking** | **B** | `chakrview/runtime/retrieval.py` (`BM25KnowledgeIndex`) | BM25 lexical index ($k_1=1.5, b=0.75$) achieves 95.45% Recall@1 on corpus. Dense bi-encoder reranking over BM25 candidate pools matches but does not exceed BM25 alone. | **VERY HIGH** |
| **Evidence Span Extraction** | **B** | `LocalKnowledgeAssistant._extract_concise_evidence` | Deterministic token-overlap sentence scoring over retrieved chunks. No neural extractive reader head exists. | **CERTAIN (Code inspection)** |
| **False-Premise Contradiction & Refutation** | **B** | `LocalKnowledgeAssistant.ask` (attribute and action alignment rules) | 100% deterministic regex and number set comparison. No neural NLI (Natural Language Inference) or contradiction classification model exists in ChakrView. | **CERTAIN (Code inspection)** |
| **Multi-Turn Context & Pronoun Binding** | **B** | `chakrview/runtime/session.py` (`ConversationContextManager`) | Deterministic concatenation of prior user turns onto current query before BM25 retrieval. | **CERTAIN (Code inspection)** |
| **Autoregressive Token Generation (Language Modeling)** | **C** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | Small 10.8M parameter Transformer (`ChakrMicro`, 6 layers, 384 dim). Generates repetitive tokens when unconstrained. Only verified on synthetic token sequence reconstruction. | **MODERATE** |
| **Plan DSL Syntax Generation** | **C** | `chakrview/planner/` | Generates valid Plan DSL syntax on narrow synthetic prompt templates. Fails slot binding and generalization when syntax deviates from training distribution. | **HIGH** |
| **Multi-Step Deductive Reasoning** | **D** | None | No chain-of-thought, reinforcement learning, or multi-step deduction is learned or implemented. | **CERTAIN (Absence of code)** |
| **Mathematical Calculation** | **D** | None | No arithmetic or quantitative calculation module in neural core; purely extractive regex numbers. | **CERTAIN (Absence of code)** |
| **Coding Ability / Code Synthesis** | **D** | None | No pre-training on code corpus; tokenizer vocab (4096 tokens) lacks programming language representations. | **CERTAIN (Absence of code)** |
| **Long-Context Reasoning (>512 tokens)** | **D** | None | Context ceiling in `ChakrMicro` is hardcoded to 512 tokens. Sliding window and compaction are handled deterministically outside the model. | **CERTAIN (Code inspection)** |

---

### 3. Deep-Dive on Specific Subsystems

#### 3.1. Language Modeling and Generation (`ChakrMicro`)
- **Architecture:** 6-layer Transformer decoder, 384 hidden dimension, 6 query heads, 2 KV heads (grouped-query attention), 1536 intermediate FFN dimension, 4096 vocabulary.
- **Trained Parameters:** 10.8 million parameters.
- **Canonical Checkpoint:** `training_checkpoint_step_10000.pt` (SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Current State:** The model learns basic token distribution priors for the narrow training vocabulary, but lacks open-ended generation capacity, coherent factual generation, or factual grounding without external retrieval.

#### 3.2. Contrastive Semantic Encoder (`ChakrBiEncoder`)
- **Architecture:** 2-layer Transformer encoder, 128 hidden dimension, mean pooling over token embeddings.
- **Trained Parameters:** ~350,000 parameters.
- **Current State:** Successfully separates contrastive positive pairs from hard negatives within its technical training domain (e.g., associating "worker threads" with "CPU execution pool"). However, because candidate pools in the local knowledge assistant are already dominated by exact technical terms (port numbers, version tags, parameters), BM25 achieves superior precision with lower compute.

#### 3.3. Refutation and Verification
- **Current State:** It is vital to state clearly that **no neural NLI model is currently present in ChakrView**. The assistant refutes false premises by parsing numerical attributes (`head`, `price`, `sla`, `ram`, `port`) and prohibited action keywords (`sudo`, `root`, `unauthenticated`) against verified evidence chunks extracted by the deterministic retriever.

---

### 4. Strategic Implications for Project Roadmapping

1. **Stop Expanding Regex Heuristics:** Adding ever more regular expressions to simulate understanding of syntax, units, or negation produces an increasingly fragile rule engine with diminishing returns.
2. **Focus Next Engineering Sprint on the Neural Core:** The indigenous neural core (`ChakrMicro`) requires rigorous, reproducible evaluation on next-token prediction and perplexity on held-out technical text, before attempting complex downstream agentic tasks.
