# WP 1011–1014: Architectural & Strategic Bottlenecks

## 1. Why Is the System Hitting a Wall in Autoregressive Planning?

### Bottleneck 1: Model Capacity & Training Data Scale
- **Current Core Model**: 6 layers, 6 attention heads, $d_{\text{model}} = 192$, ~3.8 million parameters.
- **The Reality**: An ultra-compact 3.8M model trained to 10,000 steps does not have sufficient representational capacity to simultaneously:
  1. Maintain general language coherence.
  2. Parse arbitrary complex English and Hinglish user instructions.
  3. Predict variable-length directed acyclic task graphs (DAGs).
  4. Perform precise entity grounding without subword fragmentation.
- Expecting a 3.8M parameter transformer to autonomously emit perfect, long-horizon structured Plan DSL text without external scaffolding is **unrealistic at this scale**.

### Bottleneck 2: Subword BPE Tokenizer Misalignment
- The vocabulary of 4,096 tokens splits multi-digit numbers (`600` $\to$ `['6', '00']`).
- When generating token-by-token autoregressively, the grammar state machine cannot distinguish when a multi-token literal is complete, causing severe truncation bugs.

### Bottleneck 3: Synthetic Micro-Datasets vs True Learning
- Training sets in Batches 741–980 contained only 20 to 50 examples.
- Models trained on 30 examples do not learn generalizable reasoning; they memorize syntactic transitions and experience catastrophic forgetting on everything else.

### Bottleneck 4: Heuristic Scaffolding Masking Model Limitations
- Each milestone added an extra layer of heuristic code (`grammar_constrained_decoder`, `optimized_grammar_decoder`, `unified_plan_fusion`, `neural_topology_planner`) to force compliance with a complex DSL.
- The DSL was originally invented to make tool use structured, but generating the DSL itself became harder than the actual downstream task!
