# WP 981–986: Architecture and Weight Lineage Audit

## 1. Architectural Lineage Tree

All specialized research models trace back to the canonical pretraining checkpoint:

```
[Pretraining Step 10,000]
training_checkpoint_step_10000.pt (SHA-256: 96357705a683...)
  │  (ChakrMicro: 6 layers, 6 heads, d_model=192, vocab=4096, causal transformer)
  │
  ├──> [Batches 541–580]: candidate_step250_intent_understanding.pt
  │      Fine-tuned for structured JSON intent prediction
  │
  ├──> [Batches 581–620]: candidate_step250_span_pointer.pt
  │      Added 2-span start/end extraction heads
  │
  ├──> [Batches 701–740]: candidate_step150_neural_planner.pt
  │      Added fixed-step subtask classification heads
  │
  ├──> [Batches 741–820]: candidate_step300_seed101_grammar_planner.pt
  │      Autoregressive fine-tuning on raw Plan DSL text
  │
  ├──> [Batches 821–860]: candidate_step300_seed101_pointer_planner.pt
  │      Added PointerGeneratorHead (learned copy gate + attention over prompt)
  │
  └──> [Batches 861–980]: candidate_step250_seed42_span_grounding.pt
         Added SpanGroundingHead (4 slot query embeddings, start/end linear, type classifier)
```

---

## 2. Key Finding: Catastrophic Forgetting in Specialized Candidates

Evaluating raw greedy generation across general prompts reveals that every fine-tuned candidate suffered severe **catastrophic domain collapse**:
- Prompt: `"The capital of France is"`
  - **Canonical Model**: Emits natural English continuation (`"land. The Sea is the Earth of the Arcean"`).
  - **Plan Candidates**: Emits corrupted DSL syntax (`"',opop=strip} | deps=[]\nST"`).
- Prompt: `"Question: What is 5 multiplied by 6?\nAnswer:"`
  - **Canonical Model**: Recognizes Q&A format and begins step-by-step arithmetic (`"\nStep-by-Step Solution:\nThe total number "`).
  - **Plan Candidates**: Emits DSL fragments (`"| deps=[]\nSTEP 0: text_transform |"`).

### Strategic Insight:
The fine-tuning runs in Batches 741–980 did **not expand the core intelligence** of ChakrMicro. Instead, they destroyed the model's general language capabilities to overfit a tiny, specialized DSL grammar subset.
