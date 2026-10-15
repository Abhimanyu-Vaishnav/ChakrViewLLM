# Capability Attribution & Decoupled Evaluation — Batches 281–310

**Milestone:** Batches 281–310  
**Target Architecture:** ChakrMicro (3,443,136 parameters, 6 layers, $d_{model}=192$, 6 heads)  
**Evaluated Artifacts:**
- Canonical Checkpoint: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (SHA-256: `96357705...471ee`)
- Trained Candidate Checkpoint: `experiments/batch281_310_neural_capability/checkpoints/chakrmicro_capability_candidate_step70.pt` (SHA-256: `3b6b98b821a0c1e73aa18cae25ffc866e027b6c40acd6fdb4673e25b02b6c2fe`)
- Relational Pointer Checkpoint: `experiments/batch231_240_multihop_reasoning/checkpoints/cross_attention_pointer_step1000.pt`
- Benchmark Suites:
  - Multi-Task Neural Capability Test Suite (Held-Out Test Split, Batch 286)
  - 3-Hop Multi-Hop Relational Benchmark (100 frozen records across Astronomy, Materials, Computing)
  - 200-Question Independent Held-Out Benchmark (Batch 274)

---

## 1. Decoupled Performance by Evaluation Arm

| Evaluation Arm | Method / Mechanism | Task Type | Accuracy / Metric | Primary Observation / Attribution |
| :--- | :--- | :--- | :--- | :--- |
| **Arm 1: Canonical Neural Core (Baseline)** | Greedy vocabulary decoding ($T=0$) | Neural Capability Test | Val Loss: 8.4447<br>Clean Term: 0.0%<br>EM: 0.0% | Reverts to Wikipedia table attractors (`\|200\|...\|`) due to pretraining distribution bias. |
| **Arm 2: Trained Candidate Neural Core (Step 70)** | SFT with Warmup + Cosine Annealing + Target Loss Masking | Neural Capability Test | Val Loss: 5.9164 ($-2.53$ nats)<br>Clean Term: **60.0%**<br>EM: 0.0% | Learns conversational turn termination on `<\|im_end\|>`, escaping tabular pretraining attractors without forgetting general English/Hindi tokens. |
| **Arm 3: Cross-Attention Pointer Head** | Learned cross-attention query-to-context scoring | 3-Hop Relational Grounding (100 items) | Exact Match: **51.0%**<br>Distractor Errors: 17% | Successfully chains 3 document hops (e.g. Unit Circle -> Tau -> Pi -> 3.14159) purely from context embeddings without hallucination. |
| **Arm 4: Deterministic Tools & DSLs** | AST math parsing, unit conversion, code generator | Arithmetic, Units, Syntax | Exact Match: **100.0%**<br>(68/68 math & units) | Zero dynamic `eval()`. Rejects dimensional mismatches (km to kg) and absolute zero violations. |
| **Arm 5: Full Integrated Runtime** | Cognitive Router + Session Adapter + Knowledge Base | 200 Fresh Held-Out Queries | Overall: **95.0%** (190/200)<br>Avg Latency: 7.19 ms | Dispatches queries to their appropriate specialized components with full provenance tracking. |

---

## 2. What Genuinely Improved in the Neural System

1. **Target Loss Masking and Delimiter Alignment:**
   - Previous SFT approaches stripped prompt delimiters (`prompt.strip()`) and computed unweighted loss across 88% template tokens, leading to pseudo-convergence and generation collapse (`|277`).
   - By preserving trailing newlines and calculating loss exclusively on response tokens (`shift_labels != -100`), the candidate model reduced validation cross-entropy by **2.53 nats** (Perplexity: $4650.19 \to 371.06$).
2. **ChatML Protocol Adherence & Termination:**
   - The canonical baseline produced 0% `<|im_end|>` terminations, looping through Wikitext table rows.
   - The candidate model achieved **60.0% clean termination** with no repetitive looping, while retaining valid generation on standard English and Hindi prompts.
3. **Multi-Hop Relational Pointer Reasoning:**
   - Verified that the neural cross-attention pointer head achieves **51.0% exact match** on complex 3-hop reasoning chains across science, computing, and astronomy, drastically outperforming the single-hop deterministic extractor baseline (20.0%).

---

## 3. What Remains Deterministic, Retrieval-Based, or Tool-Assisted

To prevent overstating neural capabilities:
- **Exact Multi-Step Math:** Evaluated by `CompoundArithmeticEngine` and `CalculatorTool`, NOT by neural arithmetic generation.
- **Physical Unit Conversions:** Evaluated by dimensional lookup tables, rejecting non-physical values deterministically.
- **Python AST Syntax Validation:** Validated via Python's standard `ast.parse` and sandboxed subprocess execution.
- **Internet Research Governance:** Gated by `PermissionGatedResearchWorkflow` finite-state machine requiring explicit human approval.
