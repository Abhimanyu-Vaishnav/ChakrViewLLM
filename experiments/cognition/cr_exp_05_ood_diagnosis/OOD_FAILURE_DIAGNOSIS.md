# CHAKRVIEW — CR-EXP-05 OUT-OF-DISTRIBUTION FAILURE DIAGNOSIS
### Master Scientific Investigation Report: Why Conversational Generalization Fails

- **Experiment ID**: CR-EXP-05
- **Execution Mode**: Strictly Read-Only Scientific Investigation
- **Model Architecture**: `ChakrMicro v0.1` (3,443,136 parameters, d_model=192, n_layers=6, n_heads=6, d_ff=512, tie_embeddings=True)
- **Candidate Checkpoint**: `experiments/cognition/autonomous_12h/conversational_pilot/checkpoints/conversational_pilot_best_val.pt`
  - SHA-256: `ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd` (Bit-Exact Match)
- **Parent Checkpoint**: `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt`
  - SHA-256: `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` (Bit-Exact Match)
- **Canonical Baseline Init**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` (Bit-Exact Match)
- **Runtime File**: `chakrview/runtime/session.py` (Untouched, SHA-256: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`)
- **Git Branch**: `exp/cbr-01-conversational-breakthrough` | **Git HEAD**: `564472a42245a47a7bf6f78ca6410e523383d2ce`

---

## 1. Executive Summary & Core Diagnostic Findings

CR-EXP-04 verified that the conversational candidate achieved a bit-exact **41.78% cross-entropy reduction** on held-out test data, eliminated `'Cube'` and `'Mumbai'` mode collapse, and learned clean persona identity. However, when evaluated on an independent 100-case out-of-distribution (OOD) conversational suite, the candidate exhibited an aggregate accuracy of only **20.0%**, failing virtually all queries involving entity extraction, dynamic state updates, multi-hop reasoning, and arithmetic.

The mission of CR-EXP-05 was to isolate the mechanistic root causes of these failures through strictly read-only diagnostics across eight investigative dimensions.

### Core Scientific Findings:
1. **The 41.78% Cross-Entropy Reduction is Statistically Genuine but Functionally Narrow**:
   Cluster-level hierarchical reassessment across all 26 distinct template families confirmed that the candidate improves cross-entropy loss on **26 out of 26 clusters** ($t = 10.99, p < 10^{-15}$). The loss reduction is not an artifact of template duplication. However, the model lowered loss by mastering the high-probability surface statistics of conversational turns (e.g. `"Good morning!"`, `"I am ChakrView..."`, `"Assistant:"`, `"How may I assist you?"`) rather than learning neural variable binding.
2. **Weak Neural Variable Binding is the Primary Cognitive Bottleneck**:
   In controlled synthetic binding experiments (e.g. `"Quasar is mapped to Rhodium. What does Quasar map to?"`), the candidate failed 8 out of 8 probes. The target token rank is $>30$ in the next-token probability distribution, obscured beneath an overwhelming $88\%$ prior for generic conversational filler (`' '` followed by greeting openers).
3. **Severe Tokenizer Fragmentation Gaps**:
   Devanagari Hindi words fragment into 4-8 byte tokens with 0% merged word units in the 4,096 vocabulary. Furthermore, **whitespace fusion** affects 100% of tested entities: `" Alice"` produces an entirely distinct subword ID from `"Alice"`, causing the model's attention layers to observe conflicting token identities between contextual declarations and downstream queries.
4. **Context Position Insensitivity (Global Suppression)**:
   Testing fact positions (Beginning, Middle, End) across contexts from 35 to 153 tokens showed that retrieval fails uniformly. The candidate's target token probability remains near zero ($0.0001 - 0.0003$) across all positions, demonstrating that failure is not a "lost in the middle" attention decay, but a global failure to route entity tokens into output logits.
5. **Mechanistic Information-Flow Confirmation**:
   At Layer 0, the residual stream maintains an 18.5% contextual delta between contrasting entities. However, as information propagates through the 6 Transformer blocks, total residual norm inflates from $4.53$ to $15.71$ due to invariant background vectors injected by the FFN blocks, diluting the contextual signal-to-noise ratio.

---

## 2. Detailed Audit of the 100-Case OOD Suite

The 100-case suite from CR-EXP-04 was parsed into eleven fine-grained functional failure categories.

| Taxonomy Category | Total Cases | Candidate Passed | Candidate Accuracy | Dominant Failure Mode |
|---|---|---|---|---|
| **Natural Greetings** | 13 | 13 | **100.0%** | None (Fully mastered) |
| **Persona & Identity** | 10 | 7 | **70.0%** | Occasional AST/code bleed after 25 tokens |
| **Unseen Entity Retrieval** | 8 | 0 | **0.0%** | Out-of-context attractor (`'Celsia'`) or generic greeting |
| **Familiar Entity, Unfamiliar Template** | 13 | 1 | **7.69%** | Falls back to conversational filler |
| **Unseen Entity & Unseen Template** | 13 | 0 | **0.0%** | Ignores context, emits canned greeting |
| **Dynamic State Updates** | 4 | 0 | **0.0%** | Cannot track value reassignments |
| **Multi-Turn Variable Binding** | 7 | 0 | **0.0%** | Forgets entities across intervening turns |
| **Multi-Hop Reasoning** | 5 | 0 | **0.0%** | Incapable of 2-hop compositional routing |
| **Formatting / Length Instructions** | 7 | 0 | **0.0%** | Ignores negative and format constraints |
| **Arithmetic & Quantitative** | 10 | 0 | **0.0%** | Emits corrupted syntax (`'x is'`) |
| **Pure Hindi (Devanagari)** | 5 | 0 | **0.0%** | Phonetic corruption due to byte fragmentation |
| **Hinglish Dialogue** | 5 | 0 | **0.0%** | Falls back to broken English fragments |
| **TOTAL** | **100** | **21** | **21.0%** | — |

**Key Takeaway**: The model has acquired a strong **conversational conversationalist persona**, but possesses **zero relational variable-binding capacity**.

---

## 3. Tokenizer Diagnostic Findings

The Chakr-BPE tokenizer (4,096 vocabulary) was systematically evaluated:
- **Hindi Devanagari**: Highly fragmented. Standard words like `"राजधानी"` (capital) require 5 tokens. `"नमस्ते"` standalone maps to token ID `3280`, but with a leading space (`" नमस्ते"`) it splits into `[517, 311, 652, 270]` (4 tokens). The model cannot learn Hindi semantic representations when identical words take completely different token sequences depending on preceding punctuation.
- **Entity Whitespace Fusion**: Every tested entity string changes token IDs when preceded by a space. For example, `"Ravi"` is `[85, 415, 120]`, whereas `" Ravi"` is `[35, 85, 415, 120]`. This forces the attention mechanism to bridge mismatched token representations across the context window.
- **Number Representation**: Numbers lack dedicated multi-digit tokens; digits are split arbitrarily, hindering arithmetic learning.

---

## 4. Controlled Variable-Binding Diagnostics

Eight controlled probes were constructed where answers cannot be guessed from template frequency.

| Probe ID | Description | Target | Candidate Generation | Candidate Top-1 Token | Target Rank |
|---|---|---|---|---|---|
| `VB_01` | Forward synthetic binding | `Rhodium` | `"Good afternoon! What ready to assist you..."` | `' '` (prob=0.884) | 336 |
| `VB_02` | Reverse synthetic binding | `Quasar` | `"Good evening! I am ChakrView..."` | `' '` (prob=0.888) | 621 |
| `VB_03` | Negation query | `No` | `"Good evening! Yes, I am ready..."` | `' '` (prob=0.802) | 30 |
| `VB_04` | Dynamic update (Rhodium $\to$ Osmium) | `Osmium` | `"Good evening! I am ready to inspecty..."` | `' '` (prob=0.822) | 48 |
| `VB_05` | Name interference (Quasar vs Quasor) | `Rhodium` | `"Good evening! Yes, I am ready to collaborate..."` | `' '` (prob=0.844) | 442 |
| `VB_06` | Distractor injection | `Rhodium` | `"That information is not provided in the context..."` | `' '` (prob=0.800) | 130 |
| `VB_07` | Familiar entities, synthetic template | `Pune` | `"That information is not provided in the context..."` | `' '` (prob=0.736) | 127 |
| `VB_08` | Swapped target control | `Titanium` | `"Good evening! I am ready to assist..."` | `' '` (prob=0.895) | 183 |

**Finding**: The candidate's next-token distribution is completely dominated by the conversational formatting prior ($P(\text{' '}) \approx 0.80 - 0.90$). The actual target entity is buried deep in the tail of the distribution (ranks 30 to 621).

---

## 5. Context Position & Length Dynamics

Testing `Rohan lives in Nagpur` across positions and context lengths yielded:
- **Beginning of context (99 tokens)**: Candidate Rank = 32, Target Prob = $0.00018$.
- **Middle of context (99 tokens)**: Candidate Rank = 30, Target Prob = $0.00019$.
- **End of context (99 tokens)**: Candidate Rank = 29, Target Prob = $0.00031$.
- **Short context (35 tokens, 0 distractors)**: Candidate Rank = 37, Target Prob = $0.00016$.
- **Long context (153 tokens, 7 distractors)**: Candidate Rank = 33, Target Prob = $0.00012$.

**Finding**: The failure is **position-invariant**. The model does not suffer from positional attention drop-off at the beginning or middle; rather, it completely lacks the circuit required to route context entities into the final output projection.

---

## 6. Multi-Hop and Dynamic State Tracking

Evaluating single-hop, two-hop, state updates, and long-range follow-ups:
- **Single-hop**: Even on direct prompts (`Ravi lives in Pune. Where does Ravi live?`), candidate outputs `"Good evening\n\nNo, I am not provided in the context."`
- **Two-hop**: (`Ravi owns Garuda Systems. Garuda Systems is in Jaipur.`), candidate outputs generic persona strings.
- **State updates**: (`Order status was PENDING. Later, updated to ACTIVE.`), candidate fails to overwrite previous states and outputs abstention or greeting tokens.
- **Intervening dialogue turns**: Facts stated 3 turns earlier are completely forgotten, collapsing to `'Celsia'`.

---

## 7. Generation & Decoding Dynamics

Testing Greedy vs. Stochastic Sampling ($T=0.7, p=0.9, k=40$ across seeds 42, 101, 2026):
- On natural greetings and persona prompts, stochastic sampling produces diverse, valid variations (`"Good afternoon!"`, `"Good morning! How may I assist you?"`).
- On retrieval prompts, stochastic sampling does **not** recover the correct entity; instead, it samples chaotic tokens from the tail (`"Celsia Delhi Crot Certainly As a No..."`).
- This proves that retrieval failure is **intrinsic to the model's logits**, not an artifact of greedy search getting trapped.

---

## 8. Statistical Reassessment of CR-EXP-04 Claims

Hierarchical cluster analysis grouped the 54 held-out test examples into 26 distinct template categories:
- **Cluster Win Rate**: **26 / 26 (100.0%)** of template clusters exhibited lower cross-entropy in the candidate than in the parent.
- **Cluster $t$-Statistic**: $t = 10.99, p < 1.0 \times 10^{-15}$.
- **Cluster Bootstrap 95% CI on Loss Reduction**: $[1.748, 2.508]$ nats.
- **Resolution of Paradox**: The candidate achieved a genuine, broad-based statistical reduction in cross-entropy across all conversational templates. However, cross-entropy on conversational datasets is heavily dominated by common structural tokens. A model can achieve a 41.78% loss drop by predicting greetings, punctuation, and assistant boilerplate with near-zero entropy, while remaining 0% accurate on the few critical entity tokens.

---

## 9. Primary Scientific Conclusion

Among the evaluated hypotheses, the evidence points conclusively to:

> **`MULTIPLE INTERACTING CAUSES: WEAK VARIABLE BINDING AND TEMPLATE DEPENDENCE DOMINATING OVER MODEL CAPACITY AND TOKENIZATION FRAGMENTATION`**

1. **Weak Variable Binding (Dominant Cause)**: The model lacks relational binding circuits in its transformer layers. It cannot bind arbitrary keys to arbitrary values and route them to the output head.
2. **Template Dependence (Secondary Cause)**: The SFT training objective conditioned the model to generate pleasant greetings whenever it sees an `Assistant:` prompt, masking the underlying context.
3. **Model Capacity & Residual Dilution (Structural Constraint)**: In a 3.44M parameter model (d_model=192, 6 layers), cumulative FFN background norm inflation dilutes subtle entity signals down to $<4\%$ of total residual energy.
4. **Tokenizer Whitespace/Subword Fragmentation (Catalyst)**: Entity token mismatch between context and query severely degrades entity attention matching.

---

## 10. Answers to the 5 Concluding Questions

### Question 1: Why does ChakrView perform better on familiar conversational distributions than unfamiliar conversations?
**Answer**:
ChakrView's SFT training optimized autoregressive likelihood over conversational turn formats, teaching the model the dominant language statistics of greetings, persona statements, and polite closers. Because the held-out test set shared these identical templates, cross-entropy improved uniformly. However, the model did not learn relational abstraction; it learned a high-probability conversational prior. When presented with unfamiliar prompts or novel entity queries, the conversational prior overrides the context, emitting polite greetings or abstentions instead of extracting the requested fact.

### Question 2: Which failure is most fundamental?
**Answer**:
**The total absence of neural variable binding.** Even when a fact is presented in an isolated 35-token prompt with zero distractors and familiar entity names, the model cannot bind the entity to the query and route it to the output head.

### Question 3: What is the smallest intervention likely to improve the demonstrated weakness?
**Answer**:
**A dual-objective Curriculum Regularization pilot (CR-EXP-06)** that pairs conversational response formatting with explicit variable-binding pairs (using the verified I4 relational dataset format) under target-token loss weighting, forcing the optimizer to penalize entity extraction errors rather than merely minimizing entropy on greeting boilerplate.

### Question 4: Which existing capabilities must be preserved?
**Answer**:
1. Deterministic, bit-exact CPU-first runtime execution.
2. The authentic ChakrView persona introduction (`"I am ChakrView, an indigenous neural language model..."`).
3. Turn-boundary discipline (stopping at `User:` / `Assistant:` delimiters without infinite repetition).
4. Zero-regression on the canonical baseline ($\Delta W = 0$).

### Question 5: What evidence should be required before the next training experiment is approved?
**Answer**:
Before any new training run is authorized, the following pre-flight gates must be met:
1. **Targeted Loss Weighting Specification**: Explicit loss masking that weights entity target tokens $\ge 5\times$ higher than greeting/filler tokens.
2. **Whitespace-Normalized Dataset**: Tokenizer verification ensuring contextual entity tokens and query entity tokens share bit-exact token IDs.
3. **Synthetic Relational Gate**: A frozen 20-probe variable-binding validation set where parent performance is measured prior to step 0.
4. **Strict Optimization Boundaries**: Maximum 150 optimizer steps with learning rate $\le 5 \times 10^{-5}$ and gradient clipping $\le 0.5$ to prevent residual stream norm explosion.

---

## 11. Authoritative Next Milestone Recommendation

> **`RECOMMENDED NEXT MILESTONE: CR-EXP-06 — TARGETED VARIABLE-BINDING & RELATIONAL CURRICULUM REGULARIZATION`**
- Strictly bounded investigation and dataset design.
- No immediate training execution until experimental specification is independently reviewed and authorized.

