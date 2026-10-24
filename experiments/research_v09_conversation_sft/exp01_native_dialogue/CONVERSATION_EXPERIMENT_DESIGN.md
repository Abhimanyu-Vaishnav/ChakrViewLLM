# ChakrView Experiment 13: Native Human Conversation Understanding, Context Learning, and Self-Correction
**Directory**: `experiments/research_v09_conversation_sft/exp01_native_dialogue/`  
**Date**: October 21, 2026  
**Auditor / Researcher**: Antigravity AI Engine

---

## 1. Mission & Scientific Objective

The goal of this experiment is to test and improve ChakrView's native causal language model (ChakrMicro, 3.44M parameters) on realistic human conversations without relying on external corporate APIs, hosted LLMs, or deterministic controllers.

We evaluate:
1. **Multi-turn dialogue understanding**: Follow-up references ("that one", "do the same for this"), conversational corrections ("Actually I meant X, not Y"), topic shifts, and contextual memory across turns.
2. **Context-grounded reasoning & questions**: Distractor resistance, answering from multi-sentence context, acknowledging missing information, and refusing unsupported claims.
3. **Controlled training with strict response-only loss masking**: Fine-tuning candidate checkpoints from the canonical Step-10000 parent checkpoint while keeping the canonical parent frozen.
4. **Controlled self-correction loop**: Generating initial responses, detecting concrete failures, proposing verified corrections, and measuring whether candidate retraining from verified correction traces improves behavior.

---

## 2. Invariants & Isolation Constraints

1. **Artifact Preservation**:
   - Canonical parent checkpoint: `experiments/batch16_21_combined/.../training_checkpoint_step_10000.pt` (SHA-256: `96357705...`) remains byte-identical.
   - Tokenizer vocabulary and merges (`data/experiments/vocab_4096/`) remain byte-identical.
   - Protected runtime `chakrview/runtime/session.py` is NEVER modified.
2. **Pure Native Execution**:
   - Model outputs evaluated strictly autoregressively (greedy $T=0.0$).
   - No schema controllers, no external LLM judges, no hidden heuristics.
3. **No Git Mutations**:
   - Zero commits, pushes, merges, tags, resets, or stash changes.

---

## 3. Conversational Role Formatting Contract

To maintain 100% tokenization parity without altering the frozen tokenizer vocabulary ($V=4096$), conversations are serialized using standard text role headers:

```text
User: Hello, can you help me check the power inverter?
Assistant: Hello! Yes, I can help check the power inverter. What is its current reading?
User: The reading shows 480 volts. Is that normal?
Assistant: 480 volts is within standard operational range for industrial power inverters.
```

When predicting the final assistant turn:
- **Input sequence (prompt)**: All preceding dialogue up to and including `"Assistant:"`.
- **Target sequence (response)**: The assistant's text up to newline or `<|endoftext|>`.
- **Loss Masking**: Tokens corresponding to earlier turns and the `"Assistant:"` prompt prefix are masked with `-100` so gradients flow strictly into assistant response generation.

---

## 4. 20 Realistic Conversation Categories

The dataset generator creates diverse scenarios across 20 distinct categories:
1. Casual greetings and short exchanges.
2. Informal, incomplete, and grammatically imperfect messages.
3. Requests with implicit but inferable intent.
4. Follow-up questions dependent on previous turns.
5. Anaphoric references ("that one", "the earlier unit", "do the same for this").
6. User corrections and changes to earlier requirements.
7. Topic changes and returning to previous topics.
8. Multi-turn instruction following.
9. Explicit constraints and formatting requirements.
10. Questions answerable from provided context.
11. Multi-hop context inference.
12. Distractor resistance and conflicting context.
13. Ambiguous requests requiring clarification.
14. Missing information / questions unanswerable from context.
15. Factual questions requiring evidence or uncertainty statements.
16. Simple explanation, planning, and code synthesis.
17. Multi-step task chunking.
18. Assistant self-correction (acknowledging prior mistake).
19. Refuting false / misleading user feedback.
20. Refusing to assert unsupported facts / calibrated abstention.

---

## 5. Experimental Arms & Evaluation Protocol

We evaluate across:
- **Arm 1: Frozen Canonical Baseline (Step-10000)**: Native pre-trained causal LM evaluated zero-shot on the dialogue benchmarks.
- **Arm 2: Candidate Trained Model (Step-150 / Step-200)**: Candidate model fine-tuned on the multi-turn conversational dataset.
- **Arm 3: Self-Correction Retrained Model**: Candidate fine-tuned with augmented verified correction traces.
