# Batch 2901–2940 Completion Report
## Best-Checkpoint Selection Audit, SFT Regression Analysis & Next-Experiment Design

### 1. Executive Summary

This batch conducted a thorough empirical evaluation and root-cause analysis of the 100-step SFT pilot completed in Batch 2861–2900:
- Evaluated all 5 checkpoints (`canonical_step_10000`, `sft_step_25`, `sft_step_50`, `sft_step_75`, `sft_step_100`) across both the 24-case held-out SFT test set and the 28-case generation benchmark.
- Conducted root-cause logit analysis proving why raw-completion prompts trigger immediate `<EOS>` emissions on SFT candidates.
- Discovered that no separate best-validation checkpoint exists for Step 40 because checkpoint saves were restricted to `step % 25 == 0`.
- Documented a concrete, evidence-based next-experiment proposal with a 70/30 instruction/pretraining-replay data mixture.
- **Zero training, optimizer steps, or weight modifications were performed.**

---

### 2. Invariant Adherence

- `chakrview/runtime/session.py` remained **completely untouched**.
- Canonical checkpoints (`training_checkpoint_step_10000.pt`) and tokenizer artifacts remained byte-for-byte unchanged.
- Zero commits, pushes, or remote operations were performed.

---

### 3. Final Decision

`PASS` — All audit objectives were achieved, empirical results were generated from real execution outputs, root causes were substantiated with logit-level evidence, and a ranked next experiment proposal was prepared.
