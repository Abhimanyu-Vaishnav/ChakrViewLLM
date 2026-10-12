# Batch 127–130: Experiment Plan — Independent Evidence Audit, Multi-Digit Binding Diagnosis & Controlled Experiment

## 1. Research Question & Hypothesis

### Problem Statement
In Batches 123–126, ChakrMicro v0.1 was trained and tested on two-digit ($10..99$) addition requests across 8 training templates and 2 held-out evaluation templates.
While the model achieved:
- 100% negative control refusal (0% false positives, 0 malformed calls)
- 0 operand swaps ($a \leftrightarrow b$)
- 100% exact-call accuracy on familiar pairs with held-out template 1 (`"Execute calculator addition: first={a}, second={b}."`)

it struggled on unseen operand pairs (Partition 2: 13.3% exact-call accuracy, Partition 3: 0.0% exact-call accuracy).

### Competing Hypotheses Audited
1. **Hypothesis 1 (Tokenization & BPE Segmentation)**: Numbers split into 1 vs 2 tokens, and digits merge into trailing punctuation (e.g. `17, ` is single token 3658).
   - *Audit Finding*: Comma-merger occurs across 63/90 numbers (70%). However, conditioning on comma-merging reveals it is NOT the barrier preventing generalization: comma-merged numbers achieved 36% (9/25) operand A accuracy vs 30% (3/10) for unmerged numbers. Furthermore, both-single-token pairs achieved 10% exact calls, while both-multi-token pairs achieved 33.3% exact calls.
2. **Hypothesis 2 (Prompt and Target Format Inconsistency)**:
   - *Audit Finding*: The 8 training prompt templates provide diverse syntactic frames, but each training pair was presented only 1.17 times on average (50 pairs seen once, 10 pairs seen twice).
3. **Hypothesis 3 (Operand Repetition vs Pair Space Coverage)**:
   - *Audit Finding*: Out of 8,100 possible two-digit pairs ($90 \times 90$), the training set exposed only 60 pairs across 70 tool records. A model of 4.1M parameters cannot learn universal multi-digit copy attention across 8 syntax variations from only 70 examples.
4. **Hypothesis 4 (Syntactic Anchor Dependence vs Generic Refusal)**:
   - *Audit Finding*: Held-out template 2 (`"Determine arithmetic total for inputs: A is {a} and B is {b}."`) triggered 0 tool calls because the phrase `"Determine arithmetic total for inputs"` has zero syntactic overlap with training cues, so the gating head conservatively emitted the trained refusal string.
5. **Hypothesis 5 (Subword-Level Partial Binding)**:
   - *Audit Finding*: Qualitative inspections revealed that on unseen pairs, the copy-initialized model was indeed attending to the correct input numbers and copying the tens digit or units digit correctly in 33.3% of cases (e.g., prompt `[74, 53]` -> predicted `[75, 53]`; prompt `[79, 40]` -> predicted `[79, 4]`).

### Target Hypothesis for Batch 127–130
Can multi-digit operand binding generalize to unseen two-digit pairs when the model is trained with adequate pair diversity and exposure?
Specifically, we isolate:
- **Baseline Arm (Arm 1 — Standard 70-tool exposure, replication)**: 70 tool records across 60 pairs.
- **Intervention Arm (Arm 2 — Controlled Pair-Density Curriculum, 200 tool exposure)**: 200 tool records across 160 distinct pairs and 8 templates, keeping the identical 30% negative control ratio, same optimizer, same step budget (200 steps), same seed (42), and same copy-task initialization (`copy_task_sft_step150.pt`).

## 2. Predicted Outcomes & Decision Rule
- **Prediction**: If limited pair exposure (Hypothesis 3) is the primary bottleneck, Arm 2 will significantly increase individual operand extraction and exact-call accuracy on unseen pairs (Partition 2) without degrading gating on negative controls (0% false positives).
- **Decision Rule**:
  - If Arm 2 increases Partition 2 exact-call accuracy over Arm 1 by $\ge 15$ percentage points while maintaining 0% false positives on negative controls: accept that pair-space density is a necessary condition for multi-digit generalization in ChakrMicro v0.1.
  - If Arm 2 fails to improve Partition 2 exact-call accuracy: reject Hypothesis 3 and conclude that the tiny transformer's multi-head attention capacity or optimization dynamics prevent zero-shot composition of subword arithmetic operands.

## 3. Experimental Controls & Leakage Prevention
1. **Disjoint Sets**: All test operand pairs in Partition 2 and Partition 3 are strictly disjoint from all training pairs and reversed training pairs across both arms.
2. **Held-Out Templates**: Partitions 1 and 3 use exclusively the 2 held-out templates never seen in train or val.
3. **Deterministic Seeds**: All datasets generated with seed 42.
4. **Identical Hyperparameters**: AdamW ($lr=3\times 10^{-4}$, weight decay 0.01, gradient clipping 1.0, batch size 4, 200 steps).
5. **Evaluation Protocol**: Free-running greedy decoding from cold-reloaded saved checkpoints.
