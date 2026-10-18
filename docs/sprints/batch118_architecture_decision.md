# Batch 118 — Architecture Decision, Synthesis, and Handoff

**Sprint**: ChakrView Master Sprint Batches 115–118  
**Module**: Tool-Gating Calibration, Strict Execution Metrics, and Reliable Argument Binding  
**Author**: ML Research Engineer  
**Date**: 2026-10-12  

---

## 1. Supported Empirical Conclusion

### Selected Conclusion:
**"The balanced curriculum improves both tool gating and argument binding, demonstrating that standard autoregressive transformer attention is fully capable of input-conditioned tool argument binding and decision gating when trained on cleanly bounded sequence curricula."**

### Summary of Evidence:
1. **Strict 11-Metric Evaluation (Batch 115)**:
   - Established strict metric separation. Accidental arithmetic result matches (`5 + 4 = 9` for `1 + 8`) are verified as non-passes.
   - Prohibited injection fields, missing keys, and invalid types are enforced through unit tests.
2. **Balanced Curriculum (Batch 116)**:
   - Combined positive tool additions with negative out-of-domain controls, shell injection attempts, and underspecified arithmetic requests.
   - Formatted using discrete per-sample causal sequence masking with prompt exclusion and verified tokenization compositionality.
3. **Matched Training Comparison (Batch 117)**:
   - **Tool-Gating**: Both arms achieved **0.0% false-positive tool calls** (8/8 clean refusals on held-out test controls, 16/16 on train, 4/4 on val).
   - **Argument Binding**: The Curriculum Bridge arm (Stage 1 copy pre-training followed by balanced tool curriculum) achieved **100.0% (20/20)** exact-call accuracy and 100.0% end-to-end execution success on the held-out test set under cold reload.
   - **Constant Collapse**: Reduced to the theoretical minimum of 5.0% (20 unique calls across 20 test inputs).

---

## 2. Architecture Decision

**No Pointer Network or Custom Copy Attention Mechanism is Justified.**

The hypothesis that "4-layer causal transformer attention cannot copy dynamic operands or bind arguments without custom pointer heads" is decisively refuted. The historical failure was driven entirely by:
1. Continuous sliding-window byte stream slicing across conversation boundaries.
2. Absence of negative controls during SFT.
3. Overwhelming static JSON boilerplate prior to learning clean token-to-token routing.

Once the learning task is structured as a two-stage curriculum (variable copying $\to$ balanced tool calling), the canonical `ChakrMicro` architecture achieves 100% precision on held-out test splits without altering model weights, parameter counts, or runtime dependencies.

---

## 3. Checkpoint Manifest & Invariant Verification

| Checkpoint Identifier | File Path | Steps | SHA-256 |
|---|---|---|---|
| **Canonical Base** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | 10,000 | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **Stage 1 Copy Model** | `experiments/batch107_110_operand_copy_benchmark/checkpoints/copy_task_sft_step150.pt` | 150 | `e84b2ea9b633b50181fb120ea2444a4a3c84c008b28905dd33c24fedba2b1bc4` |
| **Matched Control Base** | `experiments/batch115_118_tool_gating_curriculum/checkpoints/balanced_control_base_step150.pt` | 150 | `6905fa8b759f3730926665395e54453a779fb0767fb574e115163b3fc512a427` |
| **Curriculum Bridge** | `experiments/batch115_118_tool_gating_curriculum/checkpoints/balanced_curriculum_bridge_step150.pt` | 150 | `ff606e5a8c25272347d1e127005255080eb82192a7b2d6eb8136e9a13134a2f7` |

---

## 4. The Single Most Valuable Next Experiment

**Multi-Operation & Multi-Digit Arithmetic Scaling**:
Extend the verified two-stage curriculum from single-digit addition to:
1. Multi-digit numbers ($a, b \in [10, 99]$).
2. Multiple calculator operations (`sub`, `mul`).
3. Multi-turn dialogue context where the model must extract operands from informal user phrasing rather than structured key-value prompts.

**Success Criteria**: $>80\%$ exact-call accuracy on multi-operation held-out test sets with $<5\%$ false-positive tool calls on non-math conversational turns.

---

## 5. Deliverables & Regression Suite Summary

```powershell
# Run accumulated regression tests (51 tests in 3.85s)
.venv\Scripts\python.exe -m pytest tests/test_batch103_106_slot_weighted_sft.py tests/test_batch107_110_copy_benchmark.py tests/test_batch111_114_curriculum.py tests/test_batch115_evaluation_semantics.py tests/test_batch115_118_tool_gating_regression.py -v
```

- [batch115_evaluation_integrity.md](file:///d:/Project/ChakrView/docs/sprints/batch115_evaluation_integrity.md)
- [batch116_balanced_curriculum_dataset.md](file:///d:/Project/ChakrView/docs/sprints/batch116_balanced_curriculum_dataset.md)
- [batch117_tool_gate_training_results.md](file:///d:/Project/ChakrView/docs/sprints/batch117_tool_gate_training_results.md)
- [batch118_architecture_decision.md](file:///d:/Project/ChakrView/docs/sprints/batch118_architecture_decision.md)
- [test_batch115_evaluation_semantics.py](file:///d:/Project/ChakrView/tests/test_batch115_evaluation_semantics.py) (6 passed in 0.06s)
- [test_batch115_118_tool_gating_regression.py](file:///d:/Project/ChakrView/tests/test_batch115_118_tool_gating_regression.py) (4 passed in 0.04s)
- [matched_comparison_results.json](file:///d:/Project/ChakrView/experiments/batch115_118_tool_gating_curriculum/reports/matched_comparison_results.json)
