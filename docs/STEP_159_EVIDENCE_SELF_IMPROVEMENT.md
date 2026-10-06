# Step 159 — Evidence-Based Self-Improvement Cycle

## 1. Governance Architecture

Step 159 applies the self-improvement loop with mandatory capability gating:
$$\text{OBSERVE} \to \text{DIAGNOSE} \to \text{PROPOSE} \to \text{TRAIN} \to \text{EVALUATE} \to \text{CHECK} \to \text{DECIDE} \to \text{RECORD}$$

## 2. Mandatory Capability Gate

Unlike Step 152 where aggregate loss reduction allowed candidate promotion despite zero reasoning gain, Step 159 introduces a strict capability-specific requirement:
- **Mandatory Condition:** $\text{Held-Out Reasoning Accuracy} \ge 0.50$
- If this gate is not met, the controller MUST output:
  $$\text{REJECT\_CANDIDATE}$$
  regardless of whether cross-entropy training loss decreased.

## 3. Empirical Cycle Audit

```json
{
  "cycle_id": "cycle_step159_evidence",
  "target_weakness": "diag_step147_failure_autopsy",
  "decision": "REJECT_CANDIDATE",
  "held_out_reasoning_acc": 0.0,
  "held_out_reasoning_delta": 0.0,
  "language_loss_delta": 0.02,
  "baseline_intact": true,
  "audit_rationale": "Candidate rejected under mandatory capability gating: Held-out reasoning accuracy (0.0) below mandatory threshold (0.5)"
}
```

This demonstrates that ChakrView's governed self-improvement framework correctly rejects candidates when capability claims cannot be empirically substantiated.
