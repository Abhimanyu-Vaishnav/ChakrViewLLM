# ChakrView Exp17 Validation — Trust Gate & Reproducibility Review

**Target Implementation**: `chakrview/training/sft_engine.py`  
**Configuration Class**: `SFTConfig` ([sft_engine.py:32-55](file:///D:/Project/ChakrView/chakrview/training/sft_engine.py#L32-L55))  
**Auditor**: Independent ML Experiment Auditor  

---

## 1. Trust Gate Guardrail Analysis

### 1.1 Exact Code Invariants
In [`chakrview/training/sft_engine.py`](file:///D:/Project/ChakrView/chakrview/training/sft_engine.py#L101-L122):
```python
        for rec in raw_records:
            rid = rec.get("id", "UNKNOWN_RECORD")
            role = rec.get("data_role", "TRAIN" if self.is_training else "VALIDATION")

            # Check benchmark leakage into training
            if self.is_training and role in ("HELD_OUT_BENCHMARK", "BENCHMARK"):
                raise ValueError(
                    f"Contamination error: Record {rid} with role {role} cannot enter training dataset."
                )

            # Enforce trust gate
            if self.enforce_trust_gate:
                review_status = rec.get("review_status")
                status = rec.get("status", "")
                if review_status == "PENDING_HUMAN_REVIEW" and not self.allow_exploratory_opt_in:
                    raise PermissionError(
                        f"Trust Gate Refusal: Record {rid} is PENDING_HUMAN_REVIEW. "
                        f"Unreviewed records cannot enter training without explicit allow_exploratory_opt_in=True."
                    )
                if status == "REJECTED":
                    raise ValueError(f"Trust Gate Refusal: Record {rid} is marked REJECTED.")

            self.records.append(rec)
```

### 1.2 Structural Conclusions:
1. **Contamination Guard**: The check preventing benchmark leakage (`role in ("HELD_OUT_BENCHMARK", "BENCHMARK")`) is hardcoded before `if self.enforce_trust_gate:`. It cannot be bypassed by setting `enforce_trust_gate=False`.
2. **Trust Gate Guard**: Only checks `review_status == "PENDING_HUMAN_REVIEW"` and `status == "REJECTED"`.
3. **Exp17 Applicability**: Exp17 dataset records do not contain `review_status` or `status` tags (both evaluate to `None`). Consequently, initializing `SFTExperimentEngine` with `enforce_trust_gate=True` loads all 320 records without raising exceptions.
4. **Reproducibility Assessment**: The run is fully reproducible under both `enforce_trust_gate=True` and `enforce_trust_gate=False`.

---

## 2. Automated Test Suite Results

Two independent test discovery passes were executed:

### Test Suite 1: Full Repository Root Test Suite
- **Command**: `.venv/Scripts/python.exe -m unittest discover -s tests -p test_*.py`
- **Tests Executed**: **106 tests**
- **Failures**: 0
- **Errors**: 0
- **Execution Time**: 120.709s
- **Status**: **PASS (OK)**
- **Coverage**: Covers neural runtime adapter, BPE tokenizer, SFT collator, prompt contracts, and checkpoint serialization.

### Test Suite 2: Exp17 Focused Test Suite
- **Command**: `.venv/Scripts/python.exe -m unittest discover -s experiments/research_v09_conversation_sft/exp17_instruction_sft_v1 -p test_*.py`
- **Tests Executed**: **9 tests**
- **Failures**: 0
- **Errors**: 0
- **Execution Time**: 32.536s
- **Status**: **PASS (OK)**
- **Coverage**: Covers candidate banks, proof-of-absence verification, control grounding, response token bounds, and normal chat gate isolation.
