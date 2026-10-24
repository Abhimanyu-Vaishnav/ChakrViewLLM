# ChakrView Post-Overnight Forensic Audit — Trust Gate Analysis

**Target Run**: `experiments/overnight_runs/mission_20261023_003512/`  
**Auditor**: Independent ML Experiment Auditor  
**Investigation Timestamp**: 2026-10-23T07:10:00+05:30  
**Audit Finding**: **`VALID CONFIGURATION WITH ARCHITECTURAL SCOPE DISTINCTION`**

---

## 1. Executive Summary & Code Location

In Task 2, we investigated the invocation parameter `enforce_trust_gate=False` used during the overnight training run script ([`run_stage5_pipeline.py`](file:///C:/Users/abhim/.gemini/antigravity-ide/brain/204d74ae-0f57-4370-8f17-c1d6f0bd82ab/scratch/run_stage5_pipeline.py)).

The Trust Gate mechanism is implemented inside:
- Configuration: [`SFTConfig.enforce_trust_gate: bool = True`](file:///D:/Project/ChakrView/chakrview/training/sft_engine.py#L52)
- Enforcement Site: [`SFTDataset._load()` in `chakrview/training/sft_engine.py:101-123`](file:///D:/Project/ChakrView/chakrview/training/sft_engine.py#L101-L123)

---

## 2. Exact Implementation Evidence

Lines 101 to 123 of [`chakrview/training/sft_engine.py`](file:///D:/Project/ChakrView/chakrview/training/sft_engine.py#L101-L123) state:

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

### Key Observations from Code:
1. **Benchmark Contamination Check is UNCONDITIONAL**:
   Lines 106–109 unconditionally raise `ValueError("Contamination error: ...")` if `role in ("HELD_OUT_BENCHMARK", "BENCHMARK")` during training. Setting `enforce_trust_gate=False` does **not** disable benchmark contamination checking.
2. **What `enforce_trust_gate=False` Actually Controls**:
   The `enforce_trust_gate` flag guards only two specific fields:
   - `review_status == "PENDING_HUMAN_REVIEW"`: prevents unreviewed crowdsourced/human-review-pending records from entering training without `allow_exploratory_opt_in=True`.
   - `status == "REJECTED"`: refuses records marked explicitly as rejected.
3. **No Effect on Model, Checkpoints, or Gradients**:
   The flag does **not** bypass parameter verification, finite weight checks, loss masking, gradient clipping, tokenizer verification, or checkpoint serialization.

---

## 3. Dataset Inspection & Field Audit

We audited the actual official dataset files (`train_exp17_v1.jsonl`, `val_exp17_v1.jsonl`, `test_exp17_v1.jsonl`) generated for Exp17:
- `data_role`: Not present in records (defaults to `"TRAIN"` or `"VALIDATION"`).
- `review_status`: Not present in records (evaluates to `None`).
- `status`: Not present in records (evaluates to `""`).

Because Exp17 is a synthetic, mathematically generated instruction dataset whose records are not tagged with the legacy annotation status string `"PENDING_HUMAN_REVIEW"`, executing `SFTExperimentEngine` with `enforce_trust_gate=True` loads all 320 training records cleanly without raising any error.

### Empirical Proof:
We executed `SFTExperimentEngine` with `enforce_trust_gate=True` directly against the official dataset files:
```powershell
.venv\Scripts\python.exe -c "
from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine
cfg = SFTConfig(..., enforce_trust_gate=True)
engine = SFTExperimentEngine(cfg)
print(len(engine.train_dataset))
"
# Output: 320 records loaded successfully (Exit code: 0)
```

---

## 4. Validity Determination Under Project Protocol

- **Did `enforce_trust_gate=False` mask a defect?**: No. Even with `enforce_trust_gate=True`, zero records fail the trust gate.
- **Did it bypass contamination prevention?**: No. Contamination checks are unconditional.
- **Protocol Determination**: The overnight run is classified as **`VALID UNDER PROTOCOL`**. However, as a best-practice convention for future production runs, `enforce_trust_gate=True` should be explicitly passed to maintain standard defensive defaults across all runs.
