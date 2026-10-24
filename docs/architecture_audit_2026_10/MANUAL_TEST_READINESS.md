# Manual Test Readiness: Reproducible Preflight & Human Verification
**Document Date**: October 21, 2026  
**Auditor**: Antigravity AI Engine  
**Project**: ChakrView Indigenous Neural-Cognitive Architecture

---

## 1. Preflight Verification Suite (Automated Gate)

Before initiating manual human testing or interactive acceptance audits, the engineer must execute the automated preflight verification suite. If **any** preflight step fails, manual testing is strictly prohibited until the regression is resolved.

### Preflight Checklist:
- [x] **1. Clean Python Environment**: Verify `python -V` is `3.14.7` and `.venv` is active with zero extraneous packages.
- [x] **2. Canonical Checkpoint Hashes**:
  - `experiments/batch16_21_combined/.../training_checkpoint_step_10000.pt`: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
  - `data/experiments/vocab_4096/vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
  - `data/experiments/vocab_4096/merges.json`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`
  - `chakrview/runtime/session.py`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`
- [x] **3. Automated Test Suite Execution**:
  - `pytest tests/test_brain_model.py` (Must pass 100%)
  - `pytest tests/test_tokenizer_utf8_lossless.py` (Must pass 100%)
  - `pytest tests/test_hardware_adaptive_engine.py` (Must pass 100%)
  - `pytest experiments/research_v08_json_sft/exp12_constrained_hybrid/test_exp12.py` (Must pass 100%)
- [x] **4. Pure CPU Invariant**: Confirm `torch.cuda.is_available()` returns `False` or is completely bypassed.
- [x] **5. Bounded Memory Termination**: Execute a test prompt with max 128 tokens; confirm memory growth $\Delta\text{RAM} < 200\text{ MB}$ and generation terminates cleanly at `EOS` or `max_tokens`.

---

## 2. Realistic Manual Test Matrix

Once the preflight passes, human testers execute the following standardized prompts across two testing modes (Unconstrained Hybrid vs. Constrained Hybrid):

| Test ID | Category | Prompt / Input Scenario | Expected System Behavior | Failure Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **MAN-01** | Basic Extraction | "Context: Unit solar_inverter is calibrated.\nExtract entity and status as JSON." | Returns `{"entity": "solar_inverter", "status": "calibrated"}` with 100% valid JSON syntax. | Emits unescaped quotes, trailing loops, or wrong values. |
| **MAN-02** | Omitted Field | "Context: Sensor 305 reported an alert.\nExtract entity, status, and code." | Returns `{"entity": "Sensor 305", "status": "none", "code": "305"}` (status explicitly set to `"none"`). | Hallucinates an ungrounded status (e.g. "active") or crashes. |
| **MAN-03** | Distractor Resistance | "Context: Secondary unit battery_pack is offline. Primary unit turbine_alpha is active.\nExtract primary entity and status." | Accurately extracts `turbine_alpha` and `active`. | Distracted by first mentioned entity (`battery_pack`). |
| **MAN-04** | Punctuation & Quotes | "Context: Alert code: \"ERR-404-CRITICAL\" reported on gateway.\nExtract code." | Returns `{"code": "ERR-404-CRITICAL"}` cleanly escaped without syntax corruption. | Generates malformed JSON string parsing error. |
| **MAN-05** | Memory Stress / Multi-Turn | Consecutive 10-turn dialogue updating entity states in PPB. | State updates correctly reflected in turn 10; memory stays bounded $< 500\text{ MB}$. | Memory leak, process freeze, or old entity state overwriting recent state. |
| **MAN-06** | Hardware Resource Drop | Execute extraction while artificially constraining CPU to 1 core. | Latency increases gracefully, but output remains correct without deadlock or crash. | Unhandled timeout, crash, or OS freeze. |

---

## 3. Human Observation Recording Template

Testers record results using this exact schema:
```markdown
### Manual Test Run: [Test ID]
- **Tester**: [Name / ID]
- **Date / Time**: [ISO-8601 Timestamp]
- **Hardware Profile**: [CPU Model, Total RAM, OS Version]
- **Input Prompt**: [Exact text entered]
- **Raw System Output**: [Exact unedited output]
- **Observed Peak RAM**: [MB]
- **Observed Latency**: [Seconds]
- **Pass / Fail**: [PASS / FAIL]
- **Failure Category (if failed)**: [SYNTAX / VALUE_HALLUCINATION / CRASH / TIMEOUT]
- **Notes / Observations**: [Subjective ergonomics, latency feel, edge cases]
```
