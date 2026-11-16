# CR-EXP-20R: Phase A — Technical Audit and Recovery Report

**Mission ID:** CR-EXP-20R  
**Phase:** Phase A — Recover and Verify CR-20  
**Date:** 2026-11-15  
**Investigator:** Antigravity Autonomous Coding Agent  
**Repository Working Copy:** `D:\Project\ChakrView`  
**Git Branch:** `exp/cbr-01-conversational-breakthrough`  

---

## 1. Cryptographic Verification of Artifacts

All required canonical artifacts, release checkpoints, and intermediate experimental checkpoints were audited and verified with bit-exact SHA-256 matches:

| Subject | Path / Source | Expected SHA-256 | Actual Measured SHA-256 | Verification Status |
| :--- | :--- | :--- | :--- | :---: |
| **Canonical Stage B Checkpoint** | `experiments/batch16_21_combined/...` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **PASS (Bit-Exact)** |
| **Conversational Parent (`cbr_sft_04`)** | `experiments/cbr01_sft/runs/cbr_sft_04/...` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **PASS (Bit-Exact)** |
| **v0.9.1 Release Checkpoint** | `releases/v0.9.1_conversational_preview/...` | `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10` | `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10` | **PASS (Bit-Exact)** |
| **v0.10 Release Checkpoint** | `releases/v0.10_conversational_model/...` | `9815f235996934c9c8de4772a2ecd98db1b3b9f5f4d70e5796d3530e4a6fe2ba` | `9815f235996934c9c8de4772a2ecd98db1b3b9f5f4d70e5796d3530e4a6fe2ba` | **PASS (Bit-Exact)** |
| **v0.10 Seed 2026 Checkpoint** | `experiments/cognition/cr_exp_20/checkpoints/...` | `9815f235996934c9c8de4772a2ecd98db1b3b9f5f4d70e5796d3530e4a6fe2ba` | `9815f235996934c9c8de4772a2ecd98db1b3b9f5f4d70e5796d3530e4a6fe2ba` | **PASS (Bit-Exact)** |
| **v0.10 Seed 3030 Checkpoint** | `experiments/cognition/cr_exp_20/checkpoints/...` | `a83a1d139a7146326022aedb3e248495d87abe72df86aa896b94850db8cfab3b` | `a83a1d139a7146326022aedb3e248495d87abe72df86aa896b94850db8cfab3b` | **PASS (Bit-Exact)** |
| **Protected Runtime (`session.py`)** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS (Bit-Exact)** |

---

## 2. Audit of CR-20 Training Runs and Evaluation

1. **Training Completion:**
   - Both seed 2026 and seed 3030 completed 350 training steps on CPU without gradient explosion or NaN loss.
   - Seed 2026 best step: 350 (Val loss: 2.3223).
   - Seed 3030 best step: 350 (Val loss: 2.3196).
2. **Evaluation Status:**
   - The 700-probe frozen benchmark evaluation completed in 825.4s total across all 3 models (`parent_baseline`, `cr20_candidate_seed2026`, and `cr20_candidate_seed3030`).
   - Every single probe in `cr20_evaluation_results.json` was evaluated under greedy decoding ($T=0.0$, repetition penalty 1.15).
3. **The 0/700 Result:**
   - In CR-20, all 6 benchmarks were scored strictly on exact name/entity containment: `any(e.lower() in resp.lower() for e in exp)`.
   - Because the probes were structured as: `"User: My name is TestUser_0 and I live in TestCity_0.\nAssistant: Nice to meet you...\nUser: What is my name?\nAssistant:"` and the models replied with conversational assistance priors rather than quoting the exact entity token `TestUser_0`, the evaluator logged 0 hits.
   - Whether the model can converse naturally, answer ordinary questions, or handle conversational flow in English, Hindi, and Hinglish was **not captured** by the strict exact-match entity metric.

Phase A is complete. We now proceed to Phase B to generate and inspect actual raw conversations.

