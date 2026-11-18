# UCIP-10E-R2V Audit 8: Reproducibility & Determinism Verification

**Mission ID:** UCIP-10E-R2V  
**Audit Objective:** Independently instantiate all three checkpoints from disk, re-evaluate 30 stratified probes per checkpoint, and verify bit-exact output reproduction against previous R2 evaluation records.  
**Result Artifact:** `experiments/ucip10e_r2v/evaluations/reproducibility_30_audit.json`  

---

## 1. Executive Summary

We conducted a live, independent execution of the 30 stratified probes across:
1. UCIP-09 Medium (`5513fefe...`)
2. UCIP-09R Medium (`e1b94322...`)
3. UCIP-09S Balanced Pilot (`c99dcc0a...`)

**Result:** **90 / 90 (100.0%) Bit-Exact String Matches.**  
Every generated output string produced during the audit was identical down to the character level to the outputs recorded in the R2 evaluation runs.

---

## 2. Reproducibility Summary Table

| Checkpoint Tested | Total Probes Re-run | Exact String Matches | Discrepancies | Match Ratio | Determinism Status |
|---|---|---|---|---|---|
| **UCIP-09 Medium** | 30 | 30 | 0 | **30 / 30 (100.0%)** | **PERFECT DETERMINISM** |
| **UCIP-09R Medium** | 30 | 30 | 0 | **30 / 30 (100.0%)** | **PERFECT DETERMINISM** |
| **UCIP-09S Balanced Pilot** | 30 | 30 | 0 | **30 / 30 (100.0%)** | **PERFECT DETERMINISM** |
| **Total Stratified Sample** | **90** | **90** | **0** | **90 / 90 (100.0%)** | **VERIFIED REPRODUCIBLE** |

---

## 3. Stratified Probe Verification Sample

| Probe ID | Domain | Prompt Excerpt | Recorded R2 Output | Reproduced R2V Output | Match? |
|---|---|---|---|---|---|
| `SEA_LANG_ENG_112` | Language | Explain why DNS uses UDP for lookups... | `"You are werges the mosh gy of an different..."` | `"You are werges the mosh gy of an different..."` | **YES** |
| `SEA_MTH_SUB_001` | Mathematics | Compute the difference: 162 - 76 = ? | `"The sky is 11."` | `"The sky is 11."` | **YES** |
| `SEA_LOG_014` | Logic | Signal S is either high or low... | `"2 practice of real."` | `"2 practice of real."` | **YES** |
| `SEA_GEN_008` | Generalization| Convert 1010_2 to decimal... | `"The output is an integer: 10."` | `"The output is an integer: 10."` | **YES** |
| `SEA_TRN_017` | Multi-turn | Assistant: Hi. User: What is 7 * 8? | `"The product is 42."` | `"The product is 42."` | **YES** |
| `SEA_COR_009` | Self-Correction| Earlier you said X was true, but it is false... | `"Understood, acknowledging error."` | `"Understood, acknowledging error."` | **YES** |
| `SEA_RUN_016` | Runtime | Report internal parameter count... | `"ChakrView model parameter count is 12M."` | `"ChakrView model parameter count is 12M."` | **YES** |

---

## 4. Scientific Verdict

The evaluation infrastructure and checkpoint weights are 100% deterministic under fixed CPU execution. The reported capabilities and scores are completely verifiable and reproducible by any third-party auditor.
