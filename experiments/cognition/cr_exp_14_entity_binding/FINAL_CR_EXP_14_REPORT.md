# CR-EXP-14: Final Scientific Report
## Subject-Conditioned Entity Binding & Dynamic State Tracking

**Mission ID:** CR-EXP-14  
**Date:** November 8, 2026  
**Status:** Completed  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Working Directory:** `experiments/cognition/cr_exp_14_entity_binding/`  

---

## 1. Executive Summary

CR-EXP-14 investigated and addressed the critical evaluator flaw exposed by CR-EXP-13: previous conversational working-memory evaluations rewarded correct city values even when the model substituted familiar training names for the queried subject, resulting in a strict subject-value binding accuracy of **0.00%** on out-of-distribution entities.

CR-EXP-14 delivered:
1. **Repository Safety Audit & Immutable Lineage Verification:** Verified that canonical checkpoints and `chakrview/runtime/session.py` (`9080ff51...`) remain byte-for-byte untouched.
2. **Sealed 1,200-Probe Benchmark:** Created and froze `benchmark_1200_sealed.json` (SHA-256: `1958e94571f83c66c26f390634ffa563d0a2ff57e58943e835d7825245b964b9`) with strictly disjoint entities and locations, covering single-entity, two-entity, three-entity, repeated updates, conversational distractors, counterfactual reassignments, and calibrated abstention.
3. **Comprehensive Strict Baseline Evaluations:** Evaluated CR-11 Candidate A, CR-12 Candidate A, and CR-12 Candidate B (parent) under strict scoring, confirming that all baseline models score **0.00% (0/1,100)** on out-of-distribution entity state tracking.
4. **Three Controlled Interventions Trained:**
   - **Candidate A:** Explicit Subject-Value Conditioning (ESVC)
   - **Candidate B:** Interleaved State-Update Curriculum (ISUC)
   - **Candidate C:** Structured Entity Delimiters (SED)
5. **Mechanistic & Architectural Diagnosis:** Demonstrated why low-resource models ($3.44\text{M}$ parameters, $d_{\text{head}}=32$) fail to form abstract induction circuits for multi-token subword entities, and outlined the architectural requirements ($d_{\text{head}} \ge 64$) for v0.9.1 readiness.

---

## 2. Experimental Results Summary

| Model / Candidate | Checkpoint Hash (SHA-256) | Overall Strict Acc (%) | Strict Hits / 1200 | State-Tracking Acc (Cat 1-6) | Abstention Acc (Cat 7) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CR-11 Candidate A** | `4d14cf05df395608ed8d1ab02eeca571b6bf049574785a3d16a7ec2112a64bc5` | 5.00% | 60 / 1200 | **0.00%** (0 / 1100) | 60.0% (60 / 100) |
| **CR-12 Candidate A** | `cfb99ad7651bf64303a4fd8ab562ab9d7b0b0ed008fb3389957e97c76a05a8a2` | 8.33% | 100 / 1200 | **0.00%** (0 / 1100) | **100.0%** (100 / 100) |
| **CR-12 Candidate B (Parent)** | `60271c8b5e3d26b76237dc8f986e0d8d6ada2f08815cea57ec5d9a37102d9d2e` | **8.33%** | **100 / 1200** | **0.00%** (0 / 1100) | **100.0%** (100 / 100) |
| **Candidate A (ESVC)** | `58f51bc249aec77235bbdd07179dc130f72e24947500b05fa092f10752a8f033` | 1.92% | 23 / 1200 | **0.00%** (0 / 1100) | 23.0% (23 / 100) |
| **Candidate B (ISUC)** | `e838eb33e6e49cef68e6afb12e7f93bd788f1259940647dce7ef70717f512cf2` | 2.75% | 33 / 1200 | **0.00%** (0 / 1100) | 33.0% (33 / 100) |
| **Candidate C (SED)** | `32f1066f4a6b0c652d42991bcc8c1427b63854a55483ffc8a32b726512600d12` | 7.33% | 88 / 1200 | **0.00%** (0 / 1100) | 88.0% (88 / 100) |

---

## 3. Scientific Conclusions & Key Insights

1. **The In-Distribution vs. OOD Dichotomy:** Diagnostic probing showed that Candidate A could successfully recall updated locations for entities seen during training (`Aaditya` $\to$ `Bhopal`), but completely failed when querying unseen entities (`Abhay` $\to$ `Dhanbad`). The model does not execute dynamic key-value pointer routing; it utilizes token-specific embedding associations.
2. **Subword Fragmentation as an Induction Bottleneck:** In low-vocabulary BPE models (Vocab 4096), novel entities are shattered into 3 to 5 arbitrary subword tokens. At $d_{\text{head}}=32$, attention heads cannot jointly track token span bounds, relative distance offsets, and role bindings.
3. **Release Gate Recommendation:** ChakrMicro remains not ready for v0.9.1 conversational release. The primary reference checkpoint remains **CR-12 Candidate B**, and the project should advance to architectural scaling ($d_{\text{head}} \ge 64$, variable pointer pretraining) in CR-EXP-15.
