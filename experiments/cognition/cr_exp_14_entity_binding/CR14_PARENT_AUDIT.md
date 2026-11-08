# CR-EXP-14 PARENT AUDIT REPORT
**Auditor:** Autonomous Scientific Auditor & Neural Research Engineer  
**Audit Date:** 2026-11-08T10:10:45+05:30  
**Parent Checkpoints Audited:**
1. CR-12 Candidate B (`60271c8b5e3d26b76237dc8f986e0d8d6ada2f08815cea57ec5d9a37102d9d2e`)
2. CR-12 Candidate A (`cfb99ad7651bf64303a4fd8ab562ab9d7b0b0ed008fb3389957e97c76a05a8a2`)
3. CR-11 Candidate A (`4d14cf05df395608ed8d1ab02eeca571b6bf049574785a3d16a7ec2112a64bc5`)

---

## 1. REPRODUCTION OF CR-13 CORE FINDINGS
- **Strict Working-Memory Accuracy: 0/300 (0.00%)**:
  In CR-EXP-13, auditing of Candidate A ($43.0\%$ loose hit rate) and Candidate B ($27.33\%$ loose hit rate) showed that on queries regarding held-out entities (e.g., `Aalok`), Candidate A predicted `"Ananya is currently in Amritsar"` and Candidate B predicted `"Shreya is currently in Surat"`.
  When requiring that the response match **both the correct subject entity and the correct latest value**, accuracy was **0.00% (0/300)**.
- **Subject Accuracy vs. Value Accuracy Disconnect**:
  The causal attention mechanism successfully routes to recent values in context ($27\%-45\%$), but completely unbinds from open-vocabulary query subjects, defaulting to high-frequency training attractor subjects (`Ananya`, `Shreya`, `Aarav`, `Kabir`).

---

## 2. PARENT SELECTION DECISION
- **Primary Parent Checkpoint**: **CR-12 Candidate B** (`60271c8b...`).
- **Rationale**:
  Candidate B achieved the highest statistically confirmed overall capability on the frozen 1,500 benchmark ($34.13\%$, paired McNemar $p = 0.00026$), with the best counterfactual update ($45.0\%$) and distractor handling ($24.5\%$), while avoiding Candidate A's catastrophic collapse in logic ($10\%$) and single-hop retrieval ($0\%$).
- Working copy created and cryptographically verified at `experiments/cognition/cr_exp_14_entity_binding/checkpoints/baseline_cr12_b/cr_exp_12_candidate_b_best_val_copy.pt`.
