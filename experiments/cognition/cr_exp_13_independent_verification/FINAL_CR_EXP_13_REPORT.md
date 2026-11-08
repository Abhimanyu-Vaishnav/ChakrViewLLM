# FINAL CR-EXP-13 MISSION REPORT
## Independent Capability Verification, Statistical Reassessment & Generalization Audit

**Mission Codename:** `CR-EXP-13`  
**Execution Timestamp:** 2026-11-08T08:39:00+05:30 to 2026-11-08T10:08:00+05:30  
**Role:** Independent Scientific Auditor & Neural Research Verification Division  
**Model Architecture:** ChakrMicro v0.1 ($3,443,136$ parameters, $d_{\text{model}}=192, n_{\text{layers}}=6, n_{\text{heads}}=6$, Vocab 4096 ByteBPE, RoPE)  
**Host Runtime:** Python 3.14.7, PyTorch 2.14.0+cpu, Windows 11 (AMD64)  
**Git Branch:** `exp/cbr-01-conversational-breakthrough`  

---

## 1. MANDATORY QUESTIONS ANSWERED

### 1. Which CR-12 improvements are independently reproducible?
- All reported accuracy numerators and denominators for the 5 evaluated models on the frozen 1,500-probe benchmark were **100% numerically reproduced to the exact integer probe count**:
  - CR-10 Arm B: $17.33\%$ (260/1500)
  - CR-11 Candidate A: $29.33\%$ (440/1500)
  - CR-11 Candidate B: $28.33\%$ (425/1500)
  - CR-12 Candidate A: $31.87\%$ (478/1500)
  - CR-12 Candidate B: $34.13\%$ (512/1500)

### 2. Which improvements survive strict unseen evaluation?
- **Counterfactual State Updates ($45.00\%$)**: Confirmed genuine recency gating. The model actively selects updated locations over stale locations on unseen entities.
- **Multi-Entity Distractor Disambiguation ($24.50\%$)**: Confirmed. Successfully isolates the target entity's value amidst 3 concurrent entities.
- **Elementary Arithmetic Rebound ($5.33\%$)**: Rebounded from complete 0% output collapse.

### 3. Is multi-turn memory genuinely demonstrated?
- **NO — The Subject Substitution Illusion**:
  - Under the evaluator's loose matching criterion (`target_city in generated`), Candidate A scored $43.0\%$ and Candidate B scored $27.33\%$.
  - However, inspection of the actual generated strings reveals that the model generated familiar training names (e.g., `'Ananya is currently in Amritsar'` when the subject was `Aalok`).
  - Under strict scoring requiring **both** the correct subject and correct updated location, **all models score 0.00% (0/300)** on unseen names. The network learned recency value routing, but decoupled the value from the subject token.

### 4. Is entity binding reliable?
- **Partially in Direct Context, Completely Unbound in Multi-Turn Generation**:
  - In direct context with short spans, the network can match adjacent tokens ($24.5\%$).
  - In multi-turn dialogue generation, the network suffers from complete subject substitution ($0\%$ strict binding on unseen names).

### 5. Is Candidate A or B actually better?
- **Candidate B is objectively better**.
  - Candidate A suffered severe collapse in conditional logic ($50\% \to 10\%$) and single-hop retrieval ($8\% \to 0\%$).
  - Candidate B preserves balanced stability across all 9 domains and achieves the highest overall accuracy ($34.13\%$).

### 6. Does CR-12 justify the reported statistical significance?
- **YES, and the true paired significance is even stronger**.
  - CR-12 reported an unpaired two-proportion Z-test ($p = 0.0047$).
  - Our paired **McNemar test with continuity correction** ($N=1,500, b+c=378$) yielded $\chi^2 = 13.336, \quad \mathbf{p = 0.00026036}$ ($p < 0.001$).
  - Candidate B's advantage over Parent Candidate A is robust and survives strict Holm-Bonferroni correction ($p = 0.00078$).

### 7. Is v0.9.1 ready?
- **OFFICIAL DECISION: NOT READY — CAPABILITY GAPS REMAIN.**
- 10 out of 11 release gates are unequivocally failed. Multi-turn working memory on unseen entities sits at 0.0% strict accuracy. Release is withheld.

### 8. What is the single largest remaining capability bottleneck?
- **Dynamic Subword Attention Binding in Generative Heads**:
  The causal attention mechanism successfully routes to recent values, but fails to bind open-vocabulary subwords in the query to the corresponding entity in the prompt during generation.

### 9. What should CR-EXP-14 investigate?
- **CR-EXP-14 Recommendation**:
  Investigate **Entity-Token Cross-Attention Anchoring**: Test whether structured delimiter tagging (e.g., XML/bracket entity tags `<ent>Name</ent>`) enables 3.44M attention heads to preserve subject identity without substituting familiar training names.

---

## 2. FINAL AUDIT MATRIX
- **Canonical Runtime**: `chakrview/runtime/session.py` remains verified and unmodified (`9080ff51...`).
- **All 15 Required Deliverables** produced and validated under `experiments/cognition/cr_exp_13_independent_verification/`.
