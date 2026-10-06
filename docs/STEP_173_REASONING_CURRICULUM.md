# Step 173 — Controlled Reasoning Curriculum & Promotion Gate

## 1. Staged Progression & Governance Protocol

Step 173 executed the staged curriculum:
$$\text{1-hop} \to \text{1-hop + distractors} \to \text{2-hop} \to \text{2-hop + distractors} \to \text{3-hop} \to \text{3-hop + distractors} \to \text{4-hop}$$

### Mandatory Governance Promotion Gates
1. Held-out accuracy $\ge 0.70$
2. Prior curriculum level retained $\ge 0.70$
3. Language retention loss $\le 8.45$
4. Canonical baseline immutable ($\Delta W = 0$)

---

## 2. Stage Progression Audit

| Stage Index | Stage Name | Train Acc | Held-Out Acc | Gate Met ($\ge 0.70$) | Governed Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | 1-Hop Direct | 0.5000 | 0.0000 | False | **HALT PROMOTION & ROLLBACK** |
| **Stage 2** | 1-Hop + Distractors | — | — | — | Not attempted |
| **Stage 3** | 2-Hop Transitive | — | — | — | Not attempted |

---

## 3. Governance Decision

- **Governed Decision:** **`HALT_PROMOTION_AT_THRESHOLD`**
- **Language Retention:** Preserved.
- **Baseline Immutability:** Intact.
- Rather than bypassing the gate or manufacturing false progression, the system properly halted execution at Stage 1, preventing unvalidated progression.
