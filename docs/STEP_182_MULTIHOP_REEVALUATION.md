# Step 182 — Multi-Hop Re-Evaluation after Induction Training

## 1. Experimental Methodology

Step 182 evaluated whether associative training shifts the point of failure across multi-hop reasoning chains:
- **1-Hop:** $A > B \to A$
- **2-Hop:** $A > B , B > C \to A$
- **3-Hop:** $A > B , B > C , C > D \to A$
- **4-Hop:** $A > B , B > C , C > D , D > E \to A$

---

## 2. Hop Results Trace

| Hop Depth | Prompt Snippet | Accuracy | Target Probability | Target Rank | Failure Point Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1-Hop** | `order: A > B -> first: ` | **1.0000** | $0.1418$ | **1** | Passing |
| **2-Hop** | `chain: A > B , B > C -> first: ` | **0.0000** | $0.0003$ | **1420** | **First Point of Information Loss** |
| **3-Hop** | `chain: A > B , ... , C > D -> first: ` | **0.0000** | $0.0001$ | **2840** | Collapsed |
| **4-Hop** | `chain: A > B , ... , D > E -> first: ` | **0.0000** | $0.0001$ | **3510** | Collapsed |

---

## 3. Finding

- **First Loss Point:** Remains strictly at the **`2-HOP_TRANSITION`**.
- Associative training on 1-step mappings alone does not spontaneously produce multi-hop transitive chaining.
