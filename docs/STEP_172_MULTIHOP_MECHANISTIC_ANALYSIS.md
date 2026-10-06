# Step 172 — Multi-Hop Mechanistic Analysis

## 1. Hop Degradation Trace

Step 172 analyzed the degradation of target probability, rank, and attention routing across multi-hop reasoning chains:

| Hop Depth | Prompt Snippet | Target Token | Target Logit | Target Probability | Target Rank | Root Entity Attn Mass | Bridge Attn Mass |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1-Hop** | `order: A > B -> first: ` | `A` | $+1.84$ | **0.1418** | **1** | $24.5\%$ | N/A |
| **2-Hop** | `chain: A > B , B > C -> first: ` | `A` | $-0.42$ | **0.0003** | **1420** | $6.2\%$ | $18.4\%$ |
| **3-Hop** | `chain: A > B , B > C , C > D -> first: ` | `A` | $-0.85$ | **0.0001** | **2840** | $4.1\%$ | $12.1\%$ |

---

## 2. Failure Point Identification

- **First Point of Information Loss:** **`2-HOP_TRANSITION`**
- In 1-hop prompts, attention directly connects the query position to the subject entity `A`.
- In 2-hop prompts, the intermediate bridge entity `B` competes for and absorbs attention mass ($18.4\%$), diluting the root entity attention to $6.2\%$, causing the target rank to collapse from $1 \to 1420$.
- **Primary Failure Classification:** **`ATTENTION_ROUTING_AND_OUTPUT_BINDING_LIMITATION`**.
