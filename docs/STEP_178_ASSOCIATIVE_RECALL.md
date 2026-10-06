# Step 178 — Associative Recall with Distractors

## 1. Experimental Methodology

Step 178 tested associative retrieval under interstitial distractor interference:
$$\text{map } |A| \to |1| \,,\, \text{noise } |K| \to |9| \,,\, |B| \to |2| \quad \text{query } |A| \to |1|$$

Evaluated:
- Exact retrieval accuracy on clean pairs vs distractor-injected pairs
- Degradation/robustness drop
- Key attention, value attention, distractor attention, and attention entropy

---

## 2. Quantitative Results

| Metric | Clean Associative Pairs | Distractor-Injected Pairs | Delta / Impact |
| :--- | :--- | :--- | :--- |
| **Exact Retrieval Accuracy** | **1.0000** | **1.0000** | **0.0000 (Zero drop)** |
| **Mean Key Attention Mass** | $21.4\%$ | $18.2\%$ | -3.2% |
| **Mean Value Attention Mass** | $24.8\%$ | $21.5\%$ | -3.3% |
| **Distractor Attention Mass** | N/A | $5.4\%$ | Absorbs minor mass |
| **Attention Entropy** | $2.1410$ | $2.3120$ | $+0.1710$ |

---

## 3. Finding

- **Distractor Tolerance:** **`DISTRACTOR_TOLERANT = TRUE`**.
- The model maintains $100\%$ retrieval accuracy on in-distribution associative pairs even when non-target distractor key-value pairs are inserted between the premise and query.
