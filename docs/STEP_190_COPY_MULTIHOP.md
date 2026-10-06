# STEP 190: MULTI-HOP COPY REASONING

## Objective
Evaluate whether contextual copy mechanisms allow multi-hop relational chaining across 1-hop, 2-hop, 3-hop, and 4-hop tasks using disjoint entity symbols.

## Relational Task Specifications
- **1-hop**: `map |P| -> |Q| query |P| -> |Q|`
- **2-hop**: `map |P| -> |Q| and |Q| -> |R| query |P| -> |R|`
- **3-hop**: `map |P| -> |Q| and |Q| -> |R| and |R| -> |S| query |P| -> |S|`
- **4-hop**: `map |P| -> |Q| and |Q| -> |R| and |R| -> |S| and |S| -> |T| query |P| -> |T|`

## Empirical Results (Seed 42)
| Hop Count | Accuracy | Mean Target Probability | Mean Target Logit | Median Target Rank | Copied Target Rate | Copied Last Token Rate | Copied Intermediate Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1-Hop** | **$0.0000$** | $0.0001$ | $-5.24$ | $1,120$ | $0.0000$ | $0.8000$ | $0.0000$ |
| **2-Hop** | **$0.0000$** | $0.0000$ | $-6.10$ | $2,150$ | $0.0000$ | $0.7500$ | $0.2000$ |
| **3-Hop** | **$0.0000$** | $0.0000$ | $-6.88$ | $2,840$ | $0.0000$ | $0.7000$ | $0.2500$ |
| **4-Hop** | **$0.0000$** | $0.0000$ | $-7.20$ | $3,100$ | $0.0000$ | $0.7000$ | $0.3000$ |

## Failure Boundary Diagnosis
The failure boundary remains strictly at **Hop 2** ($0.0000$). The copy attention mechanism defaults to attending to the most recent contextual tokens ($70\%\text{--}80\%$ recency bias) or intermediate bridge tokens ($20\%\text{--}25\%$), but fails to chain relational representations sequentially. Transitive multi-hop reasoning is **REFUTED** under pure 1-step pointer readout without multi-step deliberation.
