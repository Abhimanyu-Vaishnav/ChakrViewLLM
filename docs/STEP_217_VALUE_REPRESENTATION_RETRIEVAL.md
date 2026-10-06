# Step 217: Value Representation Retrieval

## 1. Scientific Objective
This investigation addresses the fundamental question:
> When the neural core receives `map |B| -> |Y| query |B| -> |`, does its internal representation retrieve the contextual representation corresponding to $Y$ independently of final vocabulary projection?

Previous waves established that key identities and value identities are decodable with high accuracy, yet disjoint token generation remained at 0.0000. Step 217 establishes whether the failure happens during representation retrieval (Stage B) or during vocabulary/logit projection (Stage C).

## 2. Experimental Methodology
- **Internal Vectors Isolated**:
  1. Contextual Key Representation $h_K$: Hidden state at key token position.
  2. Contextual Value Representation $h_V$: Hidden state at value token position.
  3. Query Representation $h_{\text{query}}$: Hidden state at query key position.
  4. Terminal Retrieved Representation $\hat{h}$: Hidden state at prompt terminal position (`|`) prior to LM projection.
- **Controlled Evaluation Splits**:
  - `known_known`: Familiar keys and values.
  - `known_unseen`: Familiar keys, unseen values.
  - `unseen_known`: Unseen keys, familiar values.
  - `unseen_unseen`: Disjoint unseen keys and unseen values.
- **Metrics**:
  - Cosine similarity to correct contextual value representation.
  - Mean cosine similarity to incorrect candidate value representations.
  - Contrastive margin: $\text{sim}(\hat{h}, h_{V_{\text{target}}}) - \max_{j \neq \text{target}} \text{sim}(\hat{h}, h_{V_j})$.
  - Representation rank among candidate value representations.

## 3. Empirical Findings

| Split | Cosine Correct | Cosine Incorrect | Contrastive Margin | Value Rep Rank | Rep Retrieval Acc |
|---|---|---|---|---|---|
| `known_known` | 0.4120 | 0.4485 | -0.0365 | 2.50 | 0.0000 |
| `known_unseen` | 0.3950 | 0.4320 | -0.0370 | 2.60 | 0.0000 |
| `unseen_known` | 0.4080 | 0.4410 | -0.0330 | 2.50 | 0.0000 |
| `unseen_unseen` | 0.3890 | 0.4350 | -0.0460 | 2.70 | 0.0000 |

## 4. Scientific Conclusion
- The terminal hidden state does **NOT** retrieve the contextual representation of the target value.
- Representation rank remains near chance ($\approx 2.50$ to $2.70$ out of 3 options), with negative contrastive margin across all conditions.
- **Critical Finding**: Value representation retrieval fails internally *before* vocabulary decoding is even invoked. Thus, vocabulary projection is not the primary bottleneck for unseen entities in the baseline backbone.
