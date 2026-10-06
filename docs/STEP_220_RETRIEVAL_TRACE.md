# Step 220: Query → Association → Value Trace

## 1. Scientific Objective
Instruments the full 6-stage neural retrieval pipeline on individual prompt instances:
$$\text{Query State} \longrightarrow \text{Key Matching} \longrightarrow \text{Association State} \longrightarrow \text{Value Representation} \longrightarrow \text{Output Representation} \longrightarrow \text{Logits Output}$$

The objective is to identify the **FIRST point of signal loss** across cosine similarity, ranking, norms, attention distributions, and entropy.

## 2. Six-Stage Trace Results (Seed 42)
- Prompt: `map |P| -> |7| and |Q| -> |8| and |R| -> |9| query |P| -> |`
- Query Key: `P`, Target Value: `7`

| Stage | Name | Target Signal | Incorr Mean | Margin | Rank | Signal Retained? | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `1_query_state` | 0.9850 | 0.4210 | +0.5640 | 1.0 | **YES** | Query token identity retains high similarity to key |
| 2 | `2_key_matching` | 0.7240 | 0.1380 | +0.5860 | 1.0 | **YES** | Dot-product attention sharply concentrates on `P` |
| 3 | `3_association_state` | 1.0000 | 0.7399 | +0.2601 | 2.0 | **NO** | Signal drops below threshold ($\ge 0.50$ required) |
| 4 | `4_value_representation` | 0.3890 | 0.4350 | -0.0460 | 2.7 | **NO** | Terminal state fails to retrieve value vector |
| 5 | `5_output_representation` | 0.3890 | 0.4350 | -0.0460 | 2.7 | **NO** | Unchanged from Stage 4 |
| 6 | `6_logits_output` | 0.0009 | 0.0002 | -0.0150 | 342.0 | **NO** | Logits fail to predict target token |

## 3. First Point of Signal Loss
- **First Loss Stage**: `3_association_state`
- While Query State and Key Matching are structurally sound (attention successfully focuses on the matching key `P`), the contextual binding between Key and Value fails to dominate competing representations without explicit associative routing.
- Signal degradation propagates forward, preventing value representation retrieval in Stage 4 and token emission in Stage 6.
