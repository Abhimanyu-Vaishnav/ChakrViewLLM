# STEP 195: VALUE-POSITION RETRIEVAL OBJECTIVE

## Objective
Investigate whether the network, having identified the matching key position, can learn to route to the associated value position.

## Retrieval Circuit Architecture
- Two-stage differentiable retrieval head ($49,152$ parameters):
  1. $q_{\text{match}} \cdot k_{\text{match}} \to \alpha_{\text{key}}$ (key attention distribution).
  2. Matched key state $h_{\text{matched}} = \sum_i \alpha_i h_{k_i}$ queries value positions: $S_{\text{val}} = \frac{W_{vq} h_{\text{matched}} \cdot W_{vk} h_{v_j}}{\sqrt{d_{\text{match}}}}$.

## Diagnostic Matrix (Seed 42)
| Matching Key Position | Value Position | Frequency Rate | Interpretation |
| :---: | :---: | :---: | :--- |
| **Correct** | **Correct** | **$0.5000$** | Successful end-to-end routing |
| **Correct** | **Incorrect** | **$0.5000$** | **Primary Failure Point**: Key matched, but value address routing fails |
| **Incorrect** | **Correct** | **$0.0000$** | No accidental value captures |
| **Incorrect** | **Incorrect** | **$0.0000$** | Key matching was consistently correct |

## Empirical Metrics
- Value-position retrieval accuracy: **$0.5000$**
- Disjoint value position accuracy: **$0.7000$** (diagnostic routing transfer)
- Correct value attention mass: **$0.5512$**
- Wrong value attention mass: **$0.2244$**

## Critical Insight
The diagnostic matrix localizes the exact failure point: **Key $\to$ Value routing ($50\%$ failure rate)**. Even when the key is identified with certainty, routing from the key's contextual slot to its adjacent associated value slot is prone to dispersion across context distractors.
