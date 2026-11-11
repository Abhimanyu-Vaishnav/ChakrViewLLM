# CR-EXP-06 Baseline Diagnosis & Mechanistic Review

**Author**: Antigravity Autonomous Research Agent  
**Audit Target**: `experiments/cognition/cr_exp_05_ood_diagnosis/`  
**Reference Checkpoint**: `conversational_pilot_best_val.pt` (`ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd`)  

---

## 1. Review of Empirical Findings from CR-EXP-05

A detailed synthesis of the diagnostic artifacts in `cr_exp_05_ood_diagnosis/` establishes seven verified empirical facts regarding why the conversational candidate failed out-of-distribution entity retrieval:

### 1.1 The Dominance of the Conversational Prior
In `variable_binding_results.json`, testing synthetic forward binding (`"Context: Quasar is mapped to Rhodium. User: What does Quasar map to? Assistant:"`) showed:
- Candidate's Top-1 predicted token was **`' '`** (leading space for conversational sentences) with **88.44% probability**.
- Top-5 predictions were all generic conversational starters: `' A'`, `' S'`, `' 1'`, `' H'`.
- The true target token (`"Rhodium"`, starting with token ID 85) had a probability of only **0.00053** and ranked **#336**.
- The candidate learned a powerful prior that every `Assistant:` turn must begin with a polite greeting (e.g. `"Good morning!"`, `"Good afternoon!"`) regardless of whether the user asked an informational question.

### 1.2 Whitespace Fusion & Token Disconnect
Inspection of `tokenizer_diagnostic_results.json` demonstrated that leading whitespace alters tokenization for 100% of tested entity names:
- Standalone `"Alice"`: token ID `[1425]`
- Spaced `" Alice"`: token ID `[35, 1425]` or distinct merged subword
- If a dataset presents `"Alice lives in Seattle."` at sentence beginning (token `1425`), but the downstream question asks `"Where does Alice live?"` (where Alice has a leading space and becomes token `[35, 1425]`), the Transformer's self-attention layers must bridge between two distinct token embeddings.
- **Actionable Rule for Stage 2**: Dataset synthesis must normalize whitespace and explicitly teach both capitalized sentence-start and lowercased/spaced in-sentence entity mentions.

### 1.3 Uniform Context-Position Failure (Not "Lost in the Middle")
`context_position_results.json` evaluated facts placed at Beginning, Middle, and End across context lengths from 35 to 153 tokens:
- Beginning: Target Rank = 32, Prob = $0.00018$
- Middle: Target Rank = 30, Prob = $0.00019$
- End: Target Rank = 29, Prob = $0.00031$
- The failure is position-invariant. The model's failure to retrieve is not caused by RoPE positional decay, but by the complete absence of a trained circuit linking the query key to the context value.

### 1.4 Residual Stream Energy Dilution
`information_flow_results.json` revealed:
- Layer 0 residual stream contextual delta: $18.5\%$ relative divergence.
- By Layer 5, the total residual norm grew from $4.53$ to $15.71$ due to cumulative invariant FFN additions, while the contextual delta remained nearly static ($\sim 1.0$).
- Consequently, contextual signal-to-noise ratio drops to $<4\%$ at Layer 5 and $<2.9\%$ at final logits.

---

## 2. Specific Requirements for the CR-EXP-06 Curriculum & Loss

To overcome these mechanistic failure modes without destroying conversational persona:
1. **Direct Answer Cadence**:
   The model must be trained on examples where the assistant directly answers the question first (e.g., `"Ravi lives in Pune."` or `"The server port is 8080."`) rather than always beginning with a greeting.
2. **Explicit Relational Target Loss Weighting (Treatment Arm B)**:
   In standard SFT (Arm A), greeting tokens (`"Good morning!"`) and syntax delimiters outnumber relational answer tokens by $>20:1$. The optimizer minimizes loss by perfecting greeting fluency while ignoring entity errors. Arm B's 5.0x weighting on relational answer tokens directly forces the gradient to update weights associated with entity routing.
3. **Template & Entity Disjointness**:
   Strict OOD evaluation requires that test entities (names, cities, objects) and test prompt templates never appear in the training split.

