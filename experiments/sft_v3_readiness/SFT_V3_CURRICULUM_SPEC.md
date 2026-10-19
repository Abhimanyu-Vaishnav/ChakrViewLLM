# SFT V3 CURRICULUM SPECIFICATION
**Version**: 3.0-DRAFT  
**Scope**: Neural Grounding, Anti-Attractor Resistance, and Schema-Aware Structured JSON Generation  
**Status**: Proposal & Design Only (No Training Authorized)

---

## 1. Executive Motivation

In Batch 3111–3150 (JSON SFT V2), the model succeeded in learning JSON opening syntax (`{"`), but produced 0/20 semantically correct field values across the validation set.

### Identified Dataset Flaws in SFT V2:
1. **Lack of Input Context Passages**:
   - In SFT V2, prompts followed templates like:
     `"Instruction: Return a JSON object with fields 'status' and 'code' indicating server health status and return code.\nResponse:"`
   - **Crucial Problem**: The prompt contained **no factual information** stating what the status or code actually was! The target `{"status": "queued", "code": 202}` was an ungrounded hallucination from the model's perspective. The model was effectively forced to memorize arbitrary tuples per template index.
2. **Pretraining Code Attractor Dominance**:
   - The canonical base model has strong weights dedicated to Python code (`import doctest`, `def `, `self.`). Without explicit training showing when code is inappropriate, the model defaults to Python boilerplate inside or around braces.
3. **Leading Zero Integer Hallucination**:
   - The model frequently produced integer counts formatted with leading zeros (`"count": 060`), violating standard RFC 8259 JSON syntax.

---

## 2. Proposed SFT V3 Capability Categories

To move from superficial formatting to genuine neural comprehension, SFT V3 introduces 6 structured curriculum categories:

| Category ID | Name | Target Capability | Example Structure |
| :--- | :--- | :--- | :--- |
| **CAT-1** | `grounded_extraction_server_status` | Extract status string and HTTP status code from an explicit input narrative. | `Context: Node 14 reporting standby status with HTTP return code 200.\nInstruction: ...` -> `{"status": "standby", "code": 200}` |
| **CAT-2** | `grounded_telemetry_metric` | Extract metric name and float reading from measurement text. | `Context: Sensor telemetry reading packet_loss measured at 0.04 percent.\nInstruction: ...` -> `{"metric": "packet_loss", "value": 0.04}` |
| **CAT-3** | `grounded_feature_flag` | Extract feature name and boolean operational state (`true`/`false`). | `Context: Flag auto_save is currently active in configuration.\nInstruction: ...` -> `{"feature": "auto_save", "enabled": true}` |
| **CAT-4** | `grounded_inventory_item` | Extract item name and non-zero integer count without leading zeros. | `Context: Stock register records 120 units of patch_cable.\nInstruction: ...` -> `{"name": "patch_cable", "count": 120}` |
| **CAT-5** | `anti_code_attractor_guard` | Train model to suppress Python doctest boilerplate in JSON tasks. | Prompts explicitly directing `ONLY a JSON object, no Python code` paired with clean target JSON. |
| **CAT-6** | `missing_information_abstention` | Output explicit `null` fields or `{"error": "missing_information"}` when context is absent. | Context with missing data paired with `{"error": "missing_information", "status": null, "code": null}`. |

---

## 3. Negative-Example Policy & Learning Objective

### Theoretical Clarification:
In standard autoregressive SFT trained via cross-entropy next-token prediction:
$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^T \log P_\theta(y_t \mid x, y_{<t})$$
- **Placing a negative answer directly in the target sequence does NOT teach the model to reject it.** Standard SFT only maximizes the probability of whatever tokens are in the target.
- If negative examples are inserted as targets, the model will simply learn to output the negative text!

### The SFT V3 Policy:
1. **Initial Reference Architecture (Standard Causal SFT)**:
   - SFT V3 will maintain the robust, verified causal SFT objective.
   - Anti-attractor resistance will be taught by **conditioning prompts with negative constraint instructions** (e.g. `"Do NOT import doctest or write Python script. Return ONLY a JSON object..."`) paired with **clean, positive JSON targets**. This conditions the attention layers to route probability mass away from code tokens when negative instructions appear.
2. **Future Preference Optimization (DPO / Unlikelihood Training)**:
   - If explicit negative sampling is needed, it must be implemented via **Direct Preference Optimization (DPO)** or **Unlikelihood Loss**:
     $$\mathcal{L}_{\text{UL}}(\theta) = -\sum_{w \in \mathcal{C}_{\text{neg}}} \log (1 - P_\theta(w \mid x, y_{<t}))$$
   - This requires an authorized loss function extension, not naive insertion into `train.jsonl`.

---

## 4. Data Quality & Leakage Prevention Rules

1. **Context Grounding Rule**: Every structured JSON training record must contain an explicit `Context:` passage that unambiguously specifies the target values. Zero ungrounded memorization records permitted.
2. **Strict Template Disjointness**:
   - Train and validation splits must use completely disjoint prompt template stems.
   - For example:
     - Train: `"Context: {ctx}\nInstruction: Output a JSON object with keys '{k1}' and '{k2}' representing {desc}.\nResponse:"`
     - Val: `"Context: {ctx}\nInstruction: From the context above, extract fields '{k1}' and '{k2}' as a JSON object.\nResponse:"`
3. **Disjoint Entity & Value Sets**:
   - Entities and values in validation must never appear in training data.
   - If `"standby"` is used in training, validation must use different states (e.g. `"queued"`, `"draining"`).
4. **Replay Corpus Preservation**:
   - Maintain 25–30% continuation replay from disjoint raw text corpora (`corpus_stage_b_systems`, `corpus_english_cs`) to prevent immediate EOS regression and catastrophic forgetting of pretraining knowledge.

---

## 5. Success Criteria for Future Authorized Training

A future SFT V3 pilot may only be considered successful if:
1. **Unconstrained Valid Syntax**: Reaches $\ge 40\%$ (up from 0% in SFT V2).
2. **Unconstrained Value Semantic Accuracy**: Reaches $\ge 25\%$ (up from 0% in SFT V2).
3. **Schema-Constrained Completion**: Reaches $\ge 80\%$ (up from 50%).
4. **Immediate-EOS Regression Guard**: 4/4 passing on canonical replay probes.
5. **No Catastrophic Forgetting**: Continuation length on replay prompts remains $\ge 40$ tokens.
