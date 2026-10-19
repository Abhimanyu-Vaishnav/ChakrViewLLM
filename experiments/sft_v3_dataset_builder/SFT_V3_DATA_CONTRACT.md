# SFT V3 DATA CONTRACT SPECIFICATION
**Version:** 3.0.0  
**Status:** Canonical Design for SFT V3 Grounded Extraction & Formatting  
**Target Architecture:** ChakrMicro (3.44M Parameters, Context: 512, Vocab: 4096)  
**Safety Gate:** Engineering Specification Only (No Training Authorized)

---

## 1. Principles and Data Architecture

ChakrView is an indigenous neural brain developed from scratch. To achieve true learned intelligence rather than superficial token sequence mimicry, SFT V3 abandons ungrounded target generation.

### Core Data Principles:
1. **Context Grounding**: Every structured data record MUST contain a verifiable `context` string. Target fields and values must be directly derivable from this context.
2. **Separation of Model Input vs Evaluation Labels**:
   - `prompt`: The formatted text provided to the model during training and inference (e.g. `Context: ...\nInstruction: ...\nResponse:`).
   - `expected_answer`: Parsed canonical object used strictly by automated evaluators and loss metrics.
   - Evaluation labels, target key lists, and validation rules are NEVER serialized into the model prompt.
3. **Template-Family Disjointness**: Train and validation sets must utilize mutually exclusive `template_family_id`s to ensure the model generalizes to novel instruction forms rather than memorizing template stems.
4. **Anti-Attractor Formatting**:
   - Compact JSON targets without superfluous newlines or 4-space indents outside of strings (`{"key": "value"}`).
   - Dedicated anti-attractor prompts that condition the model against pretraining code attractors (`import doctest`, `def main()`).
5. **Exact Type Fidelity**:
   - String fields contain strings.
   - Integer fields contain integers (RFC 8259 compliant, no leading zeros like `060`).
   - Float fields contain decimals.
   - Boolean fields contain `true` or `false`.
   - Null fields contain `null` (used specifically for missing information abstention).

---

## 2. Record Schema Specification

Each record in `train.jsonl` and `validation.jsonl` is a JSON line conforming to the following structure:

```json
{
  "id": "SFT3_TRN_EXT_001",
  "category": "grounded_extraction_server_status",
  "template_family_id": "server_status_reporting_train",
  "context": "Cluster node 14 reporting standby status with HTTP return code 200.",
  "instruction": "Extract the node status and code as a JSON object with keys 'status' and 'code'.",
  "prompt": "Context: Cluster node 14 reporting standby status with HTTP return code 200.\nInstruction: Extract the node status and code as a JSON object with keys 'status' and 'code'.\nResponse:",
  "response": "{\"status\": \"standby\", \"code\": 200}",
  "expected_answer": {
    "status": "standby",
    "code": 200
  },
  "field_types": {
    "status": "string",
    "code": "integer"
  },
  "required_keys": ["status", "code"],
  "optional_keys": [],
  "difficulty": "standard",
  "provenance": "SYNTHETIC_GROUNDED_CURRICULUM_V3",
  "data_role": "TRAIN",
  "validation_rule": "exact_key_value_match"
}
```

### Field Definitions:

| Field Name | Type | Description | Visibility to Model |
| :--- | :--- | :--- | :--- |
| `id` | `string` | Unique identifier formatted as `SFT3_{SPLIT}_{CAT}_{IDX}` | Metadata only |
| `category` | `string` | Capability category (e.g. `grounded_extraction_server_status`, `continuation_replay`) | Metadata only |
| `template_family_id` | `string` | Identifier for the prompt phrasing family. Must be disjoint between splits | Metadata only |
| `context` | `string` or `null` | The factual narrative or source text containing the required facts | Component of Prompt |
| `instruction` | `string` or `null` | The directive instructing the model on the task and schema keys | Component of Prompt |
| `prompt` | `string` | The complete string passed to the tokenizer as input | **Model Input** |
| `response` | `string` | The canonical target sequence predicted autoregressively | **Model Target** |
| `expected_answer` | `dict` or `string` | Deterministic parsed ground truth for semantic evaluation | Evaluation only |
| `field_types` | `dict` or `null` | Expected JSON types per field (`string`, `integer`, `number`, `boolean`, `null`) | Evaluation only |
| `required_keys` | `list[string]` | Schema keys that MUST be present | Evaluation only |
| `optional_keys` | `list[string]` | Schema keys that MAY be present | Evaluation only |
| `difficulty` | `string` | `standard`, `edge_case`, `abstention`, or `replay` | Metadata only |
| `provenance` | `string` | Source marker (e.g. `SYNTHETIC_GROUNDED_CURRICULUM_V3`, `CORPUS_STAGE_B_SYSTEMS`) | Metadata only |
| `data_role` | `string` | `TRAIN` or `VAL` | Metadata only |
| `validation_rule` | `string` | Deterministic validation logic (`exact_key_value_match`, `replay_token_continuation`) | Evaluation only |

---

## 3. Delimiter and Tokenization Contract

1. **Delimiters**:
   - All instruction examples end prompt text with `\nResponse:`.
   - Structured responses begin immediately with `{"` (no leading whitespace or newline).
   - This prevents triggering the pretraining multi-line code indentation basin.
2. **Tokenization Alignment (`chakrview/training/prompt_contract.py`)**:
   - `input_seq = [BOS] + prompt_tokens + response_tokens`
   - `target_seq = [-100] * len(prompt_tokens) + response_tokens + [EOS]`
   - `input_seq` and `target_seq` have identical lengths.
   - Prompt tokens receive loss mask `-100`.
   - The first token predicting the response is strictly aligned with `response_tokens[0]`.
3. **Replay Continuation Contract**:
   - Replay continuation prompts do not use `\nResponse:`.
   - Replay records use `append_eos = False` to teach natural continuation without premature termination.

---

## 4. Capability Categories & Ground-Truth Rules

The dataset is partitioned into 6 grounded structured categories plus 1 replay continuation category:

1. **Category 1: `grounded_extraction_server_status`**
   - Context: Server/cluster telemetry narrative specifying an operating status and HTTP code.
   - Schema: `{"status": <string>, "code": <int>}`
   - Correctness Rule: `parsed["status"] == context_status and parsed["code"] == context_code`

2. **Category 2: `grounded_telemetry_metric`**
   - Context: Sensor readings specifying a physical or system metric and float measurement.
   - Schema: `{"metric": <string>, "value": <float>}`
   - Correctness Rule: `parsed["metric"] == context_metric and abs(parsed["value"] - context_val) < 1e-4`

3. **Category 3: `grounded_feature_flag`**
   - Context: Configuration status narrative stating whether a feature is active/enabled or disabled/inactive.
   - Schema: `{"feature": <string>, "enabled": <bool>}`
   - Correctness Rule: `parsed["feature"] == context_feature and parsed["enabled"] is context_bool`

4. **Category 4: `grounded_inventory_item`**
   - Context: Logistics inventory report specifying an item and integer quantity.
   - Schema: `{"name": <string>, "count": <int>}`
   - Correctness Rule: `parsed["name"] == context_name and parsed["count"] == context_count` (Strictly no leading zero in string representation).

5. **Category 5: `anti_code_attractor_guard`**
   - Context: Technical system reporting accompanied by prompt instructions explicitly commanding: *"Do NOT write Python code or import doctest. Output ONLY a valid JSON object."*
   - Schema: Varied grounded 2-key schemas.
   - Correctness Rule: Must parse as JSON, contain exact context values, and contain 0 Python code tokens.

6. **Category 6: `missing_information_abstention`**
   - Context: System report indicating an event occurred, but the requested field was not measured or unavailable.
   - Schema: `{"error": "missing_information", "<key>": null}`
   - Correctness Rule: Must explicitly return `null` for the missing field, proving the model does not invent values.

7. **Category 7: `base_continuation_replay`**
   - Context: High-quality natural text from pretraining corpora (`corpus_english_cs`, `corpus_stage_b_systems` for train; `corpus_english_systems` for val).
   - Correctness Rule: Natural language continuation maintaining pretraining language capabilities and preventing immediate-EOS collapse.
