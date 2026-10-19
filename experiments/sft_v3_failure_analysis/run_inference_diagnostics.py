"""Inference-Only Diagnostics & Prompt-Contract Ablation for SFT V3 Failure Analysis.

Investigates:
1. Prompt Format Ablations (5 controlled prompt styles) on Base vs SFT V3 Candidate.
2. Progressive Extraction Task Decomposition (Task A to Task E):
   - Task A: Direct value copy from context.
   - Task B: Single-field extraction (plain text answer).
   - Task C: Single-field minimal JSON object.
   - Task D: Two-field JSON extraction.
   - Task E: Full multi-field structured task (canonical).
3. Schema Prefix Trie reachability audit:
   - Proves whether ground-truth values are mathematically reachable under trie masking.

HARD SAFETY INVARIANTS:
- Inference only.
- ZERO training, optimizer steps, or parameter updates.
- Uses validation split only; NEVER accesses test.jsonl.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer import load_experiment_artifacts
from experiments.batch3111_3150.schema_prefix_trie import (
    JSONSchema,
    JSONType,
    PropertySpec,
    SchemaConstrainedJSONValidator,
    generate_schema_constrained,
)

BASE_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
CANDIDATE_CKPT = REPO_ROOT / "experiments" / "sft_v3_pilot_readiness" / "pilot_run_v3" / "sft_best_checkpoint.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
VAL_FILE = REPO_ROOT / "experiments" / "sft_v3_dataset_builder" / "validation.jsonl"
OUTPUT_JSON = REPO_ROOT / "experiments" / "sft_v3_failure_analysis" / "EXTRACTION_DIAGNOSTICS.json"


def load_model(ckpt_path: Path, device: torch.device = torch.device("cpu")) -> ChakrMicro:
    cfg = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=6,
        n_heads=6,
        hidden_dim=512,
        max_seq_len=512,
    )
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def generate_unconstrained(model: ChakrMicro, tokenizer: Any, prompt_str: str, max_new_tokens: int = 48) -> str:
    input_ids = tokenizer.encode(prompt_str)
    curr_ids = list(input_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = curr_ids[-512:]
            x = torch.tensor([idx_cond], dtype=torch.long, device=torch.device("cpu"))
            logits = model(x)
            next_token = int(torch.argmax(logits[0, -1, :]).item())
            if next_token == 1:  # EOS
                break
            curr_ids.append(next_token)

    gen_tokens = curr_ids[len(input_ids):]
    try:
        return tokenizer.decode(gen_tokens, errors="replace")
    except TypeError:
        return tokenizer.decode(gen_tokens)


def run_diagnostics():
    device = torch.device("cpu")
    tokenizer = load_experiment_artifacts(TOKENIZER_DIR)

    with open(VAL_FILE, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if json.loads(l)["category"] != "base_continuation_replay"]

    # Select representative records across categories
    sample_records = records[:10]

    base_model = load_model(BASE_CKPT, device)
    cand_model = load_model(CANDIDATE_CKPT, device)

    diagnostics_out: Dict[str, Any] = {
        "milestone": "SFT_V3_FAILURE_ANALYSIS",
        "sample_size": len(sample_records),
        "prompt_ablations": {},
        "progressive_extraction_tasks": {},
        "schema_trie_reachability_audit": {},
    }

    # =========================================================================
    # PART 1: Prompt Format & Delimiter Ablation
    # =========================================================================
    print("Executing Prompt Format & Delimiter Ablations...")
    ablation_formats = {
        "format_1_canonical_v3": lambda ctx, inst, k1, k2: f"Context: {ctx}\nInstruction: {inst}\nResponse:",
        "format_2_explicit_json_prefix": lambda ctx, inst, k1, k2: f"Context: {ctx}\nInstruction: {inst}\nJSON Output: {{",
        "format_3_section_delimited": lambda ctx, inst, k1, k2: f"### CONTEXT\n{ctx}\n\n### TASK\n{inst}\n\n### JSON RESPONSE\n{{",
        "format_4_chat_turn_delimited": lambda ctx, inst, k1, k2: f"User: Given {ctx}, extract {k1} and {k2}.\nAssistant: {{",
        "format_5_plain_text_question": lambda ctx, inst, k1, k2: f"Context: {ctx}\nQuestion: What is the {k1} and {k2}?\nAnswer: ",
    }

    for fmt_name, fmt_fn in ablation_formats.items():
        base_attractor_count = 0
        cand_attractor_count = 0
        base_valid_json = 0
        cand_valid_json = 0

        sample_outputs = []

        for r in sample_records:
            ctx = r["context"]
            inst = r["instruction"]
            k1 = r["required_keys"][0] if len(r["required_keys"]) > 0 else "status"
            k2 = r["required_keys"][1] if len(r["required_keys"]) > 1 else "code"

            prompt_p = fmt_fn(ctx, inst, k1, k2)

            out_b = generate_unconstrained(base_model, tokenizer, prompt_p, max_new_tokens=40)
            out_c = generate_unconstrained(cand_model, tokenizer, prompt_p, max_new_tokens=40)

            # Check code attractor
            is_attr_b = ("__main__" in out_b or "import doctest" in out_b or "def " in out_b)
            is_attr_c = ("__main__" in out_c or "import doctest" in out_c or "def " in out_c)
            if is_attr_b:
                base_attractor_count += 1
            if is_attr_c:
                cand_attractor_count += 1

            # Check valid JSON
            prefix = "{" if prompt_p.endswith("{") else ""
            try:
                json.loads((prefix + out_b).strip())
                base_valid_json += 1
            except Exception:
                pass
            try:
                json.loads((prefix + out_c).strip())
                cand_valid_json += 1
            except Exception:
                pass

            sample_outputs.append({
                "id": r["id"],
                "base_output": out_b[:80],
                "cand_output": out_c[:80],
                "cand_is_attractor": is_attr_c,
            })

        diagnostics_out["prompt_ablations"][fmt_name] = {
            "base_attractor_rate": f"{base_attractor_count}/{len(sample_records)}",
            "cand_attractor_rate": f"{cand_attractor_count}/{len(sample_records)}",
            "base_valid_json_rate": f"{base_valid_json}/{len(sample_records)}",
            "cand_valid_json_rate": f"{cand_valid_json}/{len(sample_records)}",
            "sample_outputs": sample_outputs[:2],
        }

    # =========================================================================
    # PART 2: Progressive Extraction Complexity (Task A to E)
    # =========================================================================
    print("Executing Progressive Extraction Task Decomposition...")
    # Evaluate candidate model on Tasks A-E on sample records
    tasks = {
        "task_A_direct_copy": {
            "prompt_template": "Text: {ctx}\nCopy the exact word for {k1}: ",
            "target_key": 0,
            "format": "plain",
        },
        "task_B_single_field_qa": {
            "prompt_template": "Context: {ctx}\nQuestion: What is the {k1}?\nAnswer: ",
            "target_key": 0,
            "format": "plain",
        },
        "task_C_single_field_json": {
            "prompt_template": "Context: {ctx}\nExtract {k1} as JSON: {{\"{k1}\": \"",
            "target_key": 0,
            "format": "json_suffix",
        },
        "task_D_two_field_json": {
            "prompt_template": "Context: {ctx}\nExtract {k1} and {k2} as JSON: {{\"{k1}\": ",
            "target_key": "both",
            "format": "json_mid",
        },
        "task_E_full_canonical": {
            "prompt_template": "Context: {ctx}\nInstruction: {inst}\nResponse: ",
            "target_key": "both",
            "format": "full_canonical",
        },
    }

    for task_name, task_spec in tasks.items():
        correct_count = 0
        attractor_count = 0
        details = []

        for r in sample_records:
            ctx = r["context"]
            inst = r["instruction"]
            k1 = r["required_keys"][0] if len(r["required_keys"]) > 0 else "status"
            k2 = r["required_keys"][1] if len(r["required_keys"]) > 1 else "code"
            expected_val1 = str(r["expected_answer"].get(k1, ""))

            prompt_t = task_spec["prompt_template"].format(ctx=ctx, inst=inst, k1=k1, k2=k2)
            gen = generate_unconstrained(cand_model, tokenizer, prompt_t, max_new_tokens=30).strip()

            is_attr = ("__main__" in gen or "import doctest" in gen or "def " in gen)
            if is_attr:
                attractor_count += 1

            # Check if ground truth value appears in the output
            is_correct = (expected_val1.lower() in gen.lower()) and len(expected_val1) > 0
            if is_correct:
                correct_count += 1

            details.append({
                "id": r["id"],
                "expected": expected_val1,
                "generated": gen[:60],
                "contains_expected": is_correct,
                "is_attractor": is_attr,
            })

        diagnostics_out["progressive_extraction_tasks"][task_name] = {
            "value_hit_rate": f"{correct_count}/{len(sample_records)} ({(correct_count/len(sample_records))*100:.1f}%)",
            "attractor_rate": f"{attractor_count}/{len(sample_records)} ({(attractor_count/len(sample_records))*100:.1f}%)",
            "samples": details[:3],
        }

    # =========================================================================
    # PART 3: Schema Prefix Trie Reachability Audit
    # =========================================================================
    print("Auditing Schema Prefix Trie Reachability...")
    trie_reachability_results = []
    for r in sample_records:
        rid = r["id"]
        exp = r["expected_answer"]
        req_keys = r["required_keys"]
        ftypes = r["field_types"]

        prop_specs = []
        for k in req_keys:
            t = ftypes.get(k, "string")
            jtype = JSONType.INTEGER if t == "integer" else (JSONType.NUMBER if t == "number" else (JSONType.BOOLEAN if t == "boolean" else (JSONType.NULL if t == "null" else JSONType.STRING)))
            prop_specs.append(PropertySpec(name=k, allowed_types={jtype}, required=True))

        schema = JSONSchema(properties={p.name: p for p in prop_specs})
        validator = SchemaConstrainedJSONValidator(schema=schema, strict_spacing=True)

        target_json_str = r["response"]
        reachable = True
        step_failed = None

        for idx, ch in enumerate(target_json_str):
            if not validator.is_valid_string_extension(ch):
                reachable = False
                step_failed = f"Char '{ch}' at index {idx} rejected by validator"
                break
            validator.feed_char(ch)

        trie_reachability_results.append({
            "id": rid,
            "target": target_json_str,
            "is_reachable_under_trie": reachable,
            "failure_reason": step_failed,
        })

    all_reachable = all(x["is_reachable_under_trie"] for x in trie_reachability_results)
    diagnostics_out["schema_trie_reachability_audit"] = {
        "all_ground_truth_targets_reachable": all_reachable,
        "reachable_count": f"{sum(1 for x in trie_reachability_results if x['is_reachable_under_trie'])}/{len(sample_records)}",
        "records": trie_reachability_results[:3],
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(diagnostics_out, f, indent=2)

    print(f"Extraction diagnostics completed and saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    run_diagnostics()
