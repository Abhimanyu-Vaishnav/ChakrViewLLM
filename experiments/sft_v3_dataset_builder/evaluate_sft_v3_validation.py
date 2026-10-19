"""Non-Training Capability Evaluation Harness for SFT V3 Validation Split.

Evaluates candidate checkpoint against the newly constructed 40-record SFT V3 validation split.
Measures separately:
- Layer A: Native unconstrained generation (greedy decoding, temperature=0.0).
- Layer B: Constrained decoding (using Schema Prefix Trie).
- Exact Task Match (EM).
- JSON Syntax validity.
- Per-field value accuracy.
- Category breakdowns.

SAFETY NOTICE:
- Inference only.
- ZERO training, fine-tuning, backward passes, or optimizer steps.
- Uses validation.jsonl only; NEVER accesses test.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

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
    generate_schema_constrained,
)

CANDIDATE_CKPT = REPO_ROOT / "experiments" / "batch3111_3150" / "json_sft_v2_pilot_run" / "sft_best_checkpoint.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
VAL_SPLIT_PATH = REPO_ROOT / "experiments" / "sft_v3_dataset_builder" / "validation.jsonl"
OUTPUT_REPORT_PATH = REPO_ROOT / "experiments" / "sft_v3_dataset_builder" / "VALIDATION_CAPABILITY_EVALUATION.json"


def load_candidate_model(device: torch.device = torch.device("cpu")):
    cfg = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=6,
        n_heads=6,
        hidden_dim=512,
        max_seq_len=512,
    )
    model = ChakrMicro(cfg)
    ckpt = torch.load(str(CANDIDATE_CKPT), map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate_unconstrained(model, tokenizer, prompt_str: str, max_new_tokens: int = 48) -> str:
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
    return tokenizer.decode(gen_tokens)


def evaluate_sft_v3_validation(max_records: int = 40):
    device = torch.device("cpu")
    print(f"Loading candidate model from: {CANDIDATE_CKPT}")
    model = load_candidate_model(device)
    tokenizer = load_experiment_artifacts(TOKENIZER_DIR)

    with open(VAL_SPLIT_PATH, encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f if line.strip()][:max_records]

    results = {
        "total_records": len(val_records),
        "structured_records": 0,
        "replay_records": 0,
        "native_json_valid": 0,
        "native_exact_match": 0,
        "constrained_json_valid": 0,
        "constrained_exact_match": 0,
        "constrained_deadlocks": 0,
        "per_category": {},
        "records_detail": [],
    }

    print(f"Evaluating {len(val_records)} SFT V3 validation records...")

    for idx, r in enumerate(val_records, start=1):
        rid = r["id"]
        cat = r["category"]
        prompt = r["prompt"]
        expected = r["expected_answer"]
        req_keys = r.get("required_keys", [])
        ftypes = r.get("field_types") or {}

        if cat not in results["per_category"]:
            results["per_category"][cat] = {
                "total": 0,
                "native_valid": 0,
                "constrained_valid": 0,
                "native_em": 0,
                "constrained_em": 0,
            }
        results["per_category"][cat]["total"] += 1

        if cat == "base_continuation_replay":
            results["replay_records"] += 1
            continue

        results["structured_records"] += 1

        # 1. Native Unconstrained Generation
        native_out = evaluate_unconstrained(model, tokenizer, prompt, max_new_tokens=48)
        native_valid = False
        native_em = False
        try:
            parsed_native = json.loads(native_out.strip())
            native_valid = True
            if parsed_native == expected:
                native_em = True
        except Exception:
            pass

        if native_valid:
            results["native_json_valid"] += 1
            results["per_category"][cat]["native_valid"] += 1
        if native_em:
            results["native_exact_match"] += 1
            results["per_category"][cat]["native_em"] += 1

        # 2. Constrained Decoding
        # Build JSONSchema
        prop_specs = []
        for k in req_keys:
            t = ftypes.get(k, "string")
            if t == "integer":
                jtype = JSONType.INTEGER
            elif t == "number":
                jtype = JSONType.NUMBER
            elif t == "boolean":
                jtype = JSONType.BOOLEAN
            elif t == "null":
                jtype = JSONType.NULL
            else:
                jtype = JSONType.STRING
            prop_specs.append(PropertySpec(name=k, allowed_types={jtype}, required=True))

        schema = JSONSchema(
            properties={p.name: p for p in prop_specs},
            allow_additional_properties=False,
            allow_duplicate_keys=False,
        )
        prompt_tokens = tokenizer.encode(prompt)

        dec_res = generate_schema_constrained(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            schema=schema,
            max_new_tokens=64,
            strict_spacing=True,
        )

        gen_text = dec_res.get("generated_text", "")
        stop_reason = dec_res.get("stop_reason", "")
        parsed_c = dec_res.get("parsed_object")
        constrained_valid = (parsed_c is not None)
        constrained_em = False
        if constrained_valid:
            results["constrained_json_valid"] += 1
            results["per_category"][cat]["constrained_valid"] += 1
            if parsed_c == expected:
                constrained_em = True

        if constrained_em:
            results["constrained_exact_match"] += 1
            results["per_category"][cat]["constrained_em"] += 1

        is_deadlock = (stop_reason == "DEADLOCK" or dec_res.get("deadlock", False))
        if is_deadlock:
            results["constrained_deadlocks"] += 1

        results["records_detail"].append({
            "id": rid,
            "category": cat,
            "native_output": native_out,
            "native_valid": native_valid,
            "native_em": native_em,
            "constrained_output": gen_text,
            "constrained_valid": constrained_valid,
            "constrained_em": constrained_em,
            "constrained_stop_reason": stop_reason,
        })

        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(val_records)} records...")

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n==================================================")
    print("SFT V3 VALIDATION EVALUATION COMPLETE")
    print("==================================================")
    print(f"Structured Records:       {results['structured_records']}")
    print(f"Replay Records:           {results['replay_records']}")
    print(f"Native Valid JSON:        {results['native_json_valid']}/{results['structured_records']} ({(results['native_json_valid']/max(1, results['structured_records']))*100:.1f}%)")
    print(f"Native Exact Match:       {results['native_exact_match']}/{results['structured_records']} ({(results['native_exact_match']/max(1, results['structured_records']))*100:.1f}%)")
    print(f"Constrained Valid JSON:   {results['constrained_json_valid']}/{results['structured_records']} ({(results['constrained_json_valid']/max(1, results['structured_records']))*100:.1f}%)")
    print(f"Constrained Exact Match:  {results['constrained_exact_match']}/{results['structured_records']} ({(results['constrained_exact_match']/max(1, results['structured_records']))*100:.1f}%)")
    print(f"Constrained Deadlocks:    {results['constrained_deadlocks']}")
    print(f"Report saved to:          {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    evaluate_sft_v3_validation()
