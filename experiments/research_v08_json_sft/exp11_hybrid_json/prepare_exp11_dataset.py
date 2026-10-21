"""
prepare_exp11_dataset.py
ChakrView Experiment 11 — Synthetic Structured JSON Extraction Dataset Generator

Generates a controlled, deterministic structured dataset with:
- Context containing multiple key-value pairs and distractors.
- Instruction requesting extraction of 1, 2, or 3 fields ("entity", "status", "code").
- Target structured JSON object.
- Ground truth token positions and spans for copied values.
- Identification of which tokens are structural JSON syntax vs copied context values.
- Missing field handling ("none" when field is requested but absent from context).
- Strict separation between Familiar entities (Train, Val, Test A) and Held-Out entities (Test B).

Outputs:
- train.jsonl (300 records)
- val.jsonl (60 records)
- test_a_familiar.jsonl (60 records)
- test_b_held_out.jsonl (60 records)
- dataset_manifest.json
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXP11_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP11_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

# Familiar vocabulary (seen in training, validation, Test A)
FAMILIAR_ENTITIES = [
    "subsystem_alpha", "hydraulic_actuator", "power_converter",
    "thermal_sensor", "propulsion_thruster", "solar_inverter"
]
FAMILIAR_STATUSES = [
    "nominal", "active", "degraded", "overheated", "idle",
    "calibrated", "throttled", "standby_normal"
]
FAMILIAR_CODES = [
    "101", "204", "305", "410", "512", "620"
]

# Held-out vocabulary (ONLY in Test B)
HELD_OUT_ENTITIES = [
    "cryo_pump_01", "optical_relay_02", "quantum_core_03",
    "flux_capacitor_04", "laser_emitter_05", "plasma_duct_06"
]
HELD_OUT_STATUSES = [
    "standby_degraded", "beam_misaligned", "cavity_depressurized",
    "flux_fluctuation", "phase_jitter", "quench_imminent"
]
HELD_OUT_CODES = [
    "801", "832", "870", "895", "912", "950"
]

TEMPLATES = [
    "Telemetry Report: System [{entity}] status is '{status}' with code {code}.",
    "Log Notice: Monitored [{entity}] currently '{status}', diagnostic code: {code}.",
    "System Alert: Unit [{entity}] indicated '{status}' (sys_code={code}).",
    "Status Packet: Check on [{entity}] returned '{status}', reference={code}."
]

INSTRUCTION_TEMPLATES = {
    "all": "Extract entity, status, and code as JSON.",
    "entity_status": "Extract entity and status as JSON.",
    "status_code": "Extract status and code as JSON.",
    "entity_code": "Extract entity and code as JSON.",
    "status_only": "Extract status as JSON.",
    "entity_only": "Extract entity as JSON.",
}

def generate_record(
    example_id: str,
    entities_pool: List[str],
    statuses_pool: List[str],
    codes_pool: List[str],
    rng: random.Random,
    tokenizer: Any,
    missing_field_prob: float = 0.1
) -> Dict[str, Any]:
    entity = rng.choice(entities_pool)
    status = rng.choice(statuses_pool)
    code = rng.choice(codes_pool)

    # Distractor entity and status
    distractor_entity = rng.choice([e for e in entities_pool if e != entity])
    distractor_status = rng.choice([s for s in statuses_pool if s != status])
    distractor_code = rng.choice([c for c in codes_pool if c != code])

    template = rng.choice(TEMPLATES)
    primary_clause = template.format(entity=entity, status=status, code=code)
    distractor_clause = f"Secondary check: [{distractor_entity}] reported '{distractor_status}' with code {distractor_code}."

    # Interleave or combine clauses
    if rng.random() < 0.5:
        context = f"{primary_clause} {distractor_clause}"
    else:
        context = f"{distractor_clause} {primary_clause}"

    req_type = rng.choice(list(INSTRUCTION_TEMPLATES.keys()))
    instruction = INSTRUCTION_TEMPLATES[req_type]

    # Check for missing field injection
    has_missing = False
    missing_key = None
    if rng.random() < missing_field_prob:
        has_missing = True
        missing_key = rng.choice(["code", "status"])
        if missing_key == "code":
            # Remove code from context
            context = context.replace(f"code {code}", "code not_reported")
            context = context.replace(f"sys_code={code}", "sys_code=none")
            context = context.replace(f"reference={code}", "reference=none")
        elif missing_key == "status":
            context = context.replace(f"'{status}'", "'unknown'")

    # Build target JSON object
    target_dict = {}
    if "entity" in req_type or req_type == "all":
        target_dict["entity"] = entity
    if "status" in req_type or req_type == "all":
        target_dict["status"] = "none" if (has_missing and missing_key == "status") else status
    if "code" in req_type or req_type == "all":
        target_dict["code"] = "none" if (has_missing and missing_key == "code") else code

    target_json_str = json.dumps(target_dict, separators=(", ", ": "))
    full_prompt = f"Context: {context}\nInstruction: {instruction}\nJSON: "

    # Tokenize prompt and target
    p_toks = tokenizer.encode(full_prompt)
    full_seq = [0] + p_toks  # Include BOS token (0)

    # Locate source spans in prompt
    copy_spans = {}
    for key, val in target_dict.items():
        if val != "none":
            # Find contiguous occurrences
            val_toks = tokenizer.encode(val)
            # Find in full_seq
            found = False
            for i in range(1, len(full_seq) - len(val_toks) + 1):
                if full_seq[i:i+len(val_toks)] == val_toks:
                    copy_spans[key] = {
                        "value": val,
                        "tokens": val_toks,
                        "start_token_index": i,
                        "end_token_index": i + len(val_toks) - 1,
                        "span_len": len(val_toks)
                    }
                    found = True
                    break
            if not found:
                # Text substring fallback
                idx = full_prompt.find(val)
                if idx != -1:
                    pre_toks = tokenizer.encode(full_prompt[:idx])
                    start_pos = len(pre_toks) + 1  # offset for BOS
                    copy_spans[key] = {
                        "value": val,
                        "tokens": val_toks,
                        "start_token_index": start_pos,
                        "end_token_index": start_pos + len(val_toks) - 1,
                        "span_len": len(val_toks)
                    }

    return {
        "example_id": example_id,
        "prompt": full_prompt,
        "context": context,
        "instruction": instruction,
        "target_json": target_json_str,
        "target_dict": target_dict,
        "copy_spans": copy_spans,
        "has_missing": has_missing,
        "missing_key": missing_key
    }

def main():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    rng_train = random.Random(1101)
    rng_val = random.Random(1102)
    rng_test_a = random.Random(1103)
    rng_test_b = random.Random(1104)

    # 1. Train Split (300 records, familiar pool)
    train_records = [
        generate_record(f"TR_{i:04d}", FAMILIAR_ENTITIES, FAMILIAR_STATUSES, FAMILIAR_CODES, rng_train, tokenizer)
        for i in range(300)
    ]

    # 2. Validation Split (60 records, familiar pool)
    val_records = [
        generate_record(f"VAL_{i:04d}", FAMILIAR_ENTITIES, FAMILIAR_STATUSES, FAMILIAR_CODES, rng_val, tokenizer)
        for i in range(60)
    ]

    # 3. Test A Split (60 records, familiar pool in novel context combinations)
    test_a_records = [
        generate_record(f"TA_{i:04d}", FAMILIAR_ENTITIES, FAMILIAR_STATUSES, FAMILIAR_CODES, rng_test_a, tokenizer)
        for i in range(60)
    ]

    # 4. Test B Split (60 records, completely held-out entities, statuses, and codes)
    test_b_records = [
        generate_record(f"TB_{i:04d}", HELD_OUT_ENTITIES, HELD_OUT_STATUSES, HELD_OUT_CODES, rng_test_b, tokenizer)
        for i in range(60)
    ]

    splits = {
        "train.jsonl": train_records,
        "val.jsonl": val_records,
        "test_a_familiar.jsonl": test_a_records,
        "test_b_held_out.jsonl": test_b_records,
    }

    manifest = {}
    for filename, recs in splits.items():
        path = DATA_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        manifest[filename] = {
            "count": len(recs),
            "path": str(path.relative_to(REPO_ROOT)),
        }
        print(f"Wrote {len(recs)} records to {path}")

    manifest_path = DATA_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved dataset manifest to {manifest_path}")

if __name__ == "__main__":
    main()
