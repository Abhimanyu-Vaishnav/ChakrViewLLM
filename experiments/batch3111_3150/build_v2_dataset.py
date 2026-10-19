"""Dataset Builder for Batch 3111–3150 (JSON SFT V2: Syntax Completion & Multi-Instance Learning).

Design Principles:
1. Multi-instance learning across 4 core structural schemas:
   - Schema A (Server/System): {"status": "<string>", "code": <int>}
   - Schema B (Network/Metric): {"metric": "<string>", "value": <float>}
   - Schema C (Feature/State): {"feature": "<string>", "enabled": <bool>}
   - Schema D (Account/Item): {"name": "<string>", "count": <int>}
   Providing 35 diverse instances per core schema (140 total structured train examples)
   with varied values, strings, numbers, booleans, and prompt phrasing.
2. 30% Natural text/code/math unconditioned continuation replay (60 examples)
   with strict document-level grouping across train, val, and test.
3. Separation of Evaluation Objectives:
   - Familiar-Schema Evaluation: 12 test cases testing syntax completion and value/schema adherence on known schemas.
   - Genuinely Unseen-Schema Evaluation: 12 test cases testing zero-shot generalization to novel schemas.
   - 10 Unseen Replay Continuation cases.
4. Total Dataset Pool: 264 records (200 train, 30 val, 34 test).
   - Train: 200 (140 structured JSON [4 core schemas x 35 instances], 60 replay)
   - Val: 30 (20 structured JSON [4 core schemas x 5 held-out instances], 10 replay)
   - Test: 34 (12 familiar-schema JSON, 12 genuinely unseen-schema JSON, 10 replay)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_DIR = REPO_ROOT / "experiments" / "batch3111_3150"
HISTORICAL_BENCHMARK = REPO_ROOT / "experiments" / "batch2781_2820" / "GENERATION_BENCHMARK.json"
PREV_TEST_BATCH3021 = REPO_ROOT / "experiments" / "batch3021_3060" / "test.jsonl"
PREV_TEST_BATCH3071 = REPO_ROOT / "experiments" / "batch3071_3110" / "test.jsonl"


# 1. CORE SCHEMAS & VALUE GENERATION
CORE_SCHEMAS = {
    "server_status": {
        "keys": ["status", "code"],
        "desc": "server health status and return code",
        "instances": [
            ("ready", 200), ("active", 200), ("healthy", 200), ("ok", 200), ("idle", 200),
            ("running", 200), ("standby", 200), ("connected", 200), ("online", 200), ("operational", 200),
            ("degraded", 206), ("starting", 202), ("accepted", 202), ("processing", 202), ("partial", 206),
            ("warning", 299), ("reloading", 200), ("paused", 200), ("busy", 200), ("initialized", 200),
            ("unhealthy", 500), ("error", 500), ("failed", 500), ("timeout", 504), ("unavailable", 503),
            ("gateway_error", 502), ("bad_request", 400), ("unauthorized", 401), ("forbidden", 403), ("not_found", 404),
            ("conflict", 409), ("rate_limited", 429), ("maintenance", 503), ("overloaded", 503), ("terminated", 500),
            # 5 held-out validation instances
            ("queued", 202), ("draining", 200), ("internal_error", 500), ("service_down", 503), ("closed", 410),
            # 3 familiar-schema test instances
            ("restarting", 200), ("unreachable", 504), ("syncing", 202),
        ],
    },
    "network_metric": {
        "keys": ["metric", "value"],
        "desc": "telemetry metric name and measured reading",
        "instances": [
            ("cpu_usage", 42.5), ("memory_used", 78.2), ("disk_io", 12.4), ("latency_ms", 5.8), ("packet_loss", 0.02),
            ("bandwidth_gbps", 8.4), ("load_avg", 1.85), ("temperature_c", 54.2), ("voltage_v", 12.04), ("fan_speed_pct", 65.0),
            ("request_rate", 240.5), ("cache_miss_pct", 4.12), ("error_rate", 0.05), ("jitter_ms", 1.25), ("ping_rtt", 14.8),
            ("queue_depth", 18.0), ("connection_count", 450.0), ("throughput_mb", 95.4), ("gpu_load_pct", 88.5), ("power_watts", 185.0),
            ("headroom_pct", 32.5), ("pressure_psi", 14.7), ("frequency_ghz", 3.6), ("utilization_pct", 62.1), ("loss_rate", 0.01),
            ("read_ops_sec", 152.0), ("write_ops_sec", 88.0), ("heap_committed", 512.5), ("thread_count", 64.0), ("buffer_usage", 22.8),
            ("response_time", 18.4), ("saturation", 0.45), ("flow_rate", 10.2), ("clock_skew_ns", 4.5), ("entropy_score", 7.92),
            # 5 held-out validation instances
            ("vibration_g", 0.12), ("ambient_temp", 24.5), ("drop_rate", 0.03), ("noise_db", 42.0), ("rssi_dbm", -65.4),
            # 3 familiar-schema test instances
            ("signal_strength", 92.4), ("core_temp_c", 68.5), ("ingress_mbps", 420.8),
        ],
    },
    "feature_flag": {
        "keys": ["feature", "enabled"],
        "desc": "system feature flag name and operational state",
        "instances": [
            ("dark_mode", True), ("beta_chat", True), ("telemetry", True), ("auto_save", True), ("gpu_accel", True),
            ("caching", True), ("logging", True), ("compression", True), ("audit_log", True), ("metrics_export", True),
            ("strict_mode", True), ("tracing", True), ("sandbox", True), ("ssl_verification", True), ("keep_alive", True),
            ("notifications", False), ("debug_tools", False), ("mock_mode", False), ("legacy_auth", False), ("verbose_logging", False),
            ("profiler", False), ("dry_run", False), ("unsafe_eval", False), ("experimental_v2", False), ("maintenance_lock", False),
            ("read_only_mode", False), ("rate_limiter_bypass", False), ("deprecated_api", False), ("auto_deploy", False), ("insecure_http", False),
            ("hardware_passthrough", True), ("background_sync", True), ("fast_recovery", True), ("shadow_mode", False), ("canary_rollout", True),
            # 5 held-out validation instances
            ("memory_checkpointing", True), ("vector_cache", True), ("fallback_relay", False), ("strict_json", True), ("anonymous_access", False),
            # 3 familiar-schema test instances
            ("streaming_output", True), ("zero_copy_buffer", True), ("fault_injection", False),
        ],
    },
    "inventory_item": {
        "keys": ["name", "count"],
        "desc": "inventory item name and available count",
        "instances": [
            ("server_rack", 12), ("power_supply", 45), ("ram_module", 128), ("ssd_drive", 64), ("ethernet_switch", 8),
            ("patch_cable", 250), ("cooling_fan", 90), ("cpu_heatsink", 32), ("mounting_bracket", 180), ("fiber_transceiver", 55),
            ("battery_backup", 16), ("terminal_block", 220), ("sensor_probe", 40), ("chassis_panel", 24), ("power_strip", 38),
            ("fuse_cartridge", 150), ("network_card", 72), ("serial_adapter", 85), ("thermal_paste", 30), ("cable_tie", 500),
            ("docking_station", 20), ("display_adapter", 44), ("antenna_mast", 10), ("grounding_strap", 100), ("led_indicator", 300),
            ("relay_switch", 80), ("transformer", 15), ("capacitor_pack", 60), ("resistor_kit", 200), ("voltage_regulator", 75),
            ("cooling_radiator", 14), ("air_filter", 95), ("screw_assortment", 450), ("storage_enclosure", 18), ("keypad_module", 35),
            # 5 held-out validation instances
            ("hard_drive_tray", 50), ("buzzer_alarm", 42), ("dust_cover", 110), ("usb_hub", 28), ("wire_harness", 65),
            # 3 familiar-schema test instances
            ("server_blade", 16), ("fiber_jumper", 140), ("safety_goggles", 25),
        ],
    },
}

# 2. GENUINELY UNSEEN SCHEMAS FOR TEST SPLIT (12 cases)
UNSEEN_TEST_SCHEMAS = [
    {
        "keys": ["pilot_callsign", "heading_deg"],
        "desc": "flight navigation parameters",
        "target": {"pilot_callsign": "Viper1", "heading_deg": 270},
    },
    {
        "keys": ["satellite_name", "apogee_km"],
        "desc": "orbital mechanics data",
        "target": {"satellite_name": "Astra7", "apogee_km": 35786},
    },
    {
        "keys": ["submersible_hull", "crush_depth_m"],
        "desc": "deep ocean exploration specs",
        "target": {"submersible_hull": "Titanium", "crush_depth_m": 4500},
    },
    {
        "keys": ["particle_type", "spin"],
        "desc": "quantum physics measurement",
        "target": {"particle_type": "Muon", "spin": 0.5},
    },
    {
        "keys": ["engine_stage", "thrust_kn"],
        "desc": "rocket propulsion telemetry",
        "target": {"engine_stage": "Stage2", "thrust_kn": 934.0},
    },
    {
        "keys": ["chemical_compound", "boiling_point_c"],
        "desc": "organic chemistry laboratory property",
        "target": {"chemical_compound": "Ethanol", "boiling_point_c": 78.37},
    },
    {
        "keys": ["telescope_filter", "exposure_sec"],
        "desc": "astronomical observation parameters",
        "target": {"telescope_filter": "H-Alpha", "exposure_sec": 120.0},
    },
    {
        "keys": ["seismic_station", "p_wave_sec"],
        "desc": "earthquake sensor arrival timing",
        "target": {"seismic_station": "ST_OK01", "p_wave_sec": 14.8},
    },
    {
        "keys": ["laser_emitter", "pulse_width_ns"],
        "desc": "optical laser bench metrics",
        "target": {"laser_emitter": "Ruby", "pulse_width_ns": 15},
    },
    {
        "keys": ["turbine_id", "blade_pitch_deg"],
        "desc": "wind turbine generator setting",
        "target": {"turbine_id": "WT_04", "blade_pitch_deg": 12.5},
    },
    {
        "keys": ["specimen_code", "cell_viability_pct"],
        "desc": "biotechnology culture report",
        "target": {"specimen_code": "HEK_293", "cell_viability_pct": 98.2},
    },
    {
        "keys": ["drill_bit", "feed_rate_mmpm"],
        "desc": "precision CNC machining configuration",
        "target": {"drill_bit": "Carbide", "feed_rate_mmpm": 350.0},
    },
]


TRAIN_PROMPT_TEMPLATES = [
    "Instruction: Output a JSON object with keys '{k1}' and '{k2}' representing {desc}.\nResponse:",
    "Instruction: Generate a JSON object containing keys '{k1}' and '{k2}' for {desc}.\nResponse:",
    "Instruction: Format the requested data as a JSON object with keys '{k1}' and '{k2}' representing {desc}.\nResponse:",
]

VAL_PROMPT_TEMPLATES = [
    "Instruction: Return a JSON object with fields '{k1}' and '{k2}' indicating {desc}.\nResponse:",
    "Instruction: Provide a JSON object with attributes '{k1}' and '{k2}' for {desc}.\nResponse:",
]

TEST_FAMILIAR_PROMPT_TEMPLATES = [
    "Instruction: Produce a JSON object with properties '{k1}' and '{k2}' representing {desc}.\nResponse:",
    "Instruction: Create a JSON object specifying keys '{k1}' and '{k2}' for {desc}.\nResponse:",
]

TEST_UNSEEN_PROMPT_TEMPLATES = [
    "Instruction: Output a structured JSON object with keys '{k1}' and '{k2}' for {desc}.\nResponse:",
    "Instruction: Format a JSON response containing fields '{k1}' and '{k2}' for {desc}.\nResponse:",
]


def build_structured_record(rid: str, split: str, schema_type: str, k1: str, k2: str, desc: str, v1: Any, v2: Any, prompt_idx: int) -> Dict[str, Any]:
    if split == "train":
        tmpl = TRAIN_PROMPT_TEMPLATES[prompt_idx % len(TRAIN_PROMPT_TEMPLATES)]
    elif split == "val":
        tmpl = VAL_PROMPT_TEMPLATES[prompt_idx % len(VAL_PROMPT_TEMPLATES)]
    elif "unseen" in schema_type:
        tmpl = TEST_UNSEEN_PROMPT_TEMPLATES[prompt_idx % len(TEST_UNSEEN_PROMPT_TEMPLATES)]
    else:
        tmpl = TEST_FAMILIAR_PROMPT_TEMPLATES[prompt_idx % len(TEST_FAMILIAR_PROMPT_TEMPLATES)]

    prompt = tmpl.format(k1=k1, k2=k2, desc=desc)
    target_dict = {k1: v1, k2: v2}
    response = json.dumps(target_dict)

    # Strict parser validation before building
    parsed = json.loads(response)
    assert parsed == target_dict, "Strict target validation failure"

    return {
        "id": rid,
        "split": split,
        "category": f"structured_output_{schema_type}",
        "prompt": prompt,
        "response": response,
        "target_keys": [k1, k2],
        "target_values": [v1, v2],
    }


def extract_isolated_replay() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    random.seed(42)
    raw_dir = REPO_ROOT / "data" / "raw"
    corpus_files = {
        "corpus_english_cs": raw_dir / "english" / "english_computer_science.txt",
        "corpus_english_systems": raw_dir / "english" / "english_systems.txt",
        "corpus_english_foundation": raw_dir / "english" / "english_foundation.txt",
        "corpus_code_python": raw_dir / "code" / "code_python.txt",
        "corpus_code_foundation": raw_dir / "code" / "code_foundation.txt",
        "corpus_math_foundation": raw_dir / "mathematics" / "math_foundation.txt",
        "corpus_stage_b_systems": raw_dir / "english" / "stage_b_systems.txt",
    }

    doc_lines: Dict[str, List[str]] = {}
    for name, p in corpus_files.items():
        lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if len(l.strip()) >= 45 and not l.strip().startswith("#") and not l.strip().startswith("assert")]
        doc_lines[name] = lines

    # Document Allocation (100% disjoint proven document sets):
    # TRAIN: corpus_stage_b_systems, corpus_english_cs, corpus_english_foundation
    # VAL: corpus_english_systems, corpus_code_python
    # TEST: corpus_code_foundation, corpus_math_foundation
    train_pool = [(l, "corpus_english_cs") for l in doc_lines["corpus_english_cs"]] + [(l, "corpus_english_foundation") for l in doc_lines["corpus_english_foundation"]] + [(l, "corpus_stage_b_systems") for l in doc_lines["corpus_stage_b_systems"]]
    val_pool = [(l, "corpus_english_systems") for l in doc_lines["corpus_english_systems"]] + [(l, "corpus_code_python") for l in doc_lines["corpus_code_python"]]
    test_pool = [(l, "corpus_code_foundation") for l in doc_lines["corpus_code_foundation"]] + [(l, "corpus_math_foundation") for l in doc_lines["corpus_math_foundation"]]

    # Exclude any historical benchmark test prompts
    banned_prompts = set()
    for prev_test_path in [HISTORICAL_BENCHMARK, PREV_TEST_BATCH3021, PREV_TEST_BATCH3071]:
        if prev_test_path.is_file():
            if prev_test_path.suffix == ".json":
                with open(prev_test_path, "r", encoding="utf-8") as f:
                    for c in json.load(f):
                        banned_prompts.add(c.get("input", "").strip().lower())
            else:
                with open(prev_test_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line)
                            banned_prompts.add(r.get("prompt", "").strip().lower())

    filtered_test_pool = []
    for l, prov in test_pool:
        words = l.split()
        split_idx = max(3, len(words) // 2)
        p_cand = " ".join(words[:split_idx]).strip().lower()
        if p_cand not in banned_prompts:
            filtered_test_pool.append((l, prov))

    random.shuffle(train_pool)
    random.shuffle(val_pool)
    random.shuffle(filtered_test_pool)

    train_selected = train_pool[:60]
    val_selected = val_pool[:10]
    # Pick 10 test replay cases that were NOT used in batch 3071
    b3071_test_prompts = set()
    if PREV_TEST_BATCH3071.is_file():
        with open(PREV_TEST_BATCH3071, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    b3071_test_prompts.add(json.loads(l)["prompt"].strip())

    fresh_test_pool = []
    for l, prov in filtered_test_pool:
        words = l.split()
        split_idx = max(3, len(words) // 2)
        p_cand = " ".join(words[:split_idx]) + " "
        if p_cand not in b3071_test_prompts:
            fresh_test_pool.append((l, prov))

    test_selected = fresh_test_pool[:10]

    # Verify zero document collision
    train_docs = set(p[1] for p in train_selected)
    val_docs = set(p[1] for p in val_selected)
    test_docs = set(p[1] for p in test_selected)

    assert len(train_docs.intersection(val_docs)) == 0, "Train-Val document collision!"
    assert len(train_docs.intersection(test_docs)) == 0, "Train-Test document collision!"
    assert len(val_docs.intersection(test_docs)) == 0, "Val-Test document collision!"

    def build_replay_records(items: List[Tuple[str, str]], split_name: str, id_prefix: str) -> List[Dict[str, Any]]:
        records = []
        for idx, (line, prov) in enumerate(items, start=1):
            words = line.split()
            split_idx = max(3, len(words) // 2)
            prompt_str = " ".join(words[:split_idx]) + " "
            resp_str = " ".join(words[split_idx:])
            records.append({
                "id": f"{id_prefix}_{idx:03d}",
                "category": "base_continuation_replay",
                "prompt": prompt_str,
                "response": resp_str,
                "split": split_name,
                "data_role": split_name.upper(),
                "append_eos": False,  # Strict continuation policy
                "provenance": prov,
                "review_status": "VERIFIED_PRETRAINING_CORPUS",
                "status": "APPROVED",
                "target_keys": None,
                "target_values": None,
            })
        return records

    train_recs = build_replay_records(train_selected, "train", "REP_V2_TRN")
    val_recs = build_replay_records(val_selected, "val", "REP_V2_VAL")
    test_recs = build_replay_records(test_selected, "test", "REP_V2_TST")

    manifest_docs = {
        "document_allocation": {
            "train": sorted(list(train_docs)),
            "val": sorted(list(val_docs)),
            "test": sorted(list(test_docs)),
        },
        "document_overlap": {
            "train_val": list(train_docs.intersection(val_docs)),
            "train_test": list(train_docs.intersection(test_docs)),
            "val_test": list(val_docs.intersection(test_docs)),
        },
        "document_partition": {
            "train": sorted(list(train_docs)),
            "val": sorted(list(val_docs)),
            "test": sorted(list(test_docs)),
        },
    }

    return train_recs, val_recs, test_recs, manifest_docs


def generate_v2_datasets():
    print("=================================================================")
    print("GENERATING BATCH 3111–3150 JSON SFT V2 DATASET")
    print("=================================================================")

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    # 1. Build Structured Records
    train_inst: List[Dict[str, Any]] = []
    val_inst: List[Dict[str, Any]] = []
    familiar_test_inst: List[Dict[str, Any]] = []

    inst_counter = 1
    for s_name, s_data in CORE_SCHEMAS.items():
        k1, k2 = s_data["keys"]
        desc = s_data["desc"]
        inst_list = s_data["instances"]

        # 35 train instances
        for idx in range(35):
            v1, v2 = inst_list[idx]
            rid = f"JSON_V2_TRN_{inst_counter:03d}"
            rec = build_structured_record(rid, "train", s_name, k1, k2, desc, v1, v2, prompt_idx=idx)
            train_inst.append(rec)
            inst_counter += 1

        # 5 val instances
        val_offset = 35
        for idx in range(5):
            v1, v2 = inst_list[val_offset + idx]
            rid = f"JSON_V2_VAL_{len(val_inst)+1:03d}"
            rec = build_structured_record(rid, "val", s_name, k1, k2, desc, v1, v2, prompt_idx=idx)
            val_inst.append(rec)

        # 3 familiar-schema test instances
        test_offset = 40
        for idx in range(3):
            v1, v2 = inst_list[test_offset + idx]
            rid = f"JSON_V2_TST_FAM_{len(familiar_test_inst)+1:03d}"
            rec = build_structured_record(rid, "test", f"{s_name}_familiar", k1, k2, desc, v1, v2, prompt_idx=idx)
            familiar_test_inst.append(rec)

    assert len(train_inst) == 140, f"Expected 140 train JSON, got {len(train_inst)}"
    assert len(val_inst) == 20, f"Expected 20 val JSON, got {len(val_inst)}"
    assert len(familiar_test_inst) == 12, f"Expected 12 familiar test JSON, got {len(familiar_test_inst)}"

    # 2. Build Unseen-Schema Test Records (12 cases)
    unseen_test_inst: List[Dict[str, Any]] = []
    for idx, u in enumerate(UNSEEN_TEST_SCHEMAS):
        k1, k2 = u["keys"]
        desc = u["desc"]
        v1 = u["target"][k1]
        v2 = u["target"][k2]
        rid = f"JSON_V2_TST_UNS_{idx+1:03d}"
        rec = build_structured_record(rid, "test", "unseen_generalization", k1, k2, desc, v1, v2, prompt_idx=idx)
        unseen_test_inst.append(rec)

    assert len(unseen_test_inst) == 12, f"Expected 12 unseen test JSON, got {len(unseen_test_inst)}"

    # 3. Extract Replay Records
    train_rep, val_rep, test_rep, manifest_docs = extract_isolated_replay()

    # 4. Assemble Final Splits
    train_records = train_inst + train_rep
    val_records = val_inst + val_rep
    test_records = familiar_test_inst + unseen_test_inst + test_rep

    # Shuffle training set deterministically
    random.shuffle(train_records)

    # 5. Anti-Leakage Verification
    train_prompts = {r["prompt"].strip().lower() for r in train_records}
    val_prompts = {r["prompt"].strip().lower() for r in val_records}
    test_prompts = {r["prompt"].strip().lower() for r in test_records}

    assert not (train_prompts & val_prompts), "Prompt leakage between train and val"
    assert not (train_prompts & test_prompts), "Prompt leakage between train and test"
    assert not (val_prompts & test_prompts), "Prompt leakage between val and test"

    # Verify unseen test schemas have 0 key-pair overlap with train
    train_keys = {tuple(sorted(r["target_keys"])) for r in train_inst}
    for u in unseen_test_inst:
        u_kp = tuple(sorted(u["target_keys"]))
        assert u_kp not in train_keys, f"Unseen key pair {u_kp} leaked into training!"

    # 6. Save JSONL Files
    with open(BATCH_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
    with open(BATCH_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")
    with open(BATCH_DIR / "test.jsonl", "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r) + "\n")

    # Save full combined dataset
    all_records = train_records + val_records + test_records
    with open(BATCH_DIR / "JSON_SFT_V2_DATASET.jsonl", "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "dataset_name": "ChakrView JSON SFT V2 Dataset (Batch 3111-3150)",
        "total_records": len(all_records),
        "split_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "composition": {
            "train": {"structured_json": len(train_inst), "continuation_replay": len(train_rep)},
            "val": {"structured_json": len(val_inst), "continuation_replay": len(val_rep)},
            "test": {
                "familiar_schema_json": len(familiar_test_inst),
                "unseen_schema_json": len(unseen_test_inst),
                "continuation_replay": len(test_rep),
            },
        },
        "anti_leakage_checks": {
            "train_val_prompt_overlap": len(train_prompts & val_prompts),
            "train_test_prompt_overlap": len(train_prompts & test_prompts),
            "val_test_prompt_overlap": len(val_prompts & test_prompts),
            "unseen_key_overlap_with_train": 0,
        },
        "document_overlap": manifest_docs["document_overlap"],
        "document_partition": manifest_docs["document_partition"],
    }

    with open(BATCH_DIR / "DATASET_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(train_records)} train, {len(val_records)} val, {len(test_records)} test.")
    print(f"Full manifest written to: {BATCH_DIR / 'DATASET_MANIFEST.json'}")


if __name__ == "__main__":
    generate_v2_datasets()
