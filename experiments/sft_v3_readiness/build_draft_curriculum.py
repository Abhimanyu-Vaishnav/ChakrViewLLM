"""Draft Synthetic Dataset Specification for SFT V3 Curriculum.

This file provides a small, clearly labeled, synthetic draft set for evaluating the proposed
SFT V3 curriculum format.

SAFETY NOTICE:
- This is a draft evaluation set only.
- It is NOT for training.
- It is NOT mixed into train.jsonl or val.jsonl.
- Does not copy from val.jsonl or test.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = REPO_ROOT / "experiments" / "sft_v3_readiness" / "DRAFT_SFT_V3_EXAMPLES.jsonl"

DRAFT_RECORDS = [
    # Category 1: Contextually Grounded Key-Value Extraction (Server Status)
    {
        "id": "DRAFT_SFT_V3_001",
        "category": "grounded_extraction_server_status",
        "data_role": "DRAFT_SYNTHETIC_CURRICULUM",
        "prompt": "Context: Node 14 reporting standby status with HTTP return code 200.\nInstruction: Extract the node status and code as a JSON object with keys 'status' and 'code'.\nResponse:",
        "response": "{\"status\": \"standby\", \"code\": 200}",
        "target_keys": ["status", "code"],
        "target_values": ["standby", 200],
        "validation_rules": {"status": "string", "code": "integer"},
    },
    {
        "id": "DRAFT_SFT_V3_002",
        "category": "grounded_extraction_server_status",
        "data_role": "DRAFT_SYNTHETIC_CURRICULUM",
        "prompt": "Context: Database connection failed after timeout, returning status failed and HTTP code 504.\nInstruction: Output a JSON object with keys 'status' and 'code' containing the report.\nResponse:",
        "response": "{\"status\": \"failed\", \"code\": 504}",
        "target_keys": ["status", "code"],
        "target_values": ["failed", 504],
        "validation_rules": {"status": "string", "code": "integer"},
    },
    # Category 2: Contextually Grounded Telemetry Measurement (Network Metric)
    {
        "id": "DRAFT_SFT_V3_003",
        "category": "grounded_telemetry_metric",
        "data_role": "DRAFT_SYNTHETIC_CURRICULUM",
        "prompt": "Context: Sensor telemetry reading packet_loss measured at 0.04 percent.\nInstruction: Represent this telemetry as a JSON object with keys 'metric' and 'value'.\nResponse:",
        "response": "{\"metric\": \"packet_loss\", \"value\": 0.04}",
        "target_keys": ["metric", "value"],
        "target_values": ["packet_loss", 0.04],
        "validation_rules": {"metric": "string", "value": "number"},
    },
    {
        "id": "DRAFT_SFT_V3_004",
        "category": "grounded_telemetry_metric",
        "data_role": "DRAFT_SYNTHETIC_CURRICULUM",
        "prompt": "Context: Thermal monitoring report indicates core_temp_c at 64.8 degrees Celsius.\nInstruction: Produce a JSON object with keys 'metric' and 'value' for this measurement.\nResponse:",
        "response": "{\"metric\": \"core_temp_c\", \"value\": 64.8}",
        "target_keys": ["metric", "value"],
        "target_values": ["core_temp_c", 64.8],
        "validation_rules": {"metric": "string", "value": "number"},
    },
    # Category 3: System Operational State (Feature Flag)
    {
        "id": "DRAFT_SFT_V3_005",
        "category": "grounded_feature_flag",
        "data_role": "DRAFT_SYNTHETIC_CURRICULUM",
        "prompt": "Context: Flag auto_save is currently active in the configuration.\nInstruction: Return a JSON object with keys 'feature' and 'enabled' reflecting this configuration.\nResponse:",
        "response": "{\"feature\": \"auto_save\", \"enabled\": true}",
        "target_keys": ["feature", "enabled"],
        "target_values": ["auto_save", True],
        "validation_rules": {"feature": "string", "enabled": "boolean"},
    },
    {
        "id": "DRAFT_SFT_V3_006",
        "category": "grounded_feature_flag",
        "data_role": "DRAFT_SYNTHETIC_CURRICULUM",
        "prompt": "Context: Feature unsafe_eval has been disabled for security compliance.\nInstruction: Provide a JSON object with keys 'feature' and 'enabled' showing its state.\nResponse:",
        "response": "{\"feature\": \"unsafe_eval\", \"enabled\": false}",
        "target_keys": ["feature", "enabled"],
        "target_values": ["unsafe_eval", False],
        "validation_rules": {"feature": "string", "enabled": "boolean"},
    },
    # Category 4: Inventory Accounting (Non-zero and Standard Integer Formatting)
    {
        "id": "DRAFT_SFT_V3_007",
        "category": "grounded_inventory_item",
        "data_role": "DRAFT_SYNTHETIC_CURRICULUM",
        "prompt": "Context: Warehouse audit reports 36 units of cooling_fan in storage.\nInstruction: Output a JSON object with keys 'name' and 'count' for this item.\nResponse:",
        "response": "{\"name\": \"cooling_fan\", \"count\": 36}",
        "target_keys": ["name", "count"],
        "target_values": ["cooling_fan", 36],
        "validation_rules": {"name": "string", "count": "integer"},
    },
    {
        "id": "DRAFT_SFT_V3_008",
        "category": "grounded_inventory_item",
        "data_role": "DRAFT_SYNTHETIC_CURRICULUM",
        "prompt": "Context: Stock register records 120 units of patch_cable available.\nInstruction: Generate a JSON object with keys 'name' and 'count' for this inventory.\nResponse:",
        "response": "{\"name\": \"patch_cable\", \"count\": 120}",
        "target_keys": ["name", "count"],
        "target_values": ["patch_cable", 120],
        "validation_rules": {"name": "string", "count": "integer"},
    },
    # Category 5: Negative Prompt Resistance / Anti-Code Attractor Pair
    {
        "id": "DRAFT_SFT_V3_009",
        "category": "anti_code_attractor_guard",
        "data_role": "DRAFT_SYNTHETIC_CURRICULUM",
        "prompt": "Context: Service module main is operational with exit code 0.\nInstruction: Do NOT import doctest or write Python script. Return ONLY a JSON object with keys 'status' and 'code'.\nResponse:",
        "response": "{\"status\": \"operational\", \"code\": 0}",
        "target_keys": ["status", "code"],
        "target_values": ["operational", 0],
        "validation_rules": {"status": "string", "code": "integer"},
    },
    # Category 6: Missing Information Abstention
    {
        "id": "DRAFT_SFT_V3_010",
        "category": "missing_information_abstention",
        "data_role": "DRAFT_SYNTHETIC_CURRICULUM",
        "prompt": "Context: System report mentions a server check, but no status or return code was recorded.\nInstruction: Extract status and code as a JSON object, or return error if data is missing.\nResponse:",
        "response": "{\"error\": \"missing_information\", \"status\": null, \"code\": null}",
        "target_keys": ["error", "status", "code"],
        "target_values": ["missing_information", None, None],
        "validation_rules": {"error": "string", "status": "null", "code": "null"},
    },
]


def write_draft_dataset():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in DRAFT_RECORDS:
            f.write(json.dumps(r) + "\n")
    print(f"Successfully wrote {len(DRAFT_RECORDS)} draft records to: {OUTPUT_FILE}")


if __name__ == "__main__":
    write_draft_dataset()
