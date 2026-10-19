"""
Single-Field Dataset Builder for Authorized Training Pilot.

Generates:
- train.jsonl (200 records) across 4 families: atc_train, inv_train, srv_train, met_train
- val.jsonl (40 records) across 4 families: atc_val, inv_val, srv_val, met_val

Enforces:
1. 100% target verbatim substring containment in context.
2. 0% overlap of contexts, target values, host IDs, or entity identifiers between train and val.
3. Dialog prompt framing:
   User: In the following context: "{context}"\nCopy the exact value of {target_field} from context:\nAssistant:
   Target: {target_value} (with leading space convention for BPE)
4. Held-out test split is NEVER touched or accessed.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Disjoint entity pools
# 1. Server Status
TRAIN_HOSTS = [f"srv-prod-{i:03d}" for i in range(1, 26)]
VAL_HOSTS = [f"node-val-{i:02d}" for i in range(1, 11)]

TRAIN_STATUSES = ["running", "degraded", "rebooting", "maintenance", "draining"]
VAL_STATUSES = ["standby", "throttled", "quarantined", "migrating"]

# 2. Telemetry Metrics
TRAIN_SYSTEMS = [f"cluster_{chr(c)}" for c in range(ord('a'), ord('z')+1)]
VAL_SYSTEMS = [f"region_{i:02d}" for i in range(1, 11)]

TRAIN_METRICS = [f"{v}%" for v in [12, 28, 45, 67, 89, 94, 33, 51, 78, 15, 99, 42, 60, 73, 81, 22, 37, 58, 64, 85, 91, 19, 49, 76, 88]]
VAL_METRICS = [f"{v}%" for v in [18, 35, 54, 71, 93, 27, 48, 66, 83, 97]]

# 3. Inventory Items
TRAIN_ITEMS = [f"item_sku_{i:03d}" for i in range(101, 126)]
VAL_ITEMS = [f"asset_tag_{i:03d}" for i in range(501, 511)]

TRAIN_LOCATIONS = [f"warehouse_{chr(c)}" for c in range(ord('A'), ord('Z'))]
VAL_LOCATIONS = [f"depot_{i:02d}" for i in range(1, 11)]

# 4. Anti-Attractor Guard (Context contains code-like docstring keywords but user asks for a simple entity)
TRAIN_CODE_ENTITIES = [f"routine_alpha_{i}" for i in range(1, 26)]
VAL_CODE_ENTITIES = [f"handler_omega_{i}" for i in range(1, 11)]

TRAIN_TOKENS = [f"key_hash_{i:03d}" for i in range(1, 26)]
VAL_TOKENS = [f"sig_auth_{i:03d}" for i in range(1, 11)]


def generate_records(split: str, count: int, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    
    if split == "train":
        hosts = TRAIN_HOSTS
        statuses = TRAIN_STATUSES
        systems = TRAIN_SYSTEMS
        metrics = TRAIN_METRICS
        items = TRAIN_ITEMS
        locations = TRAIN_LOCATIONS
        code_entities = TRAIN_CODE_ENTITIES
        tokens = TRAIN_TOKENS
        family_prefix = "train"
    else:
        hosts = VAL_HOSTS
        statuses = VAL_STATUSES
        systems = VAL_SYSTEMS
        metrics = VAL_METRICS
        items = VAL_ITEMS
        locations = VAL_LOCATIONS
        code_entities = VAL_CODE_ENTITIES
        tokens = VAL_TOKENS
        family_prefix = "val"

    # We distribute count equally across the 4 families
    per_family = count // 4
    
    # Family 1: Server Status (srv)
    for i in range(per_family):
        host = rng.choice(hosts)
        status = rng.choice(statuses)
        field = "status"
        templates = [
            f"Server telemetry report: Host {host} is currently reporting state {status}.",
            f"Cluster monitor alert: {host} switched condition to {status} at midnight.",
            f"Node diagnostic ping: node {host} returned health flag {status} in 4ms.",
            f"Infrastructure health: {host} registered operating status {status}."
        ]
        context = rng.choice(templates)
        records.append({
            "id": f"{family_prefix}_srv_{i:03d}",
            "family": f"srv_{family_prefix}",
            "context": context,
            "target_field": field,
            "target_value": status,
        })
        
    # Family 2: Telemetry Metric (met)
    for i in range(per_family):
        sys_name = rng.choice(systems)
        metric = rng.choice(metrics)
        field = "utilization"
        templates = [
            f"Telemetry summary: System {sys_name} measured memory utilization {metric}.",
            f"Resource tracker: {sys_name} compute usage stands at {metric}.",
            f"Capacity metric: {sys_name} recorded peak allocation {metric}.",
            f"Usage audit: {sys_name} reported buffer utilization {metric}."
        ]
        context = rng.choice(templates)
        records.append({
            "id": f"{family_prefix}_met_{i:03d}",
            "family": f"met_{family_prefix}",
            "context": context,
            "target_field": field,
            "target_value": metric,
        })
        
    # Family 3: Inventory (inv)
    for i in range(per_family):
        item = rng.choice(items)
        loc = rng.choice(locations)
        field = "location"
        templates = [
            f"Inventory tracking: Catalog entry {item} is stored in {loc}.",
            f"Supply chain log: Part {item} transferred to facility {loc}.",
            f"Warehouse manifest: Record {item} assigned to shelf {loc}.",
            f"Asset register: Equipment {item} located at site {loc}."
        ]
        context = rng.choice(templates)
        records.append({
            "id": f"{family_prefix}_inv_{i:03d}",
            "family": f"inv_{family_prefix}",
            "context": context,
            "target_field": field,
            "target_value": loc,
        })
        
    # Family 4: Anti-Attractor Guard (atc)
    # Context contains python doctest keywords (def, >>>, return, docstring) to force copy rather than doctest generation
    for i in range(per_family):
        fn = rng.choice(code_entities)
        tok = rng.choice(tokens)
        field = "signature"
        templates = [
            f"Code snippet: def {fn}(): \"\"\">>> {fn}()\"\"\" return '{tok}'. Signature string is {tok}.",
            f"Module docstring for {fn}: Contains token signature {tok} for verification >>> verify({tok}).",
            f"Test fixture: def test_{fn}(): assert get_sig() == '{tok}'. Expected signature is {tok}.",
            f"API specification: Method {fn} outputs verification signature {tok}."
        ]
        context = rng.choice(templates)
        records.append({
            "id": f"{family_prefix}_atc_{i:03d}",
            "family": f"atc_{family_prefix}",
            "context": context,
            "target_field": field,
            "target_value": tok,
        })
        
    rng.shuffle(records)
    return records


def format_record_for_sft(record: Dict[str, Any]) -> Dict[str, Any]:
    context = record["context"]
    field = record["target_field"]
    val = record["target_value"]
    
    prompt = f'User: In the following context: "{context}"\nCopy the exact value of {field} from context:\nAssistant:'
    target = f' {val}'
    
    return {
        "id": record["id"],
        "family": record["family"],
        "context": context,
        "target_field": field,
        "target_value": val,
        "prompt": prompt,
        "target": target,
        "expected_value": val
    }


def main():
    train_records = generate_records("train", 200, seed=42)
    val_records = generate_records("val", 40, seed=1337)
    
    data_dir = Path("experiments/single_field_copy_pilot/authorized_run_001/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(format_record_for_sft(r)) + "\n")
            
    with open(val_file, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(format_record_for_sft(r)) + "\n")
            
    print(f"Generated {len(train_records)} train records -> {train_file}")
    print(f"Generated {len(val_records)} val records -> {val_file}")

if __name__ == "__main__":
    main()
