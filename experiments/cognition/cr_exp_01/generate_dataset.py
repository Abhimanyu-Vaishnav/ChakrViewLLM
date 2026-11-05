"""
Deterministic Dataset Generator & Validator for CR-EXP-01 Variable Binding Experiment
Enforces strict disjoint entity pools, 10 distinct template formats, counterbalanced target depth,
and explicit reassignment semantics.
"""

import json
import random
import hashlib
from pathlib import Path
import sys

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

OUTPUT_DIR = REPO_ROOT / "data" / "sft" / "cr_exp01"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. DISJOINT ENTITY POOLS
TRAIN_KEYS = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa",
    "Port", "Host", "Node", "Cluster", "City", "Server", "Engine", "Worker", "Driver", "Client",
    "Database", "Service", "Router", "Gateway", "Registry", "Storage", "Memory", "Socket", "Queue", "Stream",
    "Metric", "Target", "Process", "Thread", "Module", "Kernel", "System", "Domain", "Packet", "Channel",
    "Record", "Device", "Agent", "Sensor", "Signal", "Volume", "Session", "Policy", "Tenant", "Account"
]

TRAIN_VALUES = [
    "101", "202", "303", "404", "505", "606", "707", "808", "909", "1001",
    "Red", "Blue", "Green", "Amber", "Falcon", "Apollo", "Garuda", "Agni", "Vayu", "Varun",
    "Delhi", "Tokyo", "Bengaluru", "Mumbai", "Paris", "London", "Berlin", "Sydney", "Rome", "Cairo",
    "Maple", "Cedar", "Pine", "Oak", "Birch", "Willow", "Ash", "Elm", "Teak", "Walnut",
    "Circle", "Square", "Sphere", "Cube", "Prism", "Pyramid", "Cone", "Cylinder", "Torus", "Helix"
]

VAL_KEYS = [
    "Buffer", "Matrix", "Tensor", "Vector", "Cursor", "Pipeline", "Schema", "Payload", "Trigger", "Handler",
    "Endpoint", "Context", "Request", "Response", "Catalog", "Instance", "Interface", "Protocol", "Artifact", "Manifest"
]

VAL_VALUES = [
    "1111", "2222", "3333", "4444", "5555", "Neon", "Cyan", "Magenta", "Silver", "Gold",
    "Cheetah", "Leopard", "Jaguar", "Panther", "Tiger", "Pune", "Jaipur", "Kochi", "Indore", "Surat"
]

TEST_KEYS = [
    "Quasar", "Pulsar", "Zenith", "Vertex", "Nexus", "Catalyst", "Cipher", "Vortex", "Apex", "Horizon",
    "Solstice", "Equinox", "Eclipse", "Orbit", "Cosmos", "Nebula", "Spectra", "Photon", "Quantum", "Gravity",
    "Chronos", "Aegis", "Vanguard", "Sentinel", "Beacon", "Relay", "Beacon", "Pylon", "Beacon", "Beacon"
]
# Clean duplicates in test keys
TEST_KEYS = list(dict.fromkeys([
    "Quasar", "Pulsar", "Zenith", "Vertex", "Nexus", "Catalyst", "Cipher", "Vortex", "Apex", "Horizon",
    "Solstice", "Equinox", "Eclipse", "Orbit", "Cosmos", "Nebula", "Spectra", "Photon", "Quantum", "Gravity",
    "Chronos", "Aegis", "Vanguard", "Sentinel", "Pylon", "Strata", "Synapse", "Flux", "VectorX", "Prisma"
]))

TEST_VALUES = [
    "8080", "9090", "7070", "6060", "5050", "Cobalt", "Obsidian", "Platinum", "Titanium", "Rhodium",
    "Phoenix", "Hydra", "Griffin", "Chimera", "Dragon", "Chennai", "Kolkata", "Ahmedabad", "Hyderabad", "Shimla",
    "Opal", "Ruby", "Emerald", "Sapphire", "Topaz", "Copper", "Bronze", "Nickel", "Zinc", "Iron"
]

# 2. TEMPLATE FAMILIES
TEMPLATES_COMMON = [
    # T1
    "Mapping: {k} is mapped to {v}. Query: What does {k} map to?\nAnswer: {v}",
    # T2
    "System config: {k} = {v}. Query: Value of {k}?\nAnswer: {v}",
    # T3
    "Record: {v} was assigned to {k}. Query: Which value belongs to {k}?\nAnswer: {v}",
    # T4
    "Attribute listing: {k} has attribute {v}. Query: What is {k}'s attribute?\nAnswer: {v}",
    # T5
    "Environment state: {k} : {v}. Query: Value of {k}?\nAnswer: {v}",
    # T6
    "Context: Key {k} has been registered as {v}. Query: What is {k}?\nAnswer: {v}",
    # T7
    "Parameter {k} was set to {v}. Query: What is the parameter {k}?\nAnswer: {v}",
    # T8
    "Association: {k} -> {v}. Query: What does {k} associate to?\nAnswer: {v}"
]

TEMPLATES_HELD_OUT = [
    # T9 (Unseen syntax formulation)
    "Initial record: {k} is {v}. Query: What value is held by {k}?\nAnswer: {v}",
    # T10 (Unseen device profile syntax)
    "Device profile: {k} registered with identifier {v}. Query: Identify {k}.\nAnswer: {v}"
]

def format_multi_pair(pairs, target_k, target_v, template_idx=0):
    # pairs is list of (k, v)
    lines = []
    for k, v in pairs:
        lines.append(f"{k} = {v}")
    prefix = "; ".join(lines) + "."
    query = f"Query: Value of {target_k}?\nAnswer: {target_v}"
    return f"Registry: {prefix}\n{query}"

def format_reassignment(k, v_old, v_new):
    return (
        f"Initial configuration: {k} = {v_old}.\n"
        f"Update notice: {k} was changed to {v_new}.\n"
        f"Query: What is the current confirmed value of {k}?\nAnswer: {v_new}"
    )

def main():
    print("=== CR-EXP-01 SYNTHETIC DATASET GENERATION ===")
    
    # Verify pool disjointness
    s_tr_k, s_val_k, s_te_k = set(TRAIN_KEYS), set(VAL_KEYS), set(TEST_KEYS)
    s_tr_v, s_val_v, s_te_v = set(TRAIN_VALUES), set(VAL_VALUES), set(TEST_VALUES)
    
    assert s_tr_k.isdisjoint(s_val_k), "Train and Val Keys overlap!"
    assert s_tr_k.isdisjoint(s_te_k), "Train and Test Keys overlap!"
    assert s_val_k.isdisjoint(s_te_k), "Val and Test Keys overlap!"
    assert s_tr_v.isdisjoint(s_val_v), "Train and Val Values overlap!"
    assert s_tr_v.isdisjoint(s_te_v), "Train and Test Values overlap!"
    assert s_val_v.isdisjoint(s_te_v), "Val and Test Values overlap!"
    print("Disjoint Lexicon Pools: 100% VERIFIED DISJOINT.")
    
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    
    # 1. Train Set: 1,000 examples (Seed 101)
    rng_train = random.Random(101)
    train_records = []
    for i in range(1000):
        # 60% single binding, 20% 1-3 distractors, 20% reassignment
        r_type = rng_train.random()
        if r_type < 0.60:
            k = rng_train.choice(TRAIN_KEYS)
            v = rng_train.choice(TRAIN_VALUES)
            tmpl = rng_train.choice(TEMPLATES_COMMON)
            text = tmpl.format(k=k, v=v)
            record = {"id": f"train_{i:04d}", "type": "single", "key": k, "target_value": v, "text": text}
        elif r_type < 0.80:
            # Multi-pair (1-3 distractors)
            n_dist = rng_train.randint(1, 3)
            sampled_k = rng_train.sample(TRAIN_KEYS, n_dist + 1)
            sampled_v = rng_train.sample(TRAIN_VALUES, n_dist + 1)
            pairs = list(zip(sampled_k, sampled_v))
            target_idx = rng_train.randint(0, len(pairs) - 1)
            t_k, t_v = pairs[target_idx]
            text = format_multi_pair(pairs, t_k, t_v)
            record = {"id": f"train_{i:04d}", "type": "multi_pair", "distractors": n_dist, "key": t_k, "target_value": t_v, "text": text}
        else:
            # Reassignment
            k = rng_train.choice(TRAIN_KEYS)
            v_old, v_new = rng_train.sample(TRAIN_VALUES, 2)
            text = format_reassignment(k, v_old, v_new)
            record = {"id": f"train_{i:04d}", "type": "reassignment", "key": k, "target_value": v_new, "text": text}
        train_records.append(record)
        
    p_train = OUTPUT_DIR / "train_1000.jsonl"
    with open(p_train, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
    print(f"Generated {len(train_records)} train records -> {p_train}")

    # 2. Val Set: 100 examples (Seed 202, Val Pool)
    rng_val = random.Random(202)
    val_records = []
    for i in range(100):
        r_type = rng_val.random()
        if r_type < 0.60:
            k = rng_val.choice(VAL_KEYS)
            v = rng_val.choice(VAL_VALUES)
            tmpl = rng_val.choice(TEMPLATES_COMMON)
            text = tmpl.format(k=k, v=v)
            record = {"id": f"val_{i:04d}", "type": "single", "key": k, "target_value": v, "text": text}
        elif r_type < 0.80:
            n_dist = rng_val.randint(1, 3)
            sampled_k = rng_val.sample(VAL_KEYS, n_dist + 1)
            sampled_v = rng_val.sample(VAL_VALUES, n_dist + 1)
            pairs = list(zip(sampled_k, sampled_v))
            target_idx = rng_val.randint(0, len(pairs) - 1)
            t_k, t_v = pairs[target_idx]
            text = format_multi_pair(pairs, t_k, t_v)
            record = {"id": f"val_{i:04d}", "type": "multi_pair", "distractors": n_dist, "key": t_k, "target_value": t_v, "text": text}
        else:
            k = rng_val.choice(VAL_KEYS)
            v_old, v_new = rng_val.sample(VAL_VALUES, 2)
            text = format_reassignment(k, v_old, v_new)
            record = {"id": f"val_{i:04d}", "type": "reassignment", "key": k, "target_value": v_new, "text": text}
        val_records.append(record)
        
    p_val = OUTPUT_DIR / "val_100.jsonl"
    with open(p_val, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")
    print(f"Generated {len(val_records)} val records -> {p_val}")

    # 3. Test Probes (Seed 303: 5 Distinct Evaluation Splits, 100 each = 500 probes)
    rng_test = random.Random(303)
    
    # Split 1: Condition A (Familiar Train Pool Probes)
    test_familiar = []
    for i in range(100):
        k = rng_test.choice(TRAIN_KEYS)
        v = rng_test.choice(TRAIN_VALUES)
        tmpl = rng_test.choice(TEMPLATES_COMMON)
        text = tmpl.format(k=k, v=v)
        test_familiar.append({"id": f"test_fam_{i:04d}", "condition": "A_familiar", "key": k, "target_value": v, "text": text})
    p_test_fam = OUTPUT_DIR / "test_familiar_100.jsonl"
    with open(p_test_fam, "w", encoding="utf-8") as f:
        for r in test_familiar:
            f.write(json.dumps(r) + "\n")
            
    # Split 2: Condition B (Novel Held-Out Pool Probes)
    test_novel = []
    for i in range(100):
        k = rng_test.choice(TEST_KEYS)
        v = rng_test.choice(TEST_VALUES)
        tmpl = rng_test.choice(TEMPLATES_COMMON)
        text = tmpl.format(k=k, v=v)
        test_novel.append({"id": f"test_nov_{i:04d}", "condition": "B_novel", "key": k, "target_value": v, "text": text})
    p_test_nov = OUTPUT_DIR / "test_novel_100.jsonl"
    with open(p_test_nov, "w", encoding="utf-8") as f:
        for r in test_novel:
            f.write(json.dumps(r) + "\n")

    # Split 3: Condition D (Dynamic Reassignment Probes - Held-Out Pool)
    test_reassign = []
    for i in range(100):
        k = rng_test.choice(TEST_KEYS)
        v_old, v_new = rng_test.sample(TEST_VALUES, 2)
        text = format_reassignment(k, v_old, v_new)
        test_reassign.append({"id": f"test_reassign_{i:04d}", "condition": "D_reassignment", "key": k, "target_value": v_new, "v_old": v_old, "text": text})
    p_test_reassign = OUTPUT_DIR / "test_reassignment_100.jsonl"
    with open(p_test_reassign, "w", encoding="utf-8") as f:
        for r in test_reassign:
            f.write(json.dumps(r) + "\n")

    # Split 4: Condition E (5 Distractors Stress Test - Held-Out Pool, Counterbalanced Depth)
    test_distractors = []
    depths = [0, 1, 2, 3, 4, 5] # 6 items total (1 target + 5 distractors)
    for i in range(100):
        sampled_k = rng_test.sample(TEST_KEYS, 6)
        sampled_v = rng_test.sample(TEST_VALUES, 6)
        pairs = list(zip(sampled_k, sampled_v))
        target_idx = depths[i % len(depths)]
        t_k, t_v = pairs[target_idx]
        text = format_multi_pair(pairs, t_k, t_v)
        test_distractors.append({"id": f"test_dist5_{i:04d}", "condition": "E_distractors_5", "target_depth_idx": target_idx, "key": t_k, "target_value": t_v, "text": text})
    p_test_dist = OUTPUT_DIR / "test_distractors_100.jsonl"
    with open(p_test_dist, "w", encoding="utf-8") as f:
        for r in test_distractors:
            f.write(json.dumps(r) + "\n")

    # Split 5: Condition I (Held-Out Templates 9-10 - Novel Pool)
    test_unseen_tmpl = []
    for i in range(100):
        k = rng_test.choice(TEST_KEYS)
        v = rng_test.choice(TEST_VALUES)
        tmpl = rng_test.choice(TEMPLATES_HELD_OUT)
        text = tmpl.format(k=k, v=v)
        test_unseen_tmpl.append({"id": f"test_tmpl_{i:04d}", "condition": "I_unseen_template", "key": k, "target_value": v, "text": text})
    p_test_tmpl = OUTPUT_DIR / "test_unseen_template_100.jsonl"
    with open(p_test_tmpl, "w", encoding="utf-8") as f:
        for r in test_unseen_tmpl:
            f.write(json.dumps(r) + "\n")

    print("Generated all 5 evaluation test probe suites (100 records each).")
    
    # Calculate SHA256 hashes of all generated files
    files = [p_train, p_val, p_test_fam, p_test_nov, p_test_reassign, p_test_dist, p_test_tmpl]
    manifest = {}
    print("\n--- GENERATED DATASET SHA-256 HASHES ---")
    for fp in files:
        h = hashlib.sha256(fp.read_bytes()).hexdigest()
        manifest[fp.name] = {"path": str(fp), "size_bytes": fp.stat().st_size, "sha256": h}
        print(f"  {fp.name:<30} | {h}")
        
    p_manifest = OUTPUT_DIR / "DATASET_HASH_MANIFEST.json"
    p_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved dataset hash manifest to {p_manifest}")

if __name__ == "__main__":
    main()
