"""CR-EXP-16 Benchmark Generator
Generates a sealed, frozen 1,000-probe synthetic algorithmic capability benchmark:
A. Exact Sequence Copying (100)
B. Single-Token Copying (100)
C. Multi-Token Copying (100)
D. Key-Value Retrieval (100)
E. Subject Identification (100)
F. Static Subject-Value Binding (100)
G. Single-Variable Reassignment (100)
H. Multiple-Variable Reassignment (100)
I. Distractor Resistance (100)
J. Multi-Turn State Tracking (100)
"""
import random
import json
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
EXP16_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining"
OUT_FILE = EXP16_DIR / "benchmark_data/cr16_synthetic_benchmark_1000.json"
MANIFEST_FILE = EXP16_DIR / "benchmark_data/cr16_benchmark_manifest.json"

random.seed(16000)

# Held-out entities and values strictly reserved for testing
TEST_ENTITIES_SINGLE = ["Dev", "Roy", "Prem", "Amit", "Alok", "Sonu", "Mona", "Ritu"]
TEST_ENTITIES_MULTI = [
    "Mukesh", "Vikas", "Gautam", "Fateh", "Govind", "Kalyan", "Pankhuri",
    "Sarvesh", "Yashwant", "Siddharth", "Vaibhav", "Badri", "Jaideep", "Lalit", "Madhav"
]
ALL_TEST_ENTITIES = TEST_ENTITIES_SINGLE + TEST_ENTITIES_MULTI

TEST_VALUES = [
    "Delhi", "Jaipur", "Pune", "Patna", "Surat", "Kanpur", "Nagpur", "Indore",
    "Thane", "Bhopal", "Visakhapatnam", "Vadodara", "Ghaziabad", "Ludhiana", "Agra",
    "Nashik", "Faridabad", "Meerut", "Rajkot", "Varanasi", "Srinagar", "Aurangabad"
]

TEST_RANDOM_WORDS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau"
]

probes = []
probe_idx = 1

# A. Exact Sequence Copying (100)
for _ in range(100):
    k = random.randint(3, 6)
    seq = random.sample(TEST_RANDOM_WORDS, k)
    seq_str = " ".join(seq)
    prompt = f"Echo sequence: {seq_str}\nOutput: "
    probes.append({
        "id": f"CR16_COPY_SEQ_{probe_idx:04d}",
        "category": "exact_sequence_copying",
        "prompt": prompt,
        "subject": "",
        "expected_target": seq_str,
        "earlier_value": "",
        "distractor_values": []
    })
    probe_idx += 1

# B. Single-Token Copying (100)
for _ in range(100):
    name = random.choice(TEST_ENTITIES_SINGLE)
    prompt = f"Repeat the name: {name}\nName: "
    probes.append({
        "id": f"CR16_COPY_SINGLE_{probe_idx:04d}",
        "category": "single_token_copying",
        "prompt": prompt,
        "subject": name,
        "expected_target": name,
        "earlier_value": "",
        "distractor_values": []
    })
    probe_idx += 1

# C. Multi-Token Copying (100)
for _ in range(100):
    name = random.choice(TEST_ENTITIES_MULTI)
    prompt = f"Repeat the name: {name}\nName: "
    probes.append({
        "id": f"CR16_COPY_MULTI_{probe_idx:04d}",
        "category": "multi_token_copying",
        "prompt": prompt,
        "subject": name,
        "expected_target": name,
        "earlier_value": "",
        "distractor_values": []
    })
    probe_idx += 1

# D. Key-Value Retrieval (100)
for _ in range(100):
    e1, e2 = random.sample(ALL_TEST_ENTITIES, 2)
    v1, v2 = random.sample(TEST_VALUES, 2)
    target_e, target_v = (e1, v1) if random.random() < 0.5 else (e2, v2)
    prompt = f"Record: {e1} -> {v1}\nRecord: {e2} -> {v2}\nQuery: What is {target_e}?\nValue: "
    probes.append({
        "id": f"CR16_KV_RETR_{probe_idx:04d}",
        "category": "key_value_retrieval",
        "prompt": prompt,
        "subject": target_e,
        "expected_target": target_v,
        "earlier_value": "",
        "distractor_values": [v1 if target_e == e2 else v2]
    })
    probe_idx += 1

# E. Subject Identification (100)
for _ in range(100):
    e1, e2 = random.sample(ALL_TEST_ENTITIES, 2)
    v1, v2 = random.sample(TEST_VALUES, 2)
    target_e, target_v = (e1, v1) if random.random() < 0.5 else (e2, v2)
    prompt = f"Data: {e1} is located in {v1}.\nData: {e2} is located in {v2}.\nQuery: Who is located in {target_v}?\nSubject: "
    probes.append({
        "id": f"CR16_SUBJ_ID_{probe_idx:04d}",
        "category": "subject_identification",
        "prompt": prompt,
        "subject": target_e,
        "expected_target": target_e,
        "earlier_value": "",
        "distractor_values": [e1 if target_e == e2 else e2]
    })
    probe_idx += 1

# F. Static Subject-Value Binding (100)
for _ in range(100):
    e = random.choice(ALL_TEST_ENTITIES)
    v = random.choice(TEST_VALUES)
    prompt = f"State: {e} lives in {v}.\nQuery: Where does {e} live?\nAnswer: "
    probes.append({
        "id": f"CR16_STAT_BIND_{probe_idx:04d}",
        "category": "static_subject_value_binding",
        "prompt": prompt,
        "subject": e,
        "expected_target": v,
        "earlier_value": "",
        "distractor_values": []
    })
    probe_idx += 1

# G. Single-Variable Reassignment (100)
for _ in range(100):
    e = random.choice(ALL_TEST_ENTITIES)
    v1, v2 = random.sample(TEST_VALUES, 2)
    prompt = f"Event 1: {e} = {v1}\nEvent 2: {e} = {v2}\nQuery: What is the current value of {e}?\nValue: "
    probes.append({
        "id": f"CR16_SING_REAS_{probe_idx:04d}",
        "category": "single_variable_reassignment",
        "prompt": prompt,
        "subject": e,
        "expected_target": v2,
        "earlier_value": v1,
        "distractor_values": []
    })
    probe_idx += 1

# H. Multiple-Variable Reassignment (100)
for _ in range(100):
    e1, e2 = random.sample(ALL_TEST_ENTITIES, 2)
    v1_old, v1_new, v2_old, v2_new = random.sample(TEST_VALUES, 4)
    target_e, target_new, target_old = (e1, v1_new, v1_old) if random.random() < 0.5 else (e2, v2_new, v2_old)
    other_new, other_old = (v2_new, v2_old) if target_e == e1 else (v1_new, v1_old)
    prompt = f"Log 1: {e1} = {v1_old}\nLog 2: {e2} = {v2_old}\nLog 3: {e1} = {v1_new}\nLog 4: {e2} = {v2_new}\nQuery: Current value of {target_e}?\nValue: "
    probes.append({
        "id": f"CR16_MULT_REAS_{probe_idx:04d}",
        "category": "multiple_variable_reassignment",
        "prompt": prompt,
        "subject": target_e,
        "expected_target": target_new,
        "earlier_value": target_old,
        "distractor_values": [other_new, other_old]
    })
    probe_idx += 1

# I. Distractor Resistance (100)
for _ in range(100):
    e = random.choice(ALL_TEST_ENTITIES)
    v = random.choice(TEST_VALUES)
    d1, d2 = random.sample(TEST_RANDOM_WORDS, 2)
    d_city = random.choice([x for x in TEST_VALUES if x != v])
    prompt = f"Notice: Noise factor {d1} observed in {d_city}.\nNotice: Item {d2} registered.\nRecord: {e} status is set to {v}.\nQuery: What is {e} status?\nAnswer: "
    probes.append({
        "id": f"CR16_DIST_RES_{probe_idx:04d}",
        "category": "distractor_resistance",
        "prompt": prompt,
        "subject": e,
        "expected_target": v,
        "earlier_value": "",
        "distractor_values": [d_city, d1, d2]
    })
    probe_idx += 1

# J. Multi-Turn State Tracking (100)
for _ in range(100):
    e1, e2 = random.sample(ALL_TEST_ENTITIES, 2)
    v1_init, v1_mid, v1_final = random.sample(TEST_VALUES, 3)
    v2_val = random.choice([x for x in TEST_VALUES if x not in [v1_init, v1_mid, v1_final]])
    prompt = f"Step 1: {e1} moved to {v1_init}.\nStep 2: {e2} stationed at {v2_val}.\nStep 3: {e1} relocated to {v1_mid}.\nStep 4: {e1} finally settled in {v1_final}.\nQuery: Where is {e1} currently located?\nAnswer: "
    probes.append({
        "id": f"CR16_MULT_STATE_{probe_idx:04d}",
        "category": "multi_turn_state_tracking",
        "prompt": prompt,
        "subject": e1,
        "expected_target": v1_final,
        "earlier_value": v1_mid,
        "distractor_values": [v1_init, v2_val]
    })
    probe_idx += 1

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(probes, f, indent=2)

file_hash = hashlib.sha256(open(OUT_FILE, "rb").read()).hexdigest()

manifest = {
    "benchmark_file": str(OUT_FILE.name),
    "total_probes": len(probes),
    "sha256": file_hash,
    "categories": {
        "exact_sequence_copying": 100,
        "single_token_copying": 100,
        "multi_token_copying": 100,
        "key_value_retrieval": 100,
        "subject_identification": 100,
        "static_subject_value_binding": 100,
        "single_variable_reassignment": 100,
        "multiple_variable_reassignment": 100,
        "distractor_resistance": 100,
        "multi_turn_state_tracking": 100
    }
}

with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Generated {len(probes)} probes.")
print(f"File SHA-256: {file_hash}")

