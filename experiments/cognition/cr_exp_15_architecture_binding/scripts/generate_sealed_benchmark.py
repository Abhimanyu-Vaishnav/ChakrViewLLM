"""
CR-EXP-15 Phase 3: Sealed 1,200-Probe Decomposed Benchmark Generator
Generates exactly 1,200 probes across 8 isolated capability categories:
1. entity_copying: 150 probes
2. entity_selection: 150 probes
3. static_binding: 150 probes
4. single_entity_updates: 150 probes
5. multi_entity_tracking: 150 probes
6. interleaved_updates: 150 probes
7. delayed_retrieval_distractors: 150 probes
8. calibrated_abstention: 150 probes
Total = 1,200 probes.

Guarantees:
- Disjoint entities and locations from all training pools
- Deterministic answer keys
- Frozen SHA-256
"""

import json
import random
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_15_architecture_binding" / "benchmark_data"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

# 40 Test entities (Unseen in CR-14 or CR-15 training)
TEST_ENTITIES = [
    "Apurva", "Badri", "Bhargav", "Chaitanya", "Devendra", "Digvijay", "Eshwar", "Fateh", "Gautam", "Govind",
    "Harish", "Himanshu", "Indrajit", "Jaideep", "Jawahar", "Kalyan", "Kuldeep", "Lalit", "Madhav", "Manvendra",
    "Mukesh", "Naveen", "Niraj", "Padma", "Pankhuri", "Pranav", "Raghav", "Rajnath", "Ranveer", "Sanjay",
    "Sarvesh", "Shashank", "Siddharth", "Somesh", "Tarun", "Umesh", "Vaibhav", "Varun", "Vikas", "Yashwant"
]

# 25 Test locations (Unseen in CR-14 or CR-15 training)
TEST_LOCATIONS = [
    "Alwar", "Bareilly", "Belgaum", "Calicut", "Durgapur", "Faridabad", "Gorakhpur", "Haridwar",
    "Imphal", "Jamnagar", "Kakinada", "Kollam", "Malda", "Muzaffarpur", "Nellore", "Pathankot",
    "Rajahmundry", "Rourkela", "Saharanpur", "Thanjavur", "Thrissur", "Udaipur", "Ujjain", "Vellore", "Warangal"
]

# Distractor conversational statements
DISTRACTOR_STATEMENTS = [
    "The quarterly business review was concluded earlier today.",
    "A heavy thunderstorm passed over the northern district.",
    "The project deliverables have been submitted to the client.",
    "New public transit guidelines were announced this morning.",
    "The library facilities will undergo scheduled renovations."
]

random.seed(1515) # Deterministic seed

probes = []
probe_id_counter = 1

def next_id(prefix):
    global probe_id_counter
    pid = f"CR15_{prefix}_{probe_id_counter:04d}"
    probe_id_counter += 1
    return pid

# 1. Entity Copying (150 probes)
# Prompt format: "Repeat the exact name: {name}.\nName:"
for _ in range(150):
    e = random.choice(TEST_ENTITIES)
    probes.append({
        "id": next_id("COPY"),
        "category": "entity_copying",
        "prompt": f"Repeat the exact name: {e}.\nName: ",
        "subject": e,
        "expected_target": e,
        "earlier_value": "",
        "distractor_values": []
    })

# 2. Entity Selection (150 probes)
# Prompt format: "Between {e1} and {e2}, {e_target} is the person mentioned first. Question: Who was mentioned first?\nAnswer:"
for _ in range(150):
    e1, e2 = random.sample(TEST_ENTITIES, 2)
    # 50% ask first, 50% ask second
    if random.random() < 0.5:
        target = e1
        q_text = "Who was mentioned first in that sentence?"
    else:
        target = e2
        q_text = "Who was mentioned second in that sentence?"
    prompt = f"Between {e1} and {e2}, two individuals were discussed.\nQuestion: {q_text}\nAnswer: "
    probes.append({
        "id": next_id("SEL"),
        "category": "entity_selection",
        "prompt": prompt,
        "subject": target,
        "expected_target": target,
        "earlier_value": "",
        "distractor_values": [e2 if target == e1 else e1]
    })

# 3. Static Entity-Value Binding (150 probes)
# Prompt format: Single context with 2 entities and their locations
for _ in range(150):
    e1, e2 = random.sample(TEST_ENTITIES, 2)
    c1, c2 = random.sample(TEST_LOCATIONS, 2)
    q_target = random.choice([e1, e2])
    c_target = c1 if q_target == e1 else c2
    c_other = c2 if q_target == e1 else c1
    prompt = f"Context: {e1} lives in {c1}. {e2} lives in {c2}.\nQuestion: Where is {q_target} located?\nAnswer: "
    probes.append({
        "id": next_id("STAT"),
        "category": "static_binding",
        "prompt": prompt,
        "subject": q_target,
        "expected_target": c_target,
        "earlier_value": "",
        "distractor_values": [c_other]
    })

# 4. Single-Entity State Updates (150 probes)
# Prompt format: Turn 1: X in Y. Turn 2: X moved to Z. Question: Where is X now?
for _ in range(150):
    e = random.choice(TEST_ENTITIES)
    c1, c2 = random.sample(TEST_LOCATIONS, 2)
    prompt = f"Turn 1:\nUser: {e} lives in {c1}.\n\nTurn 2:\nUser: {e} moved to {c2}.\n\nTurn 3:\nUser: Where is {e} currently located?\nAssistant: "
    probes.append({
        "id": next_id("SEUP"),
        "category": "single_entity_updates",
        "prompt": prompt,
        "subject": e,
        "expected_target": c2,
        "earlier_value": c1,
        "distractor_values": []
    })

# 5. Multi-Entity State Tracking (150 probes)
# Turn 1: e1 in c1. Turn 2: e2 in c2. Turn 3: e1 moved to c3. Query e2!
for _ in range(150):
    e1, e2 = random.sample(TEST_ENTITIES, 2)
    c1, c2, c3 = random.sample(TEST_LOCATIONS, 3)
    prompt = f"Turn 1:\nUser: {e1} lives in {c1}.\n\nTurn 2:\nUser: {e2} lives in {c2}.\n\nTurn 3:\nUser: {e1} moved to {c3}.\n\nTurn 4:\nUser: Where is {e2} currently located?\nAssistant: "
    probes.append({
        "id": next_id("MEST"),
        "category": "multi_entity_tracking",
        "prompt": prompt,
        "subject": e2,
        "expected_target": c2,
        "earlier_value": "",
        "distractor_values": [c1, c3]
    })

# 6. Interleaved Updates (150 probes)
# Turn 1: e1 in c1. Turn 2: e2 in c2. Turn 3: e1 moved to c3. Turn 4: e2 moved to c4. Query e1!
for _ in range(150):
    e1, e2 = random.sample(TEST_ENTITIES, 2)
    c1, c2, c3, c4 = random.sample(TEST_LOCATIONS, 4)
    q_target = random.choice([e1, e2])
    c_target = c3 if q_target == e1 else c4
    c_stale = c1 if q_target == e1 else c2
    c_other_recent = c4 if q_target == e1 else c3
    prompt = f"Turn 1:\nUser: {e1} lives in {c1}.\n\nTurn 2:\nUser: {e2} lives in {c2}.\n\nTurn 3:\nUser: {e1} moved to {c3}.\n\nTurn 4:\nUser: {e2} moved to {c4}.\n\nTurn 5:\nUser: In which city is {q_target} now?\nAssistant: "
    probes.append({
        "id": next_id("ILUP"),
        "category": "interleaved_updates",
        "prompt": prompt,
        "subject": q_target,
        "expected_target": c_target,
        "earlier_value": c_stale,
        "distractor_values": [c_other_recent]
    })

# 7. Delayed Retrieval with Distractors (150 probes)
# Same as single update but with 2 intervening conversational distractor turns
for _ in range(150):
    e = random.choice(TEST_ENTITIES)
    c1, c2 = random.sample(TEST_LOCATIONS, 2)
    d1 = random.choice(DISTRACTOR_STATEMENTS)
    d2 = random.choice([s for s in DISTRACTOR_STATEMENTS if s != d1])
    prompt = f"Turn 1:\nUser: {e} lives in {c1}.\n\nTurn 2:\nUser: {e} moved to {c2}.\n\nTurn 3:\nUser: {d1}\n\nTurn 4:\nUser: {d2}\n\nTurn 5:\nUser: Where can {e} be found now?\nAssistant: "
    probes.append({
        "id": next_id("DRDIST"),
        "category": "delayed_retrieval_distractors",
        "prompt": prompt,
        "subject": e,
        "expected_target": c2,
        "earlier_value": c1,
        "distractor_values": []
    })

# 8. Calibrated Abstention (150 probes)
# Mention e1 in c1, e2 in c2. Query unmentioned e_unmentioned, or query attribute unmentioned!
for _ in range(150):
    e1, e2 = random.sample(TEST_ENTITIES, 2)
    c1, c2 = random.sample(TEST_LOCATIONS, 2)
    e_unmentioned = random.choice([x for x in TEST_ENTITIES if x != e1 and x != e2])
    prompt = f"Turn 1:\nUser: {e1} lives in {c1}.\n\nTurn 2:\nUser: {e2} lives in {c2}.\n\nTurn 3:\nUser: What is the current location of {e_unmentioned}?\nAssistant: "
    probes.append({
        "id": next_id("ABST"),
        "category": "calibrated_abstention",
        "prompt": prompt,
        "subject": e_unmentioned,
        "expected_target": "abstain",
        "earlier_value": "",
        "distractor_values": [c1, c2]
    })

assert len(probes) == 1200, f"Expected 1200 probes, got {len(probes)}"

# Save frozen benchmark
bench_file = BENCHMARK_DIR / "benchmark_1200_sealed.json"
with open(bench_file, "w", encoding="utf-8") as f:
    json.dump(probes, f, indent=2)

bench_bytes = bench_file.read_bytes()
bench_hash = hashlib.sha256(bench_bytes).hexdigest()

manifest = {
    "benchmark_name": "CR-EXP-15 Decomposed 1,200-Probe Cognitive Benchmark",
    "total_probes": len(probes),
    "sha256": bench_hash,
    "categories": {
        "entity_copying": 150,
        "entity_selection": 150,
        "static_binding": 150,
        "single_entity_updates": 150,
        "multi_entity_tracking": 150,
        "interleaved_updates": 150,
        "delayed_retrieval_distractors": 150,
        "calibrated_abstention": 150
    },
    "test_entities_count": len(TEST_ENTITIES),
    "test_locations_count": len(TEST_LOCATIONS)
}

with open(BENCHMARK_DIR / "CR15_BENCHMARK_MANIFEST.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

with open(REPO_ROOT / "experiments" / "cognition" / "cr_exp_15_architecture_binding" / "CR15_FROZEN_TEST_HASH.txt", "w", encoding="utf-8") as f:
    f.write(bench_hash + "\n")

print(f"Generated CR-EXP-15 Benchmark with {len(probes)} probes.")
print(f"SHA-256: {bench_hash}")
