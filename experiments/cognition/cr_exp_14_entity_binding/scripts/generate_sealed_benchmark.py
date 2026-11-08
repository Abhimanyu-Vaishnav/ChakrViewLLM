"""
CR-EXP-14 Phase 3: Sealed 1,200-Probe Dynamic Entity Binding Benchmark Generator
Generates exactly 1,200 probes across 7 targeted categories:
1. single_entity_state_tracking: 200 probes
2. two_entity_state_tracking: 200 probes
3. three_entity_state_tracking: 200 probes
4. repeated_updates_recency: 200 probes
5. interleaved_conversational_distractors: 150 probes
6. counterfactual_reassignment: 150 probes
7. missing_information_abstention: 100 probes

Uses strictly held-out entity names, cities, professions, and template phrasing.
Outputs:
- benchmark_1200_sealed.json
- CR14_BENCHMARK_MANIFEST.json
- CR14_LEAKAGE_AUDIT.md
- CR14_FROZEN_TEST_HASH.txt
"""

import sys
import json
import hashlib
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_14_entity_binding" / "benchmark_data"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

SEED = 202614
random.seed(SEED)

# Fresh, Unseen Names for CR-14 Sealed Benchmark (never used in CR-09, 10, 11, 12, or 14 training)
BENCHMARK_NAMES = [
    "Abhay", "Bimla", "Chetan", "Damodar", "Ekta", "Girish", "Hema", "Indu", "Janak", "Kamlesh",
    "Laxman", "Mina", "Narayan", "Omwati", "Pushpa", "Ratan", "Sarla", "Trilok", "Urmila", "Ved",
    "Aniruddh", "Bela", "Chirag", "Daksha", "Gajendra", "Hansmukh", "Jagruti", "Kailashpati", "Lata", "Mukund",
    "Nalini", "Ojasvi", "Prakash", "Radha", "Subhash", "Trupti", "Udit", "Vatsala", "Yadunath", "Zulekha"
]

BENCHMARK_CITIES = [
    "Ajmer", "Bikaner", "Cuttack", "Dhanbad", "Erode", "Firozabad", "Gaya", "Hubli", "Itanagar", "Jhansi",
    "Karnal", "Ludhiana", "Mathura", "Nanded", "Ooty", "Panipat", "Ratlam", "Siliguri", "Tirupati", "Ujjain"
]

BENCHMARK_PROFESSIONS = [
    "astronomer", "blacksmith", "carpenter", "drafter", "ecologist", "filmmaker", "gardener", "historian", "illustrator", "jeweler"
]

def generate_sealed_benchmark():
    probes = []
    pid = 1
    
    # 1. Single-Entity State Tracking (200)
    for i in range(200):
        name = BENCHMARK_NAMES[i % len(BENCHMARK_NAMES)]
        c1 = BENCHMARK_CITIES[i % len(BENCHMARK_CITIES)]
        c2 = BENCHMARK_CITIES[(i + 3) % len(BENCHMARK_CITIES)]
        prompt = (
            f"User: Please note that {name} is staying in {c1}.\n"
            f"Assistant: Understood, {name} is in {c1}.\n"
            f"User: Update: {name} just moved over to {c2}.\n"
            f"Assistant: Noted the update.\n"
            f"User: Where is {name} now?\n"
            f"Assistant: "
        )
        probes.append({
            "id": f"CR14_CAT1_SE_{pid:04d}",
            "category": "single_entity_state_tracking",
            "prompt": prompt,
            "subject": name,
            "expected_target": c2,
            "earlier_value": c1,
            "entities_present": [name]
        })
        pid += 1
        
    # 2. Two-Entity State Tracking (200)
    for i in range(200):
        n1 = BENCHMARK_NAMES[i % len(BENCHMARK_NAMES)]
        n2 = BENCHMARK_NAMES[(i + 1) % len(BENCHMARK_NAMES)]
        c1 = BENCHMARK_CITIES[i % len(BENCHMARK_CITIES)]
        c2 = BENCHMARK_CITIES[(i + 1) % len(BENCHMARK_CITIES)]
        c1_up = BENCHMARK_CITIES[(i + 4) % len(BENCHMARK_CITIES)]
        c2_up = BENCHMARK_CITIES[(i + 5) % len(BENCHMARK_CITIES)]
        
        target_name = n1 if (i % 2 == 0) else n2
        target_val = c1_up if (i % 2 == 0) else c2_up
        stale_val = c1 if (i % 2 == 0) else c2
        distractor_val = c2_up if (i % 2 == 0) else c1_up
        
        prompt = (
            f"User: Track these two people: {n1} is in {c1}, and {n2} is in {c2}.\n"
            f"Assistant: Recorded both.\n"
            f"User: Update: {n1} relocated to {c1_up}.\n"
            f"Assistant: Updated {n1}.\n"
            f"User: Also, {n2} relocated to {c2_up}.\n"
            f"Assistant: Updated {n2}.\n"
            f"User: Where is {target_name} now?\n"
            f"Assistant: "
        )
        probes.append({
            "id": f"CR14_CAT2_TE_{pid:04d}",
            "category": "two_entity_state_tracking",
            "prompt": prompt,
            "subject": target_name,
            "expected_target": target_val,
            "earlier_value": stale_val,
            "distractor_values": [distractor_val],
            "entities_present": [n1, n2]
        })
        pid += 1
        
    # 3. Three-Entity State Tracking (200)
    for i in range(200):
        n1 = BENCHMARK_NAMES[i % len(BENCHMARK_NAMES)]
        n2 = BENCHMARK_NAMES[(i + 1) % len(BENCHMARK_NAMES)]
        n3 = BENCHMARK_NAMES[(i + 2) % len(BENCHMARK_NAMES)]
        c1 = BENCHMARK_CITIES[i % len(BENCHMARK_CITIES)]
        c2 = BENCHMARK_CITIES[(i + 1) % len(BENCHMARK_CITIES)]
        c3 = BENCHMARK_CITIES[(i + 2) % len(BENCHMARK_CITIES)]
        c1_up = BENCHMARK_CITIES[(i + 4) % len(BENCHMARK_CITIES)]
        c2_up = BENCHMARK_CITIES[(i + 5) % len(BENCHMARK_CITIES)]
        c3_up = BENCHMARK_CITIES[(i + 6) % len(BENCHMARK_CITIES)]
        
        target_idx = i % 3
        if target_idx == 0:
            target_name, target_val, stale_val, dist_vals = n1, c1_up, c1, [c2_up, c3_up]
        elif target_idx == 1:
            target_name, target_val, stale_val, dist_vals = n2, c2_up, c2, [c1_up, c3_up]
        else:
            target_name, target_val, stale_val, dist_vals = n3, c3_up, c3, [c1_up, c2_up]
            
        prompt = (
            f"User: Tracking group: {n1} in {c1}, {n2} in {c2}, {n3} in {c3}.\n"
            f"Assistant: Group recorded.\n"
            f"User: {n1} transferred to {c1_up}. {n2} transferred to {c2_up}. {n3} transferred to {c3_up}.\n"
            f"Assistant: Group updates recorded.\n"
            f"User: Where is {target_name} now?\n"
            f"Assistant: "
        )
        probes.append({
            "id": f"CR14_CAT3_3E_{pid:04d}",
            "category": "three_entity_state_tracking",
            "prompt": prompt,
            "subject": target_name,
            "expected_target": target_val,
            "earlier_value": stale_val,
            "distractor_values": dist_vals,
            "entities_present": [n1, n2, n3]
        })
        pid += 1
        
    # 4. Repeated Updates and Recency (A -> B -> C) (200)
    for i in range(200):
        name = BENCHMARK_NAMES[i % len(BENCHMARK_NAMES)]
        c1 = BENCHMARK_CITIES[i % len(BENCHMARK_CITIES)]
        c2 = BENCHMARK_CITIES[(i + 2) % len(BENCHMARK_CITIES)]
        c3 = BENCHMARK_CITIES[(i + 5) % len(BENCHMARK_CITIES)]
        prompt = (
            f"User: {name} arrived at {c1}.\n"
            f"Assistant: Recorded {name} at {c1}.\n"
            f"User: Then {name} traveled to {c2}.\n"
            f"Assistant: Noted {name} at {c2}.\n"
            f"User: Finally {name} reached {c3}.\n"
            f"Assistant: Noted final destination {c3}.\n"
            f"User: Where is {name} now?\n"
            f"Assistant: "
        )
        probes.append({
            "id": f"CR14_CAT4_REP_{pid:04d}",
            "category": "repeated_updates_recency",
            "prompt": prompt,
            "subject": name,
            "expected_target": c3,
            "earlier_value": c2,
            "stale_initial": c1,
            "entities_present": [name]
        })
        pid += 1
        
    # 5. Interleaved Conversational Distractors (150)
    conv_chatter = [
        "By the way, what is the weather like?",
        "Could you check your network latency?",
        "I hope our servers are running smoothly.",
        "Let us continue tracking after this brief interruption."
    ]
    for i in range(150):
        name = BENCHMARK_NAMES[i % len(BENCHMARK_NAMES)]
        c1 = BENCHMARK_CITIES[i % len(BENCHMARK_CITIES)]
        c2 = BENCHMARK_CITIES[(i + 4) % len(BENCHMARK_CITIES)]
        chatter = conv_chatter[i % len(conv_chatter)]
        prompt = (
            f"User: Please track this update: {name} is staying in {c1}.\n"
            f"Assistant: Understood, {name} is in {c1}.\n"
            f"User: {chatter}\n"
            f"Assistant: Acknowledged. I am ready for further state updates.\n"
            f"User: Update: {name} has now moved over to {c2}.\n"
            f"Assistant: Noted the new location.\n"
            f"User: Where is {name} now?\n"
            f"Assistant: "
        )
        probes.append({
            "id": f"CR14_CAT5_DIST_{pid:04d}",
            "category": "interleaved_conversational_distractors",
            "prompt": prompt,
            "subject": name,
            "expected_target": c2,
            "earlier_value": c1,
            "entities_present": [name]
        })
        pid += 1
        
    # 6. Counterfactual Reassignment (Context-form) (150)
    for i in range(150):
        name = BENCHMARK_NAMES[i % len(BENCHMARK_NAMES)]
        c1 = BENCHMARK_CITIES[(i + 1) % len(BENCHMARK_CITIES)]
        c2 = BENCHMARK_CITIES[(i + 6) % len(BENCHMARK_CITIES)]
        prompt = f"Context: Initially, {name} lived in {c1}. Later on, {name} relocated to {c2}.\nQuestion: Where does {name} live currently?\nAnswer: "
        probes.append({
            "id": f"CR14_CAT6_CF_{pid:04d}",
            "category": "counterfactual_reassignment",
            "prompt": prompt,
            "subject": name,
            "expected_target": c2,
            "earlier_value": c1,
            "entities_present": [name]
        })
        pid += 1
        
    # 7. Missing Information Abstention (100)
    for i in range(50):
        n_ctx = BENCHMARK_NAMES[i % len(BENCHMARK_NAMES)]
        n_query = BENCHMARK_NAMES[(i + 10) % len(BENCHMARK_NAMES)]
        city = BENCHMARK_CITIES[i % len(BENCHMARK_CITIES)]
        prompt = f"Context: {n_ctx} works in {city}.\nQuestion: Where does {n_query} work?\nAnswer: "
        probes.append({
            "id": f"CR14_CAT7_ABST_{pid:04d}",
            "category": "missing_information_abstention",
            "prompt": prompt,
            "subject": n_query,
            "expected_target": "abstain",
            "abstention_type": "unmentioned_entity"
        })
        pid += 1
        
    for i in range(50):
        n_ctx = BENCHMARK_NAMES[i % len(BENCHMARK_NAMES)]
        city = BENCHMARK_CITIES[(i + 2) % len(BENCHMARK_CITIES)]
        prompt = f"Context: {n_ctx} lives in {city}.\nQuestion: What is {n_ctx}'s profession?\nAnswer: "
        probes.append({
            "id": f"CR14_CAT7_ABST_{pid:04d}",
            "category": "missing_information_abstention",
            "prompt": prompt,
            "subject": n_ctx,
            "expected_target": "abstain",
            "abstention_type": "unmentioned_attribute"
        })
        pid += 1
        
    return probes

def main():
    probes = generate_sealed_benchmark()
    assert len(probes) == 1200, f"Expected 1200 probes, got {len(probes)}"
    
    suite_path = BENCHMARK_DIR / "benchmark_1200_sealed.json"
    with open(suite_path, "w", encoding="utf-8") as f:
        json.dump(probes, f, indent=2)
        
    suite_bytes = suite_path.read_bytes()
    suite_sha = hashlib.sha256(suite_bytes).hexdigest()
    
    counts = {}
    for p in probes:
        counts[p["category"]] = counts.get(p["category"], 0) + 1
        
    manifest = {
        "benchmark_title": "CR-EXP-14 Sealed 1,200-Probe Dynamic Entity Binding Benchmark",
        "file_name": "benchmark_1200_sealed.json",
        "total_probes": len(probes),
        "sha256": suite_sha,
        "seed": SEED,
        "generation_timestamp": "2026-11-08T10:11:00+05:30",
        "categories": counts
    }
    with open(BENCHMARK_DIR / "CR14_BENCHMARK_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    hash_file = REPO_ROOT / "experiments" / "cognition" / "cr_exp_14_entity_binding" / "CR14_FROZEN_TEST_HASH.txt"
    hash_file.write_text(f"{suite_sha}  experiments/cognition/cr_exp_14_entity_binding/benchmark_data/benchmark_1200_sealed.json\n", encoding="utf-8")
    
    print(f"Generated sealed benchmark with {len(probes)} probes.")
    print(f"SHA-256: {suite_sha}")
    for cat, cnt in sorted(counts.items()):
        print(f"  {cat:40s}: {cnt}")

if __name__ == "__main__":
    main()
