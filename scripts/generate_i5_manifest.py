import os
import sys
import json
import time
import random
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def generate_i5_benchmark_manifest():
    print("=" * 60)
    print("GENERATING WAVE 480: I5 BENCHMARK MANIFEST")
    print("=" * 60)
    
    os.makedirs("artifacts/waves_480_487_execution", exist_ok=True)
    
    standard_entities = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
    unseen_entities = ["Krypton", "Xenon", "Radon", "Argon", "Helium", "Neon"]
    relations = ["above", "below", "left", "right", "north", "south"]
    
    random.seed(42)
    manifest = {
        "1_hop": [],
        "2_hop": [],
        "3_hop": [],
        "4_hop": [],
        "adversarial": []
    }
    
    # 1-Hop Tasks (I4 Baseline)
    for _ in range(20):
        e1, e2 = random.sample(standard_entities, 2)
        rel = random.choice(relations)
        manifest["1_hop"].append({
            "chain": [(e1, rel, e2)],
            "query": (e1, rel),
            "target": e2
        })
        
    # 2-Hop Tasks (I5-2)
    for _ in range(20):
        e1, e2, e3 = random.sample(standard_entities, 3)
        r1, r2 = random.sample(relations, 2)
        manifest["2_hop"].append({
            "chain": [(e1, r1, e2), (e2, r2, e3)],
            "query": (e1, r1, r2),
            "target": e3
        })
        
    # 3-Hop Tasks (I5-3)
    for _ in range(20):
        e1, e2, e3, e4 = random.sample(standard_entities, 4)
        r1, r2, r3 = random.choices(relations, k=3)
        manifest["3_hop"].append({
            "chain": [(e1, r1, e2), (e2, r2, e3), (e3, r3, e4)],
            "query": (e1, r1, r2, r3),
            "target": e4
        })
        
    # 4-Hop Tasks (I5-4)
    for _ in range(20):
        e1, e2, e3, e4, e5 = random.sample(standard_entities, 5)
        r1, r2, r3, r4 = random.choices(relations, k=4)
        manifest["4_hop"].append({
            "chain": [(e1, r1, e2), (e2, r2, e3), (e3, r3, e4), (e4, r4, e5)],
            "query": (e1, r1, r2, r3, r4),
            "target": e5
        })
        
    # Adversarial & Edge Cases
    # Unseen entities, distractors, contradictions, unknown
    for _ in range(10):
        u1, u2, u3 = random.sample(unseen_entities, 3)
        r1, r2 = random.sample(relations, 2)
        manifest["adversarial"].append({
            "type": "unseen_entities",
            "chain": [(u1, r1, u2), (u2, r2, u3)],
            "query": (u1, r1, r2),
            "target": u3
        })
        
    manifest["adversarial"].append({
        "type": "unknown_rejection",
        "chain": [("Alpha", "above", "Beta")],
        "query": ("Omega", "above"),
        "target": "UNKNOWN"
    })
    manifest["adversarial"].append({
        "type": "contradiction_handling",
        "chain": [("Alpha", "above", "Beta"), ("Alpha", "below", "Beta")],
        "query": ("Alpha", "relation_status"),
        "target": "CONFLICTING"
    })
    
    out_file = "artifacts/waves_480_487_execution/i5_benchmark_manifest.json"
    with open(out_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest created with {sum(len(v) for v in manifest.values())} test instances.")
    print(f"Saved to {out_file}")
    return manifest

if __name__ == "__main__":
    generate_i5_benchmark_manifest()
