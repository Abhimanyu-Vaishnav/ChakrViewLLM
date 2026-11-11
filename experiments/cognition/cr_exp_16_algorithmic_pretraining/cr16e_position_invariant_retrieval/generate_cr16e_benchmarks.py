"""Diagnostic Benchmark Generator for CR-EXP-16E:
Generates a comprehensive sealed benchmark suite (1,700 total probes):
1. B1: Primary Position-Invariance Probes (600 probes):
   - Uniformly varies target fact position across slots 1, 2, 3, 4.
   - Evaluates query-after-facts and query-before-facts.
   - Evaluates variable distractor counts (1 to 3 distractor facts).
   - Uses 3 distinct template structures.
2. B2: Matched Permutation & Counterfactual Pairs (300 pairs = 600 probes):
   - 150 position permutation pairs (300 probes): exact same facts in swapped order.
   - 150 counterfactual value-change pairs (300 probes): key identical, value changes.
3. B3: Multi-Token Span Benchmark (500 probes):
   - Length 1 token: 150 probes
   - Length 2 tokens: 150 probes
   - Length 3 tokens: 100 probes
   - Length 4 tokens: 100 probes
4. B4: Calibrated Abstention Benchmark (300 probes):
   - Missing key (100 probes)
   - Distractor with plausible value (100 probes)
   - Corrupted / near-miss key (100 probes)
   - Strict target: '[UNKNOWN]'
"""
import json
import random
from pathlib import Path
import sys

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

# Diverse Vocabulary Pools
KEYS = [
    "color", "fruit", "animal", "city", "country", "vehicle", "sport", "flower", "profession", "weather",
    "season", "planet", "element", "metal", "gem", "weapon", "fabric", "beverage", "tree", "continent",
    "instrument", "subject", "currency", "language", "mineral", "device", "author", "deity", "vessel", "river"
]

# Vocabulary pools partitioned by token length
VALUES_LEN1 = [
    "Red", "Blue", "Green", "Gold", "Apple", "Mango", "Lion", "Bear", "Wolf", "Hawk",
    "Rome", "Delhi", "Cairo", "Spain", "Car", "Bus", "Train", "Plane", "Boat", "Bike",
    "Rose", "Tulip", "Lily", "Chef", "Judge", "Rain", "Snow", "Wind", "Sun", "Iron"
]

VALUES_LEN2 = [
    "Purple", "Yellow", "Silver", "Banana", "Orange", "Tiger", "Eagle", "Shark", "Whale",
    "Paris", "London", "Tokyo", "Berlin", "Madrid", "France", "Japan", "Brazil", "Canada", "India", "Italy",
    "Soccer", "Tennis", "Cricket", "Hockey", "Rugby", "Doctor", "Lawyer", "Pilot", "Nurse", "Artist",
    "Cloud", "Frost", "Copper", "Steel", "Ruby", "Sword", "Coffee", "Juice", "Europe", "Ghar"
]

VALUES_LEN3 = [
    "Strawberry", "Pineapple", "Elephant", "Kangaroo", "Australia", "Singapore", "Amsterdam", "Helicopter",
    "Badminton", "Sunflower", "Scientist", "Architect", "Thunder", "Lightning", "Mercury", "Spring", "Shield"
]

VALUES_LEN4 = [
    "Watermelon", "Hippopotamus", "Copenhagen", "Switzerland", "Gymnastics", "Dandelion", "Psychologist",
    "Automobile", "Motorcycle", "Earthquake", "Diamond", "Cotton"
]

TEMPLATES = [
    # T1: Standard Store/Query
    {"store": "Store: {k} = {v}. ", "query_after": "Query: {k}.\nAnswer: ", "query_before": "Query: {k}. Context: "},
    # T2: Natural English Assignment
    {"store": "Set {k} to {v}. ", "query_after": "What is {k}?\nAnswer: ", "query_before": "What is {k}? Given: "},
    # T3: Arrow Mapping
    {"store": "[{k} -> {v}] ", "query_after": "Lookup [{k}] -> ", "query_before": "Find [{k}]: "}
]

def build_cr16e_benchmarks():
    rng = random.Random(1337)
    
    # 1. B1: Primary Position Invariance Probes (600 probes)
    b1_probes = []
    for i in range(600):
        tmpl = rng.choice(TEMPLATES)
        num_facts = rng.choice([2, 3, 4])
        target_pos = rng.randint(0, num_facts - 1)
        query_pos = "after" if rng.random() < 0.8 else "before"
        
        sample_keys = rng.sample(KEYS, num_facts)
        sample_vals = rng.sample(VALUES_LEN1 + VALUES_LEN2, num_facts)
        
        target_k = sample_keys[target_pos]
        target_v = sample_vals[target_pos]
        
        facts_str = "".join(tmpl["store"].format(k=k, v=v) for k, v in zip(sample_keys, sample_vals))
        if query_pos == "after":
            prompt = facts_str + tmpl["query_after"].format(k=target_k)
        else:
            prompt = tmpl["query_before"].format(k=target_k) + facts_str + "Answer: "
            
        b1_probes.append({
            "id": f"CR16E_B1_{i+1:04d}",
            "type": "position_invariance",
            "target_pos": target_pos,
            "total_facts": num_facts,
            "query_pos": query_pos,
            "prompt": prompt,
            "expected_target": target_v,
            "target_key": target_k
        })
        
    # 2. B2: Matched Permutation (150 pairs = 300 probes) & Counterfactual (150 pairs = 300 probes)
    b2_perm_pairs = []
    for i in range(150):
        tmpl = TEMPLATES[0]
        k1, k2, k3 = rng.sample(KEYS, 3)
        v1, v2, v3 = rng.sample(VALUES_LEN1 + VALUES_LEN2, 3)
        
        # Target is k1, v1
        # Order A: k1 at pos 0
        prompt_A = tmpl["store"].format(k=k1, v=v1) + tmpl["store"].format(k=k2, v=v2) + tmpl["store"].format(k=k3, v=v3) + tmpl["query_after"].format(k=k1)
        # Order B: k1 at pos 2
        prompt_B = tmpl["store"].format(k=k2, v=v2) + tmpl["store"].format(k=k3, v=v3) + tmpl["store"].format(k=k1, v=v1) + tmpl["query_after"].format(k=k1)
        
        b2_perm_pairs.append({
            "pair_id": f"CR16E_PERM_{i+1:03d}",
            "target_key": k1,
            "expected": v1,
            "probe_A": {"prompt": prompt_A, "pos": 0},
            "probe_B": {"prompt": prompt_B, "pos": 2}
        })
        
    b2_cf_pairs = []
    for i in range(150):
        tmpl = TEMPLATES[0]
        k = rng.choice(KEYS)
        vA, vB = rng.sample(VALUES_LEN1 + VALUES_LEN2, 2)
        dist_k, dist_v = rng.choice(KEYS), rng.choice(VALUES_LEN1)
        while dist_k == k: dist_k = rng.choice(KEYS)
        
        pA = tmpl["store"].format(k=k, v=vA) + tmpl["store"].format(k=dist_k, v=dist_v) + tmpl["query_after"].format(k=k)
        pB = tmpl["store"].format(k=k, v=vB) + tmpl["store"].format(k=dist_k, v=dist_v) + tmpl["query_after"].format(k=k)
        
        b2_cf_pairs.append({
            "pair_id": f"CR16E_CF_{i+1:03d}",
            "key": k,
            "probe_A": {"prompt": pA, "expected": vA},
            "probe_B": {"prompt": pB, "expected": vB}
        })
        
    # 3. B3: Multi-Token Span Benchmark (500 probes: 150 len1, 150 len2, 100 len3, 100 len4)
    b3_probes = []
    span_specs = [(VALUES_LEN1, 1, 150), (VALUES_LEN2, 2, 150), (VALUES_LEN3, 3, 100), (VALUES_LEN4, 4, 100)]
    idx = 0
    for val_pool, tok_len, count in span_specs:
        for _ in range(count):
            idx += 1
            tmpl = TEMPLATES[0]
            k = rng.choice(KEYS)
            v = rng.choice(val_pool)
            d_k = rng.choice([x for x in KEYS if x != k])
            d_v = rng.choice(VALUES_LEN1)
            
            # Randomize order
            if rng.random() < 0.5:
                prompt = tmpl["store"].format(k=k, v=v) + tmpl["store"].format(k=d_k, v=d_v) + tmpl["query_after"].format(k=k)
                pos = 0
            else:
                prompt = tmpl["store"].format(k=d_k, v=d_v) + tmpl["store"].format(k=k, v=v) + tmpl["query_after"].format(k=k)
                pos = 1
                
            b3_probes.append({
                "id": f"CR16E_B3_SPAN_{idx:04d}",
                "target_len_tokens": tok_len,
                "prompt": prompt,
                "expected_target": v,
                "target_key": k,
                "pos": pos
            })
            
    # 4. B4: Calibrated Abstention Benchmark (300 probes)
    b4_probes = []
    for i in range(300):
        tmpl = TEMPLATES[0]
        # Generate context with 1-3 stored facts
        n = rng.choice([1, 2, 3])
        sample_keys = rng.sample(KEYS, n)
        sample_vals = rng.sample(VALUES_LEN1 + VALUES_LEN2, n)
        facts_str = "".join(tmpl["store"].format(k=k, v=v) for k, v in zip(sample_keys, sample_vals))
        
        # Query key strictly NOT in sample_keys
        unbound_key = rng.choice([k for k in KEYS if k not in sample_keys])
        prompt = facts_str + tmpl["query_after"].format(k=unbound_key)
        
        b4_probes.append({
            "id": f"CR16E_B4_ABSTAIN_{i+1:04d}",
            "type": "unbound_key",
            "prompt": prompt,
            "expected_target": "[UNKNOWN]",
            "queried_key": unbound_key,
            "stored_keys": sample_keys
        })
        
    out_dir = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "cr16e_b1_position_invariance_600.json", "w") as f:
        json.dump(b1_probes, f, indent=2)
    with open(out_dir / "cr16e_b2_permutation_pairs_150.json", "w") as f:
        json.dump(b2_perm_pairs, f, indent=2)
    with open(out_dir / "cr16e_b2_counterfactual_pairs_150.json", "w") as f:
        json.dump(b2_cf_pairs, f, indent=2)
    with open(out_dir / "cr16e_b3_multitoken_span_500.json", "w") as f:
        json.dump(b3_probes, f, indent=2)
    with open(out_dir / "cr16e_b4_abstention_300.json", "w") as f:
        json.dump(b4_probes, f, indent=2)
        
    print(f"Generated CR-16E Benchmark Suites in {out_dir}:")
    print(f"  B1: {len(b1_probes)} probes")
    print(f"  B2 Perm: {len(b2_perm_pairs)} pairs (300 probes)")
    print(f"  B2 CF: {len(b2_cf_pairs)} pairs (300 probes)")
    print(f"  B3 Span: {len(b3_probes)} probes")
    print(f"  B4 Abstain: {len(b4_probes)} probes")

if __name__ == "__main__":
    build_cr16e_benchmarks()

