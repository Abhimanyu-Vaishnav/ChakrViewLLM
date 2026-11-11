"""CR-EXP-16D Multi-Task Dataset Synthesizer
Generates:
1. Replay Mixtures for Arms A, B, C, D (4,000 samples each):
   - Arm A: 0% Copying, 100% Retrieval (0 / 4,000)
   - Arm B: 25% Copying, 75% Retrieval (1,000 / 3,000)
   - Arm C: 50% Copying, 50% Retrieval (2,000 / 2,000)
   - Arm D: 75% Copying, 25% Retrieval (3,000 / 1,000)
   - Validation set: 600 balanced samples (300 copy, 300 retrieval)
2. Compositional Generalization Benchmark (cr16d_compositional_benchmark_1000.json):
   - Group 1: Seen Keys, Seen Values, Unseen Combinations (250)
   - Group 2: Seen Keys, Unseen Values (250)
   - Group 3: Unseen Keys, Seen Values (250)
   - Group 4: Unseen Keys, Unseen Values (250)
Strictly audits zero leakage against all frozen benchmarks (CR-16 Copying, CR-16B, CR-16C).
"""
import sys
import json
import random
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
DATA_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Word pools for Copying
GREEK = [
    'alpha', 'beta', 'delta', 'epsilon', 'eta', 'gamma', 'iota', 'kappa', 'lambda', 'mu',
    'nu', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'theta', 'xi', 'zeta'
]
COMMON_NOUNS = [
    'apple', 'banana', 'orange', 'grape', 'mango', 'peach', 'cherry', 'lemon', 'berry', 'melon',
    'red', 'blue', 'green', 'yellow', 'white', 'black', 'silver', 'gold', 'purple', 'brown',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'cat', 'dog', 'bird', 'fish', 'horse', 'lion', 'tiger', 'bear', 'wolf', 'fox',
    'sun', 'moon', 'star', 'cloud', 'rain', 'snow', 'wind', 'storm', 'river', 'lake',
    'north', 'south', 'east', 'west', 'door', 'window', 'table', 'chair', 'road', 'hill',
    'stone', 'field', 'forest', 'valley', 'mountain', 'desert', 'ocean', 'island', 'tree', 'flower',
    'bread', 'water', 'fire', 'light', 'dark', 'night', 'day', 'time', 'year', 'house',
    'city', 'town', 'village', 'king', 'queen', 'prince', 'knight', 'castle', 'tower', 'bridge',
    'iron', 'steel', 'wood', 'glass', 'paper', 'book', 'bell', 'ring', 'song', 'word'
]

# Pools for Retrieval
KEYS_FAMILIAR = ['city', 'color', 'fruit', 'animal', 'metal', 'weather', 'country', 'number', 'tool', 'vehicle']
VALUES_FAMILIAR = {
    'city': ['Delhi', 'London', 'Paris', 'Tokyo', 'Rome', 'Berlin', 'Madrid', 'Cairo'],
    'color': ['Red', 'Blue', 'Green', 'Yellow', 'White', 'Black', 'Gold', 'Silver'],
    'fruit': ['Apple', 'Banana', 'Mango', 'Orange', 'Grape', 'Peach', 'Lemon', 'Berry'],
    'animal': ['Lion', 'Tiger', 'Bear', 'Wolf', 'Eagle', 'Horse', 'Fox', 'Deer'],
    'metal': ['Iron', 'Steel', 'Copper', 'Silver', 'Gold', 'Bronze', 'Lead', 'Tin'],
    'weather': ['Rain', 'Snow', 'Wind', 'Storm', 'Cloud', 'Sunny', 'Fog', 'Frost'],
    'country': ['India', 'France', 'Japan', 'Spain', 'Italy', 'Canada', 'Brazil', 'Egypt'],
    'number': ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight'],
    'tool': ['Hammer', 'Wrench', 'Pliers', 'Drill', 'Saw', 'Axe', 'Chisel', 'File'],
    'vehicle': ['Car', 'Truck', 'Train', 'Plane', 'Boat', 'Ship', 'Bus', 'Bike']
}

KEYS_UNSEEN = ['planet', 'gem', 'season', 'beverage', 'instrument', 'tree', 'weapon', 'fabric']
VALUES_UNSEEN = {
    'planet': ['Mars', 'Venus', 'Jupiter', 'Saturn', 'Mercury'],
    'gem': ['Ruby', 'Diamond', 'Emerald', 'Sapphire', 'Opal'],
    'season': ['Summer', 'Winter', 'Spring', 'Autumn', 'Monsoon'],
    'beverage': ['Coffee', 'Tea', 'Water', 'Juice', 'Milk'],
    'instrument': ['Guitar', 'Piano', 'Flute', 'Violin', 'Drum'],
    'tree': ['Oak', 'Pine', 'Cedar', 'Maple', 'Birch'],
    'weapon': ['Sword', 'Shield', 'Bow', 'Spear', 'Dagger'],
    'fabric': ['Silk', 'Cotton', 'Wool', 'Linen', 'Velvet']
}

# Load all frozen benchmarks to ensure ZERO prompt/sequence leakage
bench16_file = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/benchmark_data/cr16_synthetic_benchmark_1000.json"
bench16b_file = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16b_copying_generalization/data/cr16b_sealed_benchmark_1500.json"
bench16c_file = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16c_key_value_retrieval/data/cr16c_sealed_benchmark_1500.json"

with open(bench16_file) as f: b16 = json.load(f)
with open(bench16b_file) as f: b16b = json.load(f)
with open(bench16c_file) as f: b16c = json.load(f)

frozen_prompts = set(p['prompt'] for p in b16) | set(p['prompt'] for p in b16b) | set(p['prompt'] for p in b16c)
print(f"Loaded {len(frozen_prompts)} frozen benchmark prompts across CR-16, CR-16B, and CR-16C.")

random.seed(54321)

# Helper: make copying sample
COPY_TEMPLATES = [
    ("Echo sequence: {seq}\nOutput: ", "{seq}\n"),
    ("Repeat: {seq}\nOutput: ", "{seq}\n"),
    ("Copy: {seq} -> ", "{seq}\n"),
    ("Sequence: {seq} -> ", "{seq}\n"),
    ("Input: {seq}\nCopy: ", "{seq}\n")
]

def make_copying_sample():
    while True:
        L = random.randint(1, 5)
        pool = random.choice([GREEK, COMMON_NOUNS])
        seq = ' '.join(random.sample(pool, L))
        tmpl_p, tmpl_t = random.choice(COPY_TEMPLATES)
        p = tmpl_p.format(seq=seq)
        t = tmpl_t.format(seq=seq)
        if p not in frozen_prompts:
            return {"task": "copying", "prompt": p, "target": t, "expected": seq}

# Helper: make retrieval sample
RETR_TEMPLATES = [
    ("Store: {key} = {val}. Query: {query}.\nAnswer: ", "{ans}\n"),
    ("Set {key} to {val}. What is {query}?\nAnswer: ", "{ans}\n"),
    ("Assign: {key} -> {val}. Query: {query}?\nValue: ", "{ans}\n"),
    ("Record: {key} is {val}. Find {query}:\nOutput: ", "{ans}\n"),
]

def make_retrieval_sample():
    while True:
        n_facts = random.randint(1, 3)
        chosen_keys = random.sample(KEYS_FAMILIAR, n_facts)
        stores = []
        current_state = {}
        for k in chosen_keys:
            v = random.choice(VALUES_FAMILIAR[k])
            stores.append((k, v, False))
            current_state[k] = v
        query_key = chosen_keys[0]
        
        # 30% chance reassigned
        r = random.random()
        if r < 0.30:
            pool = [v for v in VALUES_FAMILIAR[query_key] if v != current_state[query_key]]
            new_v = random.choice(pool)
            stores.append((query_key, new_v, True))
            current_state[query_key] = new_v
        elif 0.30 <= r < 0.45:
            # noise distractor
            stores.insert(random.randint(0, len(stores)), ('noise', random.choice(['Alpha', 'Beta', 'Gamma']), 'noise'))
        elif 0.45 <= r < 0.55:
            # missing key
            query_key = random.choice([k for k in KEYS_FAMILIAR if k not in current_state] or ['unknown_key'])
            current_state[query_key] = "[UNKNOWN]"
            
        expected_ans = current_state[query_key]
        tmpl_p, tmpl_t = random.choice(RETR_TEMPLATES)
        
        context_parts = []
        for k, v, is_up in stores:
            if is_up == 'noise': context_parts.append(f"Notice: {k} is {v}.")
            elif is_up is True: context_parts.append(f"Update: {k} = {v}.")
            else: context_parts.append(f"Store: {k} = {v}.")
        context_str = ' '.join(context_parts)
        prompt = f"{context_str} Query: {query_key}.\nAnswer: "
        target = f"{expected_ans}\n"
        
        if prompt not in frozen_prompts:
            return {"task": "retrieval", "prompt": prompt, "target": target, "expected": expected_ans}

# 1. Synthesize Master Pools
print("Generating Master Copying Pool (4,000) and Retrieval Pool (4,000)...")
copy_pool = [make_copying_sample() for _ in range(4000)]
retr_pool = [make_retrieval_sample() for _ in range(4000)]

# 2. Build Arm Mixtures (4,000 training samples each)
arm_configs = {
    "arm_a_replay00": (0, 4000),      # 0% copy, 100% retrieval
    "arm_b_replay25": (1000, 3000),   # 25% copy, 75% retrieval
    "arm_c_replay50": (2000, 2000),   # 50% copy, 50% retrieval
    "arm_d_replay75": (3000, 1000),   # 75% copy, 25% retrieval
}

for arm_name, (n_copy, n_retr) in arm_configs.items():
    mixture = copy_pool[:n_copy] + retr_pool[:n_retr]
    random.shuffle(mixture)
    out_file = DATA_DIR / f"{arm_name}_train.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(mixture, f, indent=2)
    print(f"Saved {arm_name}_train.json: {n_copy} copy + {n_retr} retr = {len(mixture)} total.")

# 3. Validation Mixture (300 copy + 300 retr = 600 total)
val_mixture = [make_copying_sample() for _ in range(300)] + [make_retrieval_sample() for _ in range(300)]
random.shuffle(val_mixture)
with open(DATA_DIR / "cr16d_multitask_val.json", "w", encoding="utf-8") as f:
    json.dump(val_mixture, f, indent=2)
print("Saved cr16d_multitask_val.json (600 samples).")

# 4. Synthesize Compositional Generalization Benchmark (1,000 probes)
print("Generating 1,000-probe Compositional Benchmark across G1..G4...")
comp_probes = []
pid = 1

def add_comp(g_code, g_name, p, ans, meta):
    global pid
    comp_probes.append({
        "id": f"CR16D_{g_code}_{pid:04d}",
        "group_code": g_code,
        "group_name": g_name,
        "prompt": p,
        "expected_target": ans,
        "metadata": meta
    })
    pid += 1

# G1: Seen Keys, Seen Values, UNSEEN COMBINATIONS (250 probes)
# e.g. Store: city = Apple. (city with fruit value!)
for _ in range(250):
    k = random.choice(KEYS_FAMILIAR)
    other_k = random.choice([x for x in KEYS_FAMILIAR if x != k])
    v = random.choice(VALUES_FAMILIAR[other_k]) # Cross-category value!
    p = f"Store: {k} = {v}. Query: {k}.\nAnswer: "
    add_comp("G1", "unseen_key_val_combos", p, v, {"key": k, "val": v, "type": "cross_category"})

# G2: Seen Keys, UNSEEN VALUES (250 probes)
# e.g. Store: city = Mars. (city with novel value from VALUES_UNSEEN)
all_unseen_values = [v for sublist in VALUES_UNSEEN.values() for v in sublist]
for _ in range(250):
    k = random.choice(KEYS_FAMILIAR)
    v = random.choice(all_unseen_values)
    p = f"Store: {k} = {v}. Query: {k}.\nAnswer: "
    add_comp("G2", "seen_key_unseen_val", p, v, {"key": k, "val": v, "type": "unseen_val"})

# G3: UNSEEN KEYS, Seen Values (250 probes)
# e.g. Store: planet = Delhi. (novel key with familiar value)
all_familiar_values = [v for sublist in VALUES_FAMILIAR.values() for v in sublist]
for _ in range(250):
    k = random.choice(KEYS_UNSEEN)
    v = random.choice(all_familiar_values)
    p = f"Store: {k} = {v}. Query: {k}.\nAnswer: "
    add_comp("G3", "unseen_key_seen_val", p, v, {"key": k, "val": v, "type": "unseen_key"})

# G4: UNSEEN KEYS, UNSEEN VALUES (250 probes)
# e.g. Store: planet = Mars. (completely novel key-val pair)
for _ in range(250):
    k = random.choice(KEYS_UNSEEN)
    v = random.choice(VALUES_UNSEEN[k])
    p = f"Store: {k} = {v}. Query: {k}.\nAnswer: "
    add_comp("G4", "unseen_key_unseen_val", p, v, {"key": k, "val": v, "type": "both_unseen"})

assert len(comp_probes) == 1000
comp_file = DATA_DIR / "cr16d_compositional_benchmark_1000.json"
with open(comp_file, "w", encoding="utf-8") as f:
    json.dump(comp_probes, f, indent=2)

h_comp = hashlib.sha256(comp_file.read_bytes()).hexdigest()
print(f"Saved cr16d_compositional_benchmark_1000.json | SHA-256: {h_comp}")

with open(DATA_DIR / "CR16D_BENCHMARK_HASH.txt", "w") as f:
    f.write(f"cr16d_compositional_benchmark_1000.json: {h_comp}\n")

