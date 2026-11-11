"""CR-EXP-16B Factorial Dataset Generator
Generates strictly matched training and validation datasets for the 2x2 Factorial Experiment:
Factor 1: Loss Objective (Full Loss vs Target-Only Loss) -> controlled in loss function during training
Factor 2: Template Diversity (Single Template vs Multi-Template) -> controlled in dataset formatting
Everything else (underlying sequences, vocabulary pool, sequence length, token budget) is 100% IDENTICAL.
"""
import sys
import json
import random
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
DATA_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16b_copying_generalization/data"
BENCHMARK_PATH = DATA_DIR / "cr16b_sealed_benchmark_1500.json"

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    bench = json.load(f)

bench_prompts = set(p["prompt"] for p in bench)
bench_targets = set(p["expected_target"] for p in bench)

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

# Training names completely disjoint from test names
TRAIN_NAMES = [
    'Aarav', 'Aryan', 'Kabir', 'Rohan', 'Vihaan', 'Aditya', 'Ishaan', 'Shaurya', 'Atharv', 'Dhruv',
    'Ananya', 'Diya', 'Isha', 'Myra', 'Navya', 'Pari', 'Riya', 'Saanvi', 'Tanvi', 'Zara',
    'Arjun', 'Manish', 'Naveen', 'Pranav', 'Sanjay', 'Kiran', 'Deepak', 'Rahul', 'Anil', 'Sunil'
]

# Verify zero overlap with benchmark single/multi names
bench_name_probes = [p["expected_target"] for p in bench if "name" in p["category"]]
assert len(set(TRAIN_NAMES) & set(bench_name_probes)) == 0

random.seed(424242)

# Generate 3,000 core sequences (L=1 to 6)
core_sequences = []
while len(core_sequences) < 3000:
    r = random.random()
    if r < 0.35:
        # Greek sequence (L=2 to 6)
        L = random.randint(2, 6)
        seq = ' '.join(random.sample(GREEK, L))
    elif r < 0.70:
        # Common nouns (L=1 to 5)
        L = random.randint(1, 5)
        seq = ' '.join(random.sample(COMMON_NOUNS, L))
    else:
        # Names or name combos
        L = random.choice([1, 1, 2])
        seq = ' '.join(random.sample(TRAIN_NAMES, L))
        
    # Check exact benchmark overlap
    if seq in bench_targets:
        continue
    core_sequences.append(seq)

print(f"Generated {len(core_sequences)} core synthetic sequences.")

# Now format into:
# 1. Single-template dataset: 'Echo sequence: {seq}\nOutput: {seq}\n'
# 2. Multi-template dataset: 6 diverse templates over the EXACT same sequences!
multi_templates = [
    ('Echo sequence: {seq}\nOutput: ', '{seq}\n'),
    ('Repeat: {seq}\nOutput: ', '{seq}\n'),
    ('Copy: {seq} -> ', '{seq}\n'),
    ('Sequence: {seq} -> ', '{seq}\n'),
    ('Repeat the name: {seq}\nName: ', '{seq}\n'),
    ('Input: {seq}\nCopy: ', '{seq}\n')
]

single_tmpl_samples = []
multi_tmpl_samples = []

for i, seq in enumerate(core_sequences):
    # Single template
    p_single = f"Echo sequence: {seq}\nOutput: "
    t_single = f"{seq}\n"
    single_tmpl_samples.append({
        "id": f"SEQ_{i:04d}",
        "prompt": p_single,
        "target": t_single,
        "seq": seq
    })
    
    # Multi template (deterministic round-robin or hash selection)
    tmpl_p, tmpl_t = multi_templates[i % len(multi_templates)]
    p_multi = tmpl_p.format(seq=seq)
    t_multi = tmpl_t.format(seq=seq)
    multi_tmpl_samples.append({
        "id": f"SEQ_{i:04d}",
        "prompt": p_multi,
        "target": t_multi,
        "seq": seq
    })

# Save datasets
with open(DATA_DIR / "factorial_single_template_train.json", "w", encoding="utf-8") as f:
    json.dump(single_tmpl_samples, f, indent=2)

with open(DATA_DIR / "factorial_multi_template_train.json", "w", encoding="utf-8") as f:
    json.dump(multi_tmpl_samples, f, indent=2)

print("Saved factorial_single_template_train.json and factorial_multi_template_train.json")

# Generate 300 validation samples (disjoint from train and benchmark)
val_core = []
while len(val_core) < 300:
    L = random.randint(2, 5)
    pool = random.choice([GREEK, COMMON_NOUNS])
    seq = ' '.join(random.sample(pool, L))
    if seq in bench_targets or seq in core_sequences:
        continue
    val_core.append(seq)

val_samples = []
for i, seq in enumerate(val_core):
    tmpl_p, tmpl_t = multi_templates[i % len(multi_templates)]
    val_samples.append({
        "id": f"VAL_{i:04d}",
        "prompt": tmpl_p.format(seq=seq),
        "target": tmpl_t.format(seq=seq),
        "seq": seq
    })

with open(DATA_DIR / "factorial_val.json", "w", encoding="utf-8") as f:
    json.dump(val_samples, f, indent=2)

print(f"Saved {len(val_samples)} validation samples to factorial_val.json")

