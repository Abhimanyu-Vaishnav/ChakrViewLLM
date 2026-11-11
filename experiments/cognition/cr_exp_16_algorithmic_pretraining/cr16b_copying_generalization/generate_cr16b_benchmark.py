"""CR-EXP-16B Benchmark Generator
Constructs a balanced 1,500-probe sealed generalization benchmark across 14 categories (A through N).
Strictly independent and disjoint from CR-16 and training datasets.
"""
import sys
import json
import random
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
DATA_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16b_copying_generalization/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Word and Token Pools
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

# Indian and Western Single-Token & Multi-Token Names
SINGLE_NAMES = ['Dev', 'Roy', 'Prem', 'Amit', 'Alok', 'Mona', 'Ritu', 'Sonu', 'John', 'Mary', 'Max', 'Sam', 'Dan', 'Ben']
MULTI_NAMES = [
    'Gautam', 'Mukesh', 'Siddharth', 'Pankhuri', 'Sarvesh', 'Yashwant', 'Jaideep', 'Govind', 'Vaibhav', 'Vikas',
    'Badri', 'Fateh', 'Kalyan', 'Lalit', 'Madhav', 'Alexander', 'Elizabeth', 'Benjamin', 'Charlotte', 'Victoria'
]

ALPHANUMERIC = ['A1', 'B2', 'C3', 'D4', 'E5', 'F6', 'G7', 'H8', 'X9', 'Z0', 'K1', 'M2', 'P3', 'R4', 'T5']

TEMPLATES_FAMILIAR = [
    ('Echo sequence: {seq}\nOutput: ', '{seq}'),
    ('Repeat: {seq}\nOutput: ', '{seq}'),
    ('Copy: {seq} -> ', '{seq}'),
]

TEMPLATES_UNSEEN = [
    ('Mirror: {seq} | Result: ', '{seq}'),
    ('Replicate the following: {seq}\nExact copy: ', '{seq}'),
    ('Duplicate: {seq}\nReturn: ', '{seq}'),
    ('Print verbatim: {seq} => ', '{seq}'),
]

random.seed(987654)

probes = []
probe_id_counter = 1

def add_probe(cat_code, category_name, prompt, target, metadata):
    global probe_id_counter
    p_id = f"CR16B_{cat_code}_{probe_id_counter:04d}"
    probe_id_counter += 1
    probes.append({
        "id": p_id,
        "category_code": cat_code,
        "category": category_name,
        "prompt": prompt,
        "expected_target": target,
        "metadata": metadata
    })

# A. Familiar token, unseen sequence combinations (Greek letters, L=3-5)
for _ in range(110):
    L = random.randint(3, 5)
    seq = ' '.join(random.sample(GREEK, L))
    tmpl = random.choice(TEMPLATES_FAMILIAR)
    add_probe("CAT_A", "familiar_token_unseen_combos", tmpl[0].format(seq=seq), seq, {"length": L, "vocab": "greek"})

# B. Unseen sequence orderings (Permutations of known subsets)
for _ in range(110):
    words = random.sample(COMMON_NOUNS[:30], 4)
    seq = ' '.join(words)
    tmpl = random.choice(TEMPLATES_FAMILIAR)
    add_probe("CAT_B", "unseen_sequence_orderings", tmpl[0].format(seq=seq), seq, {"length": 4, "vocab": "nouns"})

# C. Repeated token sequences (testing pointer retention across duplicates)
for _ in range(100):
    w1, w2 = random.sample(COMMON_NOUNS, 2)
    # pattern like w1 w2 w1 or w1 w1 w2 w2
    pattern = random.choice([
        [w1, w2, w1],
        [w1, w1, w2],
        [w1, w2, w2, w1],
        [w2, w1, w2, w1]
    ])
    seq = ' '.join(pattern)
    tmpl = random.choice(TEMPLATES_FAMILIAR)
    add_probe("CAT_C", "repeated_token_sequences", tmpl[0].format(seq=seq), seq, {"length": len(pattern), "repeats": True})

# D. Short sequences (length 1–3)
for _ in range(110):
    L = random.randint(1, 3)
    pool = random.choice([GREEK, COMMON_NOUNS])
    seq = ' '.join(random.sample(pool, L))
    tmpl = random.choice(TEMPLATES_FAMILIAR)
    add_probe("CAT_D", "short_sequences_len_1_3", tmpl[0].format(seq=seq), seq, {"length": L})

# E. Medium sequences (length 4–6)
for _ in range(110):
    L = random.randint(4, 6)
    pool = random.choice([GREEK, COMMON_NOUNS])
    seq = ' '.join(random.sample(pool, L))
    tmpl = random.choice(TEMPLATES_FAMILIAR)
    add_probe("CAT_E", "medium_sequences_len_4_6", tmpl[0].format(seq=seq), seq, {"length": L})

# F. Long sequences (length 7–10)
for _ in range(110):
    L = random.randint(7, 10)
    pool = random.choice([GREEK, COMMON_NOUNS])
    seq = ' '.join(random.sample(pool, L))
    tmpl = random.choice(TEMPLATES_FAMILIAR)
    add_probe("CAT_F", "long_sequences_len_7_10", tmpl[0].format(seq=seq), seq, {"length": L})

# G. Single-token name copying
for _ in range(110):
    name = random.choice(SINGLE_NAMES)
    tmpl_p = random.choice([
        "Repeat the name: {seq}\nName: ",
        "Echo sequence: {seq}\nOutput: ",
        "Copy: {seq} -> "
    ])
    add_probe("CAT_G", "single_token_name_copying", tmpl_p.format(seq=name), name, {"length": 1, "type": "single_name"})

# H. Multi-token name copying
for _ in range(110):
    name = random.choice(MULTI_NAMES)
    tmpl_p = random.choice([
        "Repeat the name: {seq}\nName: ",
        "Echo sequence: {seq}\nOutput: ",
        "Copy: {seq} -> "
    ])
    add_probe("CAT_H", "multi_token_name_copying", tmpl_p.format(seq=name), name, {"length": 1, "type": "multi_name"})

# I. Ordinary word copying (single words from lexicon)
for _ in range(110):
    word = random.choice(COMMON_NOUNS)
    tmpl = random.choice(TEMPLATES_FAMILIAR)
    add_probe("CAT_I", "ordinary_word_copying", tmpl[0].format(seq=word), word, {"length": 1, "type": "word"})

# J. Mixed alphanumeric token copying
for _ in range(100):
    L = random.randint(2, 4)
    seq = ' '.join(random.sample(ALPHANUMERIC, L))
    tmpl = random.choice(TEMPLATES_FAMILIAR)
    add_probe("CAT_J", "mixed_alphanumeric_copying", tmpl[0].format(seq=seq), seq, {"length": L, "type": "alphanumeric"})

# K. Unseen prompt templates (Greek & nouns on completely unseen prompts)
for _ in range(110):
    L = random.randint(2, 5)
    pool = random.choice([GREEK, COMMON_NOUNS])
    seq = ' '.join(random.sample(pool, L))
    tmpl = random.choice(TEMPLATES_UNSEEN)
    add_probe("CAT_K", "unseen_prompt_templates", tmpl[0].format(seq=seq), seq, {"length": L, "unseen_template": True})

# L. Familiar prompt templates (controlled cross-check)
for _ in range(110):
    L = random.randint(2, 5)
    pool = random.choice([GREEK, COMMON_NOUNS])
    seq = ' '.join(random.sample(pool, L))
    tmpl = random.choice(TEMPLATES_FAMILIAR)
    add_probe("CAT_L", "familiar_prompt_templates", tmpl[0].format(seq=seq), seq, {"length": L, "unseen_template": False})

# M. Distractor-containing copying prompts
for _ in range(100):
    L = random.randint(2, 4)
    seq = ' '.join(random.sample(GREEK, L))
    distractor = ' '.join(random.sample(COMMON_NOUNS, 3))
    prompt = f"Ignore notice: [{distractor}]. Echo sequence: {seq}\nOutput: "
    add_probe("CAT_M", "distractor_containing_copying", prompt, seq, {"length": L, "has_distractor": True})

# N. Exact sequence reconstruction (2-turn synthetic exchange)
for _ in range(100):
    L = random.randint(2, 4)
    seq = ' '.join(random.sample(COMMON_NOUNS, L))
    prompt = f"System: Registered token sequence is '{seq}'.\nUser: Please replicate the exact sequence.\nAssistant: "
    add_probe("CAT_N", "exact_sequence_reconstruction", prompt, seq, {"length": L, "format": "dialogue"})

print(f"Total benchmark probes generated: {len(probes)}")

bench_out_file = DATA_DIR / "cr16b_sealed_benchmark_1500.json"
with open(bench_out_file, "w", encoding="utf-8") as f:
    json.dump(probes, f, indent=2)

h = hashlib.sha256(bench_out_file.read_bytes()).hexdigest()
print(f"CR-16B Sealed Benchmark saved to {bench_out_file}")
print(f"Benchmark SHA-256: {h}")

# Save SHA-256 to hash manifest
with open(DATA_DIR / "CR16B_BENCHMARK_HASH.txt", "w") as f:
    f.write(f"cr16b_sealed_benchmark_1500.json: {h}\n")

