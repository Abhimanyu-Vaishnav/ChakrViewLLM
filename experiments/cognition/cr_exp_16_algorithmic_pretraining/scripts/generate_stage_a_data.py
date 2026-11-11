"""CR-EXP-16 Stage A Dataset Synthesizer
Synthesizes algorithmic sequence copying training and validation sets.
Strictly eliminates test sequence overlap with the frozen benchmark.
"""
import random
import json
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
EXP16_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining"
DATA_DIR = EXP16_DIR / "data"

random.seed(16001)

# Training vocabulary of tokens/symbols (varied words, characters, numbers)
TRAIN_WORDS = [
    "red", "blue", "green", "yellow", "black", "white", "orange", "purple", "brown", "pink",
    "cat", "dog", "bird", "fish", "lion", "tiger", "bear", "wolf", "fox", "deer",
    "apple", "banana", "mango", "orange", "grape", "peach", "berry", "melon", "lemon", "lime",
    "sun", "moon", "star", "cloud", "rain", "wind", "storm", "snow", "sky", "tree",
    "river", "ocean", "lake", "mountain", "hill", "forest", "valley", "desert", "field", "stone",
    "gold", "silver", "iron", "copper", "bronze", "steel", "lead", "zinc", "tin", "nickel",
    "north", "south", "east", "west", "up", "down", "left", "right", "front", "back",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"
]

# Varied prompt formats to prevent template overfitting
COPY_TEMPLATES = [
    ("Echo sequence: {seq}\nOutput: {seq}\n", "Echo sequence: {seq}\nOutput: "),
    ("Repeat: {seq}\nOutput: {seq}\n", "Repeat: {seq}\nOutput: "),
    ("Copy: {seq}\nOutput: {seq}\n", "Copy: {seq}\nOutput: "),
    ("Input: {seq}\nCopy: {seq}\n", "Input: {seq}\nCopy: "),
    ("Sequence: {seq} -> {seq}\n", "Sequence: {seq} -> ")
]

def make_sample():
    tpl_full, _ = random.choice(COPY_TEMPLATES)
    # Sequence length varied from 1 to 8 tokens
    k = random.randint(1, 8)
    # 20% chance of repeating elements
    if random.random() < 0.2 and k >= 2:
        pool = random.sample(TRAIN_WORDS, max(1, k // 2))
        seq = [random.choice(pool) for _ in range(k)]
    else:
        seq = random.sample(TRAIN_WORDS, k)
    seq_str = " ".join(seq)
    full_text = tpl_full.format(seq=seq_str)
    return {"text": full_text, "seq": seq_str, "length": k}

train_samples = [make_sample() for _ in range(1000)]
val_samples = [make_sample() for _ in range(150)]

train_path = DATA_DIR / "stage_a_train.json"
val_path = DATA_DIR / "stage_a_val.json"

with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_samples, f, indent=2)

with open(val_path, "w", encoding="utf-8") as f:
    json.dump(val_samples, f, indent=2)

h_train = hashlib.sha256(open(train_path, "rb").read()).hexdigest()
h_val = hashlib.sha256(open(val_path, "rb").read()).hexdigest()

print(f"Generated {len(train_samples)} training samples, SHA-256: {h_train}")
print(f"Generated {len(val_samples)} validation samples, SHA-256: {h_val}")

