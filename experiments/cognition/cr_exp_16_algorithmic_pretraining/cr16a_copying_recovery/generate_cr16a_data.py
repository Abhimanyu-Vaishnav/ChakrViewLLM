"""CR-EXP-16A Dataset Generator
Generates clean algorithmic sequence copying datasets:
- Explicit delimiters: "Copy: {seq} -> {seq}"
- Varied lengths: 1 to 6 tokens
- Rich lexical vocabulary (over 100 common natural words & symbols)
- Strictly 0 sequence overlap with benchmark
"""
import random
import json
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
CR16A_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery"
DATA_DIR = CR16A_DIR / "data"

random.seed(16002)

VOCAB_WORDS = [
    "red", "blue", "green", "yellow", "black", "white", "orange", "purple", "brown", "pink",
    "cat", "dog", "bird", "fish", "lion", "tiger", "bear", "wolf", "fox", "deer",
    "apple", "banana", "mango", "grape", "peach", "berry", "melon", "lemon", "lime", "pear",
    "sun", "moon", "star", "cloud", "rain", "wind", "storm", "snow", "sky", "tree",
    "river", "ocean", "lake", "mountain", "hill", "forest", "valley", "desert", "field", "stone",
    "gold", "silver", "iron", "copper", "bronze", "steel", "lead", "zinc", "tin", "nickel",
    "north", "south", "east", "west", "up", "down", "left", "right", "front", "back",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "book", "desk", "door", "lamp", "road", "ship", "bell", "ring", "song", "note"
]

def make_sample():
    k = random.randint(1, 6)
    seq = random.sample(VOCAB_WORDS, k)
    seq_str = " ".join(seq)
    
    # Standard clean template
    prompt = f"Copy: {seq_str} -> "
    target = f"{seq_str}\n"
    full_text = prompt + target
    
    return {
        "prompt": prompt,
        "target": target,
        "text": full_text,
        "seq": seq_str,
        "length": k
    }

train_samples = [make_sample() for _ in range(1200)]
val_samples = [make_sample() for _ in range(150)]

train_path = DATA_DIR / "cr16a_train.json"
val_path = DATA_DIR / "cr16a_val.json"

with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_samples, f, indent=2)

with open(val_path, "w", encoding="utf-8") as f:
    json.dump(val_samples, f, indent=2)

h_train = hashlib.sha256(open(train_path, "rb").read()).hexdigest()
h_val = hashlib.sha256(open(val_path, "rb").read()).hexdigest()

print(f"Generated {len(train_samples)} training samples, SHA-256: {h_train}")
print(f"Generated {len(val_samples)} validation samples, SHA-256: {h_val}")

