"""CR-EXP-16E Training Data Generator:
Generates datasets for the 2x2 Factorial Design across Arms A, B, C, D:
Factor 1: Fact-Position Distribution
  - Standard/Fixed Position (target predominantly at position 1)
  - Shuffled/Uniform Position (target uniformly distributed across positions 0, 1, 2, 3)
Factor 2: Value-Span Curriculum
  - Standard Span (predominantly single-token / common words)
  - Balanced Multi-Token Spans (balanced distribution of length 1, 2, 3, 4 token values)

All arms include:
- Sequence Copying Replay: 1,500 examples (balanced compromise based on Arm D/C findings)
- Key-Value Retrieval: 2,000 examples
- Calibrated Abstention / Missing-Info Examples: 500 examples
Total dataset size: 4,000 examples per arm.
Target-only loss masking with strict anti-leakage against all sealed benchmarks.
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

DATA_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data"

# Load sealed benchmarks to enforce strict anti-leakage
BENCH_FILES = [
    REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/benchmark_data/cr16_synthetic_benchmark_1000.json",
    REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16c_key_value_retrieval/data/cr16c_sealed_benchmark_1500.json",
    REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/data/cr16d_compositional_benchmark_1000.json",
    DATA_DIR / "cr16e_b1_position_invariance_600.json",
    DATA_DIR / "cr16e_b3_multitoken_span_500.json",
    DATA_DIR / "cr16e_b4_abstention_300.json",
]

sealed_prompts = set()
for bf in BENCH_FILES:
    if bf.exists():
        with open(bf) as f:
            items = json.load(f)
            for item in items:
                if "prompt" in item:
                    sealed_prompts.add(item["prompt"].strip())
print(f"Loaded {len(sealed_prompts)} sealed benchmark prompts for strict contamination filtering.")

KEYS = [
    "color", "fruit", "animal", "city", "country", "vehicle", "sport", "flower", "profession", "weather",
    "season", "planet", "element", "metal", "gem", "weapon", "fabric", "beverage", "tree", "continent",
    "instrument", "subject", "currency", "language", "mineral", "device", "author", "deity", "vessel", "river"
]

VALS_L1 = [
    "Red", "Blue", "Green", "Gold", "Apple", "Mango", "Lion", "Bear", "Wolf", "Hawk",
    "Rome", "Delhi", "Cairo", "Spain", "Car", "Bus", "Train", "Plane", "Boat", "Bike",
    "Rose", "Tulip", "Lily", "Chef", "Judge", "Rain", "Snow", "Wind", "Sun", "Iron"
]

VALS_L2 = [
    "Purple", "Yellow", "Silver", "Banana", "Orange", "Tiger", "Eagle", "Shark", "Whale",
    "Paris", "London", "Tokyo", "Berlin", "Madrid", "France", "Japan", "Brazil", "Canada", "India", "Italy",
    "Soccer", "Tennis", "Cricket", "Hockey", "Rugby", "Doctor", "Lawyer", "Pilot", "Nurse", "Artist",
    "Cloud", "Frost", "Copper", "Steel", "Ruby", "Sword", "Coffee", "Juice", "Europe", "Ghar"
]

VALS_L3 = [
    "Strawberry", "Pineapple", "Elephant", "Kangaroo", "Australia", "Singapore", "Amsterdam", "Helicopter",
    "Badminton", "Sunflower", "Scientist", "Architect", "Thunder", "Lightning", "Mercury", "Spring", "Shield"
]

VALS_L4 = [
    "Watermelon", "Hippopotamus", "Copenhagen", "Switzerland", "Gymnastics", "Dandelion", "Psychologist",
    "Automobile", "Motorcycle", "Earthquake", "Diamond", "Cotton"
]

WORDS_FOR_COPYING = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    "uniform", "victor", "whiskey", "xray", "yankee", "zulu", "zero", "one", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "sun", "moon", "star", "tree", "river"
]

def generate_copying_examples(count, rng):
    examples = []
    templates = [
        ("Copy: {s} -> ", "{s}\n"),
        ("Repeat sequence: {s} | Output: ", "{s}\n"),
        ("Echo: {s} => ", "{s}\n"),
        ("[INPUT] {s} [COPY] ", "{s}\n")
    ]
    while len(examples) < count:
        length = rng.randint(3, 8)
        seq_words = rng.sample(WORDS_FOR_COPYING, length)
        seq_str = " ".join(seq_words)
        tmpl_in, tmpl_out = rng.choice(templates)
        prompt = tmpl_in.format(s=seq_str)
        target = tmpl_out.format(s=seq_str)
        if prompt.strip() in sealed_prompts:
            continue
        examples.append({
            "task": "copying",
            "prompt": prompt,
            "target": target,
            "expected": seq_str
        })
    return examples

def generate_retrieval_examples(count, rng, shuffled_positions=False, balanced_spans=False):
    examples = []
    templates = [
        ("Store: {k} = {v}. ", "Query: {k}.\nAnswer: ", "Query: {k}. Context: "),
        ("Set {k} to {v}. ", "What is {k}?\nAnswer: ", "What is {k}? Given: "),
        ("[{k} -> {v}] ", "Lookup [{k}] -> ", "Find [{k}]: ")
    ]
    while len(examples) < count:
        num_facts = rng.choice([2, 3, 4])
        if shuffled_positions:
            target_pos = rng.randint(0, num_facts - 1)
            query_pos = "after" if rng.random() < 0.85 else "before"
        else:
            target_pos = 0 # Fixed position 1
            query_pos = "after"
            
        sample_keys = rng.sample(KEYS, num_facts)
        
        # Span selection
        sample_vals = []
        for i in range(num_facts):
            if balanced_spans:
                pool = rng.choice([VALS_L1, VALS_L2, VALS_L3, VALS_L4])
            else:
                pool = VALS_L1 if rng.random() < 0.7 else VALS_L2
            sample_vals.append(rng.choice(pool))
            
        target_k = sample_keys[target_pos]
        target_v = sample_vals[target_pos]
        
        tmpl = rng.choice(templates)
        facts_str = "".join(tmpl[0].format(k=k, v=v) for k, v in zip(sample_keys, sample_vals))
        if query_pos == "after":
            prompt = facts_str + tmpl[1].format(k=target_k)
        else:
            prompt = tmpl[2].format(k=target_k) + facts_str + "Answer: "
            
        if prompt.strip() in sealed_prompts:
            continue
            
        examples.append({
            "task": "retrieval",
            "prompt": prompt,
            "target": target_v + "\n",
            "expected": target_v,
            "target_pos": target_pos,
            "num_facts": num_facts
        })
    return examples

def generate_abstention_examples(count, rng, shuffled_positions=False):
    examples = []
    while len(examples) < count:
        num_facts = rng.choice([1, 2, 3])
        sample_keys = rng.sample(KEYS, num_facts)
        sample_vals = [rng.choice(VALS_L1 + VALS_L2) for _ in range(num_facts)]
        facts_str = "".join(f"Store: {k} = {v}. " for k, v in zip(sample_keys, sample_vals))
        
        # Key guaranteed absent
        unbound_key = rng.choice([k for k in KEYS if k not in sample_keys])
        prompt = facts_str + f"Query: {unbound_key}.\nAnswer: "
        if prompt.strip() in sealed_prompts:
            continue
            
        examples.append({
            "task": "abstention",
            "prompt": prompt,
            "target": "[UNKNOWN]\n",
            "expected": "[UNKNOWN]"
        })
    return examples

def build_factorial_splits():
    rng = random.Random(42)
    
    # Common copying replay pool (1,500 samples)
    common_copying = generate_copying_examples(1500, rng)
    
    # Arm A: Fixed Position, Standard Span
    print("Generating Arm A (Fixed Pos, Standard Span)...")
    a_retr = generate_retrieval_examples(2000, rng, shuffled_positions=False, balanced_spans=False)
    a_abs = generate_abstention_examples(500, rng, shuffled_positions=False)
    arm_a = common_copying + a_retr + a_abs
    rng.shuffle(arm_a)
    with open(DATA_DIR / "arm_a_fixed_stdspan_train.json", "w") as f: json.dump(arm_a, f, indent=2)
    
    # Arm B: Shuffled Position, Standard Span
    print("Generating Arm B (Shuffled Pos, Standard Span)...")
    b_retr = generate_retrieval_examples(2000, rng, shuffled_positions=True, balanced_spans=False)
    b_abs = generate_abstention_examples(500, rng, shuffled_positions=True)
    arm_b = common_copying + b_retr + b_abs
    rng.shuffle(arm_b)
    with open(DATA_DIR / "arm_b_shuffled_stdspan_train.json", "w") as f: json.dump(arm_b, f, indent=2)
    
    # Arm C: Fixed Position, Balanced Spans
    print("Generating Arm C (Fixed Pos, Balanced Spans)...")
    c_retr = generate_retrieval_examples(2000, rng, shuffled_positions=False, balanced_spans=True)
    c_abs = generate_abstention_examples(500, rng, shuffled_positions=False)
    arm_c = common_copying + c_retr + c_abs
    rng.shuffle(arm_c)
    with open(DATA_DIR / "arm_c_fixed_balancedspan_train.json", "w") as f: json.dump(arm_c, f, indent=2)
    
    # Arm D: Shuffled Position, Balanced Spans (The Full Intervention Arm)
    print("Generating Arm D (Shuffled Pos, Balanced Spans)...")
    d_retr = generate_retrieval_examples(2000, rng, shuffled_positions=True, balanced_spans=True)
    d_abs = generate_abstention_examples(500, rng, shuffled_positions=True)
    arm_d = common_copying + d_retr + d_abs
    rng.shuffle(arm_d)
    with open(DATA_DIR / "arm_d_shuffled_balancedspan_train.json", "w") as f: json.dump(arm_d, f, indent=2)
    
    # Multi-task validation split (600 samples)
    print("Generating multi-task validation set...")
    val_copy = generate_copying_examples(200, rng)
    val_retr = generate_retrieval_examples(300, rng, shuffled_positions=True, balanced_spans=True)
    val_abs = generate_abstention_examples(100, rng, shuffled_positions=True)
    val_set = val_copy + val_retr + val_abs
    rng.shuffle(val_set)
    with open(DATA_DIR / "cr16e_multitask_val.json", "w") as f: json.dump(val_set, f, indent=2)
    
    print("Factorial training and validation splits successfully created.")

if __name__ == "__main__":
    build_factorial_splits()

