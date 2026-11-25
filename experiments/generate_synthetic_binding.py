"""
NR-015 Phase 3: Controlled Synthetic Task Generator & Evaluator
Generates diverse synthetic training samples with randomized variables and values,
then trains a controlled 17M pilot model to investigate whether variable binding
and rule induction generalize to held-out unseen variable names and values.
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path

def generate_synthetic_binding_corpus(seed=101, train_count=1200, val_count=200):
    random.seed(seed)
    
    # Pool of variable names: strictly partition train and val variable names
    train_vars = ['x', 'y', 'z', 'm', 'n', 'p', 'q', 'val1', 'val2', 'num_a', 'num_b']
    val_vars = ['delta', 'gamma', 'omega', 'sigma', 'theta', 'phi', 'tau', 'kappa']
    
    def make_sample(v1, v2, low=2, high=50, is_val=False):
        a = random.randint(low, high)
        b = random.randint(low, high)
        op_type = random.choice(['add', 'sub', 'mul', 'area', 'perimeter'])
        
        if op_type == 'add':
            ans = a + b
            text = f"<user>\nLet {v1} = {a} and {v2} = {b}. What is {v1} + {v2}?\n<assistant>\n{ans}"
        elif op_type == 'sub':
            ans = a - b
            text = f"<user>\nLet {v1} = {a} and {v2} = {b}. What is {v1} - {v2}?\n<assistant>\n{ans}"
        elif op_type == 'mul':
            a_s = random.randint(2, 12)
            b_s = random.randint(2, 12)
            ans = a_s * b_s
            text = f"<user>\nLet {v1} = {a_s} and {v2} = {b_s}. What is {v1} * {v2}?\n<assistant>\n{ans}"
        elif op_type == 'area':
            ans = a * b
            text = f"<user>\nFunction area({v1}, {v2}) = {v1} * {v2}. Evaluate area({a}, {b}).\n<assistant>\n{ans}"
        else: # perimeter
            ans = 2 * (a + b)
            text = f"<user>\nFunction perimeter({v1}, {v2}) = 2 * ({v1} + {v2}). Evaluate perimeter({a}, {b}).\n<assistant>\n{ans}"
            
        return {"text": text, "var1": v1, "var2": v2, "op": op_type, "answer": str(ans)}

    train_data = []
    for _ in range(train_count):
        v1, v2 = random.sample(train_vars, 2)
        train_data.append(make_sample(v1, v2, low=2, high=60, is_val=False))
        
    val_data = []
    for _ in range(val_count):
        # Val data uses strictly held-out variable names and unseen larger values (61-120)
        v1, v2 = random.sample(val_vars, 2)
        val_data.append(make_sample(v1, v2, low=61, high=120, is_val=True))
        
    out_dir = Path("data/synthetic_binding_015")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = out_dir / "train_binding.json"
    val_path = out_dir / "val_binding.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2)
        
    def hash_file(p):
        with open(p, 'rb') as fp:
            return hashlib.sha256(fp.read()).hexdigest()
            
    print(f"Train samples: {len(train_data)} | SHA-256: {hash_file(train_path)}")
    print(f"Val samples (OOD vars & values): {len(val_data)} | SHA-256: {hash_file(val_path)}")
    return train_path, val_path

if __name__ == '__main__':
    generate_synthetic_binding_corpus()
