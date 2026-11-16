import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import json
import torch
import torch.nn as nn
from scripts.run_m3_evaluations import load_tokenizer_artifacts

tok, _ = load_tokenizer_artifacts(Path('data/experiments/vocab_4096'))

# Inspect collation logic used in train_ucip04.py
def inspect_example(prompt: str, target: str):
    prompt_ids = tok.encode(prompt, add_bos=True, add_eos=False)
    target_ids = tok.encode(target, add_bos=False, add_eos=True)
    
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    
    # In training loop:
    # logits = model(inp[:, :-1])
    # shift_logits = logits.view(-1, vocab_size)
    # shift_targets = tgt[:, 1:].contiguous().view(-1)
    
    inp_seq = input_ids[:-1]
    tgt_seq = labels[1:]
    
    print("=" * 60)
    print(f"PROMPT: {repr(prompt)}")
    print(f"TARGET: {repr(target)}")
    print(f"Prompt tokens ({len(prompt_ids)}): {prompt_ids}")
    print(f"Target tokens ({len(target_ids)}): {target_ids}")
    print(f"Total seq len: {len(input_ids)}")
    
    # Show aligned pairs where loss is computed (where target != -100)
    supervised_pairs = []
    for pos, (in_tok, out_tok) in enumerate(zip(inp_seq, tgt_seq)):
        if out_tok != -100:
            try:
                in_str = tok.decode([in_tok], errors="replace")
            except Exception:
                in_str = f"raw_{in_tok}"
            try:
                out_str = tok.decode([out_tok], errors="replace")
            except Exception:
                out_str = f"raw_{out_tok}"
            supervised_pairs.append((pos, in_tok, in_str, out_tok, out_str))
            
    print(f"Supervised tokens count: {len(supervised_pairs)}")
    print("First 5 supervised transitions (given Input -> predicts Target):")
    for pos, in_tok, in_str, out_tok, out_str in supervised_pairs[:5]:
        print(f"  Pos {pos}: In [{in_tok}] {repr(in_str)} -> Predict [{out_tok}] {repr(out_str)}")
    print("Last 2 supervised transitions:")
    for pos, in_tok, in_str, out_tok, out_str in supervised_pairs[-2:]:
        print(f"  Pos {pos}: In [{in_tok}] {repr(in_str)} -> Predict [{out_tok}] {repr(out_str)}")

samples = [
    ("User: Hello ChakrView.\nAssistant:", " Hello! How can I help you today?"),
    ("User: What does a CPU do?\nAssistant:", " A central processing unit executes software instructions."),
    ("User: नमस्ते!\nAssistant:", " नमस्ते! मैं आपकी क्या सहायता कर सकता हूँ?"),
    ("User: Bhai, ML kya hai?\nAssistant:", " Machine learning data se patterns learn karne ka process hai.")
]

for p, t in samples:
    inspect_example(p, t)
