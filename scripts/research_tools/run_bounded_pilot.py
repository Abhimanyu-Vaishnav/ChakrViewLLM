import sys
import os
import json
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.pointer_generator import ChakrPointerMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

VOCAB_DIR = "data/experiments/vocab_4096"
EXP25_DATA_DIR = "experiments/overnight_runs/exp25_vocabulary_generalization_diagnostic_20261024_232014/data"
OUTPUT_DIR = "experiments/overnight_runs/exp27_forensic_pointer_research_20261025_203725"

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_dataset(tok, split_name):
    path = os.path.join(EXP25_DATA_DIR, f"{split_name}.jsonl")
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    processed = []
    for r in records:
        prompt_ids = tok.encode(r["prompt"])
        target_ids = tok.encode(r["target"])
        # Format: prompt_ids + target_ids + [eos=1]
        seq = prompt_ids + target_ids + [1]
        processed.append({
            "record": r,
            "prompt_len": len(prompt_ids),
            "seq": seq,
            "target_len": len(target_ids) + 1
        })
    return processed

def train_arm(model, processed_train, processed_val, is_pointer=False, steps=300, lr=5e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    n_train = len(processed_train)
    history = []

    start_t = time.time()
    for step in range(1, steps + 1):
        idx = (step - 1) % n_train
        item = processed_train[idx]
        seq = torch.tensor([item["seq"]], dtype=torch.long)
        p_len = item["prompt_len"]

        # Supervised target starts at p_len - 1 (predicting first target token)
        # Sequence layout: seq[:, :-1] -> seq[:, 1:]
        inputs = seq[:, :-1]
        targets = seq[:, 1:].clone()
        targets[:, :p_len - 1] = -100

        optimizer.zero_grad()
        if is_pointer:
            _, _, _, final_probs = model(inputs, prompt_len=p_len)
            # Loss = negative log likelihood of final_probs
            # clamp final_probs to avoid log(0)
            eps = 1e-9
            log_probs = torch.log(final_probs + eps)
            loss = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            logits, _ = model(inputs, return_hidden_states=True)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == steps:
            history.append({
                "step": step,
                "train_loss": loss.item(),
                "elapsed_sec": time.time() - start_t
            })
    return history

def evaluate_model(model, tok, split_name, is_pointer=False):
    model.eval()
    path = os.path.join(EXP25_DATA_DIR, f"{split_name}.jsonl")
    records = [json.loads(line) for line in open(path, encoding="utf-8")]

    em_count = 0
    pla_count = 0
    eos_count = 0
    outputs = []

    with torch.no_grad():
        for r in records:
            # 1. Main prompt prediction
            p_ids = tok.encode(r["prompt"])
            inp = torch.tensor([p_ids], dtype=torch.long)
            p_len = len(p_ids)

            if is_pointer:
                _, _, _, probs = model(inp, prompt_len=p_len)
                next_tok = torch.argmax(probs[:, -1, :], dim=-1).item()
            else:
                out = model(inp)
                logits = out[0] if isinstance(out, tuple) else out
                next_tok = torch.argmax(logits[:, -1, :], dim=-1).item()

            pred_text = tok.decode([next_tok]).strip()
            target_text = r["target"].strip()
            is_em = (pred_text == target_text)
            if is_em:
                em_count += 1
            outputs.append(pred_text)

            # 2. Counterfactual prompt prediction
            cf_ids = tok.encode(r["cf_prompt"])
            cf_inp = torch.tensor([cf_ids], dtype=torch.long)
            cf_len = len(cf_ids)

            if is_pointer:
                _, _, _, cf_probs = model(cf_inp, prompt_len=cf_len)
                cf_tok = torch.argmax(cf_probs[:, -1, :], dim=-1).item()
            else:
                cf_out = model(cf_inp)
                cf_logits = cf_out[0] if isinstance(cf_out, tuple) else cf_out
                cf_tok = torch.argmax(cf_logits[:, -1, :], dim=-1).item()

            cf_pred_text = tok.decode([cf_tok]).strip()
            cf_target_text = r["cf_target"].strip()
            is_cf_em = (cf_pred_text == cf_target_text)

            if is_em and is_cf_em:
                pla_count += 1

    n = len(records)
    return {
        "n_records": n,
        "em": em_count,
        "em_rate": em_count / n,
        "pla": pla_count,
        "pla_rate": pla_count / n,
        "outputs_sample": outputs[:5]
    }

def run_pilot():
    print("=== STARTING EXP27 BOUNDED PILOT ===")
    set_seed(42)
    tok, _ = load_tokenizer_artifacts(VOCAB_DIR)

    processed_train = prepare_dataset(tok, "train")
    processed_val = prepare_dataset(tok, "val")

    cfg = ModelConfig(n_layers=6, d_model=192, vocab_size=4096, tie_embeddings=True)

    # 1. Arm A: Baseline Causal-LM
    print("\n--- Training Arm A: Standard Causal-LM ---")
    set_seed(42)
    model_arm_a = ChakrMicro(cfg)
    hist_a = train_arm(model_arm_a, processed_train, processed_val, is_pointer=False, steps=300)

    eval_a = {}
    for s in ["test_1", "test_2a", "test_2b", "test_2c"]:
        eval_a[s] = evaluate_model(model_arm_a, tok, s, is_pointer=False)
        print(f"Arm A {s}: EM={eval_a[s]['em_rate']:.3f}, PLA={eval_a[s]['pla_rate']:.3f}")

    # 2. Arm B: Pointer-Generator Model
    print("\n--- Training Arm B: ChakrPointerMicro ---")
    set_seed(42)
    model_arm_b = ChakrPointerMicro(cfg)
    hist_b = train_arm(model_arm_b, processed_train, processed_val, is_pointer=True, steps=300)

    eval_b = {}
    for s in ["test_1", "test_2a", "test_2b", "test_2c"]:
        eval_b[s] = evaluate_model(model_arm_b, tok, s, is_pointer=True)
        print(f"Arm B {s}: EM={eval_b[s]['em_rate']:.3f}, PLA={eval_b[s]['pla_rate']:.3f}")

    results = {
        "status": "PILOT_COMPLETE",
        "steps": 300,
        "arm_a_baseline": eval_a,
        "arm_b_pointer": eval_b,
        "conclusions": {
            "test_1_em_comparison": f"Arm A: {eval_a['test_1']['em_rate']:.3f} vs Arm B: {eval_b['test_1']['em_rate']:.3f}",
            "test_2a_em_comparison": f"Arm A: {eval_a['test_2a']['em_rate']:.3f} vs Arm B: {eval_b['test_2a']['em_rate']:.3f}",
            "test_2b_em_comparison": f"Arm A: {eval_a['test_2b']['em_rate']:.3f} vs Arm B: {eval_b['test_2b']['em_rate']:.3f}",
            "test_2c_em_comparison": f"Arm A: {eval_a['test_2c']['em_rate']:.3f} vs Arm B: {eval_b['test_2c']['em_rate']:.3f}"
        }
    }

    out_file = os.path.join(OUTPUT_DIR, "PILOT_EVALUATION_RESULTS.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved Pilot Results to", out_file)

if __name__ == "__main__":
    run_pilot()
