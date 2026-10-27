import os
import sys
import json
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Resolve repository root dynamically relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.pointer_generator import ChakrPointerMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import build_causal_sequence

EXP_DIR = "experiments/overnight_runs/exp28_fair_baseline_vs_pointer_20261025_211555"
DATA_DIR = "experiments/overnight_runs/exp25_vocabulary_generalization_diagnostic_20261024_232014/data"
VOCAB_DIR = "data/experiments/vocab_4096"
CANONICAL_CKPT = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"

TOTAL_STEPS = 300
BATCH_SIZE = 2
GRAD_ACCUM = 2
LR_MAX = 5e-5
LR_MIN = 5e-6
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_tensors(records, tok):
    data = []
    for r in records:
        p_toks = tok.encode(r["prompt"])
        r_toks = tok.encode(r["target"])
        inp, tgt = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100, append_eos=True)
        data.append({
            "record": r,
            "inp": torch.tensor(inp, dtype=torch.long),
            "tgt": torch.tensor(tgt, dtype=torch.long),
            "prompt_len": len(p_toks) + 1  # includes BOS
        })
    return data

def collate_batch(batch_items, pad_id=2, ignore_index=-100):
    max_len = max(len(it["inp"]) for it in batch_items)
    B = len(batch_items)
    padded_x = torch.full((B, max_len), pad_id, dtype=torch.long)
    padded_y = torch.full((B, max_len), ignore_index, dtype=torch.long)
    prompt_lens = [it["prompt_len"] for it in batch_items]

    for i, it in enumerate(batch_items):
        l = len(it["inp"])
        padded_x[i, :l] = it["inp"]
        padded_y[i, :l] = it["tgt"]

    return padded_x, padded_y, prompt_lens

def evaluate_val_loss(model, val_data, is_pointer=False):
    model.eval()
    losses = []
    with torch.no_grad():
        for it in val_data:
            x = it["inp"].unsqueeze(0)
            y = it["tgt"].unsqueeze(0)
            p_len = it["prompt_len"]
            if is_pointer:
                _, _, _, probs = model(x, prompt_len=p_len)
                log_p = torch.log(probs + 1e-9)
                loss = F.nll_loss(log_p.view(-1, 4096), y.view(-1), ignore_index=-100)
            else:
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits.view(-1, 4096), y.view(-1), ignore_index=-100)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))

def train_arm(arm_name, model, train_data, val_data, is_pointer=False):
    print(f"\n==========================================")
    print(f"TRAINING {arm_name} (Canonical Pretrained Init)")
    print(f"==========================================")
    set_seed(SEED)
    rng = np.random.RandomState(SEED)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.01)

    init_val = evaluate_val_loss(model, val_data, is_pointer=is_pointer)
    print(f"Step 0: Initial Validation Loss = {init_val:.4f}")

    train_logs = []
    best_val_loss = float("inf")
    best_model_weights = None
    best_step = 0
    t0 = time.perf_counter()

    optimizer.zero_grad()
    for step in range(1, TOTAL_STEPS + 1):
        decay_ratio = (step - 1) / max(1, TOTAL_STEPS)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        cur_lr = LR_MIN + coeff * (LR_MAX - LR_MIN)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            batch_idxs = rng.choice(len(train_data), size=BATCH_SIZE, replace=False)
            batch_items = [train_data[i] for i in batch_idxs]
            padded_x, padded_y, p_lens = collate_batch(batch_items)

            if is_pointer:
                # Use max prompt len across batch for pointer attention slicing
                max_p_len = max(p_lens)
                _, _, _, probs = model(padded_x, prompt_len=max_p_len)
                log_p = torch.log(probs + 1e-9)
                loss = F.nll_loss(log_p.view(-1, 4096), padded_y.view(-1), ignore_index=-100)
            else:
                out = model(padded_x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits.view(-1, 4096), padded_y.view(-1), ignore_index=-100)

            loss = loss / GRAD_ACCUM
            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 25 == 0 or step == TOTAL_STEPS:
            val_loss = evaluate_val_loss(model, val_data, is_pointer=is_pointer)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_step = step
                best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            elapsed = time.perf_counter() - t0
            print(f"Step {step:3d}/{TOTAL_STEPS}: train_loss={step_loss:.4f}, val_loss={val_loss:.4f}, lr={cur_lr:.2e}, elapsed={elapsed:.1f}s {'*' if is_best else ''}")
            train_logs.append({
                "step": step,
                "train_loss": step_loss,
                "val_loss": val_loss,
                "lr": cur_lr,
                "elapsed_sec": elapsed
            })

    # Save training log
    log_file = f"{EXP_DIR}/TRAINING_LOGS/{arm_name}_TRAINING_LOG.jsonl"
    with open(log_file, "w") as f:
        for entry in train_logs:
            f.write(json.dumps(entry) + "\n")

    # Load best checkpoint
    model.load_state_dict(best_model_weights)
    print(f"Loaded best checkpoint from Step {best_step} (val_loss = {best_val_loss:.4f})")

    # Save best checkpoint
    ckpt_path = f"{EXP_DIR}/checkpoints/{arm_name}_best.pt"
    torch.save({
        "step": best_step,
        "val_loss": best_val_loss,
        "model_state_dict": model.state_dict()
    }, ckpt_path)

    return model, best_step, best_val_loss

def evaluate_arm_on_dataset(arm_name, model, tok, dataset_name, records, is_pointer=False):
    model.eval()
    predictions = []
    em_count = 0
    pla_count = 0
    eos_count = 0

    with torch.no_grad():
        for r in records:
            p_toks = tok.encode(r["prompt"])
            target = r["target"].strip()

            # Predict main
            x_main = torch.tensor([[0] + p_toks], dtype=torch.long)
            p_len_main = len(p_toks) + 1
            if is_pointer:
                _, _, p_gen, probs = model(x_main, prompt_len=p_len_main)
                tok_id = torch.argmax(probs[0, -1, :]).item()
                p_gen_val = float(p_gen[0, -1, 0].item())
            else:
                out = model(x_main)
                logits = out[0] if isinstance(out, tuple) else out
                tok_id = torch.argmax(logits[0, -1, :]).item()
                p_gen_val = 1.0

            pred_text = tok.decode([tok_id]).strip()
            is_em = (pred_text == target)
            if is_em:
                em_count += 1
            if tok_id == 1:
                eos_count += 1

            # Predict counterfactual if present
            cf_pred_text = ""
            is_cf_em = False
            if "cf_prompt" in r and "cf_target" in r:
                cf_toks = tok.encode(r["cf_prompt"])
                cf_target = r["cf_target"].strip()
                x_cf = torch.tensor([[0] + cf_toks], dtype=torch.long)
                p_len_cf = len(cf_toks) + 1
                if is_pointer:
                    _, _, _, cf_probs = model(x_cf, prompt_len=p_len_cf)
                    cf_tok_id = torch.argmax(cf_probs[0, -1, :]).item()
                else:
                    out = model(x_cf)
                    logits = out[0] if isinstance(out, tuple) else out
                    cf_tok_id = torch.argmax(logits[0, -1, :]).item()
                cf_pred_text = tok.decode([cf_tok_id]).strip()
                is_cf_em = (cf_pred_text == cf_target)
                if is_em and is_cf_em:
                    pla_count += 1

            predictions.append({
                "split": dataset_name,
                "prompt": r["prompt"],
                "target": target,
                "predicted_tok_id": tok_id,
                "predicted_text": pred_text,
                "is_em": is_em,
                "cf_target": r.get("cf_target", ""),
                "cf_predicted_text": cf_pred_text,
                "is_cf_em": is_cf_em,
                "pair_success": (is_em and is_cf_em),
                "p_gen": p_gen_val
            })

    n = len(records)
    res = {
        "n_records": n,
        "em": em_count,
        "em_rate": em_count / n if n > 0 else 0.0,
        "pla": pla_count,
        "pla_rate": pla_count / n if n > 0 else 0.0,
        "eos_rate": eos_count / n if n > 0 else 0.0,
        "sample_preds": [p["predicted_text"] for p in predictions[:5]]
    }

    # Save predictions
    pred_path = f"{EXP_DIR}/RAW_PREDICTIONS/{arm_name}_{dataset_name}_PREDICTIONS.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    return res

def run_experiment():
    print("=== STARTING EXP28 FAIR COMPARISON EXPERIMENT ===")
    tok, _ = load_tokenizer_artifacts(VOCAB_DIR)

    # Load Datasets
    train_recs = [json.loads(l) for l in open(f"{DATA_DIR}/train.jsonl")]
    val_recs = [json.loads(l) for l in open(f"{DATA_DIR}/val.jsonl")]
    test_1 = [json.loads(l) for l in open(f"{DATA_DIR}/test_1.jsonl")]
    test_2a = [json.loads(l) for l in open(f"{DATA_DIR}/test_2a.jsonl")]
    test_2b = [json.loads(l) for l in open(f"{DATA_DIR}/test_2b.jsonl")]
    test_2c = [json.loads(l) for l in open(f"{DATA_DIR}/test_2c.jsonl")]

    # Distractor position condition: prepend "[Z]" distractor
    distractor_recs = []
    for r in test_1:
        dist_prompt = "Note: Prior signal [Z] archived. " + r["prompt"]
        distractor_recs.append({
            "prompt": dist_prompt,
            "target": r["target"],
            "cf_prompt": "Note: Prior signal [Z] archived. " + r["cf_prompt"],
            "cf_target": r["cf_target"]
        })

    # Blank context condition: context without target
    blank_recs = [
        {"prompt": "Context: Stream alpha: Signal active.\nQuestion: Repeat the data token:\nAnswer: ", "target": "NONE"},
        {"prompt": "Context: Stream beta: System ready.\nQuestion: Copy the value token:\nAnswer: ", "target": "NONE"},
        {"prompt": "Context: Stream gamma: Buffer set.\nQuestion: Echo the input token:\nAnswer: ", "target": "NONE"},
        {"prompt": "Context: Stream delta: Stream open.\nQuestion: Reproduce the symbol token:\nAnswer: ", "target": "NONE"}
    ]

    train_tensors = prepare_tensors(train_recs, tok)
    val_tensors = prepare_tensors(val_recs, tok)

    ckpt_data = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)

    # ARM A: Canonical ChakrMicro initialized from Step 10,000
    model_a = ChakrMicro(ModelConfig())
    model_a.load_state_dict(ckpt_data["model_state_dict"], strict=True)
    model_a, best_step_a, best_val_a = train_arm("ARM_A_CANONICAL_LM", model_a, train_tensors, val_tensors, is_pointer=False)

    # ARM B: ChakrPointerMicro with backbone initialized from Step 10,000
    set_seed(SEED)
    model_b = ChakrPointerMicro(ModelConfig())
    model_b.core.load_state_dict(ckpt_data["model_state_dict"], strict=True)
    model_b, best_step_b, best_val_b = train_arm("ARM_B_POINTER_MICRO", model_b, train_tensors, val_tensors, is_pointer=True)

    # EVALUATION ACROSS ALL 7 CONDITIONS
    eval_splits = [
        ("test_1", test_1),
        ("test_2a", test_2a),
        ("test_2b", test_2b),
        ("test_2c", test_2c),
        ("distractor_position", distractor_recs),
        ("blank_context", blank_recs)
    ]

    results = {
        "status": "VALIDATED_FAIR_COMPARISON",
        "arm_a_canonical_sft": {
            "best_step": best_step_a,
            "best_val_loss": best_val_a,
            "splits": {}
        },
        "arm_b_pointer_generator": {
            "best_step": best_step_b,
            "best_val_loss": best_val_b,
            "splits": {}
        }
    }

    for split_name, recs in eval_splits:
        res_a = evaluate_arm_on_dataset("ARM_A_CANONICAL_LM", model_a, tok, split_name, recs, is_pointer=False)
        res_b = evaluate_arm_on_dataset("ARM_B_POINTER_MICRO", model_b, tok, split_name, recs, is_pointer=True)
        results["arm_a_canonical_sft"]["splits"][split_name] = res_a
        results["arm_b_pointer_generator"]["splits"][split_name] = res_b
        print(f"\nResults for {split_name}:")
        print(f"  Arm A (Canonical LM): EM={res_a['em_rate']:.3f} ({res_a['em']}/{res_a['n_records']}), PLA={res_a['pla_rate']:.3f}")
        print(f"  Arm B (Pointer Micro): EM={res_b['em_rate']:.3f} ({res_b['em']}/{res_b['n_records']}), PLA={res_b['pla_rate']:.3f}")

    out_path = f"{EXP_DIR}/METRICS_RECOMPUTED.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved all recomputed evaluation results to", out_path)

if __name__ == "__main__":
    run_experiment()
