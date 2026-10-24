"""
experiments/overnight_runs/mission_10h_exp26_15m_readiness_20261024_232621/run_exp26_scaling_diagnostic.py

ChakrView Experiment 26 — 15M Scaling & Representation Bottleneck Diagnostic.

Controlled Scientific Study:
- Evaluates whether increasing ChakrMicro parameter capacity from 3.44M to 15.7M
  resolves held-out single-token copying failure, or whether the output unembedding
  matrix remains an associative lookup bottleneck.
- Uses exact frozen Exp25 datasets (Train, Val, Test 1, Test 2A, Test 2B, Test 2C).
- Identical training protocol: 300 steps, lr=5e-5, cosine schedule, CPU execution.
- Evaluates top-5 output logits for held-out tokens to test Hypothesis H1 (unembedding bottleneck).
- Preserves all canonical assets, checkpoints, and protected files.
"""

import os
import sys
import json
import time
import math
import hashlib
import psutil
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path("D:/Project/ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.config_15m import ChakrMicro15MConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import build_causal_sequence

def get_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def wilson_ci(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.95996
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    spread = (z * math.sqrt((p * (1 - p) / n) + ((z**2) / (4 * n**2)))) / denom
    return p, max(0.0, center - spread), min(1.0, center + spread)

def main():
    print("=" * 70)
    print("CHAKRVIEW EXP26: 15M SCALING & REPRESENTATION DIAGNOSTIC")
    print("=" * 70)

    MISSION_DIR = REPO_ROOT / "experiments/overnight_runs/mission_10h_exp26_15m_readiness_20261024_232621"
    EXP26_DIR = MISSION_DIR / "exp26_15m_scaling_run"
    EXP26_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR = EXP26_DIR / "checkpoints"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    VOCAB_DIR = REPO_ROOT / "data/experiments/vocab_4096"
    EXP25_DATA_DIR = REPO_ROOT / "experiments/overnight_runs/exp25_vocabulary_generalization_diagnostic_20261024_232014/data"

    # Preflight Check on Data Hashes
    expected_data_hashes = {
        "train": "59fdd75469704bac2c5a39469a9c28383868d494a92fc00ea6a2e8e5456dbb3d",
        "val": "bb6c85b57266fa4f3ceeb4e582419b051f05dd6d15202477e8629b60a4fa8920",
        "test_1": "41c5c7157598769b4e51c5dcffbe21f3c0709839be102458787e8643402d9b1f",
        "test_2a": "d58b420660ea5827f7ceec358157e8b5a94c44e829dca50e7d33baea8c7322e8",
        "test_2b": "e404219cb850896ded417a464c5db95d5215b35008b497a9926cb1823b6cc747",
        "test_2c": "013ad9810b11bad8f13b4c02776283a364df4ccc30ea2de59f64578bea6fafc3"
    }

    for split_name, exp_h in expected_data_hashes.items():
        p = EXP25_DATA_DIR / f"{split_name}.jsonl"
        act_h = get_file_hash(p)
        assert act_h == exp_h, f"CRITICAL: {split_name} hash mismatch! Exp={exp_h}, Act={act_h}"
        print(f"Verified frozen {split_name}.jsonl: {act_h}")

    # Load Tokenizer
    tok, _ = load_tokenizer_artifacts(str(VOCAB_DIR))
    eos_id = 1

    # Load Records from Frozen Exp25 Splits
    def load_jsonl(path):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line.strip()))
        return records

    train_records = load_jsonl(EXP25_DATA_DIR / "train.jsonl")
    val_records = load_jsonl(EXP25_DATA_DIR / "val.jsonl")
    test_1_records = load_jsonl(EXP25_DATA_DIR / "test_1.jsonl")
    test_2a_records = load_jsonl(EXP25_DATA_DIR / "test_2a.jsonl")
    test_2b_records = load_jsonl(EXP25_DATA_DIR / "test_2b.jsonl")
    test_2c_records = load_jsonl(EXP25_DATA_DIR / "test_2c.jsonl")

    sft_tokens = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
        "0", "1", "2", "3", "4", "5", "6", "7",
        "@", "#", "$", "%", "&", "*", "+", "="
    ]

    BLANK_TEMPLATES = [
        "Context: Stream omega: Signal active.\nQuestion: Repeat the data token:\nAnswer: ",
        "Context: Stream omega: System ready.\nQuestion: Copy the value token:\nAnswer: ",
        "Context: Stream omega: Buffer set.\nQuestion: Echo the input token:\nAnswer: ",
        "Context: Stream omega: Stream open.\nQuestion: Reproduce the symbol token:\nAnswer: "
    ]

    # Instantiate ChakrMicro-15M
    torch.manual_seed(42)
    cfg15m = ChakrMicro15MConfig()
    model15m = ChakrMicro(cfg15m)
    trainable_params = sum(p.numel() for p in model15m.parameters() if p.requires_grad)
    print(f"\nInitialized ChakrMicro-15M with {trainable_params:,} trainable parameters.")
    assert trainable_params == 15735168

    # SFT Training Setup
    model15m.train()
    optimizer = optim.AdamW(model15m.parameters(), lr=5e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def prepare_dataset_tensors(records):
        data = []
        for r in records:
            p_toks = tok.encode(r["prompt"])
            r_toks = tok.encode(r["target"])
            inp, tgt = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=eos_id, ignore_index=-100, append_eos=True)
            data.append((torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)))
        return data

    train_tensors = prepare_dataset_tensors(train_records)
    val_tensors = prepare_dataset_tensors(val_records)

    TOTAL_STEPS = 300
    BATCH_SIZE = 2
    GRAD_ACCUM = 2
    LR_MAX = 5e-5
    LR_MIN = 5e-6
    SEED = 42

    rng = np.random.RandomState(SEED)
    torch.manual_seed(SEED)

    def evaluate_val_loss(model):
        model.eval()
        losses = []
        with torch.no_grad():
            for x, y in val_tensors:
                x_b = x.unsqueeze(0)
                y_b = y.unsqueeze(0)
                logits = model(x_b)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_b.reshape(-1))
                losses.append(loss.item())
        model.train()
        return float(np.mean(losses))

    init_val = evaluate_val_loss(model15m)
    print(f"Step 0: Initial Validation Loss = {init_val:.4f}")

    train_log = []
    best_val_loss = float("inf")
    best_step = 0
    t0_train = time.perf_counter()

    optimizer.zero_grad()
    for step in range(1, TOTAL_STEPS + 1):
        elapsed = time.perf_counter() - t0_train
        if elapsed > 300.0:
            raise RuntimeError(f"ABORT: Wall-clock timeout ({elapsed:.1f}s > 300s)")
        ram_gb = psutil.Process().memory_info().rss / (1024**3)
        if ram_gb > 6.0:
            raise RuntimeError(f"ABORT: Peak RAM exceeded ({ram_gb:.2f} GB > 6.0 GB)")

        decay_ratio = (step - 1) / max(1, TOTAL_STEPS)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        cur_lr = LR_MIN + coeff * (LR_MAX - LR_MIN)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            batch_indices = rng.choice(len(train_tensors), size=BATCH_SIZE, replace=False)
            batch_inps = [train_tensors[idx][0] for idx in batch_indices]
            batch_tgts = [train_tensors[idx][1] for idx in batch_indices]

            max_len = max(len(b) for b in batch_inps)
            padded_x = torch.full((BATCH_SIZE, max_len), 2, dtype=torch.long)
            padded_y = torch.full((BATCH_SIZE, max_len), -100, dtype=torch.long)

            for i in range(BATCH_SIZE):
                padded_x[i, :len(batch_inps[i])] = batch_inps[i]
                padded_y[i, :len(batch_tgts[i])] = batch_tgts[i]

            logits = model15m(padded_x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), padded_y.reshape(-1))
            loss = loss / GRAD_ACCUM
            loss.backward()
            step_loss += loss.item()

        nn.utils.clip_grad_norm_(model15m.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 25 == 0 or step == TOTAL_STEPS:
            val_loss = evaluate_val_loss(model15m)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_step = step
                torch.save(
                    {"step": step, "model_state_dict": model15m.state_dict(), "val_loss": val_loss, "config": cfg15m},
                    CKPT_DIR / "checkpoint_best.pt"
                )

            log_entry = {
                "step": step,
                "train_loss": step_loss,
                "val_loss": val_loss,
                "lr": cur_lr,
                "elapsed_sec": time.perf_counter() - t0_train,
                "ram_gb": psutil.Process().memory_info().rss / (1024**3)
            }
            train_log.append(log_entry)
            print(f"Step {step:03d} | Train: {step_loss:.4f} | Val: {val_loss:.4f} | Best Val: {best_val_loss:.4f} (Step {best_step}) | Time: {log_entry['elapsed_sec']:.1f}s | RAM: {log_entry['ram_gb']:.2f}GB")

    torch.save(
        {"step": TOTAL_STEPS, "model_state_dict": model15m.state_dict(), "val_loss": val_loss, "config": cfg15m},
        CKPT_DIR / "checkpoint_step_300.pt"
    )

    t_train_total = time.perf_counter() - t0_train
    print(f"\nTraining completed in {t_train_total:.2f}s! Best Step: {best_step} (Val Loss: {best_val_loss:.4f})")

    # -------------------------------------------------------------
    # EVALUATION SUITE
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("RUNNING LOCKED EVALUATION ON BEST 15M CHECKPOINT")
    print("=" * 50)

    best_ckpt = torch.load(CKPT_DIR / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    eval_model = ChakrMicro(best_ckpt["config"])
    eval_model.load_state_dict(best_ckpt["model_state_dict"])
    eval_model.eval()

    def generate_greedy(model, prompt_tokens, max_new_tokens=4):
        x = torch.tensor([prompt_tokens], dtype=torch.long)
        generated = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(x)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                generated.append(next_token)
                if next_token == eos_id:
                    break
                x = torch.cat([x, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
        return generated

    # Blank Context Probes
    blank_emissions = []
    for b_tmpl in BLANK_TEMPLATES:
        b_toks = tok.encode(b_tmpl)
        gen = generate_greedy(eval_model, b_toks, max_new_tokens=4)
        out_str = tok.decode(gen).replace("<|eos|>", "").strip()
        blank_emissions.append(out_str)

    all_split_results = {}
    all_predictions_to_save = []
    top5_logits_diagnostics = []

    splits_to_eval = [
        ("test_1", test_1_records),
        ("test_2a", test_2a_records),
        ("test_2b", test_2b_records),
        ("test_2c", test_2c_records)
    ]

    for split_name, dataset in splits_to_eval:
        exact_matches = 0
        pair_successes = 0
        eos_count = 0
        multi_token_count = 0
        blank_match_count = 0
        emitted_tokens = []
        latencies = []

        unique_tokens = sorted(list(set(r["target"] for r in dataset)))
        token_template_success = {t: [False] * 4 for t in unique_tokens}

        for record in dataset:
            t_start = time.perf_counter()
            p_toks = tok.encode(record["prompt"])
            target = record["target"]

            # Capture logit diagnostic for final prompt token
            with torch.no_grad():
                x_probe = torch.tensor([p_toks], dtype=torch.long)
                step_logits = eval_model(x_probe)[:, -1, :] # [1, 4096]
                topk_vals, topk_inds = torch.topk(step_logits, 5, dim=-1)
                top5_tokens = [tok.decode([int(idx)]) for idx in topk_inds[0]]
                
                target_tok_id = tok.encode(target)[0]
                target_rank = int((step_logits > step_logits[0, target_tok_id]).sum().item()) + 1
                
                if split_name in ["test_2a", "test_2c"] and len(top5_logits_diagnostics) < 10:
                    top5_logits_diagnostics.append({
                        "split": split_name,
                        "prompt": record["prompt"][:60] + "...",
                        "target": target,
                        "target_rank": target_rank,
                        "top5_predicted": top5_tokens
                    })

            # 1. Main prompt evaluation
            gen_ids = generate_greedy(eval_model, p_toks, max_new_tokens=4)
            lat = (time.perf_counter() - t_start) * 1000
            latencies.append(lat)

            first_tok_id = gen_ids[0] if len(gen_ids) > 0 else None
            first_str = tok.decode([first_tok_id]) if first_tok_id is not None else ""
            emitted_tokens.append(first_str)

            if len(gen_ids) > 0 and gen_ids[-1] == eos_id:
                eos_count += 1
            if len(gen_ids) > 2:
                multi_token_count += 1

            if first_str in blank_emissions:
                blank_match_count += 1

            is_em = (first_str == target) and (len(gen_ids) >= 2 and gen_ids[1] == eos_id)
            if is_em:
                exact_matches += 1
                token_template_success[target][record["template_idx"]] = True

            # 2. Counterfactual prompt evaluation
            cf_toks = tok.encode(record["cf_prompt"])
            cf_target = record["cf_target"]
            cf_gen_ids = generate_greedy(eval_model, cf_toks, max_new_tokens=4)
            cf_first_id = cf_gen_ids[0] if len(cf_gen_ids) > 0 else None
            cf_first_str = tok.decode([cf_first_id]) if cf_first_id is not None else ""

            cf_is_em = (cf_first_str == cf_target) and (len(cf_gen_ids) >= 2 and cf_gen_ids[1] == eos_id)
            is_pair_succ = (is_em and cf_is_em)
            if is_pair_succ:
                pair_successes += 1

            pred_entry = {
                "split": split_name,
                "pair_id": record["pair_id"],
                "prompt": record["prompt"],
                "target": target,
                "generated_ids": gen_ids,
                "generated_text": first_str,
                "is_exact_match": is_em,
                "cf_prompt": record["cf_prompt"],
                "cf_target": cf_target,
                "cf_generated_ids": cf_gen_ids,
                "cf_generated_text": cf_first_str,
                "cf_exact_match": cf_is_em,
                "pair_success": is_pair_succ,
                "latency_ms": lat
            }
            all_predictions_to_save.append(pred_entry)

        n_records = len(dataset)
        n_tokens = len(unique_tokens)
        modal_tok, modal_cnt = Counter(emitted_tokens).most_common(1)[0]
        modal_share = modal_cnt / n_records

        per_token_accuracy = {}
        k_full = 0
        k_any = 0
        for t in unique_tokens:
            if split_name == "test_1":
                acc_t = float(token_template_success[t][(sft_tokens.index(t) + 2) % 4])
            else:
                acc_t = sum(token_template_success[t]) / 4.0
            per_token_accuracy[t] = acc_t
            if acc_t == 1.0:
                k_full += 1
            if acc_t >= 0.50:
                k_any += 1

        acc_macro = sum(per_token_accuracy.values()) / n_tokens
        tcs_score = (k_full / k_any) if k_any > 0 else 0.0

        em_p, em_low, em_high = wilson_ci(exact_matches, n_records)
        ps_p, ps_low, ps_high = wilson_ci(pair_successes, n_records)

        all_split_results[split_name] = {
            "n_records": n_records,
            "n_tokens": n_tokens,
            "exact_matches": exact_matches,
            "em_micro": em_p,
            "em_ci95": [em_low, em_high],
            "pair_successes": pair_successes,
            "pla_micro": ps_p,
            "pla_ci95": [ps_low, ps_high],
            "acc_macro": acc_macro,
            "k_full": k_full,
            "k_any": k_any,
            "tcs_score": tcs_score,
            "eos_rate": eos_count / n_records,
            "multi_token_rate": multi_token_count / n_records,
            "modal_token": modal_tok,
            "modal_share": modal_share,
            "collapsed": (modal_share >= 0.35),
            "cir_blank_rate": blank_match_count / n_records,
            "per_token_accuracy": per_token_accuracy,
            "mean_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95))
        }

    # Save Prediction Logs
    pred_path = EXP26_DIR / "CANDIDATE_PER_RECORD_PREDICTIONS.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in all_predictions_to_save:
            f.write(json.dumps(p) + "\n")

    # Save Training Log
    log_path = EXP26_DIR / "TRAINING_LOG.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for l in train_log:
            f.write(json.dumps(l) + "\n")

    eval_report = {
        "experiment_id": "exp26_15m_scaling_diagnostic",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "model_architecture": "ChakrMicro-15M",
        "unique_trainable_parameters": trainable_params,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "training_duration_sec": t_train_total,
        "blank_emissions": blank_emissions,
        "splits": all_split_results,
        "top5_logits_diagnostics": top5_logits_diagnostics
    }

    report_path = EXP26_DIR / "EVALUATION_REPORT.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    manifest = {
        "experiment_id": "exp26_15m_scaling_diagnostic",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "trainable_params": trainable_params,
        "dataset_hashes": expected_data_hashes,
        "checkpoint_hashes": {
            "checkpoint_best": get_file_hash(CKPT_DIR / "checkpoint_best.pt"),
            "checkpoint_step_300": get_file_hash(CKPT_DIR / "checkpoint_step_300.pt")
        },
        "evaluation_report_hash": get_file_hash(report_path),
        "predictions_hash": get_file_hash(pred_path),
        "summary": {
            "test_1_em": all_split_results["test_1"]["em_micro"],
            "test_1_pla": all_split_results["test_1"]["pla_micro"],
            "test_2a_em": all_split_results["test_2a"]["em_micro"],
            "test_2a_pla": all_split_results["test_2a"]["pla_micro"],
            "test_2b_em": all_split_results["test_2b"]["em_micro"],
            "test_2b_pla": all_split_results["test_2b"]["pla_micro"],
            "test_2c_em": all_split_results["test_2c"]["em_micro"],
            "test_2c_pla": all_split_results["test_2c"]["pla_micro"],
        }
    }

    manifest_path = EXP26_DIR / "MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 50)
    print("EXP26 MASTER RUN COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in: {EXP26_DIR}")
    print(f"Test 1 EM: {manifest['summary']['test_1_em']:.3f} | PLA: {manifest['summary']['test_1_pla']:.3f}")
    print(f"Test 2A EM: {manifest['summary']['test_2a_em']:.3f} | PLA: {manifest['summary']['test_2a_pla']:.3f}")
    print(f"Test 2B EM: {manifest['summary']['test_2b_em']:.3f} | PLA: {manifest['summary']['test_2b_pla']:.3f}")
    print(f"Test 2C EM: {manifest['summary']['test_2c_em']:.3f} | PLA: {manifest['summary']['test_2c_pla']:.3f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
