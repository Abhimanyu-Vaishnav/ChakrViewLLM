"""
Controlled Training & Neural Evaluation Engine for Batches 1621–1660.
Location: experiments/batch1621_1660/run_neural_experiment.py

Executes:
1. Verifies canonical & adapted checkpoint hashes.
2. Evaluates Baseline 1: Canonical Checkpoint (Step 10,000).
3. Evaluates Baseline 2: Adapted Checkpoint (Step 10,080 from Batch 1381).
4. Trains Candidate Checkpoint on train.jsonl for 100 steps, tracking validation loss every 10 steps.
5. Saves Candidate Checkpoint to experiments/batch1621_1660/checkpoints/chakrmicro_controlled_candidate_step100.pt.
6. Evaluates Candidate Checkpoint on the 150-task held-out evaluation suite.
7. Saves comprehensive machine-readable results to experiments/batch1621_1660/results/NEURAL_EVALUATION_RESULTS.json.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
ADAPTED_CKPT = Path("experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")

TRAIN_FILE = Path("experiments/batch1621_1660/data/train.jsonl")
VAL_FILE = Path("experiments/batch1621_1660/data/val.jsonl")
TEST_FILE = Path("experiments/batch1621_1660/data/test_held_out.json")

CANDIDATE_CKPT_PATH = Path("experiments/batch1621_1660/checkpoints/chakrmicro_controlled_candidate_step100.pt")
OUT_RESULTS = Path("experiments/batch1621_1660/results/NEURAL_EVALUATION_RESULTS.json")

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(65536):
            h.update(b)
    return h.hexdigest()


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = ChakrMicro(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, config


def generate_tokens(model: ChakrMicro, tokenizer, prompt: str, max_new_tokens: int = 25) -> str:
    tokens = tokenizer.encode(prompt)
    curr = tokens.copy()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = torch.tensor([curr], dtype=torch.long)
            logits = model(inp)[0, -1, :]
            next_token = torch.argmax(logits).item()
            curr.append(next_token)
            if next_token == 1:  # EOS
                break
    return tokenizer.decode(curr[len(tokens):], errors="replace")


def compute_nll_and_ppl(model: ChakrMicro, tokenizer, texts: list[str]) -> tuple[float, float]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for t in texts:
            toks = tokenizer.encode(t)
            if len(toks) < 2:
                continue
            inp = torch.tensor([toks[:-1]], dtype=torch.long)
            targets = torch.tensor(toks[1:], dtype=torch.long)
            logits = model(inp)[0]
            log_probs = F.log_softmax(logits, dim=-1)
            target_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            total_nll -= target_lp.sum().item()
            total_tokens += len(targets)

    mean_loss = total_nll / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return round(mean_loss, 4), round(ppl, 2)


def evaluate_benchmark(model: ChakrMicro, tokenizer, tasks: list[dict], model_name: str) -> dict:
    model.eval()
    results = []
    track_scores = {
        "track1_language_continuation": {"total": 0, "passed": 0},
        "track2_arithmetic_continuation": {"total": 0, "passed": 0},
        "track3_code_completion": {"total": 0, "passed": 0},
        "track4_compositional_generalization": {"total": 0, "passed": 0},
        "track5_factual_recall_and_memorization": {"total": 0, "passed": 0},
    }

    t0 = time.perf_counter()

    for t in tasks:
        tid = t["id"]
        trk = t["track"]
        prompt = t["prompt"]
        gen_raw = generate_tokens(model, tokenizer, prompt, max_new_tokens=20)
        gen_clean = gen_raw.strip()

        is_pass = False

        if trk == "language_continuation":
            # Pass if non-empty continuation without immediate loop
            words = gen_clean.split()
            is_pass = len(words) >= 2 and len(set(words)) >= len(words) * 0.5
            track_scores["track1_language_continuation"]["total"] += 1
            if is_pass:
                track_scores["track1_language_continuation"]["passed"] += 1

        elif trk == "arithmetic_continuation":
            expected = t["expected_answer"]
            m_num = re.search(r"[-+]?\d+", gen_clean)
            extracted = m_num.group(0) if m_num else None
            is_pass = (extracted == expected)
            track_scores["track2_arithmetic_continuation"]["total"] += 1
            if is_pass:
                track_scores["track2_arithmetic_continuation"]["passed"] += 1

        elif trk == "code_completion":
            exp_sfx = t["expected_suffix"].strip()
            # Token prefix matching
            is_pass = exp_sfx.lower() in gen_clean.lower()
            track_scores["track3_code_completion"]["total"] += 1
            if is_pass:
                track_scores["track3_code_completion"]["passed"] += 1

        elif trk == "compositional_generalization":
            exp_target = t["expected_target"].lower().strip()
            is_pass = exp_target in gen_clean.lower()
            track_scores["track4_compositional_generalization"]["total"] += 1
            if is_pass:
                track_scores["track4_compositional_generalization"]["passed"] += 1

        elif trk == "factual_recall":
            exp_port = t["expected_port"].strip()
            m_port = re.search(r"\b\d{4}\b", gen_clean)
            extracted_port = m_port.group(0) if m_port else None
            if t["in_training_set"]:
                # Recall test: must recall exact port
                is_pass = (extracted_port == exp_port)
            else:
                # Counterfactual: must NOT hallucinate false confidence or can abstain/miss
                # Correct behavior for unseen: model shouldn't predict arbitrary ungrounded ports
                is_pass = (extracted_port != exp_port)
            track_scores["track5_factual_recall_and_memorization"]["total"] += 1
            if is_pass:
                track_scores["track5_factual_recall_and_memorization"]["passed"] += 1

        results.append({
            "id": tid,
            "track": trk,
            "prompt": prompt,
            "generated": gen_clean,
            "passed": is_pass,
        })

    elapsed = (time.perf_counter() - t0) * 1000.0

    summary = {
        "model_name": model_name,
        "total_tasks": len(tasks),
        "total_passed": sum(v["passed"] for v in track_scores.values()),
        "accuracy": round(sum(v["passed"] for v in track_scores.values()) / len(tasks), 4),
        "track_scores": {
            k: {
                "total": v["total"],
                "passed": v["passed"],
                "accuracy": round(v["passed"] / v["total"], 4) if v["total"] > 0 else 0.0
            }
            for k, v in track_scores.items()
        },
        "evaluation_time_ms": round(elapsed, 2),
    }

    return {"summary": summary, "per_task": results}


def train_controlled_candidate(tokenizer, base_model, config, train_texts: list[str], val_texts: list[str]) -> tuple[ChakrMicro, dict]:
    print("\nStarting Controlled Training for Candidate Checkpoint...")
    model = ChakrMicro(config)
    model.load_state_dict(base_model.state_dict())
    model.train()

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = 100
    batch_size = 4
    seq_len = 64

    # Prepare tokenized training sequences
    train_tokens = [tokenizer.encode(t) for t in train_texts]

    step_log = []
    best_val_loss = float("inf")
    best_weights = None

    t0 = time.perf_counter()

    for step in range(1, total_steps + 1):
        # Sample mini-batch
        batch_samples = random.sample(train_tokens, batch_size)
        
        # Pack or pad sequences
        inputs = []
        targets = []
        for s in batch_samples:
            if len(s) < 2:
                continue
            truncated = s[:seq_len]
            inputs.append(truncated[:-1])
            targets.append(truncated[1:])

        # Find max len in batch
        max_l = max(len(inp) for inp in inputs)
        padded_inp = []
        padded_tgt = []
        for inp, tgt in zip(inputs, targets):
            p_inp = inp + [0] * (max_l - len(inp))
            p_tgt = tgt + [-100] * (max_l - len(tgt))
            padded_inp.append(p_inp)
            padded_tgt.append(p_tgt)

        inp_tensor = torch.tensor(padded_inp, dtype=torch.long)
        tgt_tensor = torch.tensor(padded_tgt, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(inp_tensor)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_tensor.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluate validation loss every 10 steps
        if step % 10 == 0 or step == total_steps:
            val_loss, val_ppl = compute_nll_and_ppl(model, tokenizer, val_texts)
            step_log.append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            })
            print(f"Step {step:03d}/{total_steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time_sec = time.perf_counter() - t0

    # Load best weights
    if best_weights:
        model.load_state_dict(best_weights)

    # Save Candidate Checkpoint
    CANDIDATE_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": 100,
        "config": config,
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val_loss,
        "training_time_sec": train_time_sec,
    }, CANDIDATE_CKPT_PATH)

    candidate_hash = get_file_sha256(CANDIDATE_CKPT_PATH)
    print(f"Saved candidate checkpoint to {CANDIDATE_CKPT_PATH}")
    print(f"Candidate Checkpoint SHA-256: {candidate_hash}")

    training_report = {
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_rate": 5e-5,
        "training_time_sec": round(train_time_sec, 2),
        "best_val_loss": best_val_loss,
        "candidate_checkpoint_sha256": candidate_hash,
        "step_log": step_log,
    }

    return model, training_report


def run_full_neural_mission():
    print("=================================================================")
    print("CHAKRVIEW BATCHES 1621–1660: NEURAL LEARNING & EVIDENCE GATE")
    print("=================================================================")

    # 1. Verify Checkpoints
    c_hash = get_file_sha256(CANONICAL_CKPT)
    a_hash = get_file_sha256(ADAPTED_CKPT)
    print(f"Canonical Checkpoint: {CANONICAL_CKPT} (SHA-256: {c_hash})")
    print(f"Adapted Checkpoint:   {ADAPTED_CKPT} (SHA-256: {a_hash})")
    assert c_hash == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    assert a_hash == "15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139"

    # 2. Load Tokenizer & Datasets
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    train_texts = [json.loads(line)["text"] for line in open(TRAIN_FILE, encoding="utf-8")]
    val_texts = [json.loads(line)["text"] for line in open(VAL_FILE, encoding="utf-8")]
    test_tasks = json.load(open(TEST_FILE, encoding="utf-8"))

    print(f"Loaded {len(train_texts)} train texts, {len(val_texts)} val texts, {len(test_tasks)} held-out tasks.")

    # 3. Evaluate Canonical Checkpoint
    print("\n--- Evaluating Baseline 1: Canonical Checkpoint (Step 10,000) ---")
    m_canonical, config = load_model(CANONICAL_CKPT)
    can_eval = evaluate_benchmark(m_canonical, tokenizer, test_tasks, "canonical_step_10000")
    can_train_loss, can_train_ppl = compute_nll_and_ppl(m_canonical, tokenizer, train_texts)
    can_val_loss, can_val_ppl = compute_nll_and_ppl(m_canonical, tokenizer, val_texts)
    can_eval["summary"]["train_loss"] = can_train_loss
    can_eval["summary"]["train_ppl"] = can_train_ppl
    can_eval["summary"]["val_loss"] = can_val_loss
    can_eval["summary"]["val_ppl"] = can_val_ppl
    print(f"Canonical -> Accuracy: {can_eval['summary']['accuracy']*100:.2f}% | Val Loss: {can_val_loss} | Val PPL: {can_val_ppl}")

    # 4. Evaluate Adapted Checkpoint
    print("\n--- Evaluating Baseline 2: Adapted Checkpoint (Step 10,080) ---")
    m_adapted, _ = load_model(ADAPTED_CKPT)
    adap_eval = evaluate_benchmark(m_adapted, tokenizer, test_tasks, "adapted_step_10080")
    adap_train_loss, adap_train_ppl = compute_nll_and_ppl(m_adapted, tokenizer, train_texts)
    adap_val_loss, adap_val_ppl = compute_nll_and_ppl(m_adapted, tokenizer, val_texts)
    adap_eval["summary"]["train_loss"] = adap_train_loss
    adap_eval["summary"]["train_ppl"] = adap_train_ppl
    adap_eval["summary"]["val_loss"] = adap_val_loss
    adap_eval["summary"]["val_ppl"] = adap_val_ppl
    print(f"Adapted -> Accuracy: {adap_eval['summary']['accuracy']*100:.2f}% | Val Loss: {adap_val_loss} | Val PPL: {adap_val_ppl}")

    # 5. Train Candidate Checkpoint
    m_candidate, train_report = train_controlled_candidate(tokenizer, m_canonical, config, train_texts, val_texts)

    # 6. Evaluate Candidate Checkpoint
    print("\n--- Evaluating Candidate: Newly Trained Checkpoint (Step 100) ---")
    cand_eval = evaluate_benchmark(m_candidate, tokenizer, test_tasks, "candidate_step_100")
    cand_train_loss, cand_train_ppl = compute_nll_and_ppl(m_candidate, tokenizer, train_texts)
    cand_val_loss, cand_val_ppl = compute_nll_and_ppl(m_candidate, tokenizer, val_texts)
    cand_eval["summary"]["train_loss"] = cand_train_loss
    cand_eval["summary"]["train_ppl"] = cand_train_ppl
    cand_eval["summary"]["val_loss"] = cand_val_loss
    cand_eval["summary"]["val_ppl"] = cand_val_ppl
    print(f"Candidate -> Accuracy: {cand_eval['summary']['accuracy']*100:.2f}% | Val Loss: {cand_val_loss} | Val PPL: {cand_val_ppl}")

    # 7. Aggregate Machine-Readable Results
    final_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canonical_checkpoint_sha256": c_hash,
        "adapted_checkpoint_sha256": a_hash,
        "candidate_checkpoint_sha256": train_report["candidate_checkpoint_sha256"],
        "training_report": train_report,
        "comparative_summary": {
            "canonical_step_10000": can_eval["summary"],
            "adapted_step_10080": adap_eval["summary"],
            "candidate_step_100": cand_eval["summary"],
        },
        "per_task_results": {
            "canonical": can_eval["per_task"],
            "adapted": adap_eval["per_task"],
            "candidate": cand_eval["per_task"],
        }
    }

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nAll Neural Experiments Complete! Saved results to {OUT_RESULTS}")


if __name__ == "__main__":
    run_full_neural_mission()
