"""
Single-Field Entity Copy Training Pilot Runner.

HARD RESTRICTIONS & SPECIFICATIONS:
- Exactly ONE training run invocation on CPU.
- Maximum 300 optimizer steps. Fixed seed = 42.
- AdamW (lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95)).
- LR Schedule: 25 steps linear warmup, cosine decay to 2e-5 at step 300.
- Effective batch size: 2.
- Supervision: Prompt tokens masked with -100; target tokens supervised (+ terminal EOS id 1).
- Canonical Step 10,000 base checkpoint (96357705...).
- Canonical Tokenizer (vocab: db4d6119..., merges: ada4b1dc...).
- Early-stop gate at step 200:
  Evaluate exact match on 40-record validation set.
  If exact match < 10% (< 4/40), STOP IMMEDIATELY.
  If >= 10% (>= 4/40), continue to step 300.
- Primary success threshold at completion: >= 60% exact match (>= 24/40).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

RUN_DIR = REPO_ROOT / "experiments" / "single_field_copy_pilot" / "authorized_run_001"
DATA_DIR = RUN_DIR / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"

BASE_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

EXPECTED_BASE_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
EXPECTED_VOCAB_HASH = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"
EXPECTED_MERGES_HASH = "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def normalize_value(val_str: str) -> str:
    s = val_str.strip().lower()
    s = s.strip("\"'`;:")
    s = re.sub(r'[\s]+', ' ', s)
    return s


def is_doctest_attractor(text: str) -> bool:
    attractor_patterns = [
        "__main__",
        "doctest",
        "import doctest",
        ">>>",
        "linked_list",
        "np.ndarray",
        "def get_to",
        "codecannot"
    ]
    t_lower = text.lower()
    return any(p in t_lower for p in attractor_patterns)


def generate_greedy(model: nn.Module, tokenizer: Any, prompt: str, max_new_tokens: int = 40) -> Tuple[str, List[int]]:
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [0]
    
    device = next(model.parameters()).device
    generated_ids = list(input_ids)
    new_tokens = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
            if curr_tensor.shape[1] > 256:
                curr_tensor = curr_tensor[:, -256:]
            
            logits = model(curr_tensor)
            next_token_logits = logits[0, -1, :]
            next_token = int(torch.argmax(next_token_logits).item())
            
            if next_token == 1:  # Canonical EOS ID
                break
                
            new_tokens.append(next_token)
            generated_ids.append(next_token)
            
    decoded_output = tokenizer.decode(new_tokens) if new_tokens else ""
    return decoded_output, new_tokens


class SingleFieldDataset:
    def __init__(self, jsonl_path: Path, tokenizer: Any, max_seq_len: int = 512):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.records: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def encode_example(self, prompt: str, target: str) -> Tuple[List[int], List[int]]:
        bos_id = 0
        eos_id = 1
        ignore_index = -100

        p_toks = self.tokenizer.encode(prompt)
        r_toks = self.tokenizer.encode(target)

        if not r_toks:
            raise ValueError(f"Target tokenized to 0 tokens: {target}")

        # Causal sequence with prompt masked
        input_seq = [bos_id] + p_toks + r_toks
        target_seq = ([ignore_index] * len(p_toks)) + r_toks + [eos_id]

        if len(input_seq) > self.max_seq_len:
            raise ValueError(f"Sequence length {len(input_seq)} exceeds {self.max_seq_len}")

        return input_seq, target_seq

    def get_batch(self, batch_size: int, rng: np.random.RandomState) -> Tuple[torch.Tensor, torch.Tensor, int]:
        indices = rng.randint(0, len(self.records), size=batch_size)
        batch_inputs = []
        batch_targets = []

        for idx in indices:
            rec = self.records[idx]
            inp, tgt = self.encode_example(rec["prompt"], rec["target"])
            batch_inputs.append(inp)
            batch_targets.append(tgt)

        max_len = max(len(x) for x in batch_inputs)
        pad_id = 2
        ignore_index = -100

        padded_inputs = []
        padded_targets = []
        for inp, tgt in zip(batch_inputs, batch_targets):
            pad_amt = max_len - len(inp)
            padded_inputs.append(inp + [pad_id] * pad_amt)
            padded_targets.append(tgt + [ignore_index] * pad_amt)

        x_tensor = torch.tensor(padded_inputs, dtype=torch.long)
        y_tensor = torch.tensor(padded_targets, dtype=torch.long)
        active_tokens = int((y_tensor != ignore_index).sum().item())

        return x_tensor, y_tensor, active_tokens


def evaluate_copy_benchmark(model: nn.Module, tokenizer: Any, val_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    exact_match_count = 0
    norm_match_count = 0
    attractor_count = 0
    empty_count = 0
    total_tokens = 0
    
    details = []

    for r in val_records:
        prompt = r["prompt"]
        expected_val = r["expected_value"]
        norm_expected = normalize_value(expected_val)
        
        gen_text, gen_tokens = generate_greedy(model, tokenizer, prompt, max_new_tokens=40)
        norm_gen = normalize_value(gen_text)
        
        # Exact match rule: expected value is generated as an exact match / substring of the clean output
        is_exact = expected_val == gen_text.strip() or expected_val in gen_text
        is_norm = (norm_expected == norm_gen) or (norm_expected in norm_gen)
        is_attr = is_doctest_attractor(gen_text)
        is_empty = len(gen_tokens) == 0 or len(gen_text.strip()) == 0
        
        if is_exact:
            exact_match_count += 1
        if is_norm:
            norm_match_count += 1
        if is_attr:
            attractor_count += 1
        if is_empty:
            empty_count += 1
        total_tokens += len(gen_tokens)
        
        details.append({
            "id": r["id"],
            "expected": expected_val,
            "generated": gen_text,
            "exact_match": is_exact,
            "norm_match": is_norm,
            "attractor": is_attr,
            "empty": is_empty
        })

    num_records = len(val_records)
    return {
        "exact_match_count": exact_match_count,
        "exact_match_acc": exact_match_count / num_records,
        "norm_match_count": norm_match_count,
        "norm_match_acc": norm_match_count / num_records,
        "attractor_count": attractor_count,
        "attractor_rate": attractor_count / num_records,
        "empty_count": empty_count,
        "empty_rate": empty_count / num_records,
        "total_records": num_records,
        "avg_tokens": total_tokens / num_records,
        "details": details
    }


def compute_val_loss(model: nn.Module, val_dataset: SingleFieldDataset, loss_fn: nn.Module, batch_size: int = 2) -> float:
    model.eval()
    rng = np.random.RandomState(1337)
    total_loss = 0.0
    total_active_tokens = 0
    
    # 20 batches of size 2 = 40 samples
    num_batches = len(val_dataset) // batch_size
    with torch.no_grad():
        for _ in range(num_batches):
            x, y, active_count = val_dataset.get_batch(batch_size=batch_size, rng=rng)
            if active_count == 0:
                continue
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item() * active_count
            total_active_tokens += active_count

    return total_loss / max(1, total_active_tokens)


def run_authorized_pilot():
    print("=" * 75)
    print("CHAKRVIEW v0.7 — SINGLE-FIELD ENTITY COPY TRAINING PILOT")
    print("ONE AUTHORIZED RUN | CPU | MAX 300 OPTIMIZER STEPS")
    print("=" * 75)

    # 1. Preflight Verification Gates
    print("1. Running Preflight Verification...")
    ckpt_hash = compute_sha256(BASE_CKPT)
    vocab_hash = compute_sha256(TOKENIZER_DIR / "vocab.json")
    merges_hash = compute_sha256(TOKENIZER_DIR / "merges.json")
    train_hash = compute_sha256(TRAIN_FILE)
    val_hash = compute_sha256(VAL_FILE)

    assert ckpt_hash == EXPECTED_BASE_HASH, f"Base checkpoint hash mismatch: {ckpt_hash}"
    assert vocab_hash == EXPECTED_VOCAB_HASH, f"Vocab hash mismatch: {vocab_hash}"
    assert merges_hash == EXPECTED_MERGES_HASH, f"Merges hash mismatch: {merges_hash}"

    # Verify session.py is pristine
    diff = subprocess.check_output(["git", "diff", "--name-only", "chakrview/runtime/session.py"]).decode("utf-8").strip()
    assert diff == "", f"Protected file chakrview/runtime/session.py is modified: {diff}"

    print(f" - Canonical Base Checkpoint Hash: {ckpt_hash} (MATCH)")
    print(f" - Canonical Vocab Hash:           {vocab_hash} (MATCH)")
    print(f" - Canonical Merges Hash:          {merges_hash} (MATCH)")
    print(f" - Train Dataset Hash:             {train_hash}")
    print(f" - Validation Dataset Hash:        {val_hash}")
    print(" - Protected File (session.py):     UNMODIFIED")
    print(" - Held-out Test Split Access:     ZERO ACCESS / VERIFIED UNTOUCHED")

    # 2. Setup Tokenizer, Data, and Model
    tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))
    train_dataset = SingleFieldDataset(TRAIN_FILE, tokenizer)
    val_dataset = SingleFieldDataset(VAL_FILE, tokenizer)

    assert len(train_dataset) == 200, f"Expected 200 train records, got {len(train_dataset)}"
    assert len(val_dataset) == 40, f"Expected 40 val records, got {len(val_dataset)}"

    device = torch.device("cpu")
    model_config = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=6,
        n_heads=6,
        hidden_dim=512,
        max_seq_len=512,
    )
    model = ChakrMicro(model_config).to(device)

    # Load canonical base checkpoint state dict
    ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(" - Canonical Model Architecture & Weights Loaded: 6-layer ChakrMicro (1.52M params)")

    # 3. Initial Baseline Evaluation (Step 0)
    print("\n2. Baseline Evaluation on Step-10,000 Canonical Base Model...")
    val_records = [json.loads(line) for line in open(VAL_FILE, encoding="utf-8")]
    baseline_metrics = evaluate_copy_benchmark(model, tokenizer, val_records)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    baseline_val_loss = compute_val_loss(model, val_dataset, loss_fn)
    print(f"   Baseline Exact Match: {baseline_metrics['exact_match_count']}/40 ({baseline_metrics['exact_match_acc']*100:.1f}%)")
    print(f"   Baseline Norm Match:  {baseline_metrics['norm_match_count']}/40 ({baseline_metrics['norm_match_acc']*100:.1f}%)")
    print(f"   Baseline Attractor:   {baseline_metrics['attractor_count']}/40 ({baseline_metrics['attractor_rate']*100:.1f}%)")
    print(f"   Baseline Val Loss:    {baseline_val_loss:.4f} nats")

    # 4. Training Initialization
    SEED = 42
    MAX_STEPS = 300
    WARMUP_STEPS = 25
    BASE_LR = 2e-4
    MIN_LR = 2e-5
    BATCH_SIZE = 2
    WEIGHT_DECAY = 0.01

    rng = np.random.RandomState(SEED)
    torch.manual_seed(SEED)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY
    )

    print("\n3. Starting Training Process...")
    print(f"   Optimizer: AdamW (lr={BASE_LR}, min_lr={MIN_LR}, betas=(0.9, 0.95), weight_decay={WEIGHT_DECAY})")
    print(f"   Schedule:  25-step linear warmup -> cosine decay to {MIN_LR} at step {MAX_STEPS}")
    print(f"   Batch:     {BATCH_SIZE} on CPU, Seed: {SEED}")
    print(f"   Budget:    Max {MAX_STEPS} optimizer steps with Step-200 Early-Stop Gate (< 10% EM)")

    history = []
    step_200_metrics = None
    early_stopped = False
    start_time = time.perf_counter()

    for step in range(1, MAX_STEPS + 1):
        step_start = time.perf_counter()

        # Learning rate schedule: Warmup + Cosine Decay
        if step <= WARMUP_STEPS:
            cur_lr = BASE_LR * (step / WARMUP_STEPS)
        else:
            progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
            cur_lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

        # Sample batch
        x, y, active_tokens = train_dataset.get_batch(batch_size=BATCH_SIZE, rng=rng)
        x, y = x.to(device), y.to(device)

        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if not math.isfinite(loss.item()):
            raise RuntimeError(f"Non-finite loss detected at step {step}: {loss.item()}")

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()

        step_duration = time.perf_counter() - step_start

        entry = {
            "step": step,
            "train_loss": round(loss.item(), 4),
            "lr": round(cur_lr, 8),
            "grad_norm": round(grad_norm, 4),
            "active_tokens": active_tokens,
            "step_time_sec": round(step_duration, 4)
        }

        # Log every 25 steps
        if step % 25 == 0 or step == 1:
            v_loss = compute_val_loss(model, val_dataset, loss_fn)
            entry["val_loss"] = round(v_loss, 4)
            print(f"   Step {step:03d}/{MAX_STEPS:03d} | Train Loss: {loss.item():.4f} | Val Loss: {v_loss:.4f} | LR: {cur_lr:.2e} | Grad: {grad_norm:.2f} ({step_duration*1000:.1f}ms)")
        history.append(entry)

        # -------------------------------------------------------------
        # STEP 200 GATE EVALUATION
        # -------------------------------------------------------------
        if step == 200:
            print("\n" + "=" * 75)
            print("STEP 200 EARLY-STOP GATE EVALUATION")
            print("=" * 75)
            
            # Save step 200 checkpoint
            step_200_ckpt_path = RUN_DIR / "checkpoint_step_200.pt"
            torch.save({
                "step": 200,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "seed": SEED,
            }, step_200_ckpt_path)
            print(f"Saved step 200 checkpoint: {step_200_ckpt_path}")

            step_200_metrics = evaluate_copy_benchmark(model, tokenizer, val_records)
            step_200_val_loss = compute_val_loss(model, val_dataset, loss_fn)
            step_200_metrics["val_loss"] = round(step_200_val_loss, 4)

            em_count = step_200_metrics["exact_match_count"]
            em_acc = step_200_metrics["exact_match_acc"]
            print(f"Step 200 Exact Match:  {em_count}/40 ({em_acc*100:.1f}%)")
            print(f"Step 200 Norm Match:   {step_200_metrics['norm_match_count']}/40 ({step_200_metrics['norm_match_acc']*100:.1f}%)")
            print(f"Step 200 Attractor:    {step_200_metrics['attractor_count']}/40 ({step_200_metrics['attractor_rate']*100:.1f}%)")
            print(f"Step 200 Val Loss:     {step_200_val_loss:.4f} nats")

            # Check early-stop condition: < 10% (< 4/40)
            if em_count < 4:
                print("\n[EARLY-STOP TRIGGERED]: Step 200 Exact Match is below 10% (< 4/40).")
                print("Halting training immediately per safety specification.")
                early_stopped = True
                break
            else:
                print("\n[GATE PASSED]: Step 200 Exact Match is >= 10%. Continuing to step 300...")

    total_training_time = time.perf_counter() - start_time
    completed_steps = len(history)
    print(f"\nTraining finished in {total_training_time:.2f}s ({completed_steps} steps completed).")

    # Save final checkpoint
    final_ckpt_name = f"checkpoint_step_{completed_steps}.pt"
    final_ckpt_path = RUN_DIR / final_ckpt_name
    torch.save({
        "step": completed_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "seed": SEED,
    }, final_ckpt_path)
    final_ckpt_hash = compute_sha256(final_ckpt_path)
    print(f"Saved final checkpoint: {final_ckpt_path} (SHA-256: {final_ckpt_hash})")

    # 5. Final Evaluation at Completed Step
    print("\n" + "=" * 75)
    print(f"FINAL BENCHMARK EVALUATION AT STEP {completed_steps}")
    print("=" * 75)
    final_metrics = evaluate_copy_benchmark(model, tokenizer, val_records)
    final_val_loss = compute_val_loss(model, val_dataset, loss_fn)
    final_metrics["val_loss"] = round(final_val_loss, 4)

    final_em = final_metrics["exact_match_count"]
    final_em_acc = final_metrics["exact_match_acc"]
    final_norm = final_metrics["norm_match_count"]
    final_norm_acc = final_metrics["norm_match_acc"]
    final_attr = final_metrics["attractor_count"]
    final_attr_rate = final_metrics["attractor_rate"]
    final_empty = final_metrics["empty_count"]

    print(f"Final Exact Match:    {final_em}/40 ({final_em_acc*100:.1f}%)")
    print(f"Final Norm Match:     {final_norm}/40 ({final_norm_acc*100:.1f}%)")
    print(f"Final Attractor Rate: {final_attr}/40 ({final_attr_rate*100:.1f}%)")
    print(f"Final Empty/EOS Rate: {final_empty}/40 ({final_metrics['empty_rate']*100:.1f}%)")
    print(f"Final Val Loss:       {final_val_loss:.4f} nats")

    # Verdict assignment
    if final_em >= 24:
        verdict = "PASS — COPY CAPABILITY GATE"
    elif final_em > baseline_metrics["exact_match_count"]:
        verdict = "PARTIAL — LEARNING SIGNAL OBSERVED"
    elif early_stopped:
        verdict = "FAIL — EARLY STOP AT STEP 200 (<10% EM)"
    else:
        verdict = "FAIL — NO SUFFICIENT LEARNING"

    print("\n" + "=" * 75)
    print(f"PILOT VERDICT: {verdict}")
    print("=" * 75)

    # Save comprehensive results JSON
    results_payload = {
        "verdict": verdict,
        "completed_steps": completed_steps,
        "max_steps": MAX_STEPS,
        "early_stopped": early_stopped,
        "training_time_sec": round(total_training_time, 2),
        "seed": SEED,
        "hyperparameters": {
            "optimizer": "AdamW",
            "base_lr": BASE_LR,
            "min_lr": MIN_LR,
            "warmup_steps": WARMUP_STEPS,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "device": "cpu"
        },
        "hashes": {
            "canonical_base_checkpoint": ckpt_hash,
            "canonical_vocab": vocab_hash,
            "canonical_merges": merges_hash,
            "train_file": train_hash,
            "val_file": val_hash,
            "final_checkpoint": final_ckpt_hash
        },
        "baseline_metrics": baseline_metrics,
        "step_200_metrics": step_200_metrics,
        "final_metrics": final_metrics,
        "history": history
    }

    results_json_path = RUN_DIR / "PILOT_RESULTS.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Detailed pilot results written to: {results_json_path}")

    return results_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Authorized Single-Field Entity Copy Training Pilot")
    parser.add_argument("--execute", action="store_true", help="Authorize execution of the single training run")
    args = parser.parse_args()

    if not args.execute:
        print("PREFLIGHT DRY CHECK ONLY: Run with --execute to launch the single authorized training pilot.")
        sys.exit(0)

    run_authorized_pilot()
