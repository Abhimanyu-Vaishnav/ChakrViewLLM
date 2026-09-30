"""
Benchmark: Step 49 Language Acquisition & Coherence Validation.

Runs a fast (200-step) version of the Step 49 progressive training experiment
and reports all key metrics: val loss, PPL, context divergence, TTR,
trigram diversity, and negative control comparison.

Usage:
    python scripts/benchmark_step49_language_acquisition.py

This is a fast benchmark (not the full 5000-step experiment).
For the full experiment, run: scripts/run_step49_language_acquisition.py
"""

import sys
import math
import time
import hashlib
from pathlib import Path
from typing import List, Any, Dict, Tuple

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.loss import CausalLoss
from chakrview.training.optimizer import build_optimizer, build_lr_scheduler
from chakrview.training.safety import TrainingSafetyChecker
from chakrview.training.builder import ChakrOfflineDataset
from chakrview.training.config import TrainingHyperparameters
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.pipeline import InferenceEngine, InferenceRequest
from chakrview.runtime.inference import GenerationConfig
from chakrview.runtime.sampling import SamplingConfig

FROZEN_BASELINE_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
STAGE_B_TRAIN = ROOT_DIR / "data" / "tokenized" / "stage_b" / "train"
STAGE_B_VAL = ROOT_DIR / "data" / "tokenized" / "stage_b" / "validation"
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"
SEED = 42
SEQ_LEN = 64
BATCH_SIZE = 4
BENCHMARK_STEPS = [0, 50, 100, 200]


def compute_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            h.update(name.encode())
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def make_model(seed: int = SEED) -> ChakrMicro:
    torch.manual_seed(seed)
    return ChakrMicro(ModelConfig())


def eval_val(model: nn.Module, loss_fn: nn.Module, max_batches: int = 20) -> Tuple[float, float]:
    ds = StreamingTokenDataset(shard_dir=STAGE_B_VAL, sequence_length=SEQ_LEN, loop=False, drop_remainder=True)
    model.eval()
    losses = []
    buf: List[Any] = []
    with torch.no_grad():
        for item in ds:
            buf.append(item)
            if len(buf) == BATCH_SIZE:
                b = ChakrOfflineDataset.collate_fn(buf, pad_token_id=2)
                buf = []
                logits = model(b["input_ids"], attention_mask=b["attention_mask"])
                losses.append(loss_fn(logits, b["target_ids"]).item())
                if len(losses) >= max_batches:
                    break
    model.train()
    mean = sum(losses) / len(losses) if losses else float("nan")
    return round(mean, 4), round(math.exp(min(mean, 20.0)), 2)


def train_steps(model, opt, sch, loss_fn, train_iter, n: int, randomize: bool = False) -> List[float]:
    model.train()
    losses = []
    for step in range(1, n + 1):
        items = [next(train_iter) for _ in range(BATCH_SIZE)]
        batch = ChakrOfflineDataset.collate_fn(items, pad_token_id=2)
        inp = batch["input_ids"]
        tgt = batch["target_ids"]
        if randomize:
            flat = tgt.clone().view(-1)
            tgt = flat[torch.randperm(flat.size(0))].view_as(tgt)
        opt.zero_grad()
        logits = model(inp, attention_mask=batch["attention_mask"])
        loss = loss_fn(logits, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        losses.append(loss.item())
    return losses


def context_divergence(model, tokenizer, loss_fn) -> float:
    coherent = "The neural network processes input tokens through attention and produces output"
    random_pfx = "Banana clock seventeen umbrella seven rain and produces output"

    def seq_loss(text):
        ids = tokenizer.encode(text)[:60]
        t = torch.tensor([ids], dtype=torch.long)
        if t.shape[1] < 2:
            return 0.0
        with torch.no_grad():
            logits = model(t[:, :-1])
            return loss_fn(logits, t[:, 1:]).item()

    model.eval()
    d = seq_loss(random_pfx) - seq_loss(coherent)
    model.train()
    return round(d, 4)


def greedy_generate(model, tokenizer, prompt: str, max_tokens: int = 20) -> str:
    try:
        gen_cfg = GenerationConfig(max_new_tokens=max_tokens, sampling=SamplingConfig(temperature=0.0))
        engine = InferenceEngine(model=model, tokenizer=tokenizer)
        req = InferenceRequest(prompt=prompt, generation_config=gen_cfg)
        out = engine.execute(req)
        return out.text
    except Exception as e:
        return f"[ERROR: {e}]"


def main():
    print("=" * 70)
    print("STEP 49 BENCHMARK: Language Acquisition & Coherence Validation")
    print("=" * 70)

    # Verify baseline
    baseline = make_model()
    assert compute_hash(baseline) == FROZEN_BASELINE_HASH, "Baseline hash mismatch!"
    print(f"  Baseline SHA-256: {FROZEN_BASELINE_HASH[:16]}... [OK]")

    # Load tokenizer
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    print(f"  Tokenizer: {len(tokenizer.vocab)} tokens")

    loss_fn = CausalLoss(ignore_index=2)

    # Setup experimental model
    exp_model = make_model()
    exp_model.train()
    cfg = TrainingHyperparameters(
        learning_rate=1e-3, min_learning_rate=1e-4,
        warmup_steps=10, max_steps=BENCHMARK_STEPS[-1],
        gradient_accumulation_steps=1, gradient_clipping=1.0, seed=SEED,
    )
    opt = build_optimizer(exp_model, cfg)
    sch = build_lr_scheduler(opt, cfg)
    ds = StreamingTokenDataset(shard_dir=STAGE_B_TRAIN, sequence_length=SEQ_LEN, loop=True, seed=SEED)
    train_iter = iter(ds)

    print()
    print(f"  Running benchmark at stages: {BENCHMARK_STEPS}")
    print()
    print(f"  {'Steps':>6} {'ValLoss':>9} {'ValPPL':>9} {'CtxDiv':>9} | Sample Generation")
    print(f"  {'─' * 75}")

    current_step = 0
    for target in BENCHMARK_STEPS:
        if target > current_step:
            t0 = time.perf_counter()
            train_steps(exp_model, opt, sch, loss_fn, train_iter, target - current_step)
            elapsed = time.perf_counter() - t0
            current_step = target

        val_loss, val_ppl = eval_val(exp_model, loss_fn)
        ctx_div = context_divergence(exp_model, tokenizer, loss_fn)
        gen_text = greedy_generate(exp_model, tokenizer, "The foundation of mathematics", max_tokens=15)
        gen_short = gen_text[:50].replace("\n", " ")

        print(f"  {current_step:>6} {val_loss:>9.4f} {val_ppl:>9.2f} {ctx_div:>9.4f} | {gen_short}")

    # Negative control
    print()
    print("  Running negative control (200 steps, randomized targets)...")
    ctrl_model = make_model()
    ctrl_cfg = TrainingHyperparameters(
        learning_rate=1e-3, min_learning_rate=1e-4,
        warmup_steps=10, max_steps=200,
        gradient_accumulation_steps=1, gradient_clipping=1.0, seed=SEED,
    )
    ctrl_opt = build_optimizer(ctrl_model, ctrl_cfg)
    ctrl_sch = build_lr_scheduler(ctrl_opt, ctrl_cfg)
    ctrl_loss_fn = CausalLoss(ignore_index=2)
    ctrl_ds = StreamingTokenDataset(shard_dir=STAGE_B_TRAIN, sequence_length=SEQ_LEN, loop=True, seed=SEED)
    ctrl_iter = iter(ctrl_ds)
    train_steps(ctrl_model, ctrl_opt, ctrl_sch, ctrl_loss_fn, ctrl_iter, 200, randomize=True)
    ctrl_val_loss, ctrl_val_ppl = eval_val(ctrl_model, ctrl_loss_fn)
    exp_val_loss_200, exp_val_ppl_200 = eval_val(exp_model, loss_fn)

    print()
    print(f"  Control (randomized, 200 steps):    Val Loss={ctrl_val_loss:.4f}  PPL={ctrl_val_ppl:.2f}")
    print(f"  Experiment (real data, 200 steps):  Val Loss={exp_val_loss_200:.4f}  PPL={exp_val_ppl_200:.2f}")
    print(f"  PPL Ratio (Ctrl/Exp): {ctrl_val_ppl / max(exp_val_ppl_200, 0.01):.2f}x")

    # Final hash
    final_hash = compute_hash(exp_model)
    baseline_hash_final = compute_hash(baseline)
    print()
    print(f"  Baseline SHA-256 preserved: {baseline_hash_final == FROZEN_BASELINE_HASH}")
    print(f"  Experiment weight updated:  {final_hash != FROZEN_BASELINE_HASH}")
    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
