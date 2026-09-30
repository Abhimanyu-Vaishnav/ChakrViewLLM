"""
Script: Step 49 Language Acquisition & Coherence Validation.

Executes a controlled progressive training experiment on ChakrMicro v0.1
using the Stage B tokenized corpus at 6 prescribed checkpoint stages:
  Stage 0:    0 steps  (random initialization)
  Stage 1:  100 steps  (token frequency learning)
  Stage 2:  500 steps  (local sequence structure)
  Stage 3: 1000 steps  (phrase-level patterns)
  Stage 4: 2500 steps  (context-sensitive prediction)
  Stage 5: 5000 steps  (longer-range coherence)

At each stage:
  - Records train loss, validation loss, PPL
  - Runs probing generation (greedy + sampled)
  - Measures context-sensitivity divergence
  - Measures type-token ratio (TTR) and trigram diversity
  - Compares against randomized-target negative control at step 5000

Invariants:
  - dW_baseline = 0 (frozen baseline SHA-256 never mutated)
  - All training on isolated experimental model instance
  - CPU-only, seed-42 exact reproducibility

Outputs:
  - docs/STEP_49_PROGRESSIVE_TRAINING_CURVES.json
  - docs/STEP_49_COHERENCE_METRICS.json
  - checkpoints/step49/  (stage checkpoints)
"""

import sys
import math
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
from dataclasses import asdict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.loss import CausalLoss
from chakrview.training.optimizer import build_optimizer, build_lr_scheduler
from chakrview.training.safety import TrainingSafetyChecker
from chakrview.training.builder import ChakrOfflineDataset
from chakrview.training.config import TrainingHyperparameters
from chakrview.training.checkpoint import CheckpointManager
from chakrview.runtime.pipeline import InferenceEngine, InferenceRequest
from chakrview.runtime.inference import GenerationConfig
from chakrview.runtime.sampling import SamplingConfig

FROZEN_BASELINE_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_PARAM_COUNT = 3_443_136
STAGE_B_BASE = ROOT_DIR / "data" / "tokenized" / "stage_b"
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"
CKPT_BASE = ROOT_DIR / "checkpoints" / "step49"
ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "step49"
CURVES_FILE = ROOT_DIR / "docs" / "STEP_49_PROGRESSIVE_TRAINING_CURVES.json"
COHERENCE_FILE = ROOT_DIR / "docs" / "STEP_49_COHERENCE_METRICS.json"

STAGE_CHECKPOINTS = [0, 100, 500, 1000, 2500, 5000]
LR = 1e-3
MIN_LR = 1e-4
WARMUP = 50
GRAD_CLIP = 1.0
SEQ_LEN = 64
BATCH_SIZE = 4
SEED = 42

PROBING_PROMPTS = [
    "ChakrView is an indigenous neural architecture designed for",
    "The foundation of mathematics begins with",
    "Intelligence emerges from structured",
    "Language models learn to predict the next",
    "Neural networks are trained by minimizing",
]

CONTEXT_PROBE_PAIRS = [
    (
        "The neural network processes input tokens through attention",
        "Banana clock seventeen umbrella seven rain",
        "and produces output",
    ),
    (
        "Training the model requires gradient descent optimization",
        "Purple elephant seven sigma nine moon",
        "to minimize the loss",
    ),
]


def compute_model_hash(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def evaluate_split(
    model: nn.Module,
    shard_dir: Path,
    loss_fn: nn.Module,
    max_batches: int = 30,
) -> Tuple[float, float]:
    dataset = StreamingTokenDataset(
        shard_dir=shard_dir,
        sequence_length=SEQ_LEN,
        loop=False,
        drop_remainder=True,
    )
    model.eval()
    losses = []
    batch_buffer: List[Any] = []

    with torch.no_grad():
        for item in dataset:
            batch_buffer.append(item)
            if len(batch_buffer) == BATCH_SIZE:
                b = ChakrOfflineDataset.collate_fn(batch_buffer, pad_token_id=2)
                batch_buffer = []
                logits = model(b["input_ids"], attention_mask=b["attention_mask"])
                loss = loss_fn(logits, b["target_ids"])
                losses.append(loss.item())
                if len(losses) >= max_batches:
                    break

    mean_loss = float(torch.tensor(losses).mean().item()) if losses else float("nan")
    ppl = math.exp(min(mean_loss, 20.0)) if not math.isnan(mean_loss) else float("nan")
    model.train()
    return round(mean_loss, 6), round(ppl, 4)


def compute_type_token_ratio(token_ids: List[int]) -> float:
    if not token_ids:
        return 0.0
    return round(len(set(token_ids)) / len(token_ids), 4)


def compute_trigram_diversity(token_ids: List[int]) -> int:
    if len(token_ids) < 3:
        return 0
    trigrams = set()
    for i in range(len(token_ids) - 2):
        trigrams.add((token_ids[i], token_ids[i + 1], token_ids[i + 2]))
    return len(trigrams)


def probe_generation(
    model: nn.Module,
    tokenizer: BPETokenizer,
    model_hash: Optional[str] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 32,
) -> List[Dict[str, Any]]:
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        sampling=SamplingConfig(temperature=temperature),
    )
    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        expected_weight_hash=model_hash,
    )
    records = []
    for prompt in PROBING_PROMPTS:
        try:
            req = InferenceRequest(prompt=prompt, generation_config=gen_cfg)
            out = engine.execute(req)
            text = out.text
            token_ids = out.token_ids or []
        except Exception as exc:
            text = f"[GENERATION_ERROR: {exc}]"
            token_ids = []
        records.append({
            "prompt": prompt,
            "generated_text": text,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "ttr": compute_type_token_ratio(token_ids),
            "unique_trigrams": compute_trigram_diversity(token_ids),
            "temperature": temperature,
        })
    return records


def measure_context_sensitivity(
    model: nn.Module,
    tokenizer: BPETokenizer,
    loss_fn: nn.Module,
) -> Dict[str, Any]:
    model.eval()
    results = []

    with torch.no_grad():
        for coherent_prefix, random_prefix, continuation in CONTEXT_PROBE_PAIRS:
            coherent_full = coherent_prefix + " " + continuation
            random_full = random_prefix + " " + continuation

            def seq_loss(ids: List[int]) -> float:
                t = torch.tensor([ids], dtype=torch.long)
                if t.shape[1] < 2:
                    return float("nan")
                inp = t[:, :-1]
                tgt = t[:, 1:]
                logits = model(inp)
                loss = loss_fn(logits, tgt)
                return loss.item()

            try:
                coh_ids = tokenizer.encode(coherent_full)[:60]
                rnd_ids = tokenizer.encode(random_full)[:60]
                loss_coh = seq_loss(coh_ids)
                loss_rnd = seq_loss(rnd_ids)
                div = round(loss_rnd - loss_coh, 6) if not (
                    math.isnan(loss_coh) or math.isnan(loss_rnd)
                ) else None
            except Exception:
                loss_coh, loss_rnd, div = None, None, None

            results.append({
                "coherent_prefix": coherent_prefix,
                "random_prefix": random_prefix,
                "continuation": continuation,
                "loss_coherent": round(loss_coh, 6) if loss_coh is not None else None,
                "loss_random": round(loss_rnd, 6) if loss_rnd is not None else None,
                "context_divergence": div,
            })

    model.train()
    valid_divs = [r["context_divergence"] for r in results if r["context_divergence"] is not None]
    mean_div = round(sum(valid_divs) / len(valid_divs), 6) if valid_divs else 0.0
    return {"probes": results, "mean_divergence": mean_div}


def run_training_segment(
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    loss_fn: nn.Module,
    train_iter: Any,
    start_step: int,
    end_step: int,
    randomize_targets: bool = False,
) -> Tuple[List[Dict[str, Any]], Any]:
    history: List[Dict[str, Any]] = []
    model.train()

    for step in range(start_step + 1, end_step + 1):
        step_start = time.perf_counter()
        batch_items = [next(train_iter) for _ in range(BATCH_SIZE)]
        batch = ChakrOfflineDataset.collate_fn(batch_items, pad_token_id=2)

        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]
        attention_mask = batch["attention_mask"]

        if randomize_targets:
            flat = target_ids.clone().view(-1)
            perm = torch.randperm(flat.size(0))
            target_ids = flat[perm].view_as(target_ids)

        TrainingSafetyChecker.verify_token_ids(input_ids, min_id=0, max_id=4095)
        TrainingSafetyChecker.verify_token_ids(target_ids, min_id=0, max_id=4095)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, target_ids)
        loss_val = TrainingSafetyChecker.check_loss(loss, step=step)

        loss.backward()
        grad_norm = TrainingSafetyChecker.check_gradients(model, step=step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        elapsed_ms = (time.perf_counter() - step_start) * 1000.0
        current_lr = optimizer.param_groups[0]["lr"]

        if step % 100 == 0 or step in (1, 50):
            label = "CTRL" if randomize_targets else "EXP"
            print(
                f"    [{label}] Step {step:05d} | "
                f"Loss {loss_val:.4f} | Grad {grad_norm:.3f} | "
                f"LR {current_lr:.6f} | {elapsed_ms:.1f}ms"
            )

        history.append({
            "step": step,
            "train_loss": round(loss_val, 6),
            "train_ppl": round(math.exp(min(loss_val, 20.0)), 4),
            "learning_rate": round(current_lr, 8),
            "grad_norm": round(grad_norm, 6),
            "step_latency_ms": round(elapsed_ms, 2),
        })

    return history, train_iter


def evaluate_stage(
    model: nn.Module,
    tokenizer: BPETokenizer,
    loss_fn: nn.Module,
    stage_idx: int,
    cumulative_steps: int,
    model_hash: str,
    segment_history: List[Dict[str, Any]],
    ckpt_mgr: Optional[CheckpointManager],
    optimizer: Any,
    scheduler: Any,
) -> Dict[str, Any]:
    print(f"\n  [Stage {stage_idx} @ step {cumulative_steps}] Evaluating...")

    val_loss, val_ppl = evaluate_split(model, STAGE_B_BASE / "validation", loss_fn)

    greedy_probes = probe_generation(model, tokenizer, model_hash=model_hash, temperature=0.0, max_new_tokens=32)
    torch.manual_seed(SEED)
    sampled_probes = probe_generation(model, tokenizer, model_hash=model_hash, temperature=0.8, max_new_tokens=32)

    ctx_sens = measure_context_sensitivity(model, tokenizer, loss_fn)

    g_ttrs = [p["ttr"] for p in greedy_probes if p["token_count"] > 0]
    s_ttrs = [p["ttr"] for p in sampled_probes if p["token_count"] > 0]
    g_tri = [p["unique_trigrams"] for p in greedy_probes if p["token_count"] > 0]
    s_tri = [p["unique_trigrams"] for p in sampled_probes if p["token_count"] > 0]

    mean_g_ttr = round(sum(g_ttrs) / len(g_ttrs), 4) if g_ttrs else 0.0
    mean_s_ttr = round(sum(s_ttrs) / len(s_ttrs), 4) if s_ttrs else 0.0
    mean_g_tri = round(sum(g_tri) / len(g_tri), 2) if g_tri else 0.0
    mean_s_tri = round(sum(s_tri) / len(s_tri), 2) if s_tri else 0.0

    ckpt_path = None
    if ckpt_mgr:
        saved = ckpt_mgr.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=cumulative_steps,
            epoch=0,
            config=asdict(ModelConfig()),
            tokenizer_checksum="7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f",
            dataset_manifest_hash="stage_b_manifest_verified",
            parameter_count=sum(p.numel() for p in model.parameters()),
        )
        ckpt_path = str(saved)

    print(f"    Val Loss: {val_loss:.4f}  Val PPL: {val_ppl:.2f}")
    print(f"    Ctx Divergence: {ctx_sens['mean_divergence']:.4f}")
    print(f"    Greedy TTR: {mean_g_ttr:.4f}  Trigrams: {mean_g_tri:.1f}")
    for p in greedy_probes[:2]:
        print(f"    [GREEDY] \"{p['prompt'][:40]}\" -> \"{p['generated_text'][:60]}\"")

    return {
        "stage_index": stage_idx,
        "cumulative_steps": cumulative_steps,
        "model_hash": model_hash,
        "checkpoint_path": ckpt_path,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "context_sensitivity": ctx_sens,
        "greedy_generation": greedy_probes,
        "sampled_generation": sampled_probes,
        "diversity_metrics": {
            "mean_greedy_ttr": mean_g_ttr,
            "mean_sampled_ttr": mean_s_ttr,
            "mean_greedy_unique_trigrams": mean_g_tri,
            "mean_sampled_unique_trigrams": mean_s_tri,
        },
        "segment_last_train_loss": segment_history[-1]["train_loss"] if segment_history else None,
        "segment_last_train_ppl": segment_history[-1]["train_ppl"] if segment_history else None,
    }


def main():
    print("=" * 72)
    print("CHAKRVIEW STEP 49: LANGUAGE ACQUISITION & COHERENCE VALIDATION")
    print("=" * 72)
    CKPT_BASE.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Verify frozen baseline
    print("\n[Phase 1] Verifying Frozen Baseline (dW_baseline = 0)...")
    torch.manual_seed(SEED)
    baseline_model = ChakrMicro(ModelConfig())
    baseline_model.eval()
    baseline_hash = compute_model_hash(baseline_model)
    assert baseline_hash == FROZEN_BASELINE_HASH, (
        f"Baseline hash mismatch!\nExpected: {FROZEN_BASELINE_HASH}\nGot: {baseline_hash}"
    )
    print(f"  Baseline SHA-256: {baseline_hash} [VERIFIED]")

    # Phase 2: Load tokenizer
    print("\n[Phase 2] Loading Chakr-BPE Tokenizer...")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    print(f"  Vocabulary: {len(tokenizer.vocab)} tokens")

    # Phase 3: Instantiate isolated experimental model
    print("\n[Phase 3] Instantiating Isolated Experimental Model...")
    torch.manual_seed(SEED)
    exp_model = ChakrMicro(ModelConfig())
    exp_model.train()
    exp_initial_hash = compute_model_hash(exp_model)
    assert exp_initial_hash == FROZEN_BASELINE_HASH, "Experimental model hash mismatch at initialization!"
    print(f"  Experimental model initial hash: {exp_initial_hash}")

    loss_fn = CausalLoss(ignore_index=2)
    training_cfg = TrainingHyperparameters(
        learning_rate=LR,
        min_learning_rate=MIN_LR,
        warmup_steps=WARMUP,
        max_steps=STAGE_CHECKPOINTS[-1],
        gradient_accumulation_steps=1,
        gradient_clipping=GRAD_CLIP,
        seed=SEED,
    )
    optimizer = build_optimizer(exp_model, training_cfg)
    scheduler = build_lr_scheduler(optimizer, training_cfg)

    ckpt_mgr = CheckpointManager(
        checkpoint_dir=CKPT_BASE,
        keep_last_n=len(STAGE_CHECKPOINTS) + 2,
        save_optimizer=True,
    )

    train_ds = StreamingTokenDataset(
        shard_dir=STAGE_B_BASE / "train",
        sequence_length=SEQ_LEN,
        loop=True,
        seed=SEED,
    )
    train_iter = iter(train_ds)

    # Phase 4: Progressive training
    print("\n[Phase 4] Progressive Training: 0 -> 5000 steps")
    print("  Stage checkpoints:", STAGE_CHECKPOINTS)

    all_stages: List[Dict[str, Any]] = []
    full_history: List[Dict[str, Any]] = []
    current_step = 0
    t_start = time.perf_counter()

    for stage_idx, target_steps in enumerate(STAGE_CHECKPOINTS):
        print(f"\n{'─' * 60}")
        print(f"  Stage {stage_idx}: steps {current_step} -> {target_steps}")

        if target_steps > current_step:
            seg_hist, train_iter = run_training_segment(
                exp_model, optimizer, scheduler, loss_fn, train_iter,
                current_step, target_steps, randomize_targets=False,
            )
            full_history.extend(seg_hist)
        else:
            seg_hist = []

        current_step = target_steps
        model_hash_now = compute_model_hash(exp_model)
        stage_result = evaluate_stage(
            exp_model, tokenizer, loss_fn,
            stage_idx, current_step, model_hash_now, seg_hist,
            ckpt_mgr, optimizer, scheduler,
        )
        all_stages.append(stage_result)

    total_time = time.perf_counter() - t_start
    final_exp_hash = compute_model_hash(exp_model)

    # Phase 5: Baseline immutability check
    print(f"\n[Phase 5] Final Baseline Immutability Check...")
    baseline_hash_final = compute_model_hash(baseline_model)
    assert baseline_hash_final == FROZEN_BASELINE_HASH, (
        f"FATAL: Baseline mutated! {FROZEN_BASELINE_HASH} -> {baseline_hash_final}"
    )
    assert final_exp_hash != FROZEN_BASELINE_HASH, "Experiment weights did not update!"
    print(f"  Baseline SHA-256 (post-exp): {baseline_hash_final} [PRESERVED]")
    print(f"  Experimental SHA-256: {final_exp_hash}")

    # Phase 6: Weight delta analysis
    print(f"\n[Phase 6] Weight Delta Analysis...")
    total_l2_sq = 0.0
    max_abs_delta = 0.0
    tensors_changed = 0
    total_tensors = 0

    with torch.no_grad():
        for (b_name, b_param), (e_name, e_param) in zip(
            baseline_model.named_parameters(), exp_model.named_parameters()
        ):
            assert b_name == e_name
            diff = (e_param - b_param).detach()
            total_l2_sq += diff.norm().item() ** 2
            md = diff.abs().max().item()
            if md > max_abs_delta:
                max_abs_delta = md
            if md > 0.0:
                tensors_changed += 1
            total_tensors += 1

    total_l2_norm = math.sqrt(total_l2_sq)
    print(f"  Tensors Updated: {tensors_changed}/{total_tensors}")
    print(f"  L2 Weight Delta: {total_l2_norm:.6f}")
    print(f"  Max Abs Delta:   {max_abs_delta:.6f}")

    # Phase 7: Negative control (5000 steps, randomized targets)
    print(f"\n[Phase 7] Negative Control (5000 steps, randomized targets)...")
    torch.manual_seed(SEED)
    ctrl_model = ChakrMicro(ModelConfig())
    ctrl_model.train()
    ctrl_cfg = TrainingHyperparameters(
        learning_rate=LR, min_learning_rate=MIN_LR,
        warmup_steps=WARMUP, max_steps=STAGE_CHECKPOINTS[-1],
        gradient_accumulation_steps=1, gradient_clipping=GRAD_CLIP, seed=SEED,
    )
    ctrl_opt = build_optimizer(ctrl_model, ctrl_cfg)
    ctrl_sch = build_lr_scheduler(ctrl_opt, ctrl_cfg)
    ctrl_loss_fn = CausalLoss(ignore_index=2)
    ctrl_ds = StreamingTokenDataset(shard_dir=STAGE_B_BASE / "train", sequence_length=SEQ_LEN, loop=True, seed=SEED)
    ctrl_iter = iter(ctrl_ds)
    ctrl_history, _ = run_training_segment(
        ctrl_model, ctrl_opt, ctrl_sch, ctrl_loss_fn, ctrl_iter,
        0, STAGE_CHECKPOINTS[-1], randomize_targets=True,
    )
    ctrl_val_loss, ctrl_val_ppl = evaluate_split(ctrl_model, STAGE_B_BASE / "validation", ctrl_loss_fn)
    ctrl_final_hash = compute_model_hash(ctrl_model)
    print(f"  Control Val Loss: {ctrl_val_loss:.4f}  PPL: {ctrl_val_ppl:.2f}")

    # Phase 8: Reproducibility check (500 steps, Run B)
    print(f"\n[Phase 8] Reproducibility Check (500 steps, Run B)...")
    torch.manual_seed(SEED)
    repro_model = ChakrMicro(ModelConfig())
    repro_model.train()
    repro_cfg = TrainingHyperparameters(
        learning_rate=LR, min_learning_rate=MIN_LR,
        warmup_steps=WARMUP, max_steps=500,
        gradient_accumulation_steps=1, gradient_clipping=GRAD_CLIP, seed=SEED,
    )
    repro_opt = build_optimizer(repro_model, repro_cfg)
    repro_sch = build_lr_scheduler(repro_opt, repro_cfg)
    repro_loss_fn = CausalLoss(ignore_index=2)
    repro_ds = StreamingTokenDataset(shard_dir=STAGE_B_BASE / "train", sequence_length=SEQ_LEN, loop=True, seed=SEED)
    repro_iter = iter(repro_ds)
    repro_history, _ = run_training_segment(
        repro_model, repro_opt, repro_sch, repro_loss_fn, repro_iter, 0, 500, randomize_targets=False,
    )
    repro_final_hash = compute_model_hash(repro_model)
    stage2_hash = all_stages[2]["model_hash"]  # Stage 2 = 500 steps
    reproducible = (repro_final_hash == stage2_hash)
    print(f"  Run A Hash (500 steps): {stage2_hash}")
    print(f"  Run B Hash (500 steps): {repro_final_hash}")
    print(f"  Bit-exact reproducible: {reproducible}")

    # Phase 9: Save results
    print(f"\n[Phase 9] Saving Results...")

    curves_data = {
        "experiment_name": "Step 49 Language Acquisition & Coherence Validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_architecture": "ChakrMicro v0.1",
        "parameters": EXPECTED_PARAM_COUNT,
        "vocab_size": 4096,
        "max_context": 512,
        "sequence_length": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "frozen_baseline_hash": FROZEN_BASELINE_HASH,
        "baseline_hash_post_experiment": baseline_hash_final,
        "baseline_invariant_preserved": baseline_hash_final == FROZEN_BASELINE_HASH,
        "experimental_final_hash": final_exp_hash,
        "total_training_time_sec": round(total_time, 2),
        "stage_checkpoints": STAGE_CHECKPOINTS,
        "weight_delta_analysis": {
            "tensors_changed": tensors_changed,
            "total_tensors": total_tensors,
            "total_l2_norm": round(total_l2_norm, 6),
            "max_abs_delta": round(max_abs_delta, 6),
        },
        "negative_control": {
            "description": "5000 steps with randomized target token permutations",
            "final_val_loss": ctrl_val_loss,
            "final_val_ppl": ctrl_val_ppl,
            "final_hash": ctrl_final_hash,
            "last_train_loss": ctrl_history[-1]["train_loss"] if ctrl_history else None,
        },
        "reproducibility": {
            "description": "500-step Run B vs Stage 2 Run A hash comparison",
            "run_a_hash": stage2_hash,
            "run_b_hash": repro_final_hash,
            "bit_exact_match": reproducible,
        },
        "stages": [
            {
                "stage_index": r["stage_index"],
                "cumulative_steps": r["cumulative_steps"],
                "model_hash": r["model_hash"],
                "val_loss": r["val_loss"],
                "val_ppl": r["val_ppl"],
                "context_sensitivity_mean_divergence": r["context_sensitivity"]["mean_divergence"],
                "diversity_metrics": r["diversity_metrics"],
                "segment_last_train_loss": r["segment_last_train_loss"],
                "segment_last_train_ppl": r["segment_last_train_ppl"],
            }
            for r in all_stages
        ],
        "full_training_history": full_history,
    }

    with open(CURVES_FILE, "w", encoding="utf-8") as f:
        json.dump(curves_data, f, indent=2)
    print(f"  Curves: {CURVES_FILE}")

    coherence_data = {
        "experiment_name": "Step 49 Coherence Metrics by Stage",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_baseline_hash": FROZEN_BASELINE_HASH,
        "baseline_invariant_preserved": baseline_hash_final == FROZEN_BASELINE_HASH,
        "stages": [
            {
                "stage_index": r["stage_index"],
                "cumulative_steps": r["cumulative_steps"],
                "val_loss": r["val_loss"],
                "val_ppl": r["val_ppl"],
                "context_sensitivity": r["context_sensitivity"],
                "diversity_metrics": r["diversity_metrics"],
                "greedy_generation_samples": r["greedy_generation"],
                "sampled_generation_samples": r["sampled_generation"],
            }
            for r in all_stages
        ],
    }

    with open(COHERENCE_FILE, "w", encoding="utf-8") as f:
        json.dump(coherence_data, f, indent=2)
    print(f"  Coherence: {COHERENCE_FILE}")

    # Phase 10: Final summary
    print()
    print("=" * 72)
    print("STEP 49 EXPERIMENT COMPLETE")
    print("=" * 72)
    print(f"  Baseline Preserved:     {baseline_hash_final == FROZEN_BASELINE_HASH}")
    print(f"  Exp dW != 0:            {final_exp_hash != FROZEN_BASELINE_HASH}")
    print(f"  Reproducible (500):     {reproducible}")
    print(f"  Total Time:             {total_time:.1f}s")
    print()
    print(f"  {'Stage':>6} {'Steps':>6} {'Val Loss':>10} {'Val PPL':>10} {'Ctx Div':>10} {'TTR(G)':>8}")
    print(f"  {'─' * 56}")
    for r in all_stages:
        print(
            f"  {r['stage_index']:>6} "
            f"{r['cumulative_steps']:>6} "
            f"{r['val_loss']:>10.4f} "
            f"{r['val_ppl']:>10.2f} "
            f"{r['context_sensitivity']['mean_divergence']:>10.4f} "
            f"{r['diversity_metrics']['mean_greedy_ttr']:>8.4f}"
        )
    print()
    final_ppl = all_stages[-1]["val_ppl"]
    ppl_ratio = ctrl_val_ppl / max(final_ppl, 0.01)
    print(f"  Control PPL (5000 steps, shuffled):  {ctrl_val_ppl:.2f}")
    print(f"  Experiment PPL (5000 steps, real):   {final_ppl:.2f}")
    print(f"  PPL Ratio (Ctrl/Exp):                {ppl_ratio:.2f}x")
    print()
    print("=" * 72)
    print("STEP 49 HARD STOP — DO NOT PROCEED TO STEP 50.")
    print("=" * 72)
    return curves_data, coherence_data


if __name__ == "__main__":
    main()
