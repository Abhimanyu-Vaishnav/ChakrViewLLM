"""
Phase 17 & 19: Real Data Smoke-Training and Infrastructure Performance Benchmark.

Verifies end-to-end pipeline:
Raw Text -> Tokenizer -> Binary Shards -> Streaming Loader -> Collator -> Model -> Loss -> Optimizer -> Checkpoint
and records overhead measurements.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.config import (
    PretrainingConfig,
    TrainingHyperparameters,
    DataConfig,
    CheckpointConfig,
    EvaluationConfig,
)
from chakrview.training.sharding import ShardWriter, verify_shard_integrity
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.collator import CausalLanguageModelingCollator
from chakrview.training.trainer import Trainer
from chakrview.training.monitoring import ResourceMonitor


def run_real_data_smoke_test():
    print("=" * 70)
    print("CHAKRVIEW STEP 5 — REAL DATA SMOKE TEST & BENCHMARK")
    print("=" * 70)

    # 1. Load Step 3 BPE Tokenizer
    tok_dir = Path("data/experiments/vocab_4096")
    print(f"\n[1/7] Loading Tokenizer from {tok_dir}...")
    t0 = time.perf_counter()
    tokenizer, tok_cfg = load_tokenizer_artifacts(tok_dir)
    t_tok_load = time.perf_counter() - t0
    print(f"      Tokenizer loaded: Vocab Size = {tokenizer.vocab_size}, Time = {t_tok_load*1000:.2f} ms")

    # 2. Read Real Corpus Text
    hindi_path = Path("data/processed/train/hindi.txt")
    english_path = Path("data/processed/train/english.txt")
    print(f"\n[2/7] Reading Real Corpus Subsets ({hindi_path.name}, {english_path.name})...")
    with open(hindi_path, "r", encoding="utf-8") as f:
        hindi_lines = f.readlines()[:50]
    with open(english_path, "r", encoding="utf-8") as f:
        english_lines = f.readlines()[:50]

    all_lines = hindi_lines + english_lines
    print(f"      Selected {len(all_lines)} lines ({len(hindi_lines)} Hindi, {len(english_lines)} English)")

    # 3. Tokenize and Shard Data
    tokenized_dir = Path("data/tokenized")
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    train_dir = tokenized_dir / "train"
    val_dir = tokenized_dir / "val"

    print(f"\n[3/7] Tokenizing and Writing Shards to {tokenized_dir}...")
    t_tok_start = time.perf_counter()
    
    # Train shard writer
    train_writer = ShardWriter(
        output_dir=tokenized_dir,
        split_name="train",
        vocab_size=4096,
        max_tokens_per_shard=25000,
        tokenizer_checksum=tok_cfg.get("checksums", {}).get("vocab_sha256", ""),
    )
    # Val shard writer
    val_writer = ShardWriter(
        output_dir=tokenized_dir,
        split_name="val",
        vocab_size=4096,
        max_tokens_per_shard=25000,
        tokenizer_checksum=tok_cfg.get("checksums", {}).get("vocab_sha256", ""),
    )

    total_tokens_tokenized = 0
    # Split 80/20 train/val
    split_idx = int(len(all_lines) * 0.8)
    for i, line in enumerate(all_lines):
        line = line.strip()
        if not line:
            continue
        tokens = tokenizer.encode(line)
        total_tokens_tokenized += len(tokens)
        if i < split_idx:
            train_writer.add_document(tokens)
        else:
            val_writer.add_document(tokens)

    train_meta = train_writer.close()
    val_meta = val_writer.close()
    t_tokenization = time.perf_counter() - t_tok_start

    print(f"      Tokenized {total_tokens_tokenized} tokens in {t_tokenization*1000:.2f} ms ({total_tokens_tokenized/t_tokenization:.0f} tokens/sec)")
    print(f"      Train tokens: {train_meta['total_tokens']}, Shards: {train_meta['shard_count']}")
    print(f"      Val tokens:   {val_meta['total_tokens']}, Shards: {val_meta['shard_count']}")

    # Verify integrity
    assert verify_shard_integrity(train_dir) is True
    assert verify_shard_integrity(val_dir) is True
    print("      Shard checksums verified against metadata.json: OK")

    # 4. Create Streaming Dataset & Collator
    print("\n[4/7] Setting Up Streaming DataLoader and Collator...")
    seq_len = 64  # Test sequence length
    batch_size = 2

    train_dataset = StreamingTokenDataset(
        shard_dir=train_dir,
        sequence_length=seq_len,
        loop=True,
        drop_remainder=True,
    )
    val_dataset = StreamingTokenDataset(
        shard_dir=val_dir,
        sequence_length=seq_len,
        loop=False,
        drop_remainder=False,
    )

    collator = CausalLanguageModelingCollator(pad_token_id=2, max_context=512)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collator)

    # 5. Measure Step-by-Step Infrastructure Overhead (Phase 19)
    print("\n[5/7] Measuring Component Overheads (Single Batch)...")
    data_iter = iter(train_loader)

    # Data loading / batch creation time
    t0 = time.perf_counter()
    batch = next(data_iter)
    t_batch = time.perf_counter() - t0

    cfg = PretrainingConfig(
        model=ModelConfig(),
        training=TrainingHyperparameters(
            batch_size=batch_size,
            max_steps=5,
            learning_rate=1e-3,
            warmup_steps=2,
            gradient_accumulation_steps=1,
            gradient_clipping=1.0,
        ),
        data=DataConfig(
            train_path=str(train_dir),
            validation_path=str(val_dir),
            sequence_length=seq_len,
        ),
        checkpoint=CheckpointConfig(
            directory="checkpoints/smoke_test",
            save_interval=5,
            keep_last_n=2,
        ),
        evaluation=EvaluationConfig(
            eval_interval=5,
            eval_batches=2,
            eval_on_start=True,
        ),
    )

    trainer = Trainer(
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Forward time
    t0 = time.perf_counter()
    logits = trainer.model(batch["input_ids"], attention_mask=batch["attention_mask"])
    t_fwd = time.perf_counter() - t0

    # Loss time
    t0 = time.perf_counter()
    loss = trainer.loss_fn(logits, batch["target_ids"])
    t_loss = time.perf_counter() - t0

    # Backward time
    t0 = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t0

    # Optimizer step time
    t0 = time.perf_counter()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    t_opt = time.perf_counter() - t0

    # Checkpoint time
    t0 = time.perf_counter()
    ckpt_path = trainer.checkpoint_manager.save(
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        step=1,
    )
    t_ckpt = time.perf_counter() - t0

    print(f"      Batch loading time:   {t_batch*1000:7.2f} ms")
    print(f"      Forward pass time:    {t_fwd*1000:7.2f} ms")
    print(f"      Loss compute time:    {t_loss*1000:7.2f} ms")
    print(f"      Backward pass time:   {t_bwd*1000:7.2f} ms")
    print(f"      Optimizer step time:  {t_opt*1000:7.2f} ms")
    print(f"      Checkpoint save time: {t_ckpt*1000:7.2f} ms")

    # 6. Execute 5-Step Smoke Training Run
    print("\n[6/7] Running 5-Step Smoke Training Session...")
    smoke_trainer = Trainer(
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    result = smoke_trainer.train()
    history = result["history"]

    for entry in history:
        val_str = f" | Val Loss: {entry.get('val_loss', 'N/A')}" if "val_loss" in entry else ""
        print(f"      Step {entry['step']:2d} | Train Loss: {entry['train_loss']:.4f} | LR: {entry['learning_rate']:.6f} | PPL: {entry['train_perplexity']} | {entry['tokens_per_sec']:.0f} tok/s{val_str}")

    # 7. Check Resource Snapshot & Final Checkpoint
    print("\n[7/7] Validating Checkpoint & Resource Usage...")
    monitor = ResourceMonitor()
    res = monitor.get_snapshot()
    print(f"      Process RSS RAM: {res['process_ram_mb']:.1f} MB | System RAM: {res['system_ram_percent']}% | CPU: {res['cpu_percent']}%")

    final_ckpt = Path(result["final_checkpoint"])
    assert final_ckpt.is_file(), "Final checkpoint was not written!"
    loaded_ckpt = trainer.checkpoint_manager.load(final_ckpt)
    assert loaded_ckpt["step"] == 5
    print(f"      Final checkpoint verified: {final_ckpt.name} (Step {loaded_ckpt['step']})")

    print("\n" + "=" * 70)
    print("REAL DATA SMOKE TEST COMPLETE: PIPELINE 100% OPERATIONAL")
    print("=" * 70)


if __name__ == "__main__":
    run_real_data_smoke_test()
