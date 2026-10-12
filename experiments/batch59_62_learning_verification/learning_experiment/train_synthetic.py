import json
from pathlib import Path
from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine

repo_root = Path(".")
train_file = Path("experiments/batch59_62_learning_verification/data/train_synthetic.jsonl")
val_file = Path("experiments/batch59_62_learning_verification/data/val_synthetic.jsonl")
out_dir = Path("experiments/batch59_62_learning_verification/learning_experiment/run_50steps")
base_ckpt = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")

config = SFTConfig(
    repo_root=repo_root,
    train_file=train_file,
    val_file=val_file,
    output_dir=out_dir,
    base_checkpoint=base_ckpt,
    steps=50,
    batch_size=2,
    learning_rate=1e-4,
    min_learning_rate=1e-5,
    weight_decay=0.01,
    grad_clip=1.0,
    val_interval=10,
    save_interval=50,
    seed=42,
    device="cpu",
    tokenizer_dir="release/chakrview-v0.5.0/data/experiments/vocab_4096",
    allow_exploratory_opt_in=False,
    enforce_trust_gate=True,
)

print("Initializing SFTExperimentEngine on synthetic arithmetic dataset...")
engine = SFTExperimentEngine(config)
print("Starting training...")
summary = engine.train()
print("Training complete!")

summary_path = out_dir / "training_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Summary written to {summary_path}")
print(f"Initial Val Loss: {summary.get('initial_val_loss')}")
print(f"Final Val Loss: {summary.get('final_val_loss')}")
print(f"Final Train Loss: {summary.get('final_train_loss')}")
