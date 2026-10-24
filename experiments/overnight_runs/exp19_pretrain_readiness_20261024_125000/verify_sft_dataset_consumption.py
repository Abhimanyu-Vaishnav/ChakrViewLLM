import json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.training.sft_engine import SFTDataset
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.generate_exp18_datasets import Exp18DatasetGenerator
from experiments.overnight_runs.exp19_pretrain_readiness_20261024_125000.diversification_transform import Exp19ResponseDiversifier

AUDIT_DIR = Path(__file__).resolve().parent
TEMP_DATA_DIR = AUDIT_DIR / "temp_data_verify"
TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Generate splits and apply diversification
gen = Exp18DatasetGenerator()
splits = gen.generate_all_splits()
diversifier = Exp19ResponseDiversifier()
trans_train, _ = diversifier.transform_split("train", splits["train"])

train_file = TEMP_DATA_DIR / "exp19_train_debiased_isolated.jsonl"
with open(train_file, "w", encoding="utf-8") as f:
    for r in trans_train:
        f.write(json.dumps(r) + "\n")

# 2. Load with canonical SFTDataset
tok_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
tokenizer, _ = load_tokenizer_artifacts(str(tok_dir))

dataset = SFTDataset(
    jsonl_path=train_file,
    tokenizer=tokenizer,
    max_seq_len=512,
    allow_exploratory_opt_in=True,
    enforce_trust_gate=True,
    is_training=True,
    strip_prompt_delimiters=False,
)

print(f"SFTDataset loaded {len(dataset.records)} records from {train_file.name}")
assert len(dataset.records) == 1200

# 3. Verify loaded records match trans_train exactly
for orig_trans, loaded in zip(trans_train, dataset.records):
    assert orig_trans["id"] == loaded["id"]
    assert orig_trans["response"] == loaded["response"]
    assert orig_trans["prompt"] == loaded["prompt"]

print("Bitwise record identity confirmed: SFTDataset consumes the transformed records without any code modification!")

# 4. Clean up temporary verify folder
import shutil
shutil.rmtree(TEMP_DATA_DIR, ignore_errors=True)
print("Temporary verification files cleaned up.")
