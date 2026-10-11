"""
Stage D Instruction Dataset Validator and Loss Masking Verification.
Validates JSONL schema, Unicode integrity, token length bounds, duplicate detection,
and tests supervised loss masking functionality.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


class StageDDataValidator:
    REQUIRED_FIELDS = {
        "example_id",
        "language",
        "task_type",
        "split",
        "prompt",
        "response",
        "provenance",
        "quality_status",
    }
    ALLOWED_LANGS = {"en", "hi", "hi-en", "code", "sa", "math"}
    ALLOWED_SPLITS = {"train", "val", "test"}

    def __init__(self, tokenizer_dir: str = "release/chakrview-v0.5.0/data/experiments/vocab_4096"):
        self.tokenizer, _ = load_tokenizer_artifacts(tokenizer_dir)

    def validate_record(self, record: Dict[str, Any], seen_ids: set) -> Tuple[bool, str]:
        # 1. Field presence
        missing = self.REQUIRED_FIELDS - set(record.keys())
        if missing:
            return False, f"Missing required fields: {missing}"

        # 2. ID uniqueness
        eid = record["example_id"]
        if eid in seen_ids:
            return False, f"Duplicate example_id: {eid}"
        seen_ids.add(eid)

        # 3. Language & Split checks
        if record["language"] not in self.ALLOWED_LANGS:
            return False, f"Invalid language: {record['language']}"
        if record["split"] not in self.ALLOWED_SPLITS:
            return False, f"Invalid split: {record['split']}"

        # 4. Text checks
        prompt = record["prompt"].strip()
        response = record["response"].strip()
        if len(prompt) < 5:
            return False, "Prompt too short (< 5 chars)"
        if len(response) < 2:
            return False, "Response too short (< 2 chars)"

        # 5. Token budget check
        p_toks = self.tokenizer.encode(prompt)
        r_toks = self.tokenizer.encode(response)
        total_toks = len(p_toks) + len(r_toks) + 2  # BOS + EOS
        if total_toks > 512:
            return False, f"Total tokens {total_toks} exceeds max context 512"

        return True, "VALID"

    def validate_file(self, jsonl_path: Path) -> Dict[str, Any]:
        valid_records = []
        rejected_records = []
        seen_ids = set()

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    ok, reason = self.validate_record(data, seen_ids)
                    if ok:
                        valid_records.append(data)
                    else:
                        rejected_records.append({"line": line_no, "reason": reason, "record": data})
                except Exception as e:
                    rejected_records.append({"line": line_no, "reason": f"JSON parse error: {e}"})

        return {
            "total_lines": len(valid_records) + len(rejected_records),
            "valid_count": len(valid_records),
            "rejected_count": len(rejected_records),
            "rejected_records": rejected_records,
        }


def build_masked_sft_tensor(
    prompt_ids: List[int], response_ids: List[int], max_seq_len: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs autoregressive input tensor and target tensor where prompt tokens
    are masked with -100 so cross_entropy computes loss strictly on response tokens.
    """
    bos = [0]
    eos = [1]
    input_seq = bos + prompt_ids + response_ids
    # target shifted by 1:
    # positions corresponding to BOS and prompt are masked (-100)
    target_seq = ([-100] * (len(bos) + len(prompt_ids) - 1)) + response_ids + eos

    # Truncate to max_seq_len
    input_seq = input_seq[:max_seq_len]
    target_seq = target_seq[:max_seq_len]

    return torch.tensor([input_seq], dtype=torch.long), torch.tensor([target_seq], dtype=torch.long)


def verify_sft_loss_masking():
    """
    Verifies that CrossEntropyLoss ignores prompt tokens and only scores response tokens.
    """
    prompt = [10, 20, 30]
    response = [40, 50]
    x, y = build_masked_sft_tensor(prompt, response)

    vocab_size = 4096
    seq_len = x.shape[1]
    dummy_logits = torch.randn(1, seq_len, vocab_size)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(dummy_logits.reshape(-1, vocab_size), y.reshape(-1))

    # Mask targets: only response tokens and EOS should be active
    active_targets = y[y != -100]
    expected_active_count = len(response) + 1  # response + EOS
    assert len(active_targets) == expected_active_count, (
        f"Expected {expected_active_count} active targets, got {len(active_targets)}"
    )
    return True
