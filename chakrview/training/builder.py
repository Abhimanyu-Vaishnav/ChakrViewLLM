"""
Deterministic Dataset Builder for Offline Neural Training (Step 22).

Transforms approved learning experience records into reproducible,
bounded training datasets:
1. Filters strictly for TRAINING_APPROVED records (rejects/quarantines invalid records).
2. Tokenizes using frozen ChakrView BPETokenizer.
3. Enforces 512-token sequence limit (logs explicit truncation metadata if truncated).
4. Constructs shifted causal next-token prediction targets.
5. Computes cryptographically deterministic dataset fingerprints.
6. Produces deterministic train/validation/test partitions.
7. Generates verifiable dataset manifests.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from chakrview.intelligence.contracts import LearningRecord, LearningRecordStatus
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.training.contract import (
    TrainingExample,
    TrainingDatasetManifest,
    TokenizerFingerprint,
    TrainingRecordEligibilityError,
    validate_learning_record_for_training,
)


class ChakrOfflineDataset(Dataset):
    """
    PyTorch Dataset wrapping validated offline training examples.
    """

    def __init__(
        self,
        examples: List[TrainingExample],
        max_seq_len: int = 512,
        pad_token_id: int = 2,
    ) -> None:
        self.examples = list(examples)
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        seq = ex.sequence_token_ids

        # Ensure sequence length does not exceed max_seq_len
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]

        # Shifted causal next-token pair
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
            "example_id": ex.example_id,
        }

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, Any]],
        pad_token_id: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """
        Pad a batch of variable-length sequences to uniform maximum length in the batch.
        """
        batch_size = len(batch)
        max_len = max(b["input_ids"].size(0) for b in batch)

        padded_inputs = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        padded_targets = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        padded_masks = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, b in enumerate(batch):
            seq_len = b["input_ids"].size(0)
            padded_inputs[i, :seq_len] = b["input_ids"]
            padded_targets[i, :seq_len] = b["target_ids"]
            padded_masks[i, :seq_len] = 1

        return {
            "input_ids": padded_inputs,
            "target_ids": padded_targets,
            "attention_mask": padded_masks,
        }


class DatasetBuilder:
    """
    Deterministic dataset compiler from verified learning experience.
    """

    MAX_SEQ_LEN: int = 512
    BOS_ID: int = 0
    EOS_ID: int = 1
    PAD_ID: int = 2

    def __init__(self, tokenizer: BPETokenizer) -> None:
        self.tokenizer = tokenizer
        self.tok_fingerprint = TokenizerFingerprint.from_tokenizer(tokenizer)

    def build_from_records(
        self,
        records: List[LearningRecord],
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: int = 42,
        dataset_version: str = "v1.0",
    ) -> Tuple[ChakrOfflineDataset, ChakrOfflineDataset, TrainingDatasetManifest]:
        """
        Compile approved learning records into deterministic train/validation datasets.
        """
        examples: List[TrainingExample] = []

        # Deterministic sorting by record_id to guarantee reproducible processing order
        sorted_records = sorted(records, key=lambda r: r.record_id)

        for rec in sorted_records:
            # 1. Eligibility validation (fails closed on non-TRAINING_APPROVED)
            validate_learning_record_for_training(rec)

            # 2. Tokenization
            inp_tokens = self.tokenizer.encode(rec.input_context)
            tgt_tokens = self.tokenizer.encode(rec.target_output)

            # 3. Sequence assembly: [BOS] + input_tokens + target_tokens + [EOS]
            full_seq = [self.BOS_ID] + inp_tokens + tgt_tokens + [self.EOS_ID]
            total_tokens = len(full_seq)

            is_truncated = False
            truncation_notes = None

            if total_tokens > self.MAX_SEQ_LEN:
                is_truncated = True
                truncation_notes = (
                    f"Sequence truncated from {total_tokens} to {self.MAX_SEQ_LEN} tokens "
                    f"(input_tokens={len(inp_tokens)}, target_tokens={len(tgt_tokens)})."
                )
                full_seq = full_seq[:self.MAX_SEQ_LEN]

            ex = TrainingExample(
                example_id=f"tx_{rec.record_id}_{hashlib.sha256(rec.record_id.encode()).hexdigest()[:6]}",
                source_record_id=rec.record_id,
                owner_id=rec.owner_id,
                session_id=rec.session_id,
                input_text=rec.input_context,
                target_text=rec.target_output,
                input_token_ids=inp_tokens,
                target_token_ids=tgt_tokens,
                sequence_token_ids=full_seq,
                input_token_count=len(inp_tokens),
                target_token_count=len(tgt_tokens),
                total_token_count=len(full_seq),
                is_truncated=is_truncated,
                truncation_notes=truncation_notes,
                task_type=rec.task_type,
                quality_score=rec.quality_score,
                tokenizer_fingerprint=self.tok_fingerprint.fingerprint_hash,
                dataset_version=dataset_version,
                created_at=time.time(),
                source_provenance=rec.source_provenance,
            )
            examples.append(ex)

        if not examples:
            raise ValueError("No valid training examples generated from approved records.")

        # 4. Compute deterministic dataset fingerprint
        hasher = hashlib.sha256()
        for ex in examples:
            hasher.update(f"{ex.example_id}:{len(ex.sequence_token_ids)}:".encode("utf-8"))
            hasher.update(str(ex.sequence_token_ids[:32]).encode("utf-8"))
        dataset_fingerprint = hasher.hexdigest()[:16]

        # 5. Deterministic partition using seeded PRNG
        shuffled_indices = list(range(len(examples)))
        rng = random.Random(seed)
        rng.shuffle(shuffled_indices)

        val_count = max(1, int(len(examples) * val_ratio)) if val_ratio > 0 and len(examples) > 1 else 0
        test_count = int(len(examples) * test_ratio) if test_ratio > 0 else 0
        train_count = len(examples) - val_count - test_count

        train_indices = shuffled_indices[:train_count]
        val_indices = shuffled_indices[train_count:train_count + val_count]

        train_examples = [examples[i] for i in train_indices]
        val_examples = [examples[i] for i in val_indices] if val_indices else [examples[train_indices[0]]]

        manifest = TrainingDatasetManifest(
            manifest_id=f"manifest_{dataset_fingerprint}",
            dataset_version=dataset_version,
            dataset_fingerprint=dataset_fingerprint,
            tokenizer_fingerprint=self.tok_fingerprint.fingerprint_hash,
            total_examples=len(examples),
            total_tokens=sum(e.total_token_count for e in examples),
            train_examples_count=len(train_examples),
            val_examples_count=len(val_examples),
            test_examples_count=test_count,
            max_sequence_length=self.MAX_SEQ_LEN,
            provenance_summary={
                "seed": seed,
                "owners_represented": list(set(e.owner_id for e in examples)),
                "task_types": list(set(e.task_type for e in examples)),
                "truncated_examples_count": sum(1 for e in examples if e.is_truncated),
            },
        )

        train_ds = ChakrOfflineDataset(train_examples, max_seq_len=self.MAX_SEQ_LEN, pad_token_id=self.PAD_ID)
        val_ds = ChakrOfflineDataset(val_examples, max_seq_len=self.MAX_SEQ_LEN, pad_token_id=self.PAD_ID)

        return train_ds, val_ds, manifest

    def build_from_jsonl(
        self,
        jsonl_path: Path | str,
        owner_id: str = "default_user",
        session_id: str = "default_session",
        val_ratio: float = 0.2,
        seed: int = 42,
        dataset_version: str = "v1.0",
    ) -> Tuple[ChakrOfflineDataset, ChakrOfflineDataset, TrainingDatasetManifest]:
        """Read exported JSONL file and compile training datasets."""
        file_path = Path(jsonl_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"JSONL dataset file not found: {jsonl_path}")

        records: List[LearningRecord] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                data = json.loads(raw)
                rec = LearningRecord(
                    record_id=data.get("record_id", f"jsonl_rec_{line_no}"),
                    owner_id=data.get("owner_id", owner_id),
                    session_id=data.get("session_id", session_id),
                    input_context=data.get("input_context", ""),
                    target_output=data.get("target_output", ""),
                    task_type=data.get("task_type", "general"),
                    quality_score=data.get("quality_score", 1.0),
                    status=LearningRecordStatus.TRAINING_APPROVED,  # Exported records are approved
                )
                records.append(rec)

        return self.build_from_records(
            records=records,
            val_ratio=val_ratio,
            seed=seed,
            dataset_version=dataset_version,
        )

    @staticmethod
    def save_to_disk(
        output_dir: Path | str,
        train_ds: ChakrOfflineDataset,
        val_ds: ChakrOfflineDataset,
        manifest: TrainingDatasetManifest,
    ) -> None:
        """Persist compiled dataset and manifest to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        manifest_file = out_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        def _dump_ds(ds: ChakrOfflineDataset, filename: str):
            with open(out_path / filename, "w", encoding="utf-8") as f:
                for ex in ds.examples:
                    f.write(json.dumps(ex.to_dict()) + "\n")

        _dump_ds(train_ds, "train_examples.jsonl")
        _dump_ds(val_ds, "val_examples.jsonl")

    @classmethod
    def load_from_disk(
        cls,
        input_dir: Path | str,
    ) -> Tuple[ChakrOfflineDataset, ChakrOfflineDataset, TrainingDatasetManifest]:
        """Load compiled dataset and manifest from disk."""
        in_path = Path(input_dir)
        manifest_file = in_path / "manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Dataset manifest not found in {input_dir}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_dict = json.load(f)
        manifest = TrainingDatasetManifest.from_dict(manifest_dict)

        def _load_ds(filename: str) -> List[TrainingExample]:
            ex_list = []
            file_p = in_path / filename
            if file_p.is_file():
                with open(file_p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            ex_list.append(TrainingExample.from_dict(json.loads(line)))
            return ex_list

        train_examples = _load_ds("train_examples.jsonl")
        val_examples = _load_ds("val_examples.jsonl")

        train_ds = ChakrOfflineDataset(train_examples, max_seq_len=manifest.max_sequence_length)
        val_ds = ChakrOfflineDataset(val_examples, max_seq_len=manifest.max_sequence_length)

        return train_ds, val_ds, manifest
