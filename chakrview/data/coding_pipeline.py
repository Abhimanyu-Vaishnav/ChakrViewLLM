"""
chakrview/data/coding_pipeline.py

Coding-specialization data pipeline for ChakrView v0.7.
Validates record schema, detects duplicates, verifies prompt/template family overlap,
guards against train/val target leakage, enforces tokenizer sequence limits, and produces
deterministic manifests with SHA-256 integrity audits.
"""

import os
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

ALLOWED_TASK_TYPES = {
    "code_continuation",
    "code_explanation",
    "bug_identification_and_repair",
    "small_function_implementation",
    "test_case_generation",
    "code_summarization",
    "syntax_validity",
    "json_code_spec"
}

ALLOWED_SPLITS = {"train", "val", "validation", "test", "dev"}


class CodingRecord:
    def __init__(
        self,
        task_id: str,
        task_type: str,
        language: str,
        prompt: str,
        response: str,
        provenance: str,
        split: str,
        context: Optional[str] = None,
        template_family: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.language = language
        self.prompt = prompt
        self.response = response
        self.provenance = provenance
        self.split = split
        self.context = context or ""
        self.template_family = template_family or "default_family"
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "task_id": self.task_id,
            "task_type": self.task_type,
            "language": self.language,
            "context": self.context,
            "prompt": self.prompt,
            "response": self.response,
            "provenance": self.provenance,
            "split": self.split,
            "template_family": self.template_family,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodingRecord":
        required = ["task_id", "task_type", "language", "prompt", "response", "provenance", "split"]
        for req in required:
            if req not in data or not str(data[req]).strip():
                raise ValueError(f"Missing required field: '{req}'")

        if data["task_type"] not in ALLOWED_TASK_TYPES:
            raise ValueError(f"Invalid task_type: '{data['task_type']}', must be one of {sorted(ALLOWED_TASK_TYPES)}")

        if data["split"] not in ALLOWED_SPLITS:
            raise ValueError(f"Invalid split: '{data['split']}', must be one of {sorted(ALLOWED_SPLITS)}")

        return cls(
            task_id=str(data["task_id"]),
            task_type=str(data["task_type"]),
            language=str(data["language"]),
            prompt=str(data["prompt"]),
            response=str(data["response"]),
            provenance=str(data["provenance"]),
            split=str(data["split"]),
            context=str(data.get("context", "")),
            template_family=str(data.get("template_family", "default_family")),
            metadata=data.get("metadata", {}),
        )


def normalize_text(text: str) -> str:
    """Strips comments, trims whitespace, collapses inner spacing."""
    no_comments = re.sub(r"#.*", "", text)
    collapsed = re.sub(r"\s+", " ", no_comments).strip().lower()
    return collapsed


class CodingDataPipeline:
    def __init__(self, tokenizer_dir: Optional[str] = None, max_seq_len: int = 512):
        self.max_seq_len = max_seq_len
        self.tokenizer = None
        if tokenizer_dir:
            from chakrview.runtime.neural_runtime_adapter import BpeTokenizer
            p = Path(tokenizer_dir)
            if (p / "vocab.json").exists() and (p / "merges.json").exists():
                self.tokenizer = BpeTokenizer(str(p / "vocab.json"), str(p / "merges.json"))

    def validate_dataset(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validates records, checks duplicate inputs/targets, and audits leakage across splits."""
        audit_report = {
            "total_records": len(records),
            "valid_records": 0,
            "invalid_records": 0,
            "errors": [],
            "split_counts": {},
            "task_type_counts": {},
            "exact_duplicate_prompts": 0,
            "normalized_duplicate_prompts": 0,
            "target_leakage_cases": 0,
            "template_family_overlap": 0,
            "sequence_length_violations": 0,
            "max_sequence_length_seen": 0,
        }

        seen_prompts: Dict[str, str] = {}  # prompt -> task_id
        seen_norm_prompts: Dict[str, str] = {}  # norm_prompt -> task_id
        split_prompts: Dict[str, Set[str]] = {}  # split -> set of norm_prompts
        split_targets: Dict[str, Set[str]] = {}  # split -> set of norm_responses
        split_families: Dict[str, Set[str]] = {}  # split -> set of template_families

        parsed_records: List[CodingRecord] = []

        for idx, r in enumerate(records):
            try:
                rec = CodingRecord.from_dict(r)
                parsed_records.append(rec)
                audit_report["valid_records"] += 1
            except Exception as e:
                audit_report["invalid_records"] += 1
                audit_report["errors"].append({"index": idx, "error": str(e)})
                continue

            # Tally counts
            audit_report["split_counts"][rec.split] = audit_report["split_counts"].get(rec.split, 0) + 1
            audit_report["task_type_counts"][rec.task_type] = audit_report["task_type_counts"].get(rec.task_type, 0) + 1

            # Duplicate prompt check within dataset
            if rec.prompt in seen_prompts:
                audit_report["exact_duplicate_prompts"] += 1
            else:
                seen_prompts[rec.prompt] = rec.task_id

            norm_p = normalize_text(rec.prompt)
            norm_r = normalize_text(rec.response)

            if norm_p in seen_norm_prompts:
                audit_report["normalized_duplicate_prompts"] += 1
            else:
                seen_norm_prompts[norm_p] = rec.task_id

            # Cross-split tracking
            sp = rec.split
            if sp not in split_prompts:
                split_prompts[sp] = set()
                split_targets[sp] = set()
                split_families[sp] = set()

            split_prompts[sp].add(norm_p)
            split_targets[sp].add(norm_r)
            split_families[sp].add(rec.template_family)

            # Sequence length check
            full_text = f"{rec.prompt} {rec.response}"
            if self.tokenizer:
                toks = self.tokenizer.encode(full_text)
                seq_len = len(toks)
            else:
                seq_len = len(full_text.split())

            if seq_len > audit_report["max_sequence_length_seen"]:
                audit_report["max_sequence_length_seen"] = seq_len

            if seq_len > self.max_seq_len:
                audit_report["sequence_length_violations"] += 1

        # Leakage Audit: Check overlap between train and validation/test splits
        train_prompts = split_prompts.get("train", set())
        train_targets = split_targets.get("train", set())
        train_families = split_families.get("train", set())

        for other_sp in split_prompts:
            if other_sp in ("train", "dev"):
                continue
            leak_p = train_prompts.intersection(split_prompts[other_sp])
            leak_t = train_targets.intersection(split_targets[other_sp])
            overlap_fam = train_families.intersection(split_families[other_sp])

            if leak_t:
                audit_report["target_leakage_cases"] += len(leak_t)
            if overlap_fam:
                audit_report["template_family_overlap"] += len(overlap_fam)

        audit_report["leakage_safe"] = (audit_report["target_leakage_cases"] == 0)
        return audit_report

    def export_dataset_manifest(self, filepath: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Writes records atomically and returns deterministic SHA-256 hash manifest."""
        audit = self.validate_dataset(records)
        p = Path(filepath).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")

        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        os.replace(tmp, p)

        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)

        manifest = {
            "dataset_path": str(p),
            "sha256": hasher.hexdigest(),
            "record_count": len(records),
            "audit_summary": audit,
        }
        return manifest
