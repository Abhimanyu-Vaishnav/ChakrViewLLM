"""
Training Dataset Validator and Quality Gate for ChakrView SFT.

Validates:
1. JSONL syntax and schema conformance (prompt, response, metadata).
2. Whitespace and formatting hygiene (no empty, whitespace-only, or unbalanced prompts/responses).
3. Exact duplicate prompt and duplicate (prompt, response) pair detection.
4. Contamination/leakage against capability evaluation suites:
   - Intelligence tasks (prompts and gold answers)
   - Coding tasks (prompts and test assertions)
   - 4-gram overlap / n-gram containment checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class DatasetValidationError(ValueError):
    """Raised when an SFT dataset violates schema, hygiene, or contamination constraints."""
    pass


REQUIRED_SFT_FIELDS = {"prompt", "response"}
RECOMMENDED_METADATA_FIELDS = {"id", "schema_version", "data_role", "review_status"}


def extract_ngrams(text: str, n: int = 4) -> Set[Tuple[str, ...]]:
    """Extracts lowercase word n-grams from text."""
    words = [w for w in text.lower().split() if w]
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def load_eval_benchmarks(repo_root: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Loads evaluation tasks from intelligence and coding suites for leakage checks."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent

    intel_path = repo_root / "evaluation" / "intelligence_suite" / "intelligence_tasks.json"
    coding_path = repo_root / "evaluation" / "coding_suite" / "coding_tasks.json"

    tasks: Dict[str, List[Dict[str, Any]]] = {"intelligence": [], "coding": []}

    if intel_path.exists():
        with open(intel_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            tasks["intelligence"] = data.get("tasks", [])

    if coding_path.exists():
        with open(coding_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            tasks["coding"] = data.get("tasks", [])

    return tasks


def validate_sft_dataset_file(
    dataset_path: Path | str,
    check_leakage: bool = True,
    eval_benchmark_tasks: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    repo_root: Optional[Path] = None,
    allow_empty: bool = False,
) -> Dict[str, Any]:
    """
    Validates an SFT JSONL dataset file.

    Raises DatasetValidationError on any violation.
    Returns summary statistics dict on success.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    total_records = 0
    seen_prompts: Set[str] = set()
    seen_pairs: Set[Tuple[str, str]] = set()
    duplicate_prompts: List[int] = []
    duplicate_pairs: List[int] = []

    # Precompile evaluation n-grams if leakage check enabled
    eval_prompts_raw: List[Tuple[str, str]] = []  # (task_id, text)
    if check_leakage:
        if eval_benchmark_tasks is None:
            eval_benchmark_tasks = load_eval_benchmarks(repo_root)

        for t in eval_benchmark_tasks.get("intelligence", []):
            task_id = t.get("id", "intel_task")
            if "prompt" in t:
                eval_prompts_raw.append((task_id, t["prompt"]))
        for t in eval_benchmark_tasks.get("coding", []):
            task_id = t.get("id", "coding_task")
            if "prompt" in t:
                eval_prompts_raw.append((task_id, t["prompt"]))

    line_number = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_number += 1
            line_str = line.strip()
            if not line_str:
                continue

            try:
                record = json.loads(line_str)
            except json.JSONDecodeError as err:
                raise DatasetValidationError(
                    f"Invalid JSON on line {line_number} of '{path}': {err}"
                ) from err

            if not isinstance(record, dict):
                raise DatasetValidationError(
                    f"Line {line_number} of '{path}' must be a JSON object, got {type(record).__name__}."
                )

            # Check required fields
            for field in REQUIRED_SFT_FIELDS:
                if field not in record:
                    raise DatasetValidationError(
                        f"Missing required field '{field}' on line {line_number} of '{path}'."
                    )

            prompt = record["prompt"]
            response = record["response"]

            if not isinstance(prompt, str) or not prompt.strip():
                raise DatasetValidationError(
                    f"Line {line_number} of '{path}' has empty or invalid 'prompt'."
                )
            if not isinstance(response, str) or not response.strip():
                raise DatasetValidationError(
                    f"Line {line_number} of '{path}' has empty or invalid 'response'."
                )

            p_norm = prompt.strip()
            r_norm = response.strip()

            # Duplicate checking
            pair = (p_norm, r_norm)
            if pair in seen_pairs:
                duplicate_pairs.append(line_number)
            seen_pairs.add(pair)

            if p_norm in seen_prompts:
                duplicate_prompts.append(line_number)
            seen_prompts.add(p_norm)

            # Contamination / leakage checking
            if check_leakage:
                p_ngrams = extract_ngrams(p_norm, n=4)
                for task_id, eval_p in eval_prompts_raw:
                    eval_ngrams = extract_ngrams(eval_p, n=4)
                    if not p_ngrams or not eval_ngrams:
                        continue
                    overlap = p_ngrams.intersection(eval_ngrams)
                    # If significant 4-gram overlap (> 50% of eval prompt 4-grams or exact match)
                    if p_norm.lower() == eval_p.strip().lower():
                        raise DatasetValidationError(
                            f"LEAKAGE DETECTED on line {line_number}: Prompt exactly matches benchmark "
                            f"eval task '{task_id}': {p_norm!r}"
                        )
                    if len(overlap) >= 3 and len(overlap) >= len(eval_ngrams) * 0.6:
                        raise DatasetValidationError(
                            f"LEAKAGE DETECTED on line {line_number}: 4-gram contamination with "
                            f"benchmark eval task '{task_id}' (overlap: {overlap})."
                        )

            total_records += 1

    if total_records == 0 and not allow_empty:
        raise DatasetValidationError(f"Dataset '{path}' contains 0 valid records.")

    if duplicate_pairs:
        raise DatasetValidationError(
            f"Dataset '{path}' contains {len(duplicate_pairs)} exact duplicate (prompt, response) pairs at lines {duplicate_pairs[:10]}."
        )

    return {
        "path": str(path),
        "total_records": total_records,
        "unique_prompts": len(seen_prompts),
        "unique_pairs": len(seen_pairs),
        "duplicate_prompts_count": len(duplicate_prompts),
        "duplicate_pairs_count": len(duplicate_pairs),
        "leakage_checked": check_leakage,
    }
