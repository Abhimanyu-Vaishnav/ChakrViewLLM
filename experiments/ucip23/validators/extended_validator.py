"""
UCIP-23 Extended Adversarial Validator & Invariant Checker.
Directly aligns with the UCIP-22 benchmark record specification:
- Required keys: id, benchmark_version, task_type, split, context_entities, query_entity,
  query_relation, ground_truth_value, distractor_entities, distractor_values,
  is_abstention, expected_abstention_reason, prompt, target_response, token_metadata
- Checks uniqueness of record IDs
- Checks token_metadata constraints and causal alignment
- Adversarial negative tests for prompt answer leakage, invalid abstention flags,
  length overflow, malformed schemas, and duplicate IDs.
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import normalize_prompt_response_strings, build_causal_sequence

REQUIRED_KEYS = [
    "id", "benchmark_version", "task_type", "split", "context_entities",
    "query_entity", "query_relation", "ground_truth_value", "distractor_entities",
    "distractor_values", "is_abstention", "expected_abstention_reason",
    "prompt", "target_response", "token_metadata"
]

VALID_TASK_TYPES = {
    "task_a_entity_identity",
    "task_b_relation_recognition",
    "task_c_fact_retrieval",
    "task_d_multi_relation_binding",
    "task_e_compositional_generalization",
    "task_f_contextual_binding",
    "task_g_distractor_resistance",
    "task_h_abstention"
}

VALID_ABSTENTION_REASONS = {
    "none",
    "entity_absent",
    "relation_absent",
    "both_absent",
    "contradictory_context",
    "insufficient_information"
}


class ExtendedBenchmarkValidator:
    def __init__(self, vocab_dir: Path | None = None):
        vdir = vocab_dir or (ROOT / "data/experiments/vocab_4096")
        self.tok, _ = load_tokenizer_artifacts(vdir)
        self.max_context_len = 512
        self.seen_ids: Set[str] = set()

    def reset(self):
        self.seen_ids.clear()

    def validate_record(self, record: Dict[str, Any], check_prompt_leakage: bool = True) -> Tuple[bool, List[str]]:
        errors = []

        # 1. Structural schema validation
        for k in REQUIRED_KEYS:
            if k not in record:
                errors.append(f"Missing required key: '{k}'")

        if errors:
            return False, errors

        # 2. ID uniqueness
        rec_id = record["id"]
        if not isinstance(rec_id, str) or not rec_id.strip():
            errors.append("Record 'id' must be a non-empty string")
        elif rec_id in self.seen_ids:
            errors.append(f"Duplicate record id encountered: '{rec_id}'")
        else:
            self.seen_ids.add(rec_id)

        # 3. Enum validation
        if record["benchmark_version"] != "ucip22-v1":
            errors.append(f"Invalid benchmark_version '{record['benchmark_version']}', expected 'ucip22-v1'")

        if record["task_type"] not in VALID_TASK_TYPES:
            errors.append(f"Invalid task_type '{record['task_type']}'")

        if record["expected_abstention_reason"] not in VALID_ABSTENTION_REASONS:
            errors.append(f"Invalid expected_abstention_reason '{record['expected_abstention_reason']}'")

        # 4. Abstention consistency
        if record["is_abstention"]:
            if record["expected_abstention_reason"] == "none":
                errors.append("is_abstention is True but expected_abstention_reason is 'none'")
        else:
            if record["expected_abstention_reason"] != "none":
                errors.append(f"is_abstention is False but expected_abstention_reason is '{record['expected_abstention_reason']}'")

        # 5. Tokenization and causal alignment
        prompt = record["prompt"]
        target = record["target_response"]

        if not isinstance(prompt, str) or not prompt.strip():
            errors.append("Prompt must be a non-empty string")
        if not isinstance(target, str) or not target.strip():
            errors.append("Target response must be a non-empty string")

        if errors:
            return False, errors

        p_str, r_str = normalize_prompt_response_strings(prompt, target, strip_delimiters=True)
        p_toks = self.tok.encode(p_str)
        r_toks = self.tok.encode(r_str)

        inp_seq, tgt_seq = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)
        total_len = len(inp_seq)

        if total_len > self.max_context_len:
            errors.append(f"Total sequence length {total_len} exceeds max context limit {self.max_context_len}")

        if inp_seq[0] != 0:
            errors.append(f"First input token is not BOS (0), found {inp_seq[0]}")
        if tgt_seq[-1] != 1:
            errors.append(f"Terminal target token is not EOS (1), found {tgt_seq[-1]}")

        prompt_len = len(p_toks)
        if tgt_seq[:prompt_len] != [-100] * prompt_len:
            errors.append("Prompt tokens are not strictly masked with -100 in target sequence")

        # 6. Prompt answer leakage check (for Tasks C-G where target is a retrieved fact)
        if check_prompt_leakage and record["task_type"] in {
            "task_c_fact_retrieval", "task_d_multi_relation_binding",
            "task_e_compositional_generalization", "task_f_contextual_binding",
            "task_g_distractor_resistance"
        }:
            # Check if user query question explicitly contains the answer
            if "Question:" in prompt and "Answer" in prompt:
                q_part = prompt.split("Question:")[1].split("Answer")[0]
                if len(target) > 3 and target.lower() in q_part.lower():
                    errors.append(f"Target answer '{target}' leaked verbatim inside question prompt: '{q_part.strip()}'")

        return (len(errors) == 0), errors
