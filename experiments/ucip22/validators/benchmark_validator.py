"""
UCIP-22 Benchmark Record Validator.
Validates candidate benchmark records against tokenizer constraints, context window limits,
and structural compositionality invariants.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import normalize_prompt_response_strings, build_causal_sequence


class UCIP22BenchmarkValidator:
    def __init__(self, vocab_dir: Optional[Path] = None):
        vdir = vocab_dir or (ROOT / "data/experiments/vocab_4096")
        self.tok, _ = load_tokenizer_artifacts(vdir)
        self.max_context_len = 512

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        
        # 1. Required keys
        req_keys = [
            "id", "benchmark_version", "task_type", "split", "context_entities",
            "query_entity", "query_relation", "ground_truth_value", "distractor_entities",
            "distractor_values", "is_abstention", "expected_abstention_reason",
            "prompt", "target_response"
        ]
        for k in req_keys:
            if k not in record:
                errors.append(f"Missing required key: {k}")

        if errors:
            return False, errors

        # 2. Tokenization and sequence length
        p_str, r_str = normalize_prompt_response_strings(
            record["prompt"], record["target_response"], strip_delimiters=True
        )
        p_toks = self.tok.encode(p_str)
        r_toks = self.tok.encode(r_str)
        
        inp_seq, tgt_seq = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)
        
        total_len = len(inp_seq)
        if total_len > self.max_context_len:
            errors.append(f"Total sequence length {total_len} exceeds max context limit {self.max_context_len}")

        if inp_seq[0] != 0:
            errors.append("First input token is not BOS (ID 0)")
        if tgt_seq[-1] != 1:
            errors.append("Terminal target token is not EOS (ID 1)")

        # 3. Masking alignment
        prompt_len = len(p_toks)
        if tgt_seq[:prompt_len] != [-100] * prompt_len:
            errors.append("Prompt targets are not strictly masked with -100")
        if tgt_seq[prompt_len] != r_toks[0]:
            errors.append("First supervised target token does not match first response token")

        # 4. Abstention consistency
        if record["is_abstention"]:
            if record["expected_abstention_reason"] == "none":
                errors.append("is_abstention is True but expected_abstention_reason is 'none'")
        else:
            if record["expected_abstention_reason"] != "none":
                errors.append(f"is_abstention is False but reason is {record['expected_abstention_reason']}")

        # 5. Entity presence in context (if not entity_absent)
        if record["task_type"] in ["task_c_fact_retrieval", "task_d_multi_relation_binding", "task_f_contextual_binding"]:
            q_ent = record["query_entity"]
            if q_ent not in record["context_entities"]:
                errors.append(f"Query entity '{q_ent}' not declared in context_entities")
            if q_ent not in record["prompt"]:
                errors.append(f"Query entity '{q_ent}' surface form missing from prompt string")

        return (len(errors) == 0), errors


def create_exemplar_record() -> Dict[str, Any]:
    return {
        "id": "UCIP22-TASK-D-00001",
        "benchmark_version": "ucip22-v1",
        "task_type": "task_d_multi_relation_binding",
        "split": "dev_compositional_unseen_pair",
        "context_entities": ["NovaTech", "AeroDynamics"],
        "query_entity": "NovaTech",
        "query_relation": "founded_in",
        "ground_truth_value": "2018",
        "distractor_entities": ["AeroDynamics"],
        "distractor_values": ["Bengaluru", "orbital propulsion"],
        "is_abstention": False,
        "expected_abstention_reason": "none",
        "prompt": "User: Records:\n- NovaTech is based in Bengaluru, was founded in 2018, and produces orbital propulsion.\n- AeroDynamics is based in Hyderabad, was founded in 2012, and produces avionics.\nQuestion: In what year was NovaTech founded?\nAnswer concisely:\nAssistant: ",
        "target_response": "2018",
        "token_metadata": {
            "prompt_token_count": 68,
            "response_token_count": 2,
            "total_sequence_length": 71,
            "first_answer_token_id": 409,
            "terminal_token_id": 1,
            "is_within_context_limit": True
        }
    }
