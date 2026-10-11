"""
Batch 38 Hardened Baseline Evaluation Harness.
Derives metadata dynamically from actual runtime:
- Checkpoint SHA-256 computed from file bytes
- Checkpoint file size and path
- EOS token ID dynamically resolved from tokenizer config/vocab
- Assertion that EOS ID matches expected value
- Model explicitly set to eval() mode
- Deterministic greedy decoding with documented stopping criteria
- Context-length limit enforcement before forward pass
- Full raw token IDs, decoded outputs, stop reasons, and token counts preserved per prompt
- Aggregate metrics computed strictly from saved per-prompt results
- Fails loudly on missing files or configuration inconsistencies
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = str(Path(__file__).resolve().parents[3])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.local_runtime import DEFAULT_TOKENIZER_DIR
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def compute_file_sha256(filepath: str) -> str:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_tokenizer_eos_id(tokenizer_dir: str, expected_eos_id: int = 1) -> int:
    config_path = os.path.join(tokenizer_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Tokenizer config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    special_tokens = cfg.get("special_tokens", {})
    resolved_eos = special_tokens.get("<EOS>")
    if resolved_eos is None:
        raise ValueError(f"Special token <EOS> not found in tokenizer config at {config_path}")
    if resolved_eos != expected_eos_id:
        raise ValueError(
            f"Resolved EOS token ID ({resolved_eos}) does not match expected value ({expected_eos_id})"
        )
    return resolved_eos


class HardenedBaselineEvaluator:
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_dir: str = DEFAULT_TOKENIZER_DIR,
        expected_eos_id: int = 1,
        max_context_length: int = 512,
    ):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.checkpoint_size = os.path.getsize(checkpoint_path)
        self.checkpoint_sha256 = compute_file_sha256(checkpoint_path)

        self.tokenizer_dir = tokenizer_dir
        self.eos_token_id = resolve_tokenizer_eos_id(tokenizer_dir, expected_eos_id=expected_eos_id)
        self.max_context_length = max_context_length

        # Load tokenizer
        tok_obj, _ = load_tokenizer_artifacts(tokenizer_dir)
        self.tokenizer = tok_obj

        # Load checkpoint weights
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = checkpoint_data["model_state_dict"] if "model_state_dict" in checkpoint_data else checkpoint_data

        self.model_config = ModelConfig(
            vocab_size=4096,
            d_model=192,
            n_heads=6,
            hidden_dim=512,
            n_layers=6,
            max_seq_len=max_context_length,
            eos_token_id=self.eos_token_id,
        )
        self.model = ChakrMicro(self.model_config)
        self.model.load_state_dict(model_state)

        # Explicitly set to evaluation mode
        self.model.eval()

    def generate_greedy(
        self,
        prompt_text: str,
        max_new_tokens: int = 32,
    ) -> Dict[str, Any]:
        """
        Deterministic greedy autoregressive decoding.
        Enforces context window before each step.
        """
        prompt_tokens = self.tokenizer.encode(prompt_text)
        # Check context boundary before forward pass
        if len(prompt_tokens) >= self.max_context_length:
            return {
                "prompt_token_ids": prompt_tokens,
                "raw_generated_token_ids": [],
                "first_generated_token_id": None,
                "first_token_is_eos": False,
                "decoded_text": "",
                "generated_token_count": 0,
                "stop_reason": "context_limit_exceeded_at_prompt",
            }

        input_ids = list(prompt_tokens)
        generated: List[int] = []
        stop_reason = "max_tokens"

        with torch.no_grad():
            for step in range(max_new_tokens):
                if len(input_ids) >= self.max_context_length:
                    stop_reason = "context_limit"
                    break

                x = torch.tensor([input_ids], dtype=torch.long)
                logits = self.model(x)  # shape (1, seq_len, vocab_size)
                next_token_logits = logits[0, -1, :]  # shape (vocab_size,)
                next_token_id = int(torch.argmax(next_token_logits).item())

                generated.append(next_token_id)
                input_ids.append(next_token_id)

                if next_token_id == self.eos_token_id:
                    stop_reason = "eos"
                    break

        first_tok = generated[0] if generated else None
        first_is_eos = (first_tok == self.eos_token_id) if first_tok is not None else False
        decoded = self.tokenizer.decode(generated, errors="replace")

        return {
            "prompt_token_ids": prompt_tokens,
            "raw_generated_token_ids": generated,
            "first_generated_token_id": first_tok,
            "first_token_is_eos": first_is_eos,
            "decoded_text": decoded,
            "generated_token_count": len(generated),
            "stop_reason": stop_reason,
        }

    def evaluate_prompt_set(
        self,
        prompt_set: List[Dict[str, str]],
        max_new_tokens: int = 32,
        output_dir: str = "experiments/batch38_audit_hardening/evaluation",
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        os.makedirs(output_dir, exist_ok=True)
        results: List[Dict[str, Any]] = []

        for p_item in prompt_set:
            pid = p_item["id"]
            ptext = p_item["prompt"]
            gen_res = self.generate_greedy(ptext, max_new_tokens=max_new_tokens)
            entry = {
                "prompt_id": pid,
                "prompt": ptext,
                **gen_res,
            }
            results.append(entry)

        # Save per_prompt_results.jsonl
        jsonl_path = os.path.join(output_dir, "per_prompt_results.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Compute aggregate metrics strictly from saved results
        total_prompts = len(results)
        first_eos_count = sum(1 for r in results if r["first_token_is_eos"])
        empty_count = sum(1 for r in results if r["generated_token_count"] == 0)
        avg_tokens = sum(r["generated_token_count"] for r in results) / total_prompts if total_prompts > 0 else 0.0

        stop_dist: Dict[str, int] = {}
        for r in results:
            reason = r["stop_reason"]
            stop_dist[reason] = stop_dist.get(reason, 0) + 1

        summary = {
            "total_evaluated": total_prompts,
            "first_token_eos_count": first_eos_count,
            "first_token_eos_rate": round(first_eos_count / total_prompts, 4) if total_prompts > 0 else 0.0,
            "empty_decoded_count": empty_count,
            "average_generated_tokens": round(avg_tokens, 2),
            "stop_reason_distribution": stop_dist,
        }

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Save run manifest with dynamic properties
        run_manifest = {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_size_bytes": self.checkpoint_size,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_config": {
                "vocab_size": self.model_config.vocab_size,
                "d_model": self.model_config.d_model,
                "n_heads": self.model_config.n_heads,
                "hidden_dim": self.model_config.hidden_dim,
                "n_layers": self.model_config.n_layers,
                "max_seq_len": self.model_config.max_seq_len,
            },
            "tokenizer_directory": str(self.tokenizer_dir).replace("\\", "/"),
            "resolved_eos_id": self.eos_token_id,
            "evaluation_mode": "deterministic_greedy",
            "model_eval_mode_enforced": True,
            "max_new_tokens": max_new_tokens,
            "total_prompts": total_prompts,
            "prompt_set_labels": [p["id"] for p in prompt_set],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        manifest_path = os.path.join(output_dir, "run_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(run_manifest, f, indent=2)

        return summary, results


# Standard diagnostic prompt set (labeled as small diagnostic smoke set)
DIAGNOSTIC_PROMPT_SET = [
    {"id": "VAL-EN-01", "prompt": "How does byte-pair encoding balance sequence length and vocabulary size?"},
    {"id": "VAL-HI-01", "prompt": "कंप्यूटर विज्ञान में टोकनाइज़र का मुख्य कार्य क्या है?"},
    {"id": "VAL-HG-01", "prompt": "ChakrView model ka exact parameter size kitna hai?"},
    {"id": "VAL-CD-01", "prompt": "What is the mathematical definition of RMSNorm?"},
    {"id": "EXP-EN", "prompt": "Explain the transformer architecture."},
    {"id": "EXP-HI", "prompt": "भारत की प्राचीन संस्कृति और ज्ञान परंपरा पर टिप्पणी करें।"},
    {"id": "EXP-HG", "prompt": "ChakrView me CPU inferencing kaise optimize hoti hai?"},
    {"id": "EXP-CD", "prompt": "def scaled_dot_product_attention(q, k, v):"},
    {"id": "EXP-MA", "prompt": "State the formula for cross-entropy loss."},
    {"id": "EXP-MT", "prompt": "User: What is 2+2?\nAssistant: 4\nUser: Add 5 to it.\nAssistant:"},
]


def run_baseline_evaluation() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ckpt_path = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"
    evaluator = HardenedBaselineEvaluator(checkpoint_path=ckpt_path)
    summary, results = evaluator.evaluate_prompt_set(DIAGNOSTIC_PROMPT_SET, max_new_tokens=32)
    return summary, results


if __name__ == "__main__":
    summary, results = run_baseline_evaluation()
    print("Baseline evaluation completed:")
    print(json.dumps(summary, indent=2))
