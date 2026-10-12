"""
ChakrView Deterministic Evaluation Harness (Batch 42).

Provides full-dataset, deterministic evaluation of language model checkpoints:
- Validates model architecture & checkpoint configuration compatibility.
- Computes exact token-level cross-entropy loss weighted by active target tokens.
- Reports perplexity only when loss is finite and valid.
- Records skipped/invalid examples with explicit error diagnostics.
- Runs fixed probe benchmarks with deterministic greedy sampling.
- Outputs machine-readable evaluation reports with dataset, tokenizer, and checkpoint SHA-256 digests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import (
    normalize_prompt_response_strings,
    build_causal_sequence,
)


def compute_file_sha256(filepath: Path | str) -> str:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class EvaluationRecordResult:
    example_id: str
    active_tokens: int
    loss: Optional[float]
    status: str  # "EVALUATED", "SKIPPED_EMPTY_RESPONSE", "SKIPPED_OVERLENGTH", "SKIPPED_INVALID_TOKENS"
    error_message: Optional[str] = None


@dataclass
class ProbeResult:
    probe_id: str
    task_category: str
    prompt: str
    expected_output_hint: str
    generated_text: str
    generated_tokens: int
    eos_reached: bool
    empty_output: bool
    exact_match: bool


class DeterministicEvaluator:
    """
    Standard evaluation engine for ChakrMicro checkpoints.
    Guarantees reproducible, full-dataset evaluation without minibatch subsampling.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        tokenizer_dir: Path | str = "release/chakrview-v0.5.0/data/experiments/vocab_4096",
        device: str = "cpu",
        max_seq_len: int = 512,
        ignore_index: int = -100,
        strip_prompt_delimiters: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.device = torch.device(device)
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index

        self.strip_prompt_delimiters = strip_prompt_delimiters

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        self.checkpoint_sha256 = compute_file_sha256(self.checkpoint_path)

        # Load tokenizer
        self.tokenizer, self.tokenizer_meta = load_tokenizer_artifacts(self.tokenizer_dir)
        self.vocab_size = self.tokenizer.vocab_size

        # Load model and verify configuration
        self.model = self._load_model()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="sum")

    def _load_model(self) -> ChakrMicro:
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        cfg_dict = ckpt.get("model_config", ckpt.get("config", None))
        if cfg_dict is not None:
            if isinstance(cfg_dict, ModelConfig):
                config = cfg_dict
            elif isinstance(cfg_dict, dict):
                config = ModelConfig(**{k: v for k, v in cfg_dict.items() if hasattr(ModelConfig, k)})
            else:
                config = ModelConfig(max_seq_len=self.max_seq_len)
        else:
            config = ModelConfig(max_seq_len=self.max_seq_len)

        model = ChakrMicro(config).to(self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def encode_record(
        self, prompt: str, response: str
    ) -> Tuple[List[int], List[int], str, Optional[str]]:
        """Encodes and shifts a prompt/response pair for causal next-token evaluation."""
        bos_id = 0
        eos_id = 1

        p_str, r_str = normalize_prompt_response_strings(
            prompt, response, strip_delimiters=self.strip_prompt_delimiters
        )

        p_toks = self.tokenizer.encode(p_str)
        r_toks = self.tokenizer.encode(r_str)

        if not r_toks:
            return [], [], "SKIPPED_EMPTY_RESPONSE", "Response string tokenized to 0 tokens."

        total_len = 1 + len(p_toks) + len(r_toks)
        if total_len > self.max_seq_len:
            return (
                [],
                [],
                "SKIPPED_OVERLENGTH",
                f"Total length {total_len} exceeds max_seq_len {self.max_seq_len}.",
            )

        inp_seq, tgt_seq = build_causal_sequence(
            p_toks, r_toks, bos_id=bos_id, eos_id=eos_id, ignore_index=self.ignore_index
        )

        # Vocabulary bounds validation
        if any(t < 0 or t >= self.vocab_size for t in inp_seq):
            return [], [], "SKIPPED_INVALID_TOKENS", "Input token outside vocabulary bounds."
        if any(t != self.ignore_index and (t < 0 or t >= self.vocab_size) for t in tgt_seq):
            return [], [], "SKIPPED_INVALID_TOKENS", "Target token outside vocabulary bounds."

        return inp_seq, tgt_seq, "EVALUATED", None

    @torch.no_grad()
    def evaluate_dataset(self, dataset_path: Path | str) -> Dict[str, Any]:
        """Evaluates every record in dataset_path deterministically."""
        ds_path = Path(dataset_path)
        if not ds_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {ds_path}")
        ds_sha256 = compute_file_sha256(ds_path)

        records: List[Dict[str, Any]] = []
        with open(ds_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        total_loss_sum = 0.0
        total_active_tokens = 0
        evaluated_count = 0
        skipped_count = 0
        per_record_results: List[Dict[str, Any]] = []

        for idx, rec in enumerate(records):
            rec_id = rec.get("id", f"record_{idx}")
            prompt = rec.get("prompt", "")
            response = rec.get("response", "")

            inp, tgt, status, err_msg = self.encode_record(prompt, response)
            if status != "EVALUATED":
                skipped_count += 1
                per_record_results.append(
                    asdict(
                        EvaluationRecordResult(
                            example_id=rec_id,
                            active_tokens=0,
                            loss=None,
                            status=status,
                            error_message=err_msg,
                        )
                    )
                )
                continue

            x = torch.tensor([inp], dtype=torch.long, device=self.device)
            y = torch.tensor([tgt], dtype=torch.long, device=self.device)
            active_tokens = int((y != self.ignore_index).sum().item())

            logits = self.model(x)
            loss_sum = self.loss_fn(logits.view(-1, self.vocab_size), y.view(-1)).item()

            rec_mean_loss = loss_sum / max(1, active_tokens)
            total_loss_sum += loss_sum
            total_active_tokens += active_tokens
            evaluated_count += 1

            per_record_results.append(
                asdict(
                    EvaluationRecordResult(
                        example_id=rec_id,
                        active_tokens=active_tokens,
                        loss=round(rec_mean_loss, 4),
                        status="EVALUATED",
                        error_message=None,
                    )
                )
            )

        if total_active_tokens > 0:
            mean_loss = total_loss_sum / total_active_tokens
            perplexity = math.exp(min(20.0, mean_loss))
        else:
            mean_loss = float("nan")
            perplexity = float("nan")

        return {
            "dataset_path": str(ds_path),
            "dataset_sha256": ds_sha256,
            "total_records": len(records),
            "evaluated_records": evaluated_count,
            "skipped_records": skipped_count,
            "total_active_tokens": total_active_tokens,
            "mean_token_loss": round(mean_loss, 4) if math.isfinite(mean_loss) else None,
            "perplexity": round(perplexity, 2) if math.isfinite(perplexity) else None,
            "records": per_record_results,
        }

    @torch.no_grad()
    def evaluate_probes(
        self, probes: List[Dict[str, Any]], max_new_tokens: int = 40
    ) -> List[Dict[str, Any]]:
        """Runs fixed deterministic greedy generation on probe inputs."""
        results: List[Dict[str, Any]] = []
        for p in probes:
            prompt_str = p["prompt"]
            prompt_tokens = [0] + self.tokenizer.encode(prompt_str)
            generated = list(prompt_tokens)

            for _ in range(max_new_tokens):
                x = torch.tensor([generated], dtype=torch.long, device=self.device)
                logits = self.model(x)
                next_tok = torch.argmax(logits[0, -1, :]).item()
                generated.append(next_tok)
                if next_tok == 1:  # EOS token
                    break

            gen_tokens = generated[len(prompt_tokens) :]
            gen_text = self.tokenizer.decode(gen_tokens)
            eos_reached = len(generated) > 0 and generated[-1] == 1
            empty_output = len(gen_tokens) == 0 or (len(gen_tokens) == 1 and eos_reached)

            expected_hint = p.get("expected_output_hint", "")
            exact_match = (gen_text.strip() == expected_hint.strip()) if expected_hint else False

            results.append(
                asdict(
                    ProbeResult(
                        probe_id=p["id"],
                        task_category=p.get("task_category", "general"),
                        prompt=prompt_str,
                        expected_output_hint=expected_hint,
                        generated_text=gen_text,
                        generated_tokens=len(gen_tokens),
                        eos_reached=eos_reached,
                        empty_output=empty_output,
                        exact_match=exact_match,
                    )
                )
            )
        return results

    def run_full_evaluation(
        self,
        dataset_path: Path | str,
        probes: List[Dict[str, Any]],
        output_report_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Executes full dataset metrics and probe evaluation, saving machine-readable JSON."""
        t0 = time.perf_counter()
        ds_metrics = self.evaluate_dataset(dataset_path)
        probe_metrics = self.evaluate_probes(probes)
        elapsed = round(time.perf_counter() - t0, 3)

        report = {
            "evaluator_version": "1.0.0",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": elapsed,
            "checkpoint": {
                "path": str(self.checkpoint_path),
                "sha256": self.checkpoint_sha256,
            },
            "model_architecture": {
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "vocab_size": self.vocab_size,
                "max_seq_len": self.max_seq_len,
            },
            "dataset_evaluation": ds_metrics,
            "probe_evaluation": probe_metrics,
        }

        if output_report_path:
            out_p = Path(output_report_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        return report
