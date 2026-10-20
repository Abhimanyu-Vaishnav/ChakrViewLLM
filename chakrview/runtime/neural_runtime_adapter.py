"""
ChakrView Dedicated Neural Runtime Adapter.
Location: chakrview/runtime/neural_runtime_adapter.py

Responsibilities:
1. Encapsulate the neural model using the existing NeuralInferenceContract.
2. Load selected checkpoint and tokenizer with strict SHA-256 pre-verification.
3. Cache the loaded inference contract per adapter instance (lazy singleton on CPU).
4. Strictly enforce CPU inference, bounds validation, and Delta-W = 0 immutability.
5. Provide structured, truthful execution metadata without exposing private credentials.
6. Fail closed on missing, mismatched, corrupted, or unsupported artifacts.
7. Support modular prompt conditioning ('raw_completion', 'plain_instruction', 'context_qa').
8. Support bounded decoding controls: greedy, sampling (temperature, top_k, top_p, seed), repetition penalty, and stop tokens.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import torch

from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.runtime.neural_inference_contract import (
    NeuralInferenceContract,
    NeuralInferencePayload,
    NeuralInferenceConfig,
    NeuralInferenceOutput,
    InferenceStopReason,
    NeuralContractError,
)
from chakrview.runtime.sampling import Sampler, SamplingConfig
from chakrview.runtime.prompt_conditioning import (
    format_conditioned_prompt,
    SUPPORTED_PROMPT_MODES,
    DEFAULT_PROMPT_MODE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# Allow external canonical checkpoint root via environment variable, defaulting to standard sibling or project path
CHAKRVIEW_SOURCE_DIR = os.environ.get("CHAKRVIEW_SOURCE_DIR", "")
SOURCE_REPO_ROOT = Path(CHAKRVIEW_SOURCE_DIR) if CHAKRVIEW_SOURCE_DIR else (REPO_ROOT.parent / "ChakrView")

# Canonical Checkpoints & Expected File SHA-256 Allowlist
CANONICAL_CHECKPOINTS_ALLOWLIST: Dict[str, Tuple[Path, str, str]] = {
    "step_10000": (
        REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt",
        "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
        "Canonical Pretrained (Step 10,000)",
    ),
    "source_step_10000": (
        SOURCE_REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt",
        "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
        "Canonical Pretrained (Step 10,000 Source Path)",
    ),
    "cpt_best": (
        REPO_ROOT / "experiments" / "batch2141_2180" / "cpt" / "seed_42" / "cpt_best_checkpoint.pt",
        "d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3",
        "Seed-42 Domain CPT Candidate",
    ),
    "step_48_sft": (
        REPO_ROOT / "artifacts" / "step48" / "run_seed_42" / "checkpoint_0000100.pt",
        "58e2afa42e3d43af3810f3a835e8288aeae71f47df96487a913de1d15e9c2932",
        "Experimental Early SFT (Step 48)",
    ),
}

DEFAULT_CHECKPOINT_KEY = "step_10000"
DEFAULT_TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
EXPECTED_TOKENIZER_MERGES_HASH = "7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f"
EXPECTED_TOKENIZER_FILES_HASHES = {
    "config.json": "0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589",
    "vocab.json": "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a",
    "merges.json": "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7",
}


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Calculates SHA-256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class NeuralArtifactIntegrityError(Exception):
    """Raised when an artifact hash, configuration, or allowlist check fails."""
    pass


class NeuralRuntimeAdapter:
    """
    Dedicated runtime adapter managing neural model loading, hash pre-verification,
    CPU inference caching, prompt conditioning, decoding controls, and structured attribution.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        tokenizer_dir: Optional[Union[str, Path]] = None,
        prompt_mode: str = DEFAULT_PROMPT_MODE,
        max_new_tokens: int = 32,
        decoding: str = "greedy",
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        seed: Optional[int] = 42,
        stop_on_eos: bool = False,
    ) -> None:
        self.requested_checkpoint_path = checkpoint_path
        self.tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else DEFAULT_TOKENIZER_DIR
        
        # Validation of prompt conditioning mode
        if prompt_mode not in SUPPORTED_PROMPT_MODES:
            raise ValueError(
                f"Unsupported prompt mode '{prompt_mode}'. Supported: {list(SUPPORTED_PROMPT_MODES)}"
            )
        self.prompt_mode = prompt_mode
        
        # Validation of decoding options
        if decoding not in ("greedy", "sampling"):
            raise ValueError(f"Unsupported decoding mode '{decoding}'. Must be 'greedy' or 'sampling'.")
        self.decoding = decoding

        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        if temperature < 0.0:
            raise ValueError(f"temperature cannot be negative, got {temperature}")
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError(f"top_p must be in (0.0, 1.0], got {top_p}")
        if repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {repetition_penalty}")

        self.max_new_tokens = min(max_new_tokens, 64)  # hard cap at 64 tokens
        self.temperature = temperature
        self.top_k = max(0, top_k)
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = max(0, no_repeat_ngram_size)
        self.seed = seed
        self.stop_on_eos = stop_on_eos

        # Internal state
        self._contract: Optional[NeuralInferenceContract] = None
        self._sampler = Sampler()
        self._checkpoint_resolved_path: Optional[Path] = None
        self._checkpoint_label: Optional[str] = None
        self._checkpoint_sha256: Optional[str] = None
        self._bound_weight_digest: Optional[str] = None
        self._tokenizer: Optional[BPETokenizer] = None
        self._cold_start_load_ms: float = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._contract is not None

    def _resolve_and_verify_checkpoint(self) -> Tuple[Path, str, str]:
        """Resolves checkpoint path and verifies file SHA-256 against allowlist."""
        target_path: Optional[Path] = None
        matched_expected_hash: Optional[str] = None
        label: str = "Custom / Unlabeled Checkpoint"

        if self.requested_checkpoint_path is None:
            target_path, matched_expected_hash, label = CANONICAL_CHECKPOINTS_ALLOWLIST[DEFAULT_CHECKPOINT_KEY]
            if not target_path.is_file() and CANONICAL_CHECKPOINTS_ALLOWLIST["source_step_10000"][0].is_file():
                target_path, matched_expected_hash, label = CANONICAL_CHECKPOINTS_ALLOWLIST["source_step_10000"]
        else:
            cand_p = Path(self.requested_checkpoint_path).resolve()
            # Check against canonical allowlist
            for key, (allow_p, allow_hash, allow_label) in CANONICAL_CHECKPOINTS_ALLOWLIST.items():
                if cand_p == allow_p.resolve() or str(self.requested_checkpoint_path) == str(allow_p):
                    target_path = allow_p
                    matched_expected_hash = allow_hash
                    label = allow_label
                    break
                elif str(self.requested_checkpoint_path) == key:
                    target_path = allow_p
                    matched_expected_hash = allow_hash
                    label = allow_label
                    break

            if target_path is None:
                # Disallow unapproved arbitrary paths to preserve security & reproducibility
                raise NeuralArtifactIntegrityError(
                    f"Unsupported or unauthorized checkpoint path: '{self.requested_checkpoint_path}'. "
                    f"Allowed checkpoints: {list(CANONICAL_CHECKPOINTS_ALLOWLIST.keys())}"
                )

        if not target_path.is_file():
            raise FileNotFoundError(f"Selected checkpoint file does not exist: {target_path}")

        # Compute file SHA-256
        actual_hash = compute_file_sha256(target_path)
        if actual_hash != matched_expected_hash:
            raise NeuralArtifactIntegrityError(
                f"Checkpoint SHA-256 mismatch for {target_path.name}: "
                f"expected {matched_expected_hash}, got {actual_hash}"
            )

        return target_path, actual_hash, label

    def _verify_and_load_tokenizer(self) -> BPETokenizer:
        """Verifies tokenizer artifact file hashes and loads BPETokenizer."""
        if not self.tokenizer_dir.is_dir():
            raise FileNotFoundError(f"Tokenizer directory does not exist: {self.tokenizer_dir}")

        for fname, exp_hash in EXPECTED_TOKENIZER_FILES_HASHES.items():
            fpath = self.tokenizer_dir / fname
            if not fpath.is_file():
                raise FileNotFoundError(f"Missing required tokenizer file: {fpath}")
            act_hash = compute_file_sha256(fpath)
            if act_hash != exp_hash:
                raise NeuralArtifactIntegrityError(
                    f"Tokenizer file hash mismatch for {fname}: expected {exp_hash}, got {act_hash}"
                )

        tok, _ = load_tokenizer_artifacts(self.tokenizer_dir)
        if tok.vocab_size != 4096:
            raise NeuralArtifactIntegrityError(
                f"Tokenizer vocab size mismatch: expected 4096, got {tok.vocab_size}"
            )
        return tok

    def initialize(self) -> None:
        """Loads and validates model and tokenizer once per adapter instance."""
        if self._contract is not None:
            return

        t0 = time.perf_counter()
        ckpt_path, ckpt_hash, label = self._resolve_and_verify_checkpoint()
        tokenizer = self._verify_and_load_tokenizer()

        # Load through NeuralInferenceContract.from_checkpoint
        contract = NeuralInferenceContract.from_checkpoint(
            checkpoint_path=ckpt_path,
            tokenizer=tokenizer,
            config=ModelConfig(),
            device=torch.device("cpu"),
        )

        self._contract = contract
        self._tokenizer = tokenizer
        self._checkpoint_resolved_path = ckpt_path
        self._checkpoint_sha256 = ckpt_hash
        self._checkpoint_label = label
        self._bound_weight_digest = contract.expected_weight_hash
        self._cold_start_load_ms = (time.perf_counter() - t0) * 1000.0

    def generate(
        self,
        prompt: str,
        prompt_mode: Optional[str] = None,
        context: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        decoding: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        seed: Optional[int] = None,
        stop_on_eos: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Executes bounded neural generation using cached NeuralInferenceContract and Sampler.
        Returns dictionary containing generated text, token counts, metadata, and attribution.
        """
        if self._contract is None:
            self.initialize()

        assert self._contract is not None
        assert self._tokenizer is not None

        active_mode = prompt_mode or self.prompt_mode
        active_decoding = decoding or self.decoding
        active_max_new = max_new_tokens or self.max_new_tokens
        active_max_new = max(1, min(active_max_new, 64))

        active_temp = self.temperature if temperature is None else max(0.0, temperature)
        if active_decoding == "greedy":
            active_temp = 0.0

        active_top_k = self.top_k if top_k is None else max(0, top_k)
        active_top_p = self.top_p if top_p is None else max(0.001, min(top_p, 1.0))
        active_rep = self.repetition_penalty if repetition_penalty is None else max(1.0, min(repetition_penalty, 10.0))
        active_ngram = self.no_repeat_ngram_size if no_repeat_ngram_size is None else max(0, no_repeat_ngram_size)
        active_seed = self.seed if seed is None else seed
        active_eos_stop = self.stop_on_eos if stop_on_eos is None else stop_on_eos

        # 1. Format prompt with specified conditioning template
        formatted_prompt = format_conditioned_prompt(prompt, prompt_mode=active_mode, context=context)

        # 2. Tokenize prompt and validate context boundary
        prompt_tokens = self._contract.encode(formatted_prompt, add_bos=True, add_eos=False)
        total_horizon = len(prompt_tokens) + active_max_new
        if total_horizon > self._contract.max_context:
            raise NeuralContractError(
                f"Total context horizon {total_horizon} (prompt: {len(prompt_tokens)}, "
                f"new: {active_max_new}) exceeds model ceiling {self._contract.max_context}"
            )

        # 3. Pre-execution Delta-W = 0 verification
        pre_hash = self._contract.compute_weight_hash()
        if pre_hash != self._bound_weight_digest:
            raise NeuralContractError("Pre-execution weight mutation detected.")

        # 4. Decoding loop using Sampler
        t0 = time.perf_counter()
        sampling_cfg = SamplingConfig(
            temperature=active_temp,
            top_k=active_top_k,
            top_p=active_top_p,
            repetition_penalty=active_rep,
            no_repeat_ngram_size=active_ngram,
            seed=active_seed,
        )

        generated_tokens: List[int] = []
        curr_tokens = list(prompt_tokens)
        stop_reason = "MAX_TOKENS"

        with torch.no_grad():
            for step in range(active_max_new):
                tensor_in = torch.tensor([curr_tokens], dtype=torch.long, device=self._contract.device)
                logits = self._contract.model(tensor_in)  # [1, seq_len, vocab_size]
                next_token_logits = logits[0, -1, :]

                if not torch.isfinite(next_token_logits).all():
                    raise NeuralContractError("Model produced non-finite logits (NaN or Inf).")

                next_token = self._sampler.sample(
                    next_token_logits,
                    generated_tokens=generated_tokens,
                    config=sampling_cfg,
                    step=step,
                )

                generated_tokens.append(next_token)
                curr_tokens.append(next_token)

                if active_eos_stop and next_token == 1:  # EOS_ID
                    stop_reason = "STOP_TOKEN"
                    break

        gen_latency = (time.perf_counter() - t0) * 1000.0

        # 5. Post-execution Delta-W = 0 verification
        post_hash = self._contract.compute_weight_hash()
        if post_hash != self._bound_weight_digest:
            raise NeuralContractError("Post-execution weight mutation detected (Delta-W != 0).")

        gen_text = self._contract.decode(generated_tokens)

        return {
            "prompt": prompt,
            "formatted_prompt": formatted_prompt,
            "prompt_mode": active_mode,
            "decoding_mode": active_decoding,
            "generated_text": gen_text,
            "generated_tokens": generated_tokens,
            "prompt_tokens": prompt_tokens,
            "input_token_count": len(prompt_tokens),
            "output_token_count": len(generated_tokens),
            "total_token_count": len(curr_tokens),
            "stop_reason": stop_reason,
            "temperature": active_temp,
            "top_k": active_top_k,
            "top_p": active_top_p,
            "repetition_penalty": active_rep,
            "no_repeat_ngram_size": active_ngram,
            "seed": active_seed,
            "stop_on_eos": active_eos_stop,
            "weight_hash_verified": True,
            "latency_ms": round(gen_latency, 2),
            "cold_start_load_ms": round(self._cold_start_load_ms, 2),
            "checkpoint_name": self._checkpoint_resolved_path.name if self._checkpoint_resolved_path else "unknown",
            "checkpoint_label": self._checkpoint_label,
            "checkpoint_sha256": self._checkpoint_sha256,
            "bound_weight_digest": self._bound_weight_digest,
            "device": "cpu",
        }
