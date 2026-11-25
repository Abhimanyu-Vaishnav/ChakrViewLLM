"""
ChakrView Production 17M Inference Engine.
Mission CV-GENESIS-NR-012.

Features:
- Full support for ChakrGenesis-17M architecture with KV caching or sequence autoregression.
- Configurable sampling parameters: temperature, top_k, top_p, repetition_penalty.
- Generator-based streaming token output with incremental decoding.
- Context window truncation respecting max sequence length.
- Device agility: automatic CUDA fallback to CPU if needed.
- Strict canonical special token alignment (<BOS>=0, <EOS>=1, <PAD>=2).
"""

from __future__ import annotations
import os
import sys
import time
from typing import Generator, Dict, Any, Optional, List, Tuple
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.tokenizer.tokenizer import BPETokenizer


class ChakrViewInferenceEngine17M:
    """High-velocity neural inference engine for ChakrGenesis-17M."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        vocab_dir: str | Path = "data/experiments/vocab_4096",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_dir = Path(vocab_dir)
        if self.vocab_dir.is_file():
            self.vocab_dir = self.vocab_dir.parent
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.dtype = dtype

        # Initialize and load model
        self.config = Genesis17MConfig()
        self.model = ChakrGenesis17M(self.config)
        self._load_checkpoint()
        self.model.to(self.device)
        if self.device.type == "cuda" and self.dtype == torch.float16:
            self.model = self.model.half()
        self.model.eval()

        # Load canonical tokenizer
        self.tokenizer, _ = load_tokenizer_artifacts(str(self.vocab_dir), verify_checksums=False)
        self.bos_id = self.config.bos_token_id
        self.eos_id = self.config.eos_token_id
        self.pad_id = self.config.pad_token_id

    def _load_checkpoint(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")
        state = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        state_dict = state.get("model_state_dict", state)
        # Strip potential _orig_mod prefix if compiled
        clean_state_dict = {}
        for k, v in state_dict.items():
            key = k.replace("_orig_mod.", "")
            clean_state_dict[key] = v
        self.model.load_state_dict(clean_state_dict, strict=True)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yields tokens one by one as they are generated.
        Yielded dictionary format:
        {
            "token_id": int,
            "text": str,
            "accumulated_text": str,
            "is_finished": bool,
            "latency_ms": float
        }
        """
        prompt_tokens = self.tokenizer.encode(prompt)
        # Prepend BOS if not present
        if len(prompt_tokens) == 0 or prompt_tokens[0] != self.bos_id:
            input_tokens = [self.bos_id] + prompt_tokens
        else:
            input_tokens = list(prompt_tokens)

        # Context truncation if prompt exceeds max_seq_len - max_new_tokens
        max_prompt_len = self.config.max_seq_len - max_new_tokens - 1
        if len(input_tokens) > max_prompt_len:
            input_tokens = [self.bos_id] + input_tokens[-max_prompt_len:]

        input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
        generated_ids: List[int] = []
        accumulated_text = ""
        prev_decoded_len = 0

        start_time = time.perf_counter()

        for step in range(max_new_tokens):
            t0 = time.perf_counter()
            logits, _ = self.model(input_tensor)
            next_token_logits = logits[0, -1, :].clone()

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                all_tokens = input_tokens + generated_ids
                counts = {}
                for tok in all_tokens:
                    counts[tok] = counts.get(tok, 0) + 1
                for tok, count in counts.items():
                    if 0 <= tok < next_token_logits.shape[0]:
                        if next_token_logits[tok] > 0:
                            next_token_logits[tok] /= (repetition_penalty ** min(count, 3))
                        else:
                            next_token_logits[tok] *= (repetition_penalty ** min(count, 3))

            # Temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    topk_vals, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < topk_vals[-1]] = -float("Inf")

                # Top-p (nucleus) filtering
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).item()

            t1 = time.perf_counter()
            step_latency_ms = (t1 - t0) * 1000.0

            # Check stop token
            if next_token == self.eos_id:
                yield {
                    "token_id": next_token,
                    "text": "",
                    "accumulated_text": accumulated_text,
                    "is_finished": True,
                    "latency_ms": step_latency_ms,
                    "total_time_ms": (time.perf_counter() - start_time) * 1000.0
                }
                break

            generated_ids.append(next_token)
            # Incremental decoding to properly handle subwords/multi-byte UTF-8
            current_decoded = self.tokenizer.decode(generated_ids, errors="replace")
            token_delta = current_decoded[prev_decoded_len:]
            prev_decoded_len = len(current_decoded)
            accumulated_text = current_decoded

            is_last = (step == max_new_tokens - 1)

            yield {
                "token_id": next_token,
                "text": token_delta,
                "accumulated_text": accumulated_text,
                "is_finished": is_last,
                "latency_ms": step_latency_ms,
                "total_time_ms": (time.perf_counter() - start_time) * 1000.0
            }

            if is_last:
                break

            # Append token for next step
            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            input_tensor = torch.cat([input_tensor, next_tensor], dim=1)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ) -> Dict[str, Any]:
        """Full completion generation returning complete output and generation metadata."""
        start_time = time.perf_counter()
        full_text = ""
        tokens_count = 0
        first_token_time = None

        for chunk in self.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            if first_token_time is None and chunk["token_id"] != self.eos_id:
                first_token_time = chunk["latency_ms"]
            tokens_count += 1
            full_text = chunk["accumulated_text"]
            if chunk["is_finished"]:
                break

        total_time_sec = time.perf_counter() - start_time
        tps = tokens_count / total_time_sec if total_time_sec > 0 else 0.0

        return {
            "prompt": prompt,
            "response": full_text.strip(),
            "tokens_generated": tokens_count,
            "total_time_sec": total_time_sec,
            "tokens_per_sec": tps,
            "first_token_ms": first_token_time or 0.0,
        }
