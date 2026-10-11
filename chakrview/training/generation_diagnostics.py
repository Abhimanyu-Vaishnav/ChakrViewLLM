"""
ChakrView Generation & Repetition Diagnostic Engine (Milestone 5).

Supports:
- Decoding methods:
  1. Greedy Argmax
  2. Temperature Sampling (with seed control)
  3. Top-k Sampling
  4. Top-p (Nucleus) Sampling
- Diagnostics:
  - Token-level entropy of model distributions
  - Repetition ratio (consecutive duplicates)
  - Unique-token ratio
  - Repeated n-gram rates (2-grams, 3-grams, 4-grams)
  - Average repetition loop length
  - EOS encounter tracking
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer


@dataclass
class GenerationDiagnosticMetrics:
    prompt: str
    decoding_method: str
    temperature: float
    top_k: int
    top_p: float
    generated_text: str
    generated_token_ids: List[int]
    unique_token_ratio: float
    consecutive_repetition_ratio: float
    ngram_2_repetition_rate: float
    ngram_3_repetition_rate: float
    mean_token_entropy: float
    eos_encountered: bool


class GenerationDiagnosticEngine:
    """
    Evaluates generation behavior and analyzes distribution collapse / repetition.
    """

    @classmethod
    def sample_next_token(
        cls,
        logits: torch.Tensor,
        method: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[int, float]:
        """
        Samples next token from logits [vocab_size], returning (token_id, entropy).
        """
        # Compute entropy of predictive distribution
        probs = F.softmax(logits, dim=-1)
        # Avoid log(0)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs).item()

        if method == "greedy" or temperature <= 1e-4:
            token_id = int(torch.argmax(logits).item())
            return token_id, entropy

        # Apply temperature
        scaled_logits = logits / max(temperature, 1e-4)

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, scaled_logits.size(-1))
            val, _ = torch.topk(scaled_logits, top_k)
            min_val = val[-1]
            scaled_logits = torch.where(scaled_logits < min_val, torch.full_like(scaled_logits, -1e9), scaled_logits)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above top_p (shift by 1 to keep first token above threshold)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            scaled_logits[indices_to_remove] = -1e9

        sample_probs = F.softmax(scaled_logits, dim=-1)
        token_id = int(torch.multinomial(sample_probs, num_samples=1, generator=generator).item())
        return token_id, entropy

    @classmethod
    def generate_with_diagnostics(
        cls,
        model: ChakrMicro,
        tokenizer: BPETokenizer,
        prompt: str,
        max_new_tokens: int = 24,
        method: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int = 42,
        eos_token_id: int = 1,
    ) -> GenerationDiagnosticMetrics:
        model.eval()
        device = next(model.parameters()).device
        gen = torch.Generator(device="cpu").manual_seed(seed)

        input_ids = tokenizer.encode(prompt)
        current_tokens = list(input_ids)
        new_token_ids: List[int] = []
        entropies: List[float] = []
        eos_hit = False

        with torch.no_grad():
            for _ in range(max_new_tokens):
                inp_tensor = torch.tensor([current_tokens], dtype=torch.long, device=device)
                logits = model(inp_tensor)[0, -1, :]  # [vocab_size]

                nxt_token, ent = cls.sample_next_token(
                    logits.cpu(),
                    method=method,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=gen,
                )
                entropies.append(ent)
                new_token_ids.append(nxt_token)
                current_tokens.append(nxt_token)

                if nxt_token == eos_token_id:
                    eos_hit = True
                    break

        try:
            full_text = tokenizer.decode(current_tokens, errors="replace")
        except Exception:
            full_text = f"[Raw tokens: {current_tokens}]"

        # Repetition metrics
        if new_token_ids:
            unique_ratio = len(set(new_token_ids)) / len(new_token_ids)
            dups = sum(1 for i in range(len(new_token_ids) - 1) if new_token_ids[i] == new_token_ids[i + 1])
            consec_rep = dups / max(len(new_token_ids) - 1, 1)

            # 2-gram repetition
            bi_grams = [tuple(new_token_ids[i : i + 2]) for i in range(len(new_token_ids) - 1)]
            bi_rep = 1.0 - (len(set(bi_grams)) / max(len(bi_grams), 1)) if bi_grams else 0.0

            # 3-gram repetition
            tri_grams = [tuple(new_token_ids[i : i + 3]) for i in range(len(new_token_ids) - 2)]
            tri_rep = 1.0 - (len(set(tri_grams)) / max(len(tri_grams), 1)) if tri_grams else 0.0
        else:
            unique_ratio = 1.0
            consec_rep = 0.0
            bi_rep = 0.0
            tri_rep = 0.0

        mean_ent = float(np.mean(entropies)) if entropies else 0.0

        return GenerationDiagnosticMetrics(
            prompt=prompt,
            decoding_method=method,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generated_text=full_text,
            generated_token_ids=new_token_ids,
            unique_token_ratio=round(unique_ratio, 4),
            consecutive_repetition_ratio=round(consec_rep, 4),
            ngram_2_repetition_rate=round(bi_rep, 4),
            ngram_3_repetition_rate=round(tri_rep, 4),
            mean_token_entropy=round(mean_ent, 4),
            eos_encountered=eos_hit,
        )
