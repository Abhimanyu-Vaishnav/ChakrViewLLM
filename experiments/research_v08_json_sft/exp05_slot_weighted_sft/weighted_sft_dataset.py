"""Weighted SFT Dataset and Token-Level Loss Computation for Experiment 05.

Implements:
1. Dynamic detection of the exact status string token span in response tokens:
   - For single_field_string_extraction:
     The response is structured as `{"status": "<val>"}`.
     The status value `<val>` sits between the opening `": "` and closing `"`.
     We locate the exact subword token slice corresponding to `<val>` using serialized matching.
   - For replay records or non-extraction records:
     All active response tokens have weight 1.0.
2. Token weights tensor:
   - Prompt tokens (target = -100): weight 0.0
   - Ignored padding tokens (target = -100): weight 0.0
   - JSON delimiters (`{`, `"`, `stat`, `us`, `": "`, `"`, `}`, `<EOS>`): weight 1.0
   - Status value subwords: weight `slot_weight` (default 5.0)
3. Normalized weighted cross-entropy loss:
   $$\mathcal{L} = \frac{\sum_{i} w_i \cdot \ell_i}{\sum_{i} w_i}$$
   where $\ell_i = -\log P(y_i \mid x_{<i})$ for all active targets $y_i \ne -100$, and $w_i \in \{1.0, \text{slot\_weight}\}$.
   When all $w_i = 1.0$, this simplifies bit-exact to standard mean cross-entropy:
   $$\mathcal{L} = \frac{\sum_{i} 1.0 \cdot \ell_i}{\sum_{i} 1.0} = \frac{1}{N} \sum_{i=1}^N \ell_i$$
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.training.sft_engine import SFTDataset
from chakrview.training.prompt_contract import normalize_prompt_response_strings, build_causal_sequence


class WeightedSFTDataset(SFTDataset):
    """
    SFTDataset subclass supporting precise token-level loss weighting for target slots.
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: Any,
        max_seq_len: int = 256,
        allow_exploratory_opt_in: bool = False,
        enforce_trust_gate: bool = True,
        is_training: bool = False,
        strip_prompt_delimiters: bool = False,
        slot_weight: float = 5.0,
    ):
        self.slot_weight = slot_weight
        super().__init__(
            jsonl_path=jsonl_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            allow_exploratory_opt_in=allow_exploratory_opt_in,
            enforce_trust_gate=enforce_trust_gate,
            is_training=is_training,
            strip_prompt_delimiters=strip_prompt_delimiters,
        )

    def encode_weighted_example(
        self, record: Dict[str, Any]
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        Encodes example into input_seq, target_seq, and token_weights.
        Returns:
          input_seq: List[int]
          target_seq: List[int] (with -100 for prompt tokens)
          token_weights: List[float] (0.0 for masked positions, 1.0 for syntax/EOS, slot_weight for status value)
        """
        prompt = record["prompt"]
        response = record["response"]
        append_eos = record.get("append_eos", True)
        expected_status = record.get("expected_status", None)
        category = record.get("category", "")

        input_seq, target_seq = self.encode_example(prompt, response, append_eos=append_eos)

        # Initialize weights
        # For all prompt tokens (target == -100), weight is 0.0
        # For supervised tokens (target != -100), default weight is 1.0
        token_weights = [0.0 if t == -100 else 1.0 for t in target_seq]

        if category == "single_field_string_extraction" and expected_status is not None:
            # Locate exact subword token slice belonging to expected_status in response
            # Response string: `{"status": "<val>"}`
            # Prefix string: `{"status": "`
            # Suffix string: `"}`
            # Tokenize prefix and suffix to verify exact alignment
            p_str, r_str = normalize_prompt_response_strings(
                prompt, response, strip_delimiters=self.strip_prompt_delimiters
            )
            p_toks = self.tokenizer.encode(p_str)
            r_toks = self.tokenizer.encode(r_str)
            status_toks = self.tokenizer.encode(expected_status)

            # In response tokens, find where status_toks occurs
            # Specifically, response begins with prefix:
            prefix_toks = self.tokenizer.encode('{"status": "')
            suffix_toks = self.tokenizer.encode('"}')

            # Verify response tokens composition:
            # r_toks should start with prefix_toks and end with suffix_toks
            assert r_toks[:len(prefix_toks)] == prefix_toks, (
                f"Prefix mismatch: {r_toks[:len(prefix_toks)]} vs {prefix_toks}"
            )
            assert r_toks[-len(suffix_toks):] == suffix_toks, (
                f"Suffix mismatch: {r_toks[-len(suffix_toks):]} vs {suffix_toks}"
            )

            # The exact status tokens slice in r_toks:
            status_slice = r_toks[len(prefix_toks) : -len(suffix_toks)]
            assert status_slice == status_toks, (
                f"Status subword mismatch in record {record.get('id')}: {status_slice} vs {status_toks}"
            )

            # In target_seq, prompt tokens occupy index 0 to len(p_toks)-1 (all -100)
            # Response tokens start at index len(p_toks)
            # Within target_seq, the status tokens occupy:
            # start_idx = len(p_toks) + len(prefix_toks)
            # end_idx = start_idx + len(status_slice)
            start_idx = len(p_toks) + len(prefix_toks)
            end_idx = start_idx + len(status_slice)

            # Apply slot_weight only to this exact slice
            for idx in range(start_idx, end_idx):
                assert target_seq[idx] == status_slice[idx - start_idx]
                token_weights[idx] = float(self.slot_weight)

            # Verify that prefix tokens, closing suffix tokens, and EOS token remain weight 1.0
            for idx in range(len(p_toks), start_idx):
                assert token_weights[idx] == 1.0
            for idx in range(end_idx, len(target_seq)):
                assert token_weights[idx] == 1.0

        return input_seq, target_seq, token_weights

    def get_weighted_batch(
        self, batch_size: int, rng: np.random.RandomState
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Samples a batch, pads dynamically, and returns:
          x_tensor: [batch_size, seq_len]
          y_tensor: [batch_size, seq_len] (with -100 for ignored)
          w_tensor: [batch_size, seq_len] (loss weight per token)
          active_tokens: int
        """
        indices = rng.randint(0, len(self.records), size=batch_size)
        batch_inputs = []
        batch_targets = []
        batch_weights = []

        for idx in indices:
            rec = self.records[idx]
            inp, tgt, wgt = self.encode_weighted_example(rec)
            batch_inputs.append(inp)
            batch_targets.append(tgt)
            batch_weights.append(wgt)

        max_len = max(len(x) for x in batch_inputs)
        pad_id = 2
        ignore_index = -100

        padded_inputs = []
        padded_targets = []
        padded_weights = []

        for inp, tgt, wgt in zip(batch_inputs, batch_targets, batch_weights):
            pad_amt = max_len - len(inp)
            padded_inputs.append(inp + [pad_id] * pad_amt)
            padded_targets.append(tgt + [ignore_index] * pad_amt)
            padded_weights.append(wgt + [0.0] * pad_amt)

        x_tensor = torch.tensor(padded_inputs, dtype=torch.long)
        y_tensor = torch.tensor(padded_targets, dtype=torch.long)
        w_tensor = torch.tensor(padded_weights, dtype=torch.float32)

        active_tokens = int((y_tensor != ignore_index).sum().item())
        return x_tensor, y_tensor, w_tensor, active_tokens


def compute_weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Computes normalized token-weighted cross entropy loss:
      loss = sum(w_i * l_i) / max(1e-8, sum(w_i))
    where l_i is the standard cross-entropy loss at token position i.
    """
    # Flatten tensors
    # logits: [B, T, V] -> [B*T, V]
    # targets: [B, T] -> [B*T]
    # weights: [B, T] -> [B*T]
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_weights = weights.reshape(-1)

    # Compute unreduced cross-entropy loss per token
    ce_loss = nn.functional.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, reduction="none")

    # Mask out ignored positions in weights
    active_mask = (flat_targets != ignore_index)
    valid_weights = flat_weights * active_mask.float()

    total_weight = torch.sum(valid_weights)
    if total_weight.item() <= 0.0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    weighted_loss = torch.sum(ce_loss * valid_weights) / total_weight
    return weighted_loss
