"""
Canonical Prompt/Response Contract Helper for ChakrView.

Enforces unified formatting, delimiter handling, and tokenization parity between
training, evaluation, and inference.
"""

from __future__ import annotations

from typing import Any, List, Tuple


def normalize_prompt_response_strings(
    prompt: str,
    response: str,
    strip_delimiters: bool = False,
) -> Tuple[str, str]:
    """
    Normalizes prompt and response strings according to the delimiter policy.
    - strip_delimiters=False (Canonical): preserves trailing delimiters (such as newline
      after <|im_start|>assistant\\n and after <|im_end|>\\n) to ensure bit-exact tokenization
      parity between training and autoregressive greedy generation.
    - strip_delimiters=True (Legacy/Batches 42-62): applies .strip() for backward compatibility.
    """
    if strip_delimiters:
        return prompt.strip(), response.strip()

    p_str = prompt if prompt.endswith("\n") else prompt.strip()
    r_str = response if response.endswith("\n") else response.strip()
    return p_str, r_str


def build_causal_sequence(
    prompt_tokens: List[int],
    response_tokens: List[int],
    bos_id: int = 0,
    eos_id: int = 1,
    ignore_index: int = -100,
) -> Tuple[List[int], List[int]]:
    """
    Builds causal input sequence and masked target sequence.
    input_seq:  [BOS] + prompt_tokens + response_tokens
    target_seq: [ignore_index]*len(prompt_tokens) + response_tokens + [EOS]
    
    Causal alignment property:
      input_seq[len(prompt_tokens)] is the final prompt token, whose prediction target is
      target_seq[len(prompt_tokens)] == response_tokens[0] (the first response token).
    """
    input_seq = [bos_id] + prompt_tokens + response_tokens
    target_seq = ([ignore_index] * len(prompt_tokens)) + response_tokens + [eos_id]
    assert len(input_seq) == len(target_seq)
    return input_seq, target_seq
