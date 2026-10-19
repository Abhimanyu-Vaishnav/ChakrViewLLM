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
    append_eos: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Builds causal input sequence and masked target sequence.
    If append_eos is True (default, instruction pairs):
      input_seq:  [BOS] + prompt_tokens + response_tokens
      target_seq: [ignore_index]*len(prompt_tokens) + response_tokens + [EOS]
    If append_eos is False (continuation / replay sequences):
      input_seq:  [BOS] + prompt_tokens + response_tokens[:-1]
      target_seq: [ignore_index]*len(prompt_tokens) + response_tokens
    
    Causal alignment property:
      input_seq[len(prompt_tokens)] is the final prompt token, whose prediction target is
      target_seq[len(prompt_tokens)] == response_tokens[0] (the first response token).
    """
    if append_eos:
        input_seq = [bos_id] + prompt_tokens + response_tokens
        target_seq = ([ignore_index] * len(prompt_tokens)) + response_tokens + [eos_id]
    else:
        # Replay/continuation: response continues naturally without forced EOS boundary
        # target_seq has length len(prompt_tokens) + len(response_tokens)
        # input_seq must have identical length: [BOS] + prompt_tokens + response_tokens[:-1]
        input_seq = [bos_id] + prompt_tokens + (response_tokens[:-1] if response_tokens else [])
        target_seq = ([ignore_index] * len(prompt_tokens)) + response_tokens

    assert len(input_seq) == len(target_seq), f"Length mismatch: {len(input_seq)} vs {len(target_seq)}"
    return input_seq, target_seq
