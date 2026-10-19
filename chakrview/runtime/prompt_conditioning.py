"""
Prompt Conditioning and Template Formatting for ChakrView.
Location: chakrview/runtime/prompt_conditioning.py

Supports:
1. 'raw_completion': Returns raw prompt unchanged.
2. 'plain_instruction': Clean, minimal markdown-style task prefix:
   "Instruction: {query}\nResponse:"
3. 'context_qa': Supplied context with question:
   "Context: {context}\nQuestion: {query}\nAnswer:"
"""

from __future__ import annotations

from typing import Optional


SUPPORTED_PROMPT_MODES = ("raw_completion", "plain_instruction", "context_qa")
DEFAULT_PROMPT_MODE = "raw_completion"


def format_conditioned_prompt(
    prompt: str,
    prompt_mode: Optional[str] = None,
    context: Optional[str] = None,
    mode: Optional[str] = None,
) -> str:
    """
    Formats user prompt according to specified prompt conditioning mode.
    Fails closed / raises ValueError if prompt_mode is unrecognized.
    """
    active_mode = mode or prompt_mode or DEFAULT_PROMPT_MODE
    clean_p = prompt.strip()
    if active_mode not in SUPPORTED_PROMPT_MODES:
        raise ValueError(
            f"Unsupported prompt mode: '{active_mode}'. Supported modes: {list(SUPPORTED_PROMPT_MODES)}"
        )

    if active_mode == "raw_completion":
        return clean_p

    elif active_mode == "plain_instruction":
        return f"Instruction: {clean_p}\nResponse:"

    elif active_mode == "context_qa":
        ctx_str = (context or "").strip()
        if ctx_str:
            return f"Context: {ctx_str}\nQuestion: {clean_p}\nAnswer:"
        return f"Question: {clean_p}\nAnswer:"

    return clean_p
