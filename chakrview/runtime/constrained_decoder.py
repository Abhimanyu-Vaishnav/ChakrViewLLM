"""
Tokenizer-Aware Constrained Decoding for ChakrView.
Location: chakrview/runtime/constrained_decoder.py

Features:
1. Trie Structure: Built from encoded token sequences of candidate answer strings.
2. Prefix Matching: Restricts the next-token probability distribution strictly to valid continuation tokens.
3. Multi-Byte UTF-8 Protection: Prevents generating incomplete byte fragments in languages like Hindi.
4. Fallback Support: If constrained candidate set is empty, falls back gracefully to unconstrained decoding.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.runtime.pointer_mechanism import ChakrPointerHead


class TokenTrieNode:
    def __init__(self):
        self.children: Dict[int, TokenTrieNode] = {}
        self.is_terminal: bool = False
        self.payload: Optional[str] = None


class TokenTrie:
    """
    Trie built from token ID sequences to constrain autoregressive decoding.
    """

    def __init__(self):
        self.root = TokenTrieNode()

    def insert(self, token_ids: List[int], payload: Optional[str] = None) -> None:
        curr = self.root
        for tok in token_ids:
            if tok not in curr.children:
                curr.children[tok] = TokenTrieNode()
            curr = curr.children[tok]
        curr.is_terminal = True
        curr.payload = payload

    def get_allowed_tokens(self, prefix: List[int]) -> Set[int]:
        curr = self.root
        for tok in prefix:
            if tok not in curr.children:
                return set()
            curr = curr.children[tok]
        return set(curr.children.keys())

    def is_valid_sequence(self, token_ids: List[int]) -> bool:
        curr = self.root
        for tok in token_ids:
            if tok not in curr.children:
                return False
            curr = curr.children[tok]
        return curr.is_terminal


def decode_pointer_constrained(
    model: nn.Module,
    pointer_head: ChakrPointerHead,
    prompt_ids: List[int],
    context_start: int,
    context_end: int,
    candidate_token_sequences: List[List[int]],
    query_start: Optional[int] = None,
    query_end: Optional[int] = None,
    max_new_tokens: int = 16,
    eos_token_ids: Tuple[int, ...] = (1, 3),
    device: Optional[torch.device] = None,
) -> Tuple[List[int], List[str]]:
    """
    Autoregressive pointer decoding constrained by a candidate Trie.
    """
    if device is None:
        device = torch.device("cpu")

    if not candidate_token_sequences:
        from chakrview.runtime.pointer_mechanism import decode_pointer_greedy
        return decode_pointer_greedy(
            model=model,
            pointer_head=pointer_head,
            prompt_ids=prompt_ids,
            context_start=context_start,
            context_end=context_end,
            query_start=query_start,
            query_end=query_end,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            device=device,
        )

    trie = TokenTrie()
    for seq in candidate_token_sequences:
        trie.insert(seq)

    curr_tokens = list(prompt_ids)
    gen_tokens: List[int] = []
    attributions: List[str] = []

    model.eval()
    pointer_head.eval()

    with torch.no_grad():
        for step in range(max_new_tokens):
            allowed_tokens = trie.get_allowed_tokens(gen_tokens)
            if not allowed_tokens:
                break

            x = torch.tensor([curr_tokens], dtype=torch.long, device=device)
            T = x.shape[1]

            c_mask = [False] * T
            for idx in range(min(context_start, T), min(context_end, T)):
                c_mask[idx] = True
            mask_t = torch.tensor([c_mask], dtype=torch.bool, device=device)

            q_mask_t = None
            if query_start is not None and query_end is not None:
                q_mask = [False] * T
                for idx in range(min(query_start, T), min(query_end, T)):
                    q_mask[idx] = True
                q_mask_t = torch.tensor([q_mask], dtype=torch.bool, device=device)

            vocab_logits, hidden_states = model(x, return_hidden_states=True)
            final_probs, attn, gate = pointer_head(
                hidden_states, x, vocab_logits, mask_t, query_mask=q_mask_t, disentangle_query_overlap=True
            )

            probs_step = final_probs[0, -1, :]
            gate_val = gate[0, -1, 0].item()

            # Constrain to allowed tokens
            mask_allowed = torch.full_like(probs_step, -1e9)
            for tok_id in allowed_tokens:
                if tok_id < len(probs_step):
                    mask_allowed[tok_id] = torch.log(probs_step[tok_id] + 1e-12)

            next_tok = int(torch.argmax(mask_allowed).item())
            if next_tok in eos_token_ids:
                break

            src = "copy" if gate_val > 0.5 else "vocab"
            gen_tokens.append(next_tok)
            attributions.append(src)
            curr_tokens.append(next_tok)

    return gen_tokens, attributions
