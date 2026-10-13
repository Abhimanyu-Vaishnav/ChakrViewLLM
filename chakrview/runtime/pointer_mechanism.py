"""
Auxiliary Pointer-Generator Copy Mechanism for ChakrView.
Location: chakrview/runtime/pointer_mechanism.py

Features:
1. ChakrPointerHead: Computes attention over designated context token positions.
2. Learned Gate: Dynamic interpolation between base model vocabulary logits and context pointer probabilities.
3. Strict Causal & Mask Boundary Protection: Disallows pointing to prompt framing, padding, or future answer tokens.
4. Leaves canonical ChakrMicro model parameters 100% untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChakrPointerHead(nn.Module):
    """
    Auxiliary attention head computing pointer distribution over context positions.
    P_final = (1 - g) * P_vocab + g * P_copy
    """

    def __init__(
        self,
        d_model: int = 192,
        vocab_size: int = 4096,
        d_proj: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.d_proj = d_proj

        self.query_proj = nn.Linear(d_model, d_proj, bias=False)
        self.key_proj = nn.Linear(d_model, d_proj, bias=False)
        # Gate network takes current hidden state and attended context vector
        self.gate_proj = nn.Linear(d_model + d_model, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        vocab_logits: torch.Tensor,
        context_mask: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        disentangle_query_overlap: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        hidden_states: [B, T, d_model] from base model pre-norm
        input_ids: [B, T] token IDs corresponding to hidden_states
        vocab_logits: [B, T, vocab_size] from base model lm_head
        context_mask: [B, T] boolean tensor (True for valid context tokens to copy from, False otherwise)
        query_mask: Optional [B, T] boolean tensor (True for query tokens). Used to penalize copying entity tokens
                    that merely duplicate question terms when distinct relation value candidates exist.
        disentangle_query_overlap: If True and query_mask is given, applies a soft damping factor to context tokens
                                  whose exact IDs appear in the question.

        Returns:
            final_probs: [B, T, vocab_size] normalized probability distribution
            pointer_attn: [B, T, T] attention weights over context positions
            gate: [B, T, 1] copy gate values in [0, 1]
        """
        B, T, D = hidden_states.shape

        q = self.query_proj(hidden_states)  # [B, T, d_proj]
        k = self.key_proj(hidden_states)    # [B, T, d_proj]

        # Dot-product attention scores between all positions: [B, T, T]
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_proj)

        # Mask positions where context_mask is False (only allow pointing to context tokens)
        # context_mask shape is [B, T], expand to [B, 1, T] for broadcasting across query steps
        mask_expanded = context_mask.unsqueeze(1).expand(B, T, T)
        scores = scores.masked_fill(~mask_expanded, -1e9)

        # Disentangled Query Overlap Adjustment:
        # If query_mask is present, identify context tokens whose IDs overlap with question tokens.
        # Apply a calibrated soft penalty (-3.0 in logit space) to question-echo tokens to favor relation values.
        if disentangle_query_overlap and query_mask is not None:
            # Construct boolean mask of shape [B, T] where context token ID is also in question
            overlap_mask = torch.zeros(B, T, dtype=torch.bool, device=hidden_states.device)
            for b_idx in range(B):
                q_ids_set = set(input_ids[b_idx, query_mask[b_idx]].tolist())
                # Disentangle only non-stopwords / alphanumeric tokens
                for pos_idx in range(T):
                    if context_mask[b_idx, pos_idx] and input_ids[b_idx, pos_idx].item() in q_ids_set:
                        overlap_mask[b_idx, pos_idx] = True
            
            overlap_expanded = overlap_mask.unsqueeze(1).expand(B, T, T)
            scores = scores - 3.5 * overlap_expanded.float()

        pointer_attn = F.softmax(scores, dim=-1)  # [B, T, T]
        # Clean NaNs if a row has zero valid context tokens
        pointer_attn = torch.nan_to_num(pointer_attn, nan=0.0)

        # Compute attended context vector: [B, T, d_model]
        context_summary = torch.bmm(pointer_attn, hidden_states)

        # Gate computation: [B, T, 1] in [0, 1]
        gate_input = torch.cat([hidden_states, context_summary], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))

        # Vocabulary probabilities: [B, T, vocab_size]
        vocab_probs = F.softmax(vocab_logits, dim=-1)

        # Map pointer attention positions to vocabulary token IDs
        # input_ids: [B, T]. Expand to [B, T, T] to scatter into [B, T, vocab_size]
        copy_probs = torch.zeros(B, T, self.vocab_size, device=hidden_states.device, dtype=vocab_probs.dtype)
        
        # For each query step t, scatter pointer_attn[b, t, j] into copy_probs[b, t, input_ids[b, j]]
        expanded_input_ids = input_ids.unsqueeze(1).expand(B, T, T)  # [B, T, T]
        copy_probs.scatter_add_(dim=2, index=expanded_input_ids, src=pointer_attn)

        # Final mixture distribution: [B, T, vocab_size]
        final_probs = (1.0 - gate) * vocab_probs + gate * copy_probs

        return final_probs, pointer_attn, gate


@dataclass
class BeamHypothesis:
    """Represents an active search hypothesis in pointer beam decoding."""
    token_ids: List[int]
    log_prob: float
    token_sources: List[str]  # "copy", "vocab", or "mixture"
    completed: bool = False

    @property
    def score(self) -> float:
        """Length-normalized log-probability score."""
        length = max(len(self.token_ids), 1)
        # Standard length penalty alpha = 0.7
        return self.log_prob / (length ** 0.7)


def decode_pointer_greedy(
    model: nn.Module,
    pointer_head: ChakrPointerHead,
    prompt_ids: List[int],
    context_start: int,
    context_end: int,
    query_start: Optional[int] = None,
    query_end: Optional[int] = None,
    max_new_tokens: int = 16,
    eos_token_ids: Tuple[int, ...] = (1, 3),
    device: Optional[torch.device] = None,
) -> Tuple[List[int], List[str]]:
    """
    Greedy decoding using ChakrPointerHead.
    Returns:
        (generated_token_ids, token_attribution_list)
    """
    if device is None:
        device = torch.device("cpu")

    curr_tokens = list(prompt_ids)
    gen_tokens: List[int] = []
    attributions: List[str] = []

    model.eval()
    pointer_head.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([curr_tokens], dtype=torch.long, device=device)
            T = x.shape[1]

            # Build context mask
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
            final_probs, attn, gate = pointer_head(hidden_states, x, vocab_logits, mask_t, query_mask=q_mask_t)

            probs_step = final_probs[0, -1, :]
            gate_val = gate[0, -1, 0].item()

            next_tok = int(torch.argmax(probs_step).item())
            if next_tok in eos_token_ids:
                break

            # Attribution
            if gate_val > 0.6:
                source = "copy"
            elif gate_val < 0.2:
                source = "vocab"
            else:
                source = "mixture"

            gen_tokens.append(next_tok)
            attributions.append(source)
            curr_tokens.append(next_tok)

    return gen_tokens, attributions


def decode_pointer_beam(
    model: nn.Module,
    pointer_head: ChakrPointerHead,
    prompt_ids: List[int],
    context_start: int,
    context_end: int,
    query_start: Optional[int] = None,
    query_end: Optional[int] = None,
    beam_width: int = 3,
    max_new_tokens: int = 16,
    eos_token_ids: Tuple[int, ...] = (1, 3),
    device: Optional[torch.device] = None,
) -> Tuple[List[int], List[str]]:
    """
    Bounded Beam Search for ChakrPointerHead.
    Supports beam_width=1 (equivalent to greedy) and beam_width > 1.
    """
    if device is None:
        device = torch.device("cpu")

    if beam_width <= 1:
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

    model.eval()
    pointer_head.eval()

    beams: List[BeamHypothesis] = [BeamHypothesis(token_ids=[], log_prob=0.0, token_sources=[], completed=False)]

    with torch.no_grad():
        for step in range(max_new_tokens):
            all_candidates: List[BeamHypothesis] = []

            for hyp in beams:
                if hyp.completed:
                    all_candidates.append(hyp)
                    continue

                full_seq = list(prompt_ids) + hyp.token_ids
                x = torch.tensor([full_seq], dtype=torch.long, device=device)
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
                final_probs, _, gate = pointer_head(hidden_states, x, vocab_logits, mask_t, query_mask=q_mask_t)

                probs_step = final_probs[0, -1, :]
                gate_val = gate[0, -1, 0].item()

                log_probs = torch.log(probs_step + 1e-12)
                topk_log_probs, topk_indices = torch.topk(log_probs, k=beam_width)

                for k_idx in range(beam_width):
                    tok_id = int(topk_indices[k_idx].item())
                    tok_log_p = float(topk_log_probs[k_idx].item())

                    is_eos = tok_id in eos_token_ids
                    if gate_val > 0.6:
                        src = "copy"
                    elif gate_val < 0.2:
                        src = "vocab"
                    else:
                        src = "mixture"

                    cand = BeamHypothesis(
                        token_ids=hyp.token_ids if is_eos else hyp.token_ids + [tok_id],
                        log_prob=hyp.log_prob + tok_log_p,
                        token_sources=hyp.token_sources if is_eos else hyp.token_sources + [src],
                        completed=is_eos,
                    )
                    all_candidates.append(cand)

            # Prune to top beam_width
            all_candidates.sort(key=lambda h: h.score, reverse=True)
            beams = all_candidates[:beam_width]

            if all(h.completed for h in beams):
                break

    best_beam = max(beams, key=lambda h: h.score)
    return best_beam.token_ids, best_beam.token_sources

