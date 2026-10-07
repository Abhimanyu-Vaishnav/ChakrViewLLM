"""Step 274: Contextual Token Candidate Extraction.

Extracts candidate token positions and representations directly from episode context.
STRICT INVARIANTS:
- Candidate positions are derived from episode structural syntax, NOT answer labels.
- Hidden representations come from the neural model's adapted hidden states.
- The target information is used ONLY for computing training loss/evaluation metrics,
  NEVER injected into model forward scoring.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

from chakrview.cognition.randomized_associative_episodes import (
    RandomizedAssociativeEpisode,
)


@dataclasses.dataclass
class ContextualCandidateBundle:
    """Encapsulates neural candidate hidden states and structural positions."""
    candidate_positions: List[int]             # Prompt token indices of candidate tokens
    candidate_token_ids: List[int]             # Token IDs at candidate positions (for evaluation)
    candidate_states: torch.Tensor             # [1, N, D] extracted contextual hidden states
    target_candidate_idx: Optional[int]        # Index of the correct candidate in [0, N-1]
    candidate_mask: torch.Tensor               # [1, N] mask indicating valid candidates


def extract_contextual_candidates_from_episode(
    episode: RandomizedAssociativeEpisode,
    hidden_states: torch.Tensor,
    tok: Any,
) -> ContextualCandidateBundle:
    """
    Extracts candidate token representations from the prompt hidden states.
    For an associative episode with premises:
    e.g. map |P| -> |#| and |Q| -> |%| query |P| -> |
    The candidate values present in context are the value tokens of the premise pairs.
    
    hidden_states: [1, seq_len, D]
    Returns ContextualCandidateBundle with candidate hidden vectors.
    """
    prompt_tokens = episode.prompt_tokens
    # Identify value token positions from all premise pairs in the prompt
    cand_positions: List[int] = []
    cand_tokens: List[int] = []
    
    for pair in episode.pairs:
        val_tok = tok.encode(pair.val)[0]
        # Find position of val_tok in prompt (excluding query prompt if any)
        matches = [i for i, t in enumerate(prompt_tokens[:-1]) if t == val_tok]
        if matches:
            pos = matches[0]
            if pos not in cand_positions:
                cand_positions.append(pos)
                cand_tokens.append(val_tok)

    # In case there are distractors in prompt, add them as well if present
    # Candidates are ordered by their appearance in the premise pairs
    if not cand_positions:
        # Fallback to non-query non-key tokens
        cand_positions = [0]
        cand_tokens = [prompt_tokens[0]]

    # Extract target index
    target_token = episode.target_token
    target_idx = None
    for idx, t_id in enumerate(cand_tokens):
        if t_id == target_token:
            target_idx = idx
            break

    # Extract hidden vectors [1, N, D]
    N = len(cand_positions)
    D = hidden_states.shape[-1]
    cand_states = torch.stack(
        [hidden_states[0, pos] for pos in cand_positions], dim=0
    ).unsqueeze(0)  # [1, N, D]

    mask = torch.ones((1, N), dtype=torch.bool, device=hidden_states.device)

    return ContextualCandidateBundle(
        candidate_positions=cand_positions,
        candidate_token_ids=cand_tokens,
        candidate_states=cand_states,
        target_candidate_idx=target_idx,
        candidate_mask=mask,
    )
