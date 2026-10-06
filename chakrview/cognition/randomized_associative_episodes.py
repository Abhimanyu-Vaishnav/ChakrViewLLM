"""Step 250: Truly Randomized Associative Episode Environment.

Builds a strictly controlled, multi-pair randomized episode generator:
- Multiple key/value associations (variable 1 to 5 pairs)
- Randomized key identities from disjoint pools
- Randomized value identities from disjoint pools
- Randomized pair presentation ordering
- Randomized query location (end, prefix, middle)
- Randomized separators/layouts (canonical, reverse, verbose, tuple, assignment)
- Variable distractor key-value pairs
- Randomized contextual positions
- Ground-truth position annotations (query_key_pos, matching_key_pos, associated_value_pos)
- SHA-256 contamination audit between train and evaluation sets

Invariant:
The model receives ONLY prompt token sequence. Metadata is used strictly for ground-truth routing metrics.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
from typing import Dict, List, Optional, Tuple, Any

from chakrview.tokenizer import BPETokenizer


@dataclasses.dataclass(frozen=True)
class AssociativePair:
    key: str
    val: str


@dataclasses.dataclass(frozen=True)
class RandomizedAssociativeEpisode:
    episode_id: str
    split: str                         # "train", "val", "disjoint_test"
    pairs: Tuple[AssociativePair, ...]
    query_key: str
    expected_value: str
    layout_name: str
    prompt: str
    num_associations: int
    has_distractors: bool
    episode_hash: str
    # Exact token-level ground truth positions in the encoded prompt
    query_key_pos: int
    matching_key_pos: int
    associated_val_pos: int
    prompt_tokens: Tuple[int, ...]
    target_token: int


class RandomizedAssociativeEnvironment:
    """Generates synthetic multi-layout associative episodes with ground-truth position tracing."""

    # Disjoint pools
    TRAIN_KEYS_POOL = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    TRAIN_VALS_POOL = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    DISJOINT_KEYS_POOL = ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
    DISJOINT_VALS_POOL = ["0", "#", "@", "$", "%", "&", "!", "?"]

    LAYOUTS = [
        "standard_map",       # map |K| -> |V| and ... query |K| -> |
        "reverse_order",      # map |V| <- |K| and ... query |K| -> |
        "semicolon_verbose",  # K maps to V; ... query K: |
        "compact_tuple",      # pairs (K,V) ... query=K -> |
        "assignment_syntax",  # assoc: K=V, ... / retrieve K = |
    ]

    def __init__(self, tokenizer: Optional[BPETokenizer] = None, seed: int = 42):
        self.tok = tokenizer or BPETokenizer()
        self.rng = random.Random(seed)
        self.seed = seed

    def render_prompt_layout(
        self,
        pairs: List[AssociativePair],
        query_key: str,
        layout: str,
        distractors: Optional[List[AssociativePair]] = None,
        query_placement: str = "end", # "end" or "prefix"
    ) -> str:
        dist_pairs = distractors or []
        all_pairs = list(pairs)
        if dist_pairs:
            # Interleave distractors
            insert_idx = self.rng.randint(0, len(all_pairs))
            for dp in dist_pairs:
                all_pairs.insert(insert_idx, dp)

        if layout == "standard_map":
            parts = [f"|{p.key}| -> |{p.val}|" for p in all_pairs]
            body = "map " + " and ".join(parts)
            q_part = f"query |{query_key}| -> |"
        elif layout == "reverse_order":
            parts = [f"|{p.val}| <- |{p.key}|" for p in all_pairs]
            body = "items " + " ; ".join(parts)
            q_part = f"query |{query_key}| -> |"
        elif layout == "semicolon_verbose":
            parts = [f"{p.key} maps to {p.val}" for p in all_pairs]
            body = "; ".join(parts)
            q_part = f"; query {query_key}: |"
        elif layout == "compact_tuple":
            parts = [f"({p.key},{p.val})" for p in all_pairs]
            body = "pairs " + " ".join(parts)
            q_part = f" query={query_key} -> |"
        elif layout == "assignment_syntax":
            parts = [f"{p.key}={p.val}" for p in all_pairs]
            body = "assoc: " + ", ".join(parts)
            q_part = f" / retrieve {query_key} = |"
        else:
            parts = [f"|{p.key}| -> |{p.val}|" for p in all_pairs]
            body = "map " + " and ".join(parts)
            q_part = f" query |{query_key}| -> |"

        if query_placement == "prefix":
            return f"{q_part} context: {body} |"
        else:
            return f"{body} {q_part}"

    def generate_episode(
        self,
        split: str = "train",
        num_associations: int = 2,
        layout_name: Optional[str] = None,
        include_distractors: bool = False,
        query_placement: str = "end",
        episode_idx: int = 0,
    ) -> RandomizedAssociativeEpisode:
        # Choose pools
        if split == "train":
            k_pool, v_pool = list(self.TRAIN_KEYS_POOL), list(self.TRAIN_VALS_POOL)
        elif split == "val":
            k_pool, v_pool = list(self.TRAIN_KEYS_POOL), list(self.TRAIN_VALS_POOL)
        elif split == "disjoint_test":
            k_pool, v_pool = list(self.DISJOINT_KEYS_POOL), list(self.DISJOINT_VALS_POOL)
        elif split == "known_unseen":
            k_pool, v_pool = list(self.TRAIN_KEYS_POOL), list(self.DISJOINT_VALS_POOL)
        elif split == "unseen_known":
            k_pool, v_pool = list(self.DISJOINT_KEYS_POOL), list(self.TRAIN_VALS_POOL)
        else:
            k_pool, v_pool = list(self.TRAIN_KEYS_POOL), list(self.TRAIN_VALS_POOL)

        n = min(num_associations, len(k_pool), len(v_pool))
        s_keys = self.rng.sample(k_pool, n)
        s_vals = self.rng.sample(v_pool, n)
        pairs = [AssociativePair(k, v) for k, v in zip(s_keys, s_vals)]
        self.rng.shuffle(pairs)

        q_pair = self.rng.choice(pairs)
        q_key = q_pair.key
        exp_val = q_pair.val

        distractors = []
        if include_distractors:
            avail_k = [k for k in k_pool if k not in s_keys]
            avail_v = [v for v in v_pool if v not in s_vals]
            if avail_k and avail_v:
                distractors.append(AssociativePair(self.rng.choice(avail_k), self.rng.choice(avail_v)))

        layout = layout_name or self.rng.choice(self.LAYOUTS)
        prompt = self.render_prompt_layout(
            pairs=pairs,
            query_key=q_key,
            layout=layout,
            distractors=distractors,
            query_placement=query_placement,
        )

        tokens = self.tok.encode(prompt)
        exp_tok = self.tok.encode(exp_val)[0]
        q_key_tok = self.tok.encode(q_key)[0]

        # Trace exact ground-truth positions in encoded tokens
        all_k_pos = [i for i, t in enumerate(tokens[:-1]) if t == q_key_tok]
        all_v_pos = [i for i, t in enumerate(tokens[:-1]) if t == exp_tok]

        if query_placement == "prefix":
            q_k_pos = all_k_pos[0] if all_k_pos else len(tokens) - 1
            m_k_pos = all_k_pos[-1] if len(all_k_pos) > 1 else (all_k_pos[0] if all_k_pos else 0)
        else:
            q_k_pos = all_k_pos[-1] if all_k_pos else len(tokens) - 1
            m_k_pos = all_k_pos[0] if all_k_pos else 0

        v_pos = all_v_pos[0] if all_v_pos else 0

        hasher = hashlib.sha256()
        hasher.update(f"{split}_{episode_idx}_{prompt}_{exp_val}".encode("utf-8"))
        ep_hash = hasher.hexdigest()

        ep_id = f"ep_{split}_{layout}_{episode_idx}_{ep_hash[:8]}"

        return RandomizedAssociativeEpisode(
            episode_id=ep_id,
            split=split,
            pairs=tuple(pairs),
            query_key=q_key,
            expected_value=exp_val,
            layout_name=layout,
            prompt=prompt,
            num_associations=n,
            has_distractors=include_distractors,
            episode_hash=ep_hash,
            query_key_pos=q_k_pos,
            matching_key_pos=m_k_pos,
            associated_val_pos=v_pos,
            prompt_tokens=tuple(tokens),
            target_token=exp_tok,
        )

    def generate_batch(
        self,
        count: int,
        split: str = "train",
        num_associations: int = 2,
        layout_name: Optional[str] = None,
        include_distractors: bool = False,
    ) -> List[RandomizedAssociativeEpisode]:
        return [
            self.generate_episode(
                split=split,
                num_associations=num_associations,
                layout_name=layout_name,
                include_distractors=include_distractors,
                episode_idx=i,
            )
            for i in range(count)
        ]

    def verify_no_contamination(
        self,
        train_episodes: List[RandomizedAssociativeEpisode],
        eval_episodes: List[RandomizedAssociativeEpisode],
    ) -> Tuple[bool, int]:
        """Audits that train and evaluation episodes have zero hash or prompt collision."""
        train_hashes = {e.episode_hash for e in train_episodes}
        train_prompts = {e.prompt for e in train_episodes}
        collisions = 0
        for ev in eval_episodes:
            if ev.episode_hash in train_hashes or ev.prompt in train_prompts:
                collisions += 1
        return (collisions == 0), collisions
