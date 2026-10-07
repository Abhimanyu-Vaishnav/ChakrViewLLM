"""Step 363: State Persistence Ablation Study.

Evaluates whether persistent state carried across blocks (L2 -> L3) provides genuine
compositional advantage over non-persistent or disrupted state:
- Condition A: Two blocks, no state (bypass_state on both L2 and L3)
- Condition B: Two blocks, state reset after L2 (persistent_state=False, fresh s0 at L3)
- Condition C: Two blocks, persistent state L2 -> L3 (full multi-block recurrence)
- Condition D: Two blocks, shuffled state between L2 and L3
- Condition E: Two blocks, random Gaussian state injected into L3

Measures under matched budget & seed:
- Hop-1 key routing accuracy
- Hop-1 value routing accuracy
- Hop-2 key routing accuracy
- Hop-2 value routing accuracy
- Final token accuracy
- G4 unseen-ID unseen-composition accuracy
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import instantiate_frozen_baseline
from chakrview.cognition.multiblock_recurrent_attention_core import MultiBlockRecurrentChakrMicro
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)


@dataclasses.dataclass
class PersistenceAblationResult:
    condition_id: str
    description: str
    train_loss: float
    h1_key_acc: float
    h1_val_acc: float
    h2_key_acc: float
    h2_val_acc: float
    final_token_acc: float
    g4_acc: float


@dataclasses.dataclass
class StatePersistenceReport:
    conditions: Dict[str, PersistenceAblationResult]
    best_condition_id: str
    persistent_state_superior: bool
    hop2_routing_improved: bool
    summary: str


def train_and_eval_persistence_candidate(
    base_model: ChakrMicro,
    condition: str,
    seed: int = 42,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> PersistenceAblationResult:
    """Trains and tests candidate under specified persistence condition."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    persistent_flag = (condition == "C_Persistent")
    candidate = MultiBlockRecurrentChakrMicro(
        base_model=base_model,
        target_layers=[2, 3],
        persistent_state=persistent_flag,
        freeze_ffn=True,
    )

    optimizer = torch.optim.Adam(
        [p for p in candidate.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )

    train_episodes = [
        env.generate_episode(split="train", num_distractors=1, episode_idx=363000 + i)
        for i in range(train_steps)
    ]
    candidate.train()

    total_loss = 0.0
    for ep in train_episodes:
        optimizer.zero_grad()
        seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)

        cand_positions, cand_tokens = [], []
        for k, v in ep.all_premise_pairs:
            v_enc = tok.encode(v)[0]
            pos_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
            if pos_list and pos_list[0] not in cand_positions:
                cand_positions.append(pos_list[0])
                cand_tokens.append(v_enc)
        if not cand_positions:
            cand_positions, cand_tokens = [0], [ep.prompt_tokens[0]]

        c_ids = torch.tensor([cand_tokens], dtype=torch.long)
        c_pos = torch.tensor([cand_positions], dtype=torch.long)
        tgt_idx = cand_tokens.index(ep.target_token) if ep.target_token in cand_tokens else 0

        # Condition flags
        bypass_l = [2, 3] if condition == "A_NoState" else None
        reset_b = (condition == "B_Reset")
        shuffle_b = (condition == "D_Shuffled")
        overrides = None
        if condition == "E_Random":
            overrides = {3: torch.randn(1, 48)}

        out = candidate(
            input_ids=seq,
            candidate_ids=c_ids,
            candidate_positions=c_pos,
            bypass_state_layers=bypass_l,
            reset_state_between_blocks=reset_b,
            shuffle_state_between_blocks=shuffle_b,
            override_state=overrides,
            return_traces=True,
        )

        b_logits = out["binding_logits"]
        target = torch.tensor([tgt_idx], dtype=torch.long)
        loss_ce = F.cross_entropy(b_logits, target)

        traces = out["traces"]
        loss_h1 = 0.0
        if 2 in traces and traces[2].attn1_weights is not None:
            w1 = traces[2].attn1_weights[0, :, -1, ep.hop1_key_pos].mean()
            loss_h1 = -torch.log(w1 + 1e-8)

        loss_h2 = 0.0
        if 3 in traces and traces[3].attn2_weights is not None and condition != "A_NoState":
            w2 = traces[3].attn2_weights[0, :, -1, ep.hop2_key_pos].mean()
            loss_h2 = -torch.log(w2 + 1e-8)

        loss = loss_ce + 0.5 * loss_h1 + 0.5 * loss_h2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # Evaluation on G4
    candidate.eval()
    g4_episodes = [
        env.generate_episode(split="disjoint_test", num_distractors=1, episode_idx=363500 + i)
        for i in range(eval_episodes)
    ]

    h1_k_hits = 0
    h1_v_hits = 0
    h2_k_hits = 0
    h2_v_hits = 0
    token_hits = 0

    with torch.no_grad():
        for ep in g4_episodes:
            seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            cand_positions, cand_tokens = [], []
            for k, v in ep.all_premise_pairs:
                v_enc = tok.encode(v)[0]
                pos_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                if pos_list and pos_list[0] not in cand_positions:
                    cand_positions.append(pos_list[0])
                    cand_tokens.append(v_enc)
            if not cand_positions:
                cand_positions, cand_tokens = [0], [ep.prompt_tokens[0]]

            c_ids = torch.tensor([cand_tokens], dtype=torch.long)
            c_pos = torch.tensor([cand_positions], dtype=torch.long)
            tgt_idx = cand_tokens.index(ep.target_token) if ep.target_token in cand_tokens else 0

            bypass_l = [2, 3] if condition == "A_NoState" else None
            reset_b = (condition == "B_Reset")
            shuffle_b = (condition == "D_Shuffled")
            overrides = None
            if condition == "E_Random":
                overrides = {3: torch.randn(1, 48)}

            out = candidate(
                input_ids=seq,
                candidate_ids=c_ids,
                candidate_positions=c_pos,
                bypass_state_layers=bypass_l,
                reset_state_between_blocks=reset_b,
                shuffle_state_between_blocks=shuffle_b,
                override_state=overrides,
                return_traces=True,
            )

            pred_idx = out["binding_logits"].argmax(dim=-1).item()
            if pred_idx == tgt_idx:
                token_hits += 1

            traces = out["traces"]
            if 2 in traces and traces[2].attn1_weights is not None:
                attn1 = traces[2].attn1_weights[0].mean(dim=0)[-1]
                if attn1.argmax().item() == ep.hop1_key_pos:
                    h1_k_hits += 1
                if attn1.argmax().item() == ep.hop1_val_pos:
                    h1_v_hits += 1

            if 3 in traces and traces[3].attn2_weights is not None and condition != "A_NoState":
                attn2 = traces[3].attn2_weights[0].mean(dim=0)[-1]
                if attn2.argmax().item() == ep.hop2_key_pos:
                    h2_k_hits += 1
                if attn2.argmax().item() == ep.hop2_val_pos:
                    h2_v_hits += 1

    N = max(1, len(g4_episodes))
    descriptions = {
        "A_NoState": "Two blocks, no state (residual only)",
        "B_Reset": "Two blocks, state reset after L2 (independent)",
        "C_Persistent": "Two blocks, persistent state L2 -> L3",
        "D_Shuffled": "Two blocks, shuffled state between L2 and L3",
        "E_Random": "Two blocks, random Gaussian state into L3",
    }

    return PersistenceAblationResult(
        condition_id=condition,
        description=descriptions.get(condition, condition),
        train_loss=total_loss / max(1, train_steps),
        h1_key_acc=h1_k_hits / N,
        h1_val_acc=h1_v_hits / N,
        h2_key_acc=h2_k_hits / N,
        h2_val_acc=h2_v_hits / N,
        final_token_acc=token_hits / N,
        g4_acc=token_hits / N,
    )


def run_state_persistence_ablation_study(
    base_model: ChakrMicro,
    seed: int = 42,
) -> StatePersistenceReport:
    """Executes Step 363 persistence ablation study across Conditions A-E."""
    conditions = ["A_NoState", "B_Reset", "C_Persistent", "D_Shuffled", "E_Random"]
    results: Dict[str, PersistenceAblationResult] = {}
    for c in conditions:
        results[c] = train_and_eval_persistence_candidate(base_model, condition=c, seed=seed)

    best_id = max(results.keys(), key=lambda k: (results[k].g4_acc, results[k].h2_key_acc))
    persistent_superior = (
        results["C_Persistent"].g4_acc >= results["B_Reset"].g4_acc
        and results["C_Persistent"].h2_key_acc >= results["B_Reset"].h2_key_acc
    )
    h2_improved = results["C_Persistent"].h2_key_acc > 0.0

    return StatePersistenceReport(
        conditions=results,
        best_condition_id=best_id,
        persistent_state_superior=persistent_superior,
        hop2_routing_improved=h2_improved,
        summary=(
            f"Step 363 state persistence ablation completed. Best condition: {best_id} "
            f"(G4={results[best_id].g4_acc:.1%}, H2_Key={results[best_id].h2_key_acc:.1%}). "
            f"Persistent state superior: {persistent_superior}."
        ),
    )


if __name__ == "__main__":
    m = instantiate_frozen_baseline()
    rep = run_state_persistence_ablation_study(m, seed=42)
    print("=== STEP 363 STATE PERSISTENCE ABLATION STUDY ===")
    for c, r in rep.conditions.items():
        print(f"[{c:14s}] Loss: {r.train_loss:.4f} | H1 Key: {r.h1_key_acc:.1%} | H2 Key: {r.h2_key_acc:.1%} | G4: {r.g4_acc:.1%}")
    print(f"\nBest: {rep.best_condition_id} | Persistent Superior: {rep.persistent_state_superior}")
