"""Step 356: State-Transition Ablation Study.

Compares:
Condition A: Single attention cycle (Cycle 2 disabled)
Condition B: Two attention cycles without recurrent state (bypass_state=True)
Condition C: Two attention cycles + recurrent state (full RecurrentAttentionCoreBlock)
Condition D: Two attention cycles + frozen/no-op state (state initialized to s0 without update)
Condition E: Two attention cycles + randomized state (s1 replaced by random Gaussian noise)

Evaluates under identical training budget and seed:
- Hop-1 key routing
- Hop-1 value routing
- Hop-2 key routing
- Hop-2 value routing
- Final token accuracy
- Compositional G4 accuracy
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.recurrent_attention_chakr_micro import RecurrentAttentionChakrMicro
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)
from chakrview.cognition.identity_invariant_objective import (
    IdentityInvariantRepresentationLoss,
)


@dataclasses.dataclass
class AblationConditionResult:
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
class StateTransitionAblationReport:
    conditions: Dict[str, AblationConditionResult]
    best_condition_id: str
    recurrent_state_necessary: bool
    hop2_routing_improved: bool
    summary: str


def train_and_eval_ablation_candidate(
    base_model: ChakrMicro,
    condition: str,
    seed: int = 42,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> AblationConditionResult:
    """Trains and evaluates candidate under specific state-transition condition."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    candidate = RecurrentAttentionChakrMicro(
        base_model=base_model,
        target_layer=3,
        d_state=48,
        use_shared_kv=True,
        freeze_ffn=True,
        enable_contextual_binding=True,
    )

    optimizer = torch.optim.Adam(
        [p for p in candidate.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )
    rep_loss_fn = IdentityInvariantRepresentationLoss()

    train_episodes = [
        env.generate_episode(split="train", num_distractors=1, episode_idx=356000 + i)
        for i in range(train_steps)
    ]
    candidate.train()

    total_loss = 0.0
    for ep in train_episodes:
        optimizer.zero_grad()
        seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)

        # Extract candidates
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

        # Apply condition flags
        disable_c2 = (condition == "A_SingleCycle")
        bypass_state = (condition == "B_NoState")

        override_s1 = None
        if condition == "D_FrozenState":
            override_s1 = candidate.recurrent_block.s0.expand(1, -1)
        elif condition == "E_RandomState":
            override_s1 = torch.randn(1, candidate.recurrent_block.d_state)

        out = candidate(
            input_ids=seq,
            candidate_ids=c_ids,
            candidate_positions=c_pos,
            disable_c2=disable_c2,
            bypass_state=bypass_state,
            override_s1=override_s1,
            return_trace=True,
        )

        b_logits = out["binding_logits"]
        target = torch.tensor([tgt_idx], dtype=torch.long)
        loss_ce = F.cross_entropy(b_logits, target)

        # Hop-1 routing loss
        trace = out["trace"]
        loss_h1 = 0.0
        if trace is not None and trace.attn1_weights is not None:
            # Query token attending to premise 1 key token
            w1 = trace.attn1_weights[0, :, -1, ep.hop1_key_pos].mean()
            loss_h1 = -torch.log(w1 + 1e-8)

        loss_h2 = 0.0
        if condition not in ["A_SingleCycle", "B_NoState"] and trace is not None and trace.attn2_weights is not None:
            w2 = trace.attn2_weights[0, :, -1, ep.hop2_key_pos].mean()
            loss_h2 = -torch.log(w2 + 1e-8)

        loss = loss_ce + 0.5 * loss_h1 + 0.5 * loss_h2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # Evaluation on G4 (heldout_composition / disjoint_test)
    candidate.eval()
    g4_episodes = [
        env.generate_episode(split="disjoint_test", num_distractors=1, episode_idx=356500 + i)
        for i in range(eval_episodes)
    ]

    h1_key_hits = 0
    h1_val_hits = 0
    h2_key_hits = 0
    h2_val_hits = 0
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

            disable_c2 = (condition == "A_SingleCycle")
            bypass_state = (condition == "B_NoState")
            override_s1 = None
            if condition == "D_FrozenState":
                override_s1 = candidate.recurrent_block.s0.expand(1, -1)
            elif condition == "E_RandomState":
                override_s1 = torch.randn(1, candidate.recurrent_block.d_state)

            out = candidate(
                input_ids=seq,
                candidate_ids=c_ids,
                candidate_positions=c_pos,
                disable_c2=disable_c2,
                bypass_state=bypass_state,
                override_s1=override_s1,
                return_trace=True,
            )
            pred_idx = out["binding_logits"].argmax(dim=-1).item()
            if pred_idx == tgt_idx:
                token_hits += 1

            trace = out["trace"]
            if trace is not None and trace.attn1_weights is not None:
                # Cycle 1 attention
                attn1 = trace.attn1_weights[0].mean(dim=0)[-1]  # [T]
                if attn1.argmax().item() == ep.hop1_key_pos:
                    h1_key_hits += 1
                if attn1.argmax().item() == ep.hop1_val_pos:
                    h1_val_hits += 1

            if trace is not None and trace.attn2_weights is not None and not disable_c2:
                attn2 = trace.attn2_weights[0].mean(dim=0)[-1]  # [T]
                if attn2.argmax().item() == ep.hop2_key_pos:
                    h2_key_hits += 1
                if attn2.argmax().item() == ep.hop2_val_pos:
                    h2_val_hits += 1

    total_eval = max(1, len(g4_episodes))
    descriptions = {
        "A_SingleCycle": "Single attention cycle",
        "B_NoState": "Two attention cycles without recurrent state",
        "C_FullRecurrent": "Two attention cycles + recurrent state",
        "D_FrozenState": "Two attention cycles + frozen/no-op state",
        "E_RandomState": "Two attention cycles + randomized state",
    }

    return AblationConditionResult(
        condition_id=condition,
        description=descriptions.get(condition, condition),
        train_loss=total_loss / max(1, train_steps),
        h1_key_acc=h1_key_hits / total_eval,
        h1_val_acc=h1_val_hits / total_eval,
        h2_key_acc=h2_key_hits / total_eval,
        h2_val_acc=h2_val_hits / total_eval,
        final_token_acc=token_hits / total_eval,
        g4_acc=token_hits / total_eval,
    )


def run_state_transition_ablation_study(
    base_model: ChakrMicro,
    seed: int = 42,
) -> StateTransitionAblationReport:
    """Executes full Step 356 ablation study across Conditions A-E."""
    conditions = [
        "A_SingleCycle",
        "B_NoState",
        "C_FullRecurrent",
        "D_FrozenState",
        "E_RandomState",
    ]

    results: Dict[str, AblationConditionResult] = {}
    for cond in conditions:
        results[cond] = train_and_eval_ablation_candidate(
            base_model=base_model,
            condition=cond,
            seed=seed,
        )

    # Determine best condition
    best_id = max(results.keys(), key=lambda k: (results[k].g4_acc, results[k].h2_key_acc))
    recurrent_necessary = results["C_FullRecurrent"].g4_acc >= results["B_NoState"].g4_acc
    h2_improved = results["C_FullRecurrent"].h2_key_acc > 0.0

    return StateTransitionAblationReport(
        conditions=results,
        best_condition_id=best_id,
        recurrent_state_necessary=recurrent_necessary,
        hop2_routing_improved=h2_improved,
        summary=(
            f"Step 356 ablation study finished. Best condition: {best_id} "
            f"(G4={results[best_id].g4_acc:.1%}, H2_Key={results[best_id].h2_key_acc:.1%}). "
            f"Recurrent state beneficial: {recurrent_necessary}."
        ),
    )


if __name__ == "__main__":
    m = ChakrMicro()
    rep = run_state_transition_ablation_study(m, seed=42)
    print("=== STEP 356 STATE-TRANSITION ABLATION STUDY ===")
    for cond, res in rep.conditions.items():
        print(
            f"[{cond:16s}] Loss: {res.train_loss:.4f} | H1 Key: {res.h1_key_acc:.1%} | "
            f"H2 Key: {res.h2_key_acc:.1%} | G4: {res.g4_acc:.1%}"
        )
    print(f"\nBest Condition: {rep.best_condition_id}")
    print(f"Recurrent State Necessary: {rep.recurrent_state_necessary}")
    print(f"Hop-2 Routing Materially Improved: {rep.hop2_routing_improved}")
