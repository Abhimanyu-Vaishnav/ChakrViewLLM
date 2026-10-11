"""Step 365: Three-Block Controlled Escalation Study.

Compares:
1. Two-Block Candidate (Layers [2, 3])
2. Three-Block Candidate (Layers [2, 3, 4])

Evaluates under identical dataset, seed, and evaluation rules:
- Trainable parameter counts:
  * Two-block: 448,131 params
  * Three-block: 3 * 217,537 + 13,057 = 665,668 params
- Hop-1 key & value routing
- Hop-2 key & value routing
- Final token accuracy across G1, G2, G3, G4
- Evaluates whether adding a third block genuinely deepens multi-hop routing
  or merely increases memorization capacity.
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
class EscalationComparisonResult:
    config_name: str
    target_layers: List[int]
    trainable_params: int
    train_loss: float
    g1_acc: float
    g2_acc: float
    g3_acc: float
    g4_acc: float
    h1_key_acc: float
    h2_key_acc: float
    h2_val_acc: float


@dataclasses.dataclass
class ThreeBlockEscalationReport:
    two_block_result: EscalationComparisonResult
    three_block_result: EscalationComparisonResult
    g4_delta: float
    hop2_delta: float
    three_block_beneficial: bool
    summary: str


def train_and_eval_escalation_candidate(
    base_model: ChakrMicro,
    target_layers: List[int],
    config_name: str,
    seed: int = 42,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> EscalationComparisonResult:
    """Trains and tests candidate with specified target layers."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    candidate = MultiBlockRecurrentChakrMicro(
        base_model=base_model,
        target_layers=target_layers,
        persistent_state=True,
        freeze_ffn=True,
    )

    optimizer = torch.optim.Adam(
        [p for p in candidate.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )

    train_episodes = [
        env.generate_episode(split="train", num_distractors=1, episode_idx=365000 + i)
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

        out = candidate(
            input_ids=seq,
            candidate_ids=c_ids,
            candidate_positions=c_pos,
            return_traces=True,
        )

        b_logits = out["binding_logits"]
        target = torch.tensor([tgt_idx], dtype=torch.long)
        loss_ce = F.cross_entropy(b_logits, target)

        traces = out["traces"]
        loss_h1 = 0.0
        first_layer = target_layers[0]
        if first_layer in traces and traces[first_layer].attn1_weights is not None:
            w1 = traces[first_layer].attn1_weights[0, :, -1, ep.hop1_key_pos].mean()
            loss_h1 = -torch.log(w1 + 1e-8)

        loss_h2 = 0.0
        second_layer = target_layers[1]
        if second_layer in traces and traces[second_layer].attn2_weights is not None:
            w2 = traces[second_layer].attn2_weights[0, :, -1, ep.hop2_key_pos].mean()
            loss_h2 = -torch.log(w2 + 1e-8)

        loss = loss_ce + 0.5 * loss_h1 + 0.5 * loss_h2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # Evaluation on G1, G2, G3, G4
    candidate.eval()

    def eval_split(split_name: str) -> Tuple[float, float, float, float]:
        eps = [
            env.generate_episode(split=split_name, num_distractors=1, episode_idx=365500 + i)
            for i in range(eval_episodes)
        ]
        t_hits = 0
        h1_k_hits = 0
        h2_k_hits = 0
        h2_v_hits = 0
        with torch.no_grad():
            for ep in eps:
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

                out = candidate(
                    input_ids=seq,
                    candidate_ids=c_ids,
                    candidate_positions=c_pos,
                    return_traces=True,
                )
                pred_idx = out["binding_logits"].argmax(dim=-1).item()
                if pred_idx == tgt_idx:
                    t_hits += 1

                traces = out["traces"]
                if target_layers[0] in traces and traces[target_layers[0]].attn1_weights is not None:
                    attn1 = traces[target_layers[0]].attn1_weights[0].mean(dim=0)[-1]
                    if attn1.argmax().item() == ep.hop1_key_pos:
                        h1_k_hits += 1

                last_recurrent = target_layers[-1]
                if last_recurrent in traces and traces[last_recurrent].attn2_weights is not None:
                    attn2 = traces[last_recurrent].attn2_weights[0].mean(dim=0)[-1]
                    if attn2.argmax().item() == ep.hop2_key_pos:
                        h2_k_hits += 1
                    if attn2.argmax().item() == ep.hop2_val_pos:
                        h2_v_hits += 1

        N = max(1, len(eps))
        return t_hits / N, h1_k_hits / N, h2_k_hits / N, h2_v_hits / N

    g1, _, _, _ = eval_split("train")
    g2, _, _, _ = eval_split("val")
    g3, _, _, _ = eval_split("heldout_composition")
    g4, h1_k, h2_k, h2_v = eval_split("disjoint_test")

    return EscalationComparisonResult(
        config_name=config_name,
        target_layers=target_layers,
        trainable_params=candidate.trainable_param_count,
        train_loss=total_loss / max(1, train_steps),
        g1_acc=g1,
        g2_acc=g2,
        g3_acc=g3,
        g4_acc=g4,
        h1_key_acc=h1_k,
        h2_key_acc=h2_k,
        h2_val_acc=h2_v,
    )


def run_three_block_escalation_study(
    base_model: ChakrMicro,
    seed: int = 42,
) -> ThreeBlockEscalationReport:
    """Runs Step 365 comparison between two-block (L2+L3) and three-block (L2+L3+L4) candidates."""
    res_2b = train_and_eval_escalation_candidate(
        base_model=base_model,
        target_layers=[2, 3],
        config_name="TwoBlock_L2_L3",
        seed=seed,
    )

    res_3b = train_and_eval_escalation_candidate(
        base_model=base_model,
        target_layers=[2, 3, 4],
        config_name="ThreeBlock_L2_L3_L4",
        seed=seed,
    )

    g4_delta = res_3b.g4_acc - res_2b.g4_acc
    h2_delta = res_3b.h2_key_acc - res_2b.h2_key_acc
    beneficial = (g4_delta > 0.0 or h2_delta > 0.0)

    return ThreeBlockEscalationReport(
        two_block_result=res_2b,
        three_block_result=res_3b,
        g4_delta=g4_delta,
        hop2_delta=h2_delta,
        three_block_beneficial=beneficial,
        summary=(
            f"Step 365 escalation study completed: Two-block G4={res_2b.g4_acc:.1%}, H2={res_2b.h2_key_acc:.1%} "
            f"vs Three-block G4={res_3b.g4_acc:.1%}, H2={res_3b.h2_key_acc:.1%}. "
            f"G4 Delta = {g4_delta:+.1%}, H2 Delta = {h2_delta:+.1%}. Three-block beneficial: {beneficial}."
        ),
    )


if __name__ == "__main__":
    m = instantiate_frozen_baseline()
    rep = run_three_block_escalation_study(m, seed=42)
    print("=== STEP 365 THREE-BLOCK CONTROLLED ESCALATION STUDY ===")
    print(f"Two-Block   (L2+L3):    Params={rep.two_block_result.trainable_params:,} | G4={rep.two_block_result.g4_acc:.1%} | H2_K={rep.two_block_result.h2_key_acc:.1%}")
    print(f"Three-Block (L2+L3+L4): Params={rep.three_block_result.trainable_params:,} | G4={rep.three_block_result.g4_acc:.1%} | H2_K={rep.three_block_result.h2_key_acc:.1%}")
    print(f"Deltas: G4={rep.g4_delta:+.1%}, Hop-2={rep.hop2_delta:+.1%}")
    print(f"Three-Block Beneficial: {rep.three_block_beneficial}")
