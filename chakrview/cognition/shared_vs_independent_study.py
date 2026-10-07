"""Step 364: Shared vs Independent State Transitions Study.

Compares 4 architectural weight-sharing configurations across the two-block core:
- Option A: Shared recurrent transition weights (W_z, W_r, W_n shared across L2 & L3)
- Option B: Independent recurrent transition weights (separate W_z, W_r, W_n for L2 & L3)
- Option C: Shared transition + independent query projection state_to_q2
- Option D: Independent transition + independent query projection state_to_q2

Evaluates under matched compute and data budget:
- Trainable parameter counts
- G1, G2, G3, G4 accuracies
- Hop-1 & Hop-2 key routing
- Determines whether compositional reasoning benefits from a single unified transition rule
  or stage-specific transformations.
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
class WeightSharingResult:
    option_id: str
    description: str
    trainable_params: int
    train_loss: float
    g1_acc: float
    g2_acc: float
    g3_acc: float
    g4_acc: float
    h1_key_acc: float
    h2_key_acc: float


@dataclasses.dataclass
class SharedVsIndependentReport:
    options: Dict[str, WeightSharingResult]
    best_option_id: str
    sharing_beneficial: bool
    summary: str


def train_and_eval_sharing_candidate(
    base_model: ChakrMicro,
    option_id: str,
    seed: int = 42,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> WeightSharingResult:
    """Trains and tests candidate under specified weight-sharing configuration."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    shared_flag = (option_id in ["A_SharedTransition", "C_SharedTransIndQ"])
    candidate = MultiBlockRecurrentChakrMicro(
        base_model=base_model,
        target_layers=[2, 3],
        persistent_state=True,
        shared_transition=shared_flag,
        freeze_ffn=True,
    )

    optimizer = torch.optim.Adam(
        [p for p in candidate.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )

    train_episodes = [
        env.generate_episode(split="train", num_distractors=1, episode_idx=364000 + i)
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
        if 2 in traces and traces[2].attn1_weights is not None:
            w1 = traces[2].attn1_weights[0, :, -1, ep.hop1_key_pos].mean()
            loss_h1 = -torch.log(w1 + 1e-8)

        loss_h2 = 0.0
        if 3 in traces and traces[3].attn2_weights is not None:
            w2 = traces[3].attn2_weights[0, :, -1, ep.hop2_key_pos].mean()
            loss_h2 = -torch.log(w2 + 1e-8)

        loss = loss_ce + 0.5 * loss_h1 + 0.5 * loss_h2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # Evaluation on G1, G2, G3, G4
    candidate.eval()

    def eval_split(split_name: str) -> Tuple[float, float, float]:
        eps = [
            env.generate_episode(split=split_name, num_distractors=1, episode_idx=364500 + i)
            for i in range(eval_episodes)
        ]
        t_hits = 0
        h1_k_hits = 0
        h2_k_hits = 0
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
                if 2 in traces and traces[2].attn1_weights is not None:
                    attn1 = traces[2].attn1_weights[0].mean(dim=0)[-1]
                    if attn1.argmax().item() == ep.hop1_key_pos:
                        h1_k_hits += 1

                if 3 in traces and traces[3].attn2_weights is not None:
                    attn2 = traces[3].attn2_weights[0].mean(dim=0)[-1]
                    if attn2.argmax().item() == ep.hop2_key_pos:
                        h2_k_hits += 1

        N = max(1, len(eps))
        return t_hits / N, h1_k_hits / N, h2_k_hits / N

    g1, _, _ = eval_split("train")
    g2, _, _ = eval_split("val")
    g3, _, _ = eval_split("heldout_composition")
    g4, h1_k, h2_k = eval_split("disjoint_test")

    descriptions = {
        "A_SharedTransition": "Shared recurrent transition weights",
        "B_IndepTransition": "Independent transition weights",
        "C_SharedTransIndQ": "Shared transition + independent query projection",
        "D_IndepTransIndQ": "Independent transition + independent query projection",
    }

    return WeightSharingResult(
        option_id=option_id,
        description=descriptions.get(option_id, option_id),
        trainable_params=candidate.trainable_param_count,
        train_loss=total_loss / max(1, train_steps),
        g1_acc=g1,
        g2_acc=g2,
        g3_acc=g3,
        g4_acc=g4,
        h1_key_acc=h1_k,
        h2_key_acc=h2_k,
    )


def run_shared_vs_independent_study(
    base_model: ChakrMicro,
    seed: int = 42,
) -> SharedVsIndependentReport:
    """Executes Step 364 study across Options A-D."""
    options = [
        "A_SharedTransition",
        "B_IndepTransition",
        "C_SharedTransIndQ",
        "D_IndepTransIndQ",
    ]
    results: Dict[str, WeightSharingResult] = {}
    for opt in options:
        results[opt] = train_and_eval_sharing_candidate(base_model, option_id=opt, seed=seed)

    best_id = max(results.keys(), key=lambda k: (results[k].g4_acc, results[k].h2_key_acc))
    sharing_beneficial = (results["A_SharedTransition"].g4_acc >= results["B_IndepTransition"].g4_acc)

    return SharedVsIndependentReport(
        options=results,
        best_option_id=best_id,
        sharing_beneficial=sharing_beneficial,
        summary=(
            f"Step 364 study finished: Best option is {best_id} "
            f"(G4={results[best_id].g4_acc:.1%}, H2_Key={results[best_id].h2_key_acc:.1%}). "
            f"Weight sharing beneficial: {sharing_beneficial}."
        ),
    )


if __name__ == "__main__":
    m = instantiate_frozen_baseline()
    rep = run_shared_vs_independent_study(m, seed=42)
    print("=== STEP 364 SHARED VS INDEPENDENT STUDY ===")
    for o, r in rep.options.items():
        print(f"[{o:18s}] Params: {r.trainable_params:,} | G4: {r.g4_acc:.1%} | H2_K: {r.h2_key_acc:.1%}")
    print(f"\nBest Option: {rep.best_option_id} | Sharing Beneficial: {rep.sharing_beneficial}")
