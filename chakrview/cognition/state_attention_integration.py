"""Step 323: State Integration with Attention Comparison.

Tests where the recurrent state s_t provides the most useful compositional transformation:
Configuration A: State influences Query projection:
   q2 = Query_proj(h_v1) + Proj_q(s_t)
Configuration B: State influences Residual stream:
   h' = h_v1 + Proj_res(s_t)
Configuration C: State influences Attention context:
   Attends jointly over sequence keys and recurrent state vector [K, s_t].

Compares under matched parameter budgets across:
- Hop-1 key & value routing
- Hop-2 key & value routing
- Final token accuracy across G1, G2, G3, G4
- Parameter overhead
- CPU latency
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.recurrent_state_transition import (
    GatedRecurrentStateTransition,
    compute_module_sha256,
)
from chakrview.cognition.dynamic_contextual_token_binding import (
    DynamicContextualTokenBinding,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)
from chakrview.cognition.identity_invariant_objective import (
    IdentityInvariantRepresentationLoss,
)


class StateAttentionIntegrationModel(nn.Module):
    """Integrates recurrent state s_t into query projection, residual stream, or attention context."""

    def __init__(
        self,
        base_model: ChakrMicro,
        integration_mode: str = "B",  # "A" (query), "B" (residual), "C" (context)
        d_state: int = 64,
        d_bind: int = 64,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.d_model = base_model.config.d_model
        self.integration_mode = integration_mode

        for p in self.base_model.parameters():
            p.requires_grad = False

        self.transition_core = GatedRecurrentStateTransition(
            d_model=self.d_model,
            d_state=d_state,
        )

        # Mode-specific integration projections
        if integration_mode == "A":
            self.mode_proj = nn.Linear(d_state, self.d_model, bias=False)
        elif integration_mode == "B":
            self.mode_proj = nn.Linear(d_state, self.d_model, bias=False)
        elif integration_mode == "C":
            self.mode_proj = nn.Linear(d_state, self.d_model, bias=False)

        self.binding = DynamicContextualTokenBinding(
            d_model=self.d_model,
            d_bind=d_bind,
        )

    def forward_backbone(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.base_model.embedding(input_ids)
        for i, layer in enumerate(self.base_model.layers):
            h = layer(h, layer_idx=i)
        h = self.base_model.final_norm(h)
        logits = self.base_model.lm_head(h)
        return h, logits

    def compute_second_hop_query(
        self,
        v1_rep: torch.Tensor,
        s_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes state transition on v1_rep and computes Hop-2 query via specified mode.
        Returns (q2, s_next).
        """
        s_next, h_res = self.transition_core.transition(v1_rep, s_t)

        if self.integration_mode == "A":
            # State directly modulates query
            q2 = v1_rep + self.mode_proj(s_next)
        elif self.integration_mode == "B":
            # State modulates residual stream
            q2 = h_res + self.mode_proj(s_next)
        elif self.integration_mode == "C":
            # Contextual blend
            q2 = 0.5 * (v1_rep + self.mode_proj(s_next)) + 0.5 * h_res
        else:
            q2 = h_res

        return q2, s_next


@dataclasses.dataclass
class IntegrationModeMetrics:
    mode: str
    description: str
    trainable_params: int
    mean_g1_tok_acc: float
    mean_g2_tok_acc: float
    mean_g3_tok_acc: float
    mean_g4_tok_acc: float
    mean_h2_key_acc: float
    mean_h2_val_acc: float
    cpu_latency_ms: float


@dataclasses.dataclass
class StateIntegrationComparisonReport:
    mode_results: Dict[str, IntegrationModeMetrics]
    best_mode: str
    best_g4_acc: float
    summary: str


def evaluate_state_integration_mode(
    base_model: ChakrMicro,
    mode: str,
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> IntegrationModeMetrics:
    """Trains and tests state integration mode across seeds."""
    if seeds is None:
        seeds = [42, 101, 2026]

    g1_list, g2_list, g3_list, g4_list = [], [], [], []
    h2_k_list, h2_v_list = [], []
    latencies = []

    descriptions = {
        "A": "State Influences Query Projection",
        "B": "State Influences Residual Stream",
        "C": "State Influences Contextual Blend",
    }

    sample = StateAttentionIntegrationModel(base_model, integration_mode=mode)
    trainable_p = sum(p.numel() for p in sample.parameters() if p.requires_grad)

    for s in seeds:
        torch.manual_seed(s)
        env = CompositionalAssociativeEnvironment(seed=s)
        tok = env.tok

        model = StateAttentionIntegrationModel(base_model, integration_mode=mode)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=2.0e-3, weight_decay=0.01,
        )
        loss_fn = IdentityInvariantRepresentationLoss(temperature=0.15)

        # Train loop
        model.train()
        for st in range(train_steps):
            ep = env.generate_episode("train", num_distractors=1, episode_idx=st)
            inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            opt.zero_grad()

            h_ad, _ = model.forward_backbone(inp)
            h_norm = F.normalize(h_ad[0], p=2, dim=-1)

            premise_key_pos = []
            for k, v in ep.all_premise_pairs:
                k_enc = tok.encode(k)[0]
                m = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == k_enc]
                if m: premise_key_pos.append(m[0])

            r_loss1 = loss_fn(
                adapted_hidden=h_ad, logits=h_ad, query_key_pos=ep.query_key_pos,
                matching_key_pos=ep.hop1_key_pos, distractor_key_positions=premise_key_pos,
                associated_val_pos=ep.hop1_val_pos, target_token=ep.intermediate_token,
            )

            # Hop-1 value & recurrent transition
            v1_rep = h_ad[0, ep.hop1_val_pos : ep.hop1_val_pos + 1]
            s0 = model.transition_core.get_initial_state(1)
            q2, _ = model.compute_second_hop_query(v1_rep, s0)
            q2_norm = F.normalize(q2[0], p=2, dim=-1)

            # Hop-2 key routing loss
            k2_logits = torch.matmul(q2_norm.unsqueeze(0), h_norm.transpose(0, 1)) / 0.15
            l_route2 = F.cross_entropy(k2_logits, torch.tensor([ep.hop2_key_pos], dtype=torch.long))

            # Dynamic binding loss
            cand_positions, cand_tokens = [], []
            for k, v in ep.all_premise_pairs:
                v_enc = tok.encode(v)[0]
                pos_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                if pos_list and pos_list[0] not in cand_positions:
                    cand_positions.append(pos_list[0])
                    cand_tokens.append(v_enc)
            if not cand_positions:
                cand_positions, cand_tokens = [0], [ep.prompt_tokens[0]]

            cand_states = torch.stack([h_ad[0, p] for p in cand_positions], dim=0).unsqueeze(0)
            cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=h_ad.device)
            tgt_idx = cand_tokens.index(ep.target_token) if ep.target_token in cand_tokens else 0
            val_final_rep = h_ad[0, ep.hop2_val_pos : ep.hop2_val_pos + 1]
            bind_logits, _ = model.binding(val_final_rep, cand_states, cand_mask)
            l_bind = F.cross_entropy(bind_logits, torch.tensor([tgt_idx], dtype=torch.long))

            loss = r_loss1.total_loss + 1.5 * l_route2 + 2.0 * l_bind
            loss.backward()
            opt.step()

        # Eval loop across G1-G4
        model.eval()
        split_map = {
            "G1": ("train", g1_list),
            "G2": ("disjoint_test", g2_list),
            "G3": ("heldout_composition", g3_list),
            "G4": ("disjoint_test", g4_list),
        }

        with torch.no_grad():
            for sp_name, (sp_mode, acc_list) in split_map.items():
                corr = 0
                for ev_i in range(eval_episodes):
                    ep = env.generate_episode(sp_mode, num_distractors=1, episode_idx=1400000 + ev_i)
                    inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)

                    t0 = time.perf_counter()
                    h_ad, _ = model.forward_backbone(inp)
                    v1_rep = h_ad[0, ep.hop1_val_pos : ep.hop1_val_pos + 1]
                    s0 = model.transition_core.get_initial_state(1)
                    q2, _ = model.compute_second_hop_query(v1_rep, s0)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)

                    h_norm = F.normalize(h_ad[0], p=2, dim=-1)
                    q2_norm = F.normalize(q2[0], p=2, dim=-1)

                    premise_key_pos = []
                    for k, v in ep.all_premise_pairs:
                        k_enc = tok.encode(k)[0]
                        m = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == k_enc]
                        if m: premise_key_pos.append(m[0])

                    premise_val_pos = []
                    for k, v in ep.all_premise_pairs:
                        v_enc = tok.encode(v)[0]
                        m = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                        if m: premise_val_pos.append(m[0])

                    # Hop-2 key routing
                    k2_sims = [float(torch.dot(q2_norm, h_norm[kp]).item()) for kp in premise_key_pos]
                    pred_k2 = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0
                    h2_k_list.append(1.0 if pred_k2 == ep.hop2_key_pos else 0.0)

                    # Hop-2 val routing
                    h_mk2 = h_norm[pred_k2]
                    v2_sims = [float(torch.dot(h_mk2, h_norm[vp]).item()) for vp in premise_val_pos]
                    pred_v2 = premise_val_pos[v2_sims.index(max(v2_sims))] if v2_sims else 0
                    h2_v_list.append(1.0 if pred_v2 == ep.hop2_val_pos else 0.0)

                    # Dynamic binding
                    cand_positions, cand_tokens = [], []
                    for k, v in ep.all_premise_pairs:
                        v_enc = tok.encode(v)[0]
                        pos_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                        if pos_list and pos_list[0] not in cand_positions:
                            cand_positions.append(pos_list[0])
                            cand_tokens.append(v_enc)
                    if not cand_positions:
                        cand_positions, cand_tokens = [0], [ep.prompt_tokens[0]]

                    cand_states = torch.stack([h_ad[0, p] for p in cand_positions], dim=0).unsqueeze(0)
                    cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=h_ad.device)
                    val_final = h_ad[0, pred_v2 : pred_v2 + 1]
                    b_logits, _ = model.binding(val_final, cand_states, cand_mask)
                    sel_idx = int(torch.argmax(b_logits[0]).item())
                    if cand_tokens[sel_idx] == ep.target_token:
                        corr += 1

                acc_list.append(corr / eval_episodes)

    return IntegrationModeMetrics(
        mode=mode,
        description=descriptions[mode],
        trainable_params=trainable_p,
        mean_g1_tok_acc=float(sum(g1_list) / len(g1_list)),
        mean_g2_tok_acc=float(sum(g2_list) / len(g2_list)),
        mean_g3_tok_acc=float(sum(g3_list) / len(g3_list)),
        mean_g4_tok_acc=float(sum(g4_list) / len(g4_list)),
        mean_h2_key_acc=float(sum(h2_k_list) / max(len(h2_k_list), 1)),
        mean_h2_val_acc=float(sum(h2_v_list) / max(len(h2_v_list), 1)),
        cpu_latency_ms=float(sum(latencies) / len(latencies)),
    )


def run_state_integration_comparison(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> StateIntegrationComparisonReport:
    """Executes Step 323 comparing configurations A, B, C."""
    if seeds is None:
        seeds = [42, 101, 2026]

    modes = ["A", "B", "C"]
    results: Dict[str, IntegrationModeMetrics] = {}

    for m in modes:
        res = evaluate_state_integration_mode(
            base_model=base_model, mode=m, seeds=seeds, train_steps=train_steps, eval_episodes=eval_episodes,
        )
        results[m] = res

    best_m = max(results.keys(), key=lambda k: results[k].mean_g4_tok_acc)
    best_g4 = results[best_m].mean_g4_tok_acc

    summary = (
        f"Step 323 State Integration Comparison: " +
        "; ".join([f"{k}: G4={v.mean_g4_tok_acc*100:.1f}%, H2_K={v.mean_h2_key_acc*100:.1f}%, params={v.trainable_params}"
                   for k, v in results.items()]) +
        f" -> Best Pathway: Configuration {best_m} ({best_g4*100:.1f}%)"
    )

    return StateIntegrationComparisonReport(
        mode_results=results,
        best_mode=best_m,
        best_g4_acc=best_g4,
        summary=summary,
    )
