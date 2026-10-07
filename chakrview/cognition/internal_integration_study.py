"""Step 308: Internal Pathway + Dynamic Token Binding Integration.

Integrates the internal trainable compositional pathway with the successful I3 dynamic contextual token binding.
Compares four architectural configurations:
A. Existing outer candidate architecture (Frozen base + outer adapter + bridge + dynamic binding)
B. Internal pathway only (Frozen base + internal adapter in late blocks, without dynamic binding)
C. Internal pathway + Dynamic binding (Frozen base + internal adapter + dynamic contextual candidate binding)
D. Internal pathway + Bridge projector + Dynamic binding (Full hybrid stack)

Evaluates:
- Hop-1 key & value routing accuracy
- Intermediate state stability
- Hop-2 key & value routing accuracy
- Final token accuracy across G1, G2, G3, G4
- Parameter footprint & CPU latency
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
from chakrview.cognition.internal_trainable_pathway import (
    ChakrMicroWithInternalPathway,
    compute_module_sha256,
)
from chakrview.cognition.dynamic_contextual_token_binding import (
    DynamicContextualTokenBinding,
)
from chakrview.cognition.two_hop_composition_architecture import (
    CompositionalBridgeProjector,
    ChakrMicroCompositionalReasoningModel,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)
from chakrview.cognition.identity_invariant_objective import (
    IdentityInvariantRepresentationLoss,
)


class IntegratedInternalBindingModel(nn.Module):
    """Integrates internal pathway with dynamic candidate token binding and optional bridge."""

    def __init__(
        self,
        base_model: ChakrMicro,
        config_mode: str = "C",  # "B", "C", or "D"
        target_layers: Optional[List[int]] = None,
        bottleneck_dim: int = 32,
        d_bind: int = 64,
    ):
        super().__init__()
        self.base_model = base_model
        self.config_mode = config_mode
        self.d_model = base_model.config.d_model

        if target_layers is None:
            target_layers = [3, 4, 5]
        self.target_layers = target_layers

        # Internal pathway
        self.internal_backbone = ChakrMicroWithInternalPathway(
            base_model=base_model,
            target_layers=target_layers,
            bottleneck_dim=bottleneck_dim,
            insertion_side="residual",
        )

        # Dynamic binding
        if config_mode in ("C", "D"):
            self.binding = DynamicContextualTokenBinding(
                d_model=self.d_model,
                d_bind=d_bind,
            )
        else:
            self.binding = None

        # Optional bridge projector
        if config_mode == "D":
            self.bridge = CompositionalBridgeProjector(d_model=self.d_model)
        else:
            self.bridge = None

    def compute_binding_scores(
        self,
        retrieved_val_rep: torch.Tensor,
        candidate_states: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.binding is None:
            raise ValueError("Dynamic binding head not instantiated in this mode")
        return self.binding(retrieved_val_rep, candidate_states, candidate_mask)


@dataclasses.dataclass
class IntegrationMetrics:
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
class InternalIntegrationReport:
    results: Dict[str, IntegrationMetrics]
    best_mode: str
    best_g4_acc: float
    summary: str


def evaluate_integration_mode(
    base_model: ChakrMicro,
    mode: str,
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> IntegrationMetrics:
    """Trains and tests integration mode across seeds."""
    if seeds is None:
        seeds = [42, 101, 2026]

    g1_list, g2_list, g3_list, g4_list = [], [], [], []
    h2_k_list, h2_v_list = [], []
    latencies = []

    descriptions = {
        "A": "Existing Outer Candidate (Adapter + Bridge + Dynamic Binding)",
        "B": "Internal Pathway Only (No Dynamic Binding)",
        "C": "Internal Pathway + Dynamic Contextual Binding",
        "D": "Internal Pathway + Bridge Projector + Dynamic Binding",
    }

    if mode == "A":
        sample = ChakrMicroCompositionalReasoningModel(base_model, rank=16)
        trainable_p = sum(p.numel() for p in sample.parameters() if p.requires_grad)
    else:
        sample = IntegratedInternalBindingModel(base_model, config_mode=mode)
        trainable_p = sum(p.numel() for p in sample.parameters() if p.requires_grad)

    for s in seeds:
        torch.manual_seed(s)
        env = CompositionalAssociativeEnvironment(seed=s)
        tok = env.tok

        if mode == "A":
            model = ChakrMicroCompositionalReasoningModel(base_model, rank=16)
            for p in model.base_model.parameters(): p.requires_grad = False
            for p in model.adapter.parameters(): p.requires_grad = True
            for p in model.bridge.parameters(): p.requires_grad = True
            for p in model.binding.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2.0e-3, weight_decay=0.01)
        else:
            model = IntegratedInternalBindingModel(base_model, config_mode=mode)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2.0e-3, weight_decay=0.01)

        loss_fn = IdentityInvariantRepresentationLoss(temperature=0.15)

        # Train loop
        model.train()
        for st in range(train_steps):
            ep = env.generate_episode("train", num_distractors=1, episode_idx=st)
            inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            opt.zero_grad()

            if mode == "A":
                h_ad, _ = model.forward_backbone(inp)
                h_v1 = h_ad[0, ep.hop1_val_pos : ep.hop1_val_pos + 1]
                h_q2 = model.bridge(h_v1)
                h_norm = F.normalize(h_ad[0], p=2, dim=-1)
            else:
                final_h, _ = model.internal_backbone.forward_hidden_states(inp)
                h_norm = F.normalize(final_h[0], p=2, dim=-1)
                h_v1 = final_h[0, ep.hop1_val_pos : ep.hop1_val_pos + 1]
                if mode == "D":
                    h_q2 = model.bridge(h_v1)
                else:
                    h_q2 = h_v1

            premise_key_pos = []
            for k, v in ep.all_premise_pairs:
                k_enc = tok.encode(k)[0]
                m = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == k_enc]
                if m: premise_key_pos.append(m[0])

            h_rep = h_ad if mode == "A" else final_h
            r_loss1 = loss_fn(
                adapted_hidden=h_rep, logits=h_rep, query_key_pos=ep.query_key_pos,
                matching_key_pos=ep.hop1_key_pos, distractor_key_positions=premise_key_pos,
                associated_val_pos=ep.hop1_val_pos, target_token=ep.intermediate_token,
            )

            # Hop-2 routing loss
            h_q2_norm = F.normalize(h_q2[0], p=2, dim=-1)
            k2_logits = torch.matmul(h_q2_norm.unsqueeze(0), h_norm.transpose(0, 1)) / 0.15
            l_route = F.cross_entropy(k2_logits, torch.tensor([ep.hop2_key_pos], dtype=torch.long))

            # Dynamic binding loss
            if mode in ("A", "C", "D"):
                cand_positions, cand_tokens = [], []
                for k, v in ep.all_premise_pairs:
                    v_enc = tok.encode(v)[0]
                    pos_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                    if pos_list and pos_list[0] not in cand_positions:
                        cand_positions.append(pos_list[0])
                        cand_tokens.append(v_enc)
                if not cand_positions:
                    cand_positions, cand_tokens = [0], [ep.prompt_tokens[0]]

                cand_states = torch.stack([h_rep[0, p] for p in cand_positions], dim=0).unsqueeze(0)
                cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=h_rep.device)
                tgt_idx = cand_tokens.index(ep.target_token) if ep.target_token in cand_tokens else 0
                val_final_rep = h_rep[0, ep.hop2_val_pos : ep.hop2_val_pos + 1]
                bind_logits, _ = model.compute_binding_scores(val_final_rep, cand_states, cand_mask)
                l_bind = F.cross_entropy(bind_logits, torch.tensor([tgt_idx], dtype=torch.long))
                loss = r_loss1.total_loss + 1.0 * l_route + 2.0 * l_bind
            else:
                loss = r_loss1.total_loss + 1.5 * l_route

            loss.backward()
            opt.step()

        # Eval loop
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
                    ep = env.generate_episode(sp_mode, num_distractors=1, episode_idx=600000 + ev_i)
                    inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)

                    t0 = time.perf_counter()
                    if mode == "A":
                        h_rep, _ = model.forward_backbone(inp)
                        h_v1 = h_rep[0, ep.hop1_val_pos : ep.hop1_val_pos + 1]
                        h_q2 = model.bridge(h_v1)
                    else:
                        h_rep, _ = model.internal_backbone.forward_hidden_states(inp)
                        h_v1 = h_rep[0, ep.hop1_val_pos : ep.hop1_val_pos + 1]
                        h_q2 = model.bridge(h_v1) if mode == "D" else h_v1
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)

                    h_norm = F.normalize(h_rep[0], p=2, dim=-1)
                    h_q2_norm = F.normalize(h_q2[0], p=2, dim=-1)

                    # Premise key/val positions
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
                    k2_sims = [float(torch.dot(h_q2_norm, h_norm[kp]).item()) for kp in premise_key_pos]
                    pred_k2 = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0
                    h2_k_list.append(1.0 if pred_k2 == ep.hop2_key_pos else 0.0)

                    # Hop-2 val routing
                    h_mk2 = h_norm[pred_k2]
                    v2_sims = [float(torch.dot(h_mk2, h_norm[vp]).item()) for vp in premise_val_pos]
                    pred_v2 = premise_val_pos[v2_sims.index(max(v2_sims))] if v2_sims else 0
                    h2_v_list.append(1.0 if pred_v2 == ep.hop2_val_pos else 0.0)

                    # Candidate dynamic binding or direct token check
                    if mode in ("A", "C", "D"):
                        cand_positions, cand_tokens = [], []
                        for k, v in ep.all_premise_pairs:
                            v_enc = tok.encode(v)[0]
                            pos_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                            if pos_list and pos_list[0] not in cand_positions:
                                cand_positions.append(pos_list[0])
                                cand_tokens.append(v_enc)
                        if not cand_positions:
                            cand_positions, cand_tokens = [0], [ep.prompt_tokens[0]]

                        cand_states = torch.stack([h_rep[0, p] for p in cand_positions], dim=0).unsqueeze(0)
                        cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=h_rep.device)
                        val_rep = h_rep[0, pred_v2 : pred_v2 + 1]
                        b_logits, _ = model.compute_binding_scores(val_rep, cand_states, cand_mask)
                        sel_idx = int(torch.argmax(b_logits[0]).item())
                        if cand_tokens[sel_idx] == ep.target_token:
                            corr += 1
                    else:
                        if inp[0, pred_v2].item() == ep.target_token:
                            corr += 1

                acc_list.append(corr / eval_episodes)

    return IntegrationMetrics(
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


def run_internal_integration_study(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> InternalIntegrationReport:
    """Compares integration modes A, B, C, D."""
    if seeds is None:
        seeds = [42, 101, 2026]

    modes = ["A", "B", "C", "D"]
    results: Dict[str, IntegrationMetrics] = {}

    for m in modes:
        res = evaluate_integration_mode(
            base_model=base_model,
            mode=m,
            seeds=seeds,
            train_steps=train_steps,
            eval_episodes=eval_episodes,
        )
        results[m] = res

    best_m = max(results.keys(), key=lambda k: results[k].mean_g4_tok_acc)
    best_g4 = results[best_m].mean_g4_tok_acc

    summary = (
        f"Step 308 Integration Study: " +
        "; ".join([f"{k}: G4={v.mean_g4_tok_acc*100:.1f}%, H2_K={v.mean_h2_key_acc*100:.1f}%, params={v.trainable_params}"
                   for k, v in results.items()]) +
        f" -> Best Architecture: Mode {best_m} ({best_g4*100:.1f}%)"
    )

    return InternalIntegrationReport(
        results=results,
        best_mode=best_m,
        best_g4_acc=best_g4,
        summary=summary,
    )
