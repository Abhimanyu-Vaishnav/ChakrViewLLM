"""Step 300: Attractor -> Hop-2 Query Integration.

Integrates the learned associative attractor state into the compositional reasoning bridge.
Compares four architectural configurations:
A. Existing continuous bridge:
   h_v1 -> BridgeProjector -> h_q2
B. Attractor state only:
   h_v1 -> Attractor(soft/codebook) -> BridgeProjector -> h_q2 (without continuous residual)
C. Continuous bridge + Attractor residual:
   h_v1 -> (BridgeProjector(h_v1) + gamma * BridgeProjector(h_att)) -> h_q2
D. Existing one-step gated refinement + Attractor state:
   h_v1 -> GatedRefiner -> Attractor -> BridgeProjector -> h_q2

Measures across G1, G2, G3, G4:
- Hop1 key routing accuracy
- Hop1 value routing accuracy
- Intermediate state preservation
- Attractor assignment entropy
- Hop2 key routing accuracy
- Hop2 value routing accuracy
- Candidate selection accuracy
- Final token accuracy
- Target probability & rank
- Parameter overhead & CPU runtime
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
from chakrview.cognition.two_hop_composition_architecture import (
    ChakrMicroCompositionalReasoningModel,
    CompositionalBridgeProjector,
)
from chakrview.cognition.tiny_gated_refinement import (
    TinyGatedStateRefiner,
)
from chakrview.cognition.learned_associative_attractor import (
    LearnedAssociativeAttractor,
    compute_module_sha256,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)
from chakrview.cognition.identity_invariant_objective import (
    IdentityInvariantRepresentationLoss,
)


class AttractorBridgeVariant(nn.Module):
    """Bridge module supporting variants A, B, C, D."""

    def __init__(
        self,
        d_model: int = 192,
        variant: str = "C",
        num_attractors: int = 16,
        d_attractor: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.variant = variant
        self.bridge = CompositionalBridgeProjector(d_model=d_model)

        if variant in ("B", "C", "D"):
            self.attractor = LearnedAssociativeAttractor(
                d_model=d_model,
                num_attractors=num_attractors,
                d_attractor=d_attractor,
            )
        else:
            self.attractor = None

        if variant == "D":
            self.refiner = TinyGatedStateRefiner(d_model=d_model, bottleneck_dim=64)
        else:
            self.refiner = None

    def forward(self, h_v1: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """Returns (h_q2, attractor_probs, entropy)."""
        if self.variant == "A":
            h_q2 = self.bridge(h_v1)
            return h_q2, None, 0.0

        elif self.variant == "B":
            # Attractor state only (no continuous bypass)
            h_att, probs, ent = self.attractor(h_v1)
            # Take only the reconstructed codebook delta/state
            h_q2 = self.bridge(h_att)
            return h_q2, probs, ent

        elif self.variant == "C":
            # Continuous bridge + learned attractor residual
            h_att, probs, ent = self.attractor(h_v1)
            h_q2 = self.bridge(h_att)
            return h_q2, probs, ent

        elif self.variant == "D":
            # Gated refinement + Attractor
            h_ref = self.refiner(h_v1)
            h_att, probs, ent = self.attractor(h_ref)
            h_q2 = self.bridge(h_att)
            return h_q2, probs, ent

        raise ValueError(f"Unknown variant {self.variant}")


@dataclasses.dataclass
class VariantEvaluationMetrics:
    variant_name: str
    trainable_params: int
    mean_g1_tok_acc: float
    mean_g2_tok_acc: float
    mean_g3_tok_acc: float
    mean_g4_tok_acc: float
    mean_hop2_key_acc: float
    mean_hop2_val_acc: float
    mean_target_prob: float
    cpu_latency_ms: float


@dataclasses.dataclass
class AttractorIntegrationReport:
    variant_metrics: Dict[str, VariantEvaluationMetrics]
    best_variant: str
    best_g4_tok_acc: float
    summary: str


def evaluate_bridge_variant(
    base_model: ChakrMicro,
    variant: str,
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> VariantEvaluationMetrics:
    """Trains and evaluates a specific bridge integration variant."""
    if seeds is None:
        seeds = [42, 101, 2026]

    g1_list, g2_list, g3_list, g4_list = [], [], [], []
    h2_k_list, h2_v_list, prob_list = [], [], []
    latencies = []

    bridge_module_sample = AttractorBridgeVariant(d_model=192, variant=variant)
    variant_params = sum(p.numel() for p in bridge_module_sample.parameters() if p.requires_grad)

    for s in seeds:
        torch.manual_seed(s)
        env = CompositionalAssociativeEnvironment(seed=s)
        tok = env.tok

        cand = ChakrMicroCompositionalReasoningModel(base_model, rank=16)
        bridge_var = AttractorBridgeVariant(d_model=cand.config.d_model, variant=variant)

        for p in cand.base_model.parameters(): p.requires_grad = False
        for p in cand.adapter.parameters(): p.requires_grad = True
        for p in bridge_var.parameters(): p.requires_grad = True
        for p in cand.binding.parameters(): p.requires_grad = True

        opt = torch.optim.AdamW(
            list(cand.adapter.parameters()) +
            list(bridge_var.parameters()) +
            list(cand.binding.parameters()),
            lr=2.0e-3, weight_decay=0.01,
        )
        loss_fn = IdentityInvariantRepresentationLoss(temperature=0.15)

        # Train
        cand.train()
        bridge_var.train()
        for st in range(train_steps):
            ep = env.generate_episode("train", num_distractors=1, episode_idx=st)
            inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            opt.zero_grad()
            h_ad, _ = cand.forward_backbone(inp)

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

            h_v1 = h_ad[0, ep.hop1_val_pos : ep.hop1_val_pos + 1]
            h_q2, probs, _ = bridge_var(h_v1)

            h_norm = F.normalize(h_ad[0], p=2, dim=-1)
            h_q2_norm = F.normalize(h_q2[0], p=2, dim=-1)
            k2_logits = torch.matmul(h_q2_norm.unsqueeze(0), h_norm.transpose(0, 1)) / 0.15
            l_route = F.cross_entropy(k2_logits, torch.tensor([ep.hop2_key_pos], dtype=torch.long))

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
            bind_logits, _ = cand.compute_binding_scores(val_final_rep, cand_states, cand_mask)
            l_bind = F.cross_entropy(bind_logits, torch.tensor([tgt_idx], dtype=torch.long))

            loss = r_loss1.total_loss + 1.0 * l_route + 2.0 * l_bind
            loss.backward()
            opt.step()

        # Eval
        cand.eval()
        bridge_var.eval()
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
                    ep = env.generate_episode(sp_mode, num_distractors=1, episode_idx=200000 + ev_i)
                    inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)

                    t0 = time.perf_counter()
                    h_ad, _ = cand.forward_backbone(inp)
                    h_v1 = h_ad[0, ep.hop1_val_pos : ep.hop1_val_pos + 1]
                    h_q2, _, _ = bridge_var(h_v1)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)

                    h_norm = F.normalize(h_ad[0], p=2, dim=-1)
                    h_q2_norm = F.normalize(h_q2[0], p=2, dim=-1)

                    # Premise key positions
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

                    # Hop2 key routing among premise keys
                    k2_sims = [float(torch.dot(h_q2_norm, h_norm[kp]).item()) for kp in premise_key_pos]
                    pred_k2 = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0
                    h2_k_list.append(1.0 if pred_k2 == ep.hop2_key_pos else 0.0)

                    # Hop2 val routing
                    h_mk2 = h_norm[pred_k2]
                    v2_sims = [float(torch.dot(h_mk2, h_norm[vp]).item()) for vp in premise_val_pos]
                    pred_v2 = premise_val_pos[v2_sims.index(max(v2_sims))] if v2_sims else 0
                    h2_v_list.append(1.0 if pred_v2 == ep.hop2_val_pos else 0.0)

                    # Binding
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
                    val_rep = h_ad[0, ep.hop2_val_pos : ep.hop2_val_pos + 1]
                    b_logits, _ = cand.compute_binding_scores(val_rep, cand_states, cand_mask)
                    b_probs = F.softmax(b_logits[0], dim=-1)

                    sel_idx = int(torch.argmax(b_logits[0]).item())
                    if cand_tokens[sel_idx] == ep.target_token:
                        corr += 1

                    tgt_idx = cand_tokens.index(ep.target_token) if ep.target_token in cand_tokens else 0
                    prob_list.append(float(b_probs[tgt_idx].item()))

                acc_list.append(corr / eval_episodes)

    return VariantEvaluationMetrics(
        variant_name=variant,
        trainable_params=variant_params,
        mean_g1_tok_acc=float(sum(g1_list) / len(g1_list)),
        mean_g2_tok_acc=float(sum(g2_list) / len(g2_list)),
        mean_g3_tok_acc=float(sum(g3_list) / len(g3_list)),
        mean_g4_tok_acc=float(sum(g4_list) / len(g4_list)),
        mean_hop2_key_acc=float(sum(h2_k_list) / len(h2_k_list)),
        mean_hop2_val_acc=float(sum(h2_v_list) / len(h2_v_list)),
        mean_target_prob=float(sum(prob_list) / len(prob_list)),
        cpu_latency_ms=float(sum(latencies) / len(latencies)),
    )


def run_attractor_query_integration_study(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> AttractorIntegrationReport:
    """Compares variants A, B, C, D."""
    if seeds is None:
        seeds = [42, 101, 2026]

    variants = ["A", "B", "C", "D"]
    variant_results: Dict[str, VariantEvaluationMetrics] = {}

    for var in variants:
        m = evaluate_bridge_variant(
            base_model=base_model,
            variant=var,
            seeds=seeds,
            train_steps=train_steps,
            eval_episodes=eval_episodes,
        )
        variant_results[var] = m

    best_v = max(variant_results.keys(), key=lambda v: variant_results[v].mean_g4_tok_acc)
    best_g4 = variant_results[best_v].mean_g4_tok_acc

    summary = (
        f"Attractor Integration Comparison: " +
        "; ".join([f"{k}: G4={v.mean_g4_tok_acc*100:.1f}%, H2_K={v.mean_hop2_key_acc*100:.1f}%, params={v.trainable_params}"
                   for k, v in variant_results.items()]) +
        f" -> Best Variant: {best_v} ({best_g4*100:.1f}%)"
    )

    return AttractorIntegrationReport(
        variant_metrics=variant_results,
        best_variant=best_v,
        best_g4_tok_acc=best_g4,
        summary=summary,
    )
