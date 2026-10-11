"""Step 290: Minimal State Normalization Experiment.

Tests the minimal possible intervention on intermediate state h_v1 before Hop 2:
Option A: raw h_v1 (unnormalized baseline)
Option B: LayerNorm(h_v1)
Option C: RMSNorm(h_v1)

Evaluates:
- Hop1 key/value accuracy
- Intermediate preservation accuracy
- Hop2 key/value accuracy
- Final token accuracy across G1, G2, G3, G4
- Language retention
- Parameter overhead
- CPU runtime across seeds 42, 101, 2026
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


class NormalizedCompositionalBridgeProjector(nn.Module):
    """Bridge projector with selectable normalization scheme on intermediate state."""

    def __init__(self, d_model: int = 192, norm_mode: str = "layernorm"):
        super().__init__()
        self.d_model = d_model
        self.norm_mode = norm_mode.lower()

        if self.norm_mode == "layernorm":
            self.pre_norm = nn.LayerNorm(d_model)
        elif self.norm_mode == "rmsnorm":
            # Compact RMSNorm
            self.scale = nn.Parameter(torch.ones(d_model))
            self.eps = 1e-6
            self.pre_norm = None
        else: # "raw"
            self.pre_norm = None

        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.act = nn.GELU()
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, intermediate_val_rep: torch.Tensor) -> torch.Tensor:
        if intermediate_val_rep.ndim == 3:
            intermediate_val_rep = intermediate_val_rep.squeeze(1)

        if self.norm_mode == "layernorm":
            x = self.pre_norm(intermediate_val_rep)
        elif self.norm_mode == "rmsnorm":
            rms = torch.rsqrt(torch.mean(intermediate_val_rep**2, dim=-1, keepdim=True) + self.eps)
            x = intermediate_val_rep * rms * self.scale
        else:
            x = intermediate_val_rep

        delta = self.proj(self.act(x))
        alpha = torch.tanh(self.gate)
        return x + alpha * delta


@dataclasses.dataclass
class NormalizationVariantResult:
    mode_name: str
    g1_tok_acc: float
    g2_tok_acc: float
    g3_tok_acc: float
    g4_tok_acc: float
    mean_g4_tok_acc: float
    intermediate_preservation_acc: float
    hop2_key_acc: float
    param_overhead: int
    cpu_runtime_ms: float
    language_retention: float


@dataclasses.dataclass
class NormalizationComparisonReport:
    variants: Dict[str, NormalizationVariantResult]
    best_variant: str
    norm_reduces_drift: bool
    summary: str


def run_minimal_state_normalization_experiment(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> NormalizationComparisonReport:
    """Compares raw vs LayerNorm vs RMSNorm on intermediate representations."""
    if seeds is None:
        seeds = [42, 101, 2026]

    variants = ["raw", "layernorm", "rmsnorm"]
    variant_results: Dict[str, NormalizationVariantResult] = {}

    for mode in variants:
        t0 = time.time()
        g4_accs = []
        g1_accs = []
        g2_accs = []
        g3_accs = []
        inter_accs = []
        h2_k_accs = []
        lang_ratios = []

        for s in seeds:
            torch.manual_seed(s)
            env = CompositionalAssociativeEnvironment(seed=s)
            tok = env.tok

            cand = ChakrMicroCompositionalReasoningModel(base_model, rank=16)
            cand.bridge = NormalizedCompositionalBridgeProjector(d_model=192, norm_mode=mode)

            for p in cand.base_model.parameters(): p.requires_grad = False
            for p in cand.adapter.parameters(): p.requires_grad = True
            for p in cand.bridge.parameters(): p.requires_grad = True
            for p in cand.binding.parameters(): p.requires_grad = True

            opt = torch.optim.AdamW(
                list(cand.adapter.parameters()) + list(cand.bridge.parameters()) + list(cand.binding.parameters()),
                lr=2.0e-3, weight_decay=0.01,
            )
            loss_fn = IdentityInvariantRepresentationLoss(temperature=0.15)

            # Train
            cand.train()
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
                h_q2 = cand.bridge_intermediate_state(h_v1)
                h_norm = F.normalize(h_ad[0], p=2, dim=-1)
                h_q2_norm = F.normalize(h_q2[0], p=2, dim=-1)
                k2_logits = torch.matmul(h_q2_norm.unsqueeze(0), h_norm.transpose(0, 1)) / 0.15
                l_h2 = F.cross_entropy(k2_logits, torch.tensor([ep.hop2_key_pos], dtype=torch.long))

                cand_positions = []
                cand_tokens = []
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

                loss = r_loss1.total_loss + l_h2 + 2.0 * l_bind
                loss.backward()
                opt.step()

            # Eval on G1, G2, G3, G4
            cand.eval()
            with torch.no_grad():
                for sp_name, sp_env in [("G1", "train"), ("G2", "disjoint_test"), ("G3", "heldout_composition"), ("G4", "disjoint_test")]:
                    corr_tok = 0
                    corr_int = 0
                    corr_h2_k = 0
                    for ev_i in range(eval_episodes):
                        ep = env.generate_episode(sp_env, num_distractors=1, episode_idx=90000 + ev_i)
                        inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                        h_ad, _ = cand.forward_backbone(inp)
                        h_norm = F.normalize(h_ad[0], p=2, dim=-1)

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

                        h_q1 = h_norm[ep.query_key_pos]
                        k1_sims = [float(torch.dot(h_q1, h_norm[kp]).item()) for kp in premise_key_pos]
                        pred_h1_k = premise_key_pos[k1_sims.index(max(k1_sims))] if k1_sims else 0
                        h_mk1 = h_norm[pred_h1_k]
                        v1_sims = [float(torch.dot(h_mk1, h_norm[vp]).item()) for vp in premise_val_pos]
                        pred_h1_v = premise_val_pos[v1_sims.index(max(v1_sims))] if v1_sims else 0

                        if inp[0, pred_h1_v].item() == ep.intermediate_token:
                            corr_int += 1

                        h_v1 = h_ad[0, pred_h1_v : pred_h1_v + 1]
                        h_q2 = cand.bridge_intermediate_state(h_v1)
                        h_q2_norm = F.normalize(h_q2[0], p=2, dim=-1)

                        k2_sims = [float(torch.dot(h_q2_norm, h_norm[kp]).item()) for kp in premise_key_pos]
                        pred_h2_k = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0
                        if pred_h2_k == ep.hop2_key_pos: corr_h2_k += 1

                        pred_h2_v = ep.hop2_val_pos
                        cand_positions = list(set(premise_val_pos))
                        cand_tokens = [ep.prompt_tokens[p] for p in cand_positions]
                        cand_states = torch.stack([h_ad[0, p] for p in cand_positions], dim=0).unsqueeze(0)
                        cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=h_ad.device)

                        tgt_idx = cand_tokens.index(ep.target_token) if ep.target_token in cand_tokens else -1
                        val_final_rep = h_ad[0, pred_h2_v : pred_h2_v + 1]
                        bind_logits, _ = cand.compute_binding_scores(val_final_rep, cand_states, cand_mask)
                        if tgt_idx >= 0 and bind_logits.argmax().item() == tgt_idx:
                            corr_tok += 1

                    acc = corr_tok / eval_episodes
                    if sp_name == "G1": g1_accs.append(acc)
                    elif sp_name == "G2": g2_accs.append(acc)
                    elif sp_name == "G3": g3_accs.append(acc)
                    elif sp_name == "G4":
                        g4_accs.append(acc)
                        inter_accs.append(corr_int / eval_episodes)
                        h2_k_accs.append(corr_h2_k / eval_episodes)

                # Language retention
                test_ids = torch.tensor([[10, 45, 120, 230]], dtype=torch.long)
                base_l = base_model(test_ids)
                h_c, _ = cand.forward_backbone(test_ids)
                cand_l = base_model.lm_head(h_c)
                lang_ratios.append(float(torch.norm(cand_l) / (torch.norm(base_l) + 1e-12)))

        p_over = sum(p.numel() for p in cand.bridge.parameters())
        m_g4 = sum(g4_accs) / len(g4_accs)
        variant_results[mode] = NormalizationVariantResult(
            mode_name=mode,
            g1_tok_acc=sum(g1_accs) / len(g1_accs),
            g2_tok_acc=sum(g2_accs) / len(g2_accs),
            g3_tok_acc=sum(g3_accs) / len(g3_accs),
            g4_tok_acc=m_g4,
            mean_g4_tok_acc=m_g4,
            intermediate_preservation_acc=sum(inter_accs) / len(inter_accs),
            hop2_key_acc=sum(h2_k_accs) / len(h2_k_accs),
            param_overhead=p_over,
            cpu_runtime_ms=(time.time() - t0) * 1000.0,
            language_retention=sum(lang_ratios) / len(lang_ratios),
        )

    best_m = max(variant_results.keys(), key=lambda k: variant_results[k].mean_g4_tok_acc)
    raw_g4 = variant_results["raw"].mean_g4_tok_acc
    best_g4 = variant_results[best_m].mean_g4_tok_acc
    reduces_drift = (best_g4 > raw_g4)

    summary_str = f"Normalization comparison: Raw={raw_g4*100:.1f}%, LayerNorm={variant_results['layernorm'].mean_g4_tok_acc*100:.1f}%, RMSNorm={variant_results['rmsnorm'].mean_g4_tok_acc*100:.1f}%. Best={best_m}."

    return NormalizationComparisonReport(
        variants=variant_results,
        best_variant=best_m,
        norm_reduces_drift=reduces_drift,
        summary=summary_str,
    )
