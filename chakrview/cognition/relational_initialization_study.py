"""Step 393: Hop-1 Relational Initialization & Geometry Study.

Evaluates 7 controlled initialization and attention geometry alternatives for Hop-1 relational routing:
A. Standard random initialization (Normal(0, 0.02))
B. Normalized Q/K initialization (scaling weights by 1/sqrt(fan_in))
C. Identity-aligned Q/K initialization (adding eye/identity bias so initial Q/K preserve token coordinate alignment)
D. Orthogonal Q/K initialization (orthogonal matrices for Q and K projections)
E. Shared query/key role projection (coupling Q and K subspace projections)
F. Learned attention temperature (per-head trainable inverse temperature parameter)
G. Q/K normalization (unit L2 normalization of Q and K representations before dot-product)

Measures across identical seeds:
- H1 key routing accuracy
- H1 value routing accuracy
- H2 key routing accuracy
- G4 final-token accuracy on held-out compositions
- Inter-seed routing variance
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.normalization import RMSNorm
from chakrview.brain.rotary import RotaryEmbedding
from chakrview.brain.masking import CausalMask
from chakrview.brain.output import LMHead
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)
from chakrview.cognition.adaptive_learnability import (
    generate_mixed_hop_episode,
)


class RelationalAttentionBlock(nn.Module):
    """
    Attention block supporting the 7 initialization and normalization modes:
    A: standard, B: normalized, C: identity_aligned, D: orthogonal, E: shared_qk, F: learned_temp, G: qk_norm
    """
    def __init__(
        self,
        d_model: int = 96,
        n_heads: int = 4,
        max_seq_len: int = 512,
        mode: str = "A_standard",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 24
        self.mode = mode

        self.rotary = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)
        self.causal_mask = CausalMask(max_seq_len=max_seq_len)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Mode F: learned per-head temperature
        if mode == "F_learned_temp":
            self.inv_temp = nn.Parameter(torch.ones(1, n_heads, 1, 1) / math.sqrt(self.head_dim))
        else:
            self.inv_temp = None

        self._init_projections()

    def _init_projections(self) -> None:
        mode = self.mode
        std = 0.02
        if mode == "A_standard":
            for p in [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.out_proj.weight]:
                nn.init.normal_(p, mean=0.0, std=std)
        elif mode == "B_normalized":
            for p in [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.out_proj.weight]:
                nn.init.xavier_uniform_(p)
        elif mode == "C_identity_aligned":
            for p in [self.v_proj.weight, self.out_proj.weight]:
                nn.init.normal_(p, mean=0.0, std=std)
            # Add identity matrix bias to Q and K
            nn.init.eye_(self.q_proj.weight)
            nn.init.eye_(self.k_proj.weight)
            self.q_proj.weight.data.mul_(0.5).add_(torch.randn_like(self.q_proj.weight) * 0.01)
            self.k_proj.weight.data.mul_(0.5).add_(torch.randn_like(self.k_proj.weight) * 0.01)
        elif mode == "D_orthogonal":
            for p in [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.out_proj.weight]:
                nn.init.orthogonal_(p)
        elif mode == "E_shared_qk":
            nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
            # Tie K directly to Q initial weights
            self.k_proj.weight.data.copy_(self.q_proj.weight.data)
            for p in [self.v_proj.weight, self.out_proj.weight]:
                nn.init.normal_(p, mean=0.0, std=std)
        else:
            # Default normal for F and G
            for p in [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.out_proj.weight]:
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = q_in.shape
        q = self.q_proj(q_in).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_in).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_in).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rotary(q, T)
        k = self.rotary(k, T)

        # Mode G: Q/K normalization
        if self.mode == "G_qk_norm":
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            scale = 10.0  # Cosine similarity temperature
        elif self.inv_temp is not None:
            scale = self.inv_temp.abs()
        else:
            scale = 1.0 / math.sqrt(self.head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores + self.causal_mask(T)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(context)
        return out, weights


class InitializedRelationalCore(nn.Module):
    """
    Testbed core wrapping RelationalAttentionBlock under a specific initialization/geometry mode.
    """
    def __init__(
        self,
        mode: str = "A_standard",
        vocab_size: int = 4096,
        d_model: int = 96,
        n_heads: int = 4,
        d_state: int = 48,
        d_bind: int = 32,
    ):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.d_state = d_state
        self.d_bind = d_bind

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.norm_in = RMSNorm(d_model)
        self.norm_c1 = RMSNorm(d_model)

        # Cycle 1 Attention with specific mode
        self.attn1 = RelationalAttentionBlock(d_model=d_model, n_heads=n_heads, mode=mode)

        # Recurrent state transition
        self.norm_r = nn.LayerNorm(d_model)
        self.val_to_state = nn.Linear(d_model, d_state, bias=False)
        self.w_z_h = nn.Linear(d_state, d_state)
        self.w_z_s = nn.Linear(d_state, d_state, bias=False)
        self.w_r_h = nn.Linear(d_state, d_state)
        self.w_r_s = nn.Linear(d_state, d_state, bias=False)
        self.w_n_h = nn.Linear(d_state, d_state)
        self.w_n_s = nn.Linear(d_state, d_state, bias=False)
        self.s0 = nn.Parameter(torch.zeros(1, d_state))

        # Cycle 2 state-conditioned query & attention
        self.state_to_q = nn.Linear(d_state, d_model, bias=False)
        self.attn2 = RelationalAttentionBlock(d_model=d_model, n_heads=n_heads, mode=mode)
        self.norm_c2 = RMSNorm(d_model)

        # Dynamic binding
        self.norm_bind_val = nn.LayerNorm(d_model)
        self.bind_query_proj = nn.Linear(d_model, d_bind, bias=False)
        self.norm_bind_cand = nn.LayerNorm(d_model)
        self.bind_cand_proj = nn.Linear(d_model, d_bind, bias=False)
        self.bind_scale = nn.Parameter(torch.tensor([4.0]))

        self.lm_head = LMHead(self.embedding.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        candidate_positions: Optional[torch.Tensor] = None,
        query_pos: Optional[int] = None,
    ) -> Dict[str, Any]:
        B, T = input_ids.shape
        pos = query_pos if query_pos is not None else (T - 1)

        x = self.embedding(input_ids)
        h0 = self.norm_in(x)

        # Cycle 1
        a1, w1 = self.attn1(h0, h0, h0)
        h1 = h0 + a1

        # Intermediate value & state
        h_q = h1[:, pos : pos + 1, :]
        r1 = self.norm_r(h_q).squeeze(1)
        r_feat = self.val_to_state(r1)

        s_prev = self.s0.expand(B, -1)
        z_t = torch.sigmoid(self.w_z_h(r_feat) + self.w_z_s(s_prev))
        r_t = torch.sigmoid(self.w_r_h(r_feat) + self.w_r_s(s_prev))
        n_t = torch.tanh(self.w_n_h(r_feat) + self.w_n_s(r_t * s_prev))
        s1 = (1.0 - z_t) * s_prev + z_t * n_t

        # Cycle 2
        q2_mod = self.state_to_q(s1).unsqueeze(1)
        h1_norm = self.norm_c1(h1)
        q2 = h1_norm.clone()
        q2[:, pos : pos + 1, :] = q2[:, pos : pos + 1, :] + q2_mod

        a2, w2 = self.attn2(q2, h1_norm, h1_norm)
        h2 = h1 + a2
        h_final = self.norm_c2(h2)

        out = {
            "final_hidden": h_final,
            "w1": w1,
            "w2": w2,
            "s1": s1,
            "binding_logits": None,
        }

        if candidate_positions is not None:
            B_c, N_c = candidate_positions.shape
            c_states = torch.stack([h_final[b, candidate_positions[b]] for b in range(B_c)], dim=0)
            val_rep = h_final[:, pos, :]

            q_b = F.normalize(self.bind_query_proj(self.norm_bind_val(val_rep)), p=2, dim=-1)
            k_b = F.normalize(self.bind_cand_proj(self.norm_bind_cand(c_states)), p=2, dim=-1)
            scores = torch.bmm(k_b, q_b.unsqueeze(-1)).squeeze(-1) * self.bind_scale.abs()
            out["binding_logits"] = scores

        return out


@dataclasses.dataclass
class InitStudyResult:
    mode_id: str
    description: str
    h1_key_routing: float
    h1_val_routing: float
    h2_key_routing: float
    g4_acc: float
    routing_variance: float


@dataclasses.dataclass
class Step393InitStudyReport:
    modes: Dict[str, InitStudyResult]
    best_mode: str
    h1_routing_boost: float
    summary: str


def run_relational_initialization_study(
    seeds: Tuple[int, ...] = (42, 101),
    train_steps: int = 15,
    eval_episodes: int = 6,
) -> Step393InitStudyReport:
    """Executes Step 393 controlled comparison across modes A through G."""
    modes = [
        ("A_standard", "Standard Normal (0, 0.02)"),
        ("B_normalized", "Xavier Uniform Normalization"),
        ("C_identity_aligned", "Identity-Aligned Q/K Projections"),
        ("D_orthogonal", "Orthogonal Matrix Initialization"),
        ("E_shared_qk", "Shared Q/K Initialization Subspace"),
        ("F_learned_temp", "Learned Per-Head Attention Temperature"),
        ("G_qk_norm", "Unit L2 Normalization on Q and K (Cosine Attention)"),
    ]

    results: Dict[str, InitStudyResult] = {}

    for mode_id, desc in modes:
        h1_k_seeds = []
        h1_v_seeds = []
        h2_k_seeds = []
        g4_seeds = []

        for s in seeds:
            torch.manual_seed(s)
            env = CompositionalAssociativeEnvironment(seed=s)
            core = InitializedRelationalCore(mode=mode_id)
            opt = torch.optim.Adam(core.parameters(), lr=1e-3, weight_decay=1e-4)

            # Quick train on 2-hop episodes
            core.train()
            for step in range(train_steps):
                opt.zero_grad()
                ep = generate_mixed_hop_episode(env, hop_count=2, split="train", num_distractors=1)
                seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
                tgt = torch.tensor([ep.target_idx], dtype=torch.long)

                out = core(seq, candidate_positions=c_pos)
                loss = F.cross_entropy(out["binding_logits"], tgt)
                loss.backward()
                opt.step()

            # Eval on held-out disjoint split
            core.eval()
            h1_k_hits = 0
            h1_v_hits = 0
            h2_k_hits = 0
            g4_hits = 0

            with torch.no_grad():
                for _ in range(eval_episodes):
                    ep = generate_mixed_hop_episode(env, hop_count=2, split="disjoint_test", num_distractors=1)
                    seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                    c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)

                    out = core(seq, candidate_positions=c_pos)
                    pred_idx = torch.argmax(out["binding_logits"][0]).item()
                    if pred_idx == ep.target_idx:
                        g4_hits += 1

                    w1 = out["w1"][0].mean(dim=0)[-1, :]
                    if len(ep.key_positions) >= 1 and ep.key_positions[0] >= 0:
                        if torch.argmax(w1).item() == ep.key_positions[0]:
                            h1_k_hits += 1
                    if len(ep.val_positions) >= 1 and ep.val_positions[0] >= 0:
                        if torch.argmax(w1).item() == ep.val_positions[0]:
                            h1_v_hits += 1

                    w2 = out["w2"][0].mean(dim=0)[-1, :]
                    if len(ep.key_positions) >= 2 and ep.key_positions[1] >= 0:
                        if torch.argmax(w2).item() == ep.key_positions[1]:
                            h2_k_hits += 1

            N = max(1, eval_episodes)
            h1_k_seeds.append(h1_k_hits / N)
            h1_v_seeds.append(h1_v_hits / N)
            h2_k_seeds.append(h2_k_hits / N)
            g4_seeds.append(g4_hits / N)

        m_h1_k = sum(h1_k_seeds) / len(h1_k_seeds)
        m_h1_v = sum(h1_v_seeds) / len(h1_v_seeds)
        m_h2_k = sum(h2_k_seeds) / len(h2_k_seeds)
        m_g4 = sum(g4_seeds) / len(g4_seeds)
        var = float(torch.tensor(h1_k_seeds).var().item()) if len(h1_k_seeds) > 1 else 0.0

        results[mode_id] = InitStudyResult(
            mode_id=mode_id,
            description=desc,
            h1_key_routing=m_h1_k,
            h1_val_routing=m_h1_v,
            h2_key_routing=m_h2_k,
            g4_acc=m_g4,
            routing_variance=var,
        )

    best_m = max(results.keys(), key=lambda k: (results[k].h1_key_routing, results[k].g4_acc))
    base_h1 = results["A_standard"].h1_key_routing
    best_h1 = results[best_m].h1_key_routing
    boost = best_h1 - base_h1

    summary = (
        f"Relational Initialization Study: Best={best_m} ({results[best_m].description}). "
        f"Standard H1={base_h1:.2%} vs Best H1={best_h1:.2%} (Boost={boost:+.2%}). "
        f"Best G4={results[best_m].g4_acc:.2%}, H2 Key={results[best_m].h2_key_routing:.2%}."
    )

    return Step393InitStudyReport(
        modes=results,
        best_mode=best_m,
        h1_routing_boost=boost,
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 393: Running Hop-1 Relational Initialization & Geometry Study...")
    rep = run_relational_initialization_study(seeds=(42, 101), train_steps=12, eval_episodes=4)
    print("Report Summary:", rep.summary)
    for m, r in rep.modes.items():
        print(f"  [{m:18s}] H1_K={r.h1_key_routing:.2%}, H2_K={r.h2_key_routing:.2%}, G4={r.g4_acc:.2%}, Var={r.routing_variance:.4f}")
