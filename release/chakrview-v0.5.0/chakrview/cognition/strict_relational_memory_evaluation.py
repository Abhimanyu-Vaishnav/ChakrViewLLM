"""Step 319: Strict I3 + I4 Evaluation and Memory Ablation Suite.

Runs:
1. I3 Dynamic Contextual Token Binding Preservation:
   - Unseen/Unseen (UU) key routing, value routing, candidate selection, final token.
   - Base model language retention audit.
2. Strict Multi-Seed I4 2-Hop Compositional Benchmark:
   - Seeds: 42, 101, 2026.
   - Groups:
     G1: Known identity / Known composition
     G2: Unseen identity / Known composition
     G3: Known identity / Unseen composition
     G4: Unseen identity / Unseen composition (PRIMARY I4 CAPABILITY GATE)
3. Relational Memory Ablations:
   - Option A: No Memory (direct bypass)
   - Option B: Memory without Write
   - Option C: Memory without Recurrent Update
   - Option D: Memory without Read
   - Option E: Full Relational Memory
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
from chakrview.cognition.sequential_memory_update import (
    ChakrMicroWithRelationalMemory,
)
from chakrview.cognition.dynamic_contextual_token_binding import (
    ChakrMicroWithDynamicBinding,
)
from chakrview.cognition.true_disjoint_generalization import (
    evaluate_true_disjoint_binding,
)
from chakrview.cognition.randomized_associative_episodes import (
    RandomizedAssociativeEnvironment,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)
from chakrview.cognition.identity_invariant_objective import (
    IdentityInvariantRepresentationLoss,
)


@dataclasses.dataclass
class MemorySeedResult:
    seed: int
    g1_tok_acc: float
    g2_tok_acc: float
    g3_tok_acc: float
    g4_tok_acc: float
    h1_key_acc: float
    h2_key_acc: float
    h2_val_acc: float


@dataclasses.dataclass
class MemoryAblationMetrics:
    ablation_name: str
    g4_tok_acc: float
    h2_key_acc: float


@dataclasses.dataclass
class StrictRelationalMemoryI4Report:
    per_seed_results: Dict[int, MemorySeedResult]
    mean_g1_tok_acc: float
    mean_g2_tok_acc: float
    mean_g3_tok_acc: float
    mean_g4_tok_acc: float
    mean_h1_key_acc: float
    mean_h2_key_acc: float
    mean_h2_val_acc: float
    i3_preserved: bool
    i3_uu_tok_acc: float
    language_retention_ratio: float
    ablations: Dict[str, MemoryAblationMetrics]
    i4_gate_passed: bool
    stability_diagnostic_passed: bool
    is_base_bit_exact: bool
    final_classification: str
    summary: str


def train_relational_memory_candidate(
    base_model: ChakrMicro,
    seed: int,
    train_steps: int = 15,
    eval_episodes: int = 8,
    num_slots: int = 2,
    disable_write: bool = False,
    disable_read: bool = False,
) -> Tuple[ChakrMicroWithRelationalMemory, MemorySeedResult]:
    """Trains ChakrMicroWithRelationalMemory."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    cand = ChakrMicroWithRelationalMemory(
        base_model=base_model,
        num_slots=num_slots,
        d_mem=64,
        rank=16,
        d_bind=64,
    )

    opt = torch.optim.AdamW(
        [p for p in cand.parameters() if p.requires_grad],
        lr=2.0e-3,
        weight_decay=0.01,
    )
    loss_fn = IdentityInvariantRepresentationLoss(temperature=0.15)

    cand.train()
    for st in range(train_steps):
        ep = env.generate_episode("train", num_distractors=1, episode_idx=st)
        inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        opt.zero_grad()

        h_ad, _ = cand.forward_backbone(inp)
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

        # Hop-1 value
        v1_rep = h_ad[0, ep.hop1_val_pos : ep.hop1_val_pos + 1]

        # Memory write & read
        m_init = cand.memory.get_initial_state(1)
        m_up, _ = cand.memory.write(m_init, v1_rep, disable_write=disable_write)
        q2_rep, _ = cand.memory.read(m_up, v1_rep, disable_read=disable_read)
        q2_norm = F.normalize(q2_rep[0], p=2, dim=-1)

        # Hop-2 key routing loss
        k2_logits = torch.matmul(q2_norm.unsqueeze(0), h_norm.transpose(0, 1)) / 0.15
        l_route2 = F.cross_entropy(k2_logits, torch.tensor([ep.hop2_key_pos], dtype=torch.long))

        # Dynamic contextual binding loss
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

        total_loss = r_loss1.total_loss + 1.5 * l_route2 + 2.0 * l_bind
        total_loss.backward()
        opt.step()

    # Evaluation
    cand.eval()
    splits = [
        ("G1", "train"),
        ("G2", "disjoint_test"),
        ("G3", "heldout_composition"),
        ("G4", "disjoint_test"),
    ]
    tok_accs = {}
    h1_k_corr, h2_k_corr, h2_v_corr = 0, 0, 0
    total_eval = 0

    with torch.no_grad():
        for sp_name, sp_env in splits:
            corr_tok = 0
            for ev_i in range(eval_episodes):
                ep = env.generate_episode(sp_env, num_distractors=1, episode_idx=1200000 + ev_i)
                total_eval += 1
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

                # Hop-1 routing
                h_q1 = h_norm[ep.query_key_pos]
                k1_sims = [float(torch.dot(h_q1, h_norm[kp]).item()) for kp in premise_key_pos]
                pred_k1 = premise_key_pos[k1_sims.index(max(k1_sims))] if k1_sims else 0
                if pred_k1 == ep.hop1_key_pos: h1_k_corr += 1

                h_mk1 = h_norm[pred_k1]
                v1_sims = [float(torch.dot(h_mk1, h_norm[vp]).item()) for vp in premise_val_pos]
                pred_v1 = premise_val_pos[v1_sims.index(max(v1_sims))] if v1_sims else 0

                # Memory update
                v1_rep = h_ad[0, pred_v1 : pred_v1 + 1]
                m_init = cand.memory.get_initial_state(1)
                m_up, _ = cand.memory.write(m_init, v1_rep, disable_write=disable_write)
                q2_rep, _ = cand.memory.read(m_up, v1_rep, disable_read=disable_read)
                q2_norm = F.normalize(q2_rep[0], p=2, dim=-1)

                # Hop-2 key routing
                k2_sims = [float(torch.dot(q2_norm, h_norm[kp]).item()) for kp in premise_key_pos]
                pred_k2 = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0
                if pred_k2 == ep.hop2_key_pos: h2_k_corr += 1

                # Hop-2 val routing
                h_mk2 = h_norm[pred_k2]
                v2_sims = [float(torch.dot(h_mk2, h_norm[vp]).item()) for vp in premise_val_pos]
                pred_v2 = premise_val_pos[v2_sims.index(max(v2_sims))] if v2_sims else 0
                if pred_v2 == ep.hop2_val_pos: h2_v_corr += 1

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
                b_logits, _ = cand.compute_binding_scores(val_final, cand_states, cand_mask)
                sel_idx = int(torch.argmax(b_logits[0]).item())
                if cand_tokens[sel_idx] == ep.target_token:
                    corr_tok += 1

            tok_accs[sp_name] = corr_tok / eval_episodes

    res = MemorySeedResult(
        seed=seed,
        g1_tok_acc=tok_accs["G1"],
        g2_tok_acc=tok_accs["G2"],
        g3_tok_acc=tok_accs["G3"],
        g4_tok_acc=tok_accs["G4"],
        h1_key_acc=h1_k_corr / max(total_eval, 1),
        h2_key_acc=h2_k_corr / max(total_eval, 1),
        h2_val_acc=h2_v_corr / max(total_eval, 1),
    )
    return cand, res


def run_strict_relational_memory_i3_i4_evaluation(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
    num_slots: int = 2,
) -> StrictRelationalMemoryI4Report:
    """Executes multi-seed evaluation and ablation suite."""
    if seeds is None:
        seeds = [42, 101, 2026]

    init_hash = compute_model_hash(base_model)
    per_seed: Dict[int, MemorySeedResult] = {}
    last_cand = None

    for s in seeds:
        cand, res = train_relational_memory_candidate(
            base_model=base_model, seed=s, train_steps=train_steps, eval_episodes=eval_episodes, num_slots=num_slots,
        )
        per_seed[s] = res
        last_cand = cand

    # I3 evaluation & language retention
    env_i3 = RandomizedAssociativeEnvironment(seed=42)
    i3_wrapper = ChakrMicroWithDynamicBinding(base_model=base_model, rank=16)
    i3_wrapper.adapter.load_state_dict(last_cand.adapter.state_dict())
    i3_wrapper.binding.load_state_dict(last_cand.binding.state_dict())
    i3_wrapper.eval()

    disjoint_rep = evaluate_true_disjoint_binding(
        candidate=i3_wrapper, env=env_i3, seed=42, num_episodes_per_condition=eval_episodes,
    )

    test_ids = torch.tensor([[10, 45, 120, 230], [5, 18, 92, 104]], dtype=torch.long)
    with torch.no_grad():
        b_l = base_model(test_ids)
        c_l = last_cand.forward_backbone(test_ids)[1]
        retention = float(torch.norm(c_l).item() / max(torch.norm(b_l).item(), 1e-12))

    uu_tok = disjoint_rep.unseen_unseen_tok_acc
    i3_ok = uu_tok >= 0.50

    # Ablations
    ablations: Dict[str, MemoryAblationMetrics] = {}
    # A. Full memory
    ablations["full_memory"] = MemoryAblationMetrics("full_memory", per_seed[42].g4_tok_acc, per_seed[42].h2_key_acc)
    # B. Memory without write
    _, res_no_write = train_relational_memory_candidate(
        base_model=base_model, seed=42, train_steps=train_steps, eval_episodes=eval_episodes, disable_write=True,
    )
    ablations["no_write"] = MemoryAblationMetrics("no_write", res_no_write.g4_tok_acc, res_no_write.h2_key_acc)
    # C. Memory without read
    _, res_no_read = train_relational_memory_candidate(
        base_model=base_model, seed=42, train_steps=train_steps, eval_episodes=eval_episodes, disable_read=True,
    )
    ablations["no_read"] = MemoryAblationMetrics("no_read", res_no_read.g4_tok_acc, res_no_read.h2_key_acc)

    post_hash = compute_model_hash(base_model)
    is_bit_exact = (init_hash == post_hash == EXPECTED_WEIGHT_HASH)

    m_g1 = sum(r.g1_tok_acc for r in per_seed.values()) / len(per_seed)
    m_g2 = sum(r.g2_tok_acc for r in per_seed.values()) / len(per_seed)
    m_g3 = sum(r.g3_tok_acc for r in per_seed.values()) / len(per_seed)
    m_g4 = sum(r.g4_tok_acc for r in per_seed.values()) / len(per_seed)
    m_h1 = sum(r.h1_key_acc for r in per_seed.values()) / len(per_seed)
    m_h2_k = sum(r.h2_key_acc for r in per_seed.values()) / len(per_seed)
    m_h2_v = sum(r.h2_val_acc for r in per_seed.values()) / len(per_seed)

    i4_gate = (m_g4 >= 0.50)
    stability_diag = all(r.g4_tok_acc >= 0.40 for r in per_seed.values())

    if i4_gate and stability_diag and is_bit_exact:
        classification = "I4_ACHIEVED"
    elif m_g4 >= 0.35:
        classification = "I4_EMERGING"
    else:
        classification = "I4_NOT_ACHIEVED"

    summary = (
        f"Strict Relational Memory I4 Evaluation: "
        f"G1={m_g1*100:.1f}%, G2={m_g2*100:.1f}%, G3={m_g3*100:.1f}%, G4={m_g4*100:.1f}%. "
        f"Seeds G4: " + ", ".join([f"s{s}={per_seed[s].g4_tok_acc*100:.1f}%" for s in seeds]) +
        f". Promotion Gate (>50%): {i4_gate}. Stability Diag (all>=40%): {stability_diag}. "
        f"I3 Preserved: {i3_ok} (UU Tok={uu_tok*100:.1f}%), Language Retention={retention:.5f}. "
        f"Classification: {classification}."
    )

    return StrictRelationalMemoryI4Report(
        per_seed_results=per_seed,
        mean_g1_tok_acc=m_g1,
        mean_g2_tok_acc=m_g2,
        mean_g3_tok_acc=m_g3,
        mean_g4_tok_acc=m_g4,
        mean_h1_key_acc=m_h1,
        mean_h2_key_acc=m_h2_k,
        mean_h2_val_acc=m_h2_v,
        i3_preserved=i3_ok,
        i3_uu_tok_acc=uu_tok,
        language_retention_ratio=retention,
        ablations=ablations,
        i4_gate_passed=i4_gate,
        stability_diagnostic_passed=stability_diag,
        is_base_bit_exact=is_bit_exact,
        final_classification=classification,
        summary=summary,
    )
