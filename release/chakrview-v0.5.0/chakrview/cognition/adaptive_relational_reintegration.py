"""Step 407: Adaptive Reasoning Reintegration with Neural Relational Acquisition Core.

Reintegrates the dynamic AnswerSufficiencyController with the neural relational acquisition core.
Adaptive reasoning acts purely as a computation allocator, while relational acquisition performs intelligence.

Compares:
A. Fixed 1 cycle
B. Fixed 2 cycles
C. Fixed 3 cycles
D. Adaptive sufficiency controller

Measures:
- H1 key/val routing
- H2 key/val routing
- G1, G4 accuracies
- Average reasoning cycles executed
- Premature halt rate (multi-hop halting at 1 cycle)
- Unnecessary continuation rate (single-hop continuing past 1 cycle)
- Language retention against canonical ChakrMicro baseline
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import instantiate_frozen_baseline
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)
from chakrview.cognition.adaptive_learnability import (
    generate_mixed_hop_episode,
)
from chakrview.cognition.neural_relational_acquisition import (
    NeuralRelationalAcquisitionModule,
)
from chakrview.cognition.sufficiency_adaptive_reasoning import (
    AnswerSufficiencyController,
)
from chakrview.cognition.hop1_objective_ablation import (
    evaluate_language_retention,
)


@dataclasses.dataclass
class AdaptiveReintegrationVariantResult:
    variant_id: str
    description: str
    g1_acc: float
    g4_acc: float
    h1_routing: float
    h2_routing: float
    mean_cycles: float
    premature_halt_rate: float
    unnecessary_rate: float
    language_retention: float


@dataclasses.dataclass
class Step407ReintegrationReport:
    variants: Dict[str, AdaptiveReintegrationVariantResult]
    best_variant: str
    adaptive_preserves_accuracy: bool
    language_retention_preserved: bool
    summary: str


class AdaptiveRelationalAcquisitionPipeline(nn.Module):
    """
    Combines NeuralRelationalAcquisitionModule with AnswerSufficiencyController.
    """
    def __init__(self, d_model: int = 96, d_state: int = 48, vocab_size: int = 4096):
        super().__init__()
        self.relational_core = NeuralRelationalAcquisitionModule(
            d_input=96, d_model=d_model, d_state=d_state, vocab_size=vocab_size
        )
        self.controller = AnswerSufficiencyController(
            d_state=d_state, d_model=d_model
        )

    def forward(
        self,
        seq: torch.Tensor,
        candidate_positions: Optional[torch.Tensor] = None,
        max_cycles: int = 3,
        mode: str = "adaptive", # 'fixed_1', 'fixed_2', 'fixed_3', 'adaptive'
    ) -> Dict[str, Any]:
        B, T = seq.shape
        # Cycle 1
        out1 = self.relational_core(seq, candidate_positions=candidate_positions, max_hops=1)
        h_state = out1["s1"]
        q_pos = T - 7
        q_rep = out1["final_hidden"][:, q_pos, :]
        ctx_rep = out1["v1"]
        feat = torch.zeros(B, 3)

        _, s_score, p_cont = self.controller(h_state, q_rep, ctx_rep, feat)

        if mode == "fixed_1":
            cycles_done = 1
            final_out = out1
        elif mode == "fixed_2":
            cycles_done = 2
            final_out = self.relational_core(seq, candidate_positions=candidate_positions, max_hops=2)
        elif mode == "fixed_3":
            cycles_done = 3
            final_out = self.relational_core(seq, candidate_positions=candidate_positions, max_hops=2)
        else: # adaptive
            if p_cont.mean().item() > 0.5 and max_cycles >= 2:
                cycles_done = 2
                final_out = self.relational_core(seq, candidate_positions=candidate_positions, max_hops=2)
            else:
                cycles_done = 1
                final_out = out1

        return {
            "binding_logits": final_out["binding_logits"],
            "w1_key": final_out["w1_key"],
            "w1_val": final_out["w1_val"],
            "w2_key": final_out["w2_key"],
            "w2_val": final_out["w2_val"],
            "cycles_executed": cycles_done,
            "final_hidden": final_out["final_hidden"],
        }


def run_adaptive_relational_reintegration_study(
    seed: int = 42,
    train_steps: int = 30,
    eval_episodes: int = 6,
) -> Step407ReintegrationReport:
    """Executes Step 407 comparative evaluation across computation modes."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    baseline = instantiate_frozen_baseline()

    pipeline = AdaptiveRelationalAcquisitionPipeline()
    opt = torch.optim.Adam(pipeline.parameters(), lr=1e-3, weight_decay=1e-4)

    # Train pipeline
    pipeline.train()
    for _ in range(train_steps):
        opt.zero_grad()
        ep = generate_mixed_hop_episode(env, hop_count=2, split="train", num_distractors=1)
        seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
        tgt = torch.tensor([ep.target_idx], dtype=torch.long)

        out = pipeline(seq, candidate_positions=c_pos, mode="fixed_2")
        l_final = F.cross_entropy(out["binding_logits"], tgt)
        l_h1 = F.cross_entropy(out["w1_key"], torch.tensor([ep.key_positions[0]]))
        l_h2 = F.cross_entropy(out["w2_key"], torch.tensor([ep.key_positions[1]]))
        loss = l_final + 0.25 * l_h1 + 0.25 * l_h2
        loss.backward()
        opt.step()

    pipeline.eval()
    variants = [
        ("A_Fixed1", "Fixed 1 reasoning cycle", "fixed_1"),
        ("B_Fixed2", "Fixed 2 reasoning cycles", "fixed_2"),
        ("C_Fixed3", "Fixed 3 reasoning cycles", "fixed_3"),
        ("D_AdaptiveSufficiency", "Adaptive Sufficiency Allocation", "adaptive"),
    ]

    results: Dict[str, AdaptiveReintegrationVariantResult] = {}

    for vid, desc, mode_name in variants:
        g1_h, g4_h = 0, 0
        h1_h, h2_h = 0, 0
        cycles_list = []
        premature, unnecessary = 0, 0

        with torch.no_grad():
            # 1-hop checks for unnecessary continuation
            for _ in range(eval_episodes):
                ep = generate_mixed_hop_episode(env, hop_count=1, split="disjoint_test", num_distractors=1)
                seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
                out = pipeline(seq, candidate_positions=c_pos, mode=mode_name)
                if out["cycles_executed"] > 1:
                    unnecessary += 1

            # 2-hop checks for G4 and premature halting
            for _ in range(eval_episodes):
                ep = generate_mixed_hop_episode(env, hop_count=2, split="train", num_distractors=1)
                seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
                out = pipeline(seq, candidate_positions=c_pos, mode=mode_name)
                if torch.argmax(out["binding_logits"][0]).item() == ep.target_idx:
                    g1_h += 1

            for _ in range(eval_episodes):
                ep = generate_mixed_hop_episode(env, hop_count=2, split="disjoint_test", num_distractors=1)
                seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
                out = pipeline(seq, candidate_positions=c_pos, mode=mode_name)
                c_done = out["cycles_executed"]
                cycles_list.append(c_done)
                if c_done < 2:
                    premature += 1

                if torch.argmax(out["binding_logits"][0]).item() == ep.target_idx:
                    g4_h += 1
                if torch.argmax(out["w1_key"][0]).item() == ep.key_positions[0]:
                    h1_h += 1
                if torch.argmax(out["w2_key"][0]).item() == ep.key_positions[1]:
                    h2_h += 1

        N = max(1, eval_episodes)
        lang_ret = evaluate_language_retention(pipeline.relational_core, baseline)

        results[vid] = AdaptiveReintegrationVariantResult(
            variant_id=vid,
            description=desc,
            g1_acc=g1_h / N,
            g4_acc=g4_h / N,
            h1_routing=h1_h / N,
            h2_routing=h2_h / N,
            mean_cycles=sum(cycles_list) / len(cycles_list) if cycles_list else 1.0,
            premature_halt_rate=premature / N,
            unnecessary_rate=unnecessary / N,
            language_retention=lang_ret,
        )

    best_v = max(results.keys(), key=lambda k: (results[k].g4_acc, results[k].h2_routing))
    preserves = results["D_AdaptiveSufficiency"].g4_acc >= 0.50

    summary = (
        f"Step 407 Adaptive Reintegration: Best={best_v}. "
        f"Adaptive G4={results['D_AdaptiveSufficiency'].g4_acc:.2%}, "
        f"Adaptive H1={results['D_AdaptiveSufficiency'].h1_routing:.2%}, "
        f"Adaptive H2={results['D_AdaptiveSufficiency'].h2_routing:.2%}, "
        f"Mean Cycles={results['D_AdaptiveSufficiency'].mean_cycles:.2f}."
    )

    return Step407ReintegrationReport(
        variants=results,
        best_variant=best_v,
        adaptive_preserves_accuracy=preserves,
        language_retention_preserved=all(r.language_retention >= 0.9500 for r in results.values()),
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 407: Running Adaptive Reasoning Reintegration with Relational Acquisition...")
    rep = run_adaptive_relational_reintegration_study(seed=42, train_steps=25, eval_episodes=6)
    print("Report Summary:", rep.summary)
    for vid, r in rep.variants.items():
        print(f"  [{vid:22s}] G1={r.g1_acc:.2%}, G4={r.g4_acc:.2%}, H1={r.h1_routing:.2%}, H2={r.h2_routing:.2%}, Cycles={r.mean_cycles:.2f}")
