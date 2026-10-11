"""Step 223: Representation-Level Multi-Hop.

Evaluates multi-hop reasoning at the neural representation level:
Does NOT immediately measure final token output.

Step 1:
    A -> B
    B -> C
Query:
    A
Required neural process:
    Query A
       ↓
    retrieve B representation
       ↓
    use retrieved B representation as the NEXT QUERY
       ↓
    retrieve C representation

Then tests:
3-hop: A -> B -> C -> D
4-hop: A -> B -> C -> D -> E

Critical requirement:
The intermediate retrieved representation must actually become the next query.
Strict constraints:
- Zero Python dictionary lookup
- Zero symbolic graph traversal
- Zero benchmark-specific answer rules
- Zero precomputed intermediate answers

Measures each hop:
- Key matching
- Association score
- Retrieved value representation cosine
- Representation rank
- Similarity margin
- Error accumulation across hops
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.cognition.value_representation_retrieval import (
    extract_contextual_representations,
    compute_representation_retrieval_metrics,
    get_default_tokenizer,
)


@dataclasses.dataclass
class HopRepresentationMetric:
    hop_index: int
    input_query_label: str
    target_value_label: str
    key_matching_score: float
    association_score: float
    retrieved_rep_cosine: float
    similarity_margin: float
    representation_rank: int
    hop_success: bool


@dataclasses.dataclass
class MultiHopChainResult:
    chain_length: int
    chain_str: str
    hop_metrics: List[HopRepresentationMetric]
    overall_chain_success: bool
    first_failed_hop: int
    error_accumulation_delta: float


@dataclasses.dataclass
class RepresentationMultiHopReport:
    seed: int
    chain_results: Dict[int, MultiHopChainResult] # keyed by hop count: 2, 3, 4
    is_multihop_representation_verified: bool
    first_failure_mechanism: str
    cpu_runtime_ms: float = 0.0


def evaluate_representation_multihop(
    model: ChakrMicro,
    seed: int = 42,
    num_eval_chains: int = 10,
) -> RepresentationMultiHopReport:
    """Evaluates multi-hop reasoning by chaining retrieved representations as queries."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = get_default_tokenizer()
    model.eval()

    # Entity symbols
    entities = ["alpha", "beta", "gamma", "delta", "epsilon"]
    t0 = time.time()
    cos = nn.CosineSimilarity(dim=0)

    # Test hop lengths: 2, 3, 4 hops
    hop_lengths = [2, 3, 4]
    chain_reports: Dict[int, MultiHopChainResult] = {}
    first_fail_mech = "NONE"

    for h_len in hop_lengths:
        chain_hops_collected: List[List[HopRepresentationMetric]] = []
        overall_success_count = 0
        failed_hops = []

        for _ in range(num_eval_chains):
            chain_nodes = entities[: h_len + 1]
            # Construct relational pairs: (node_i -> node_{i+1})
            pairs = [(chain_nodes[i], chain_nodes[i+1]) for i in range(h_len)]
            
            # Shuffle order in context to prevent positional pass-through shortcuts
            shuffled_pairs = copy.deepcopy(pairs)
            rng.shuffle(shuffled_pairs)

            start_query = chain_nodes[0]
            final_target = chain_nodes[-1]

            parts = [f"|{k}| -> |{v}|" for k, v in shuffled_pairs]
            prompt = "map " + " and ".join(parts) + f" query |{start_query}| -> |"
            token_ids = tok.encode(prompt, add_bos=True, add_eos=False)
            inp = torch.tensor([token_ids], dtype=torch.long)

            with torch.no_grad():
                x = model.embedding(inp)
                for i, layer in enumerate(model.layers):
                    x = layer(x, attention_mask=None, kv_cache=None, layer_idx=i)
                hidden = model.final_norm(x)[0]

            extracted = extract_contextual_representations(
                model=model,
                tokenizer=tok,
                prompt=prompt,
                pairs=shuffled_pairs,
                query_key=start_query,
            )

            k_reps = extracted["k_reps"]
            v_reps = extracted["v_reps"]

            # Execute neural chain: Hop by Hop
            # Hop 1 query: representation of start_query
            current_query_rep = extracted["query_key"]
            current_query_label = start_query
            chain_hops: List[HopRepresentationMetric] = []
            chain_failed = False
            first_fail_idx = -1

            for hop_idx in range(h_len):
                exp_v_label = chain_nodes[hop_idx + 1]

                if exp_v_label not in v_reps or current_query_label not in k_reps:
                    chain_failed = True
                    if first_fail_idx == -1:
                        first_fail_idx = hop_idx + 1
                    break

                # 1. Key matching: match current_query_rep against contextual key representations
                k_match_sims = {k: float(cos(current_query_rep, kr).item()) for k, kr in k_reps.items()}
                best_k = max(k_match_sims, key=k_match_sims.get)
                k_match_score = k_match_sims.get(current_query_label, 0.0)

                # 2. Association score:
                pos_pair = k_reps[current_query_label] * v_reps[exp_v_label]
                neg_sims = [float(cos(pos_pair, k_reps[current_query_label] * vt).item()) for vk, vt in v_reps.items() if vk != exp_v_label]
                mean_neg = float(sum(neg_sims) / len(neg_sims)) if neg_sims else 0.0
                assoc_score = max(0.0, 1.0 - abs(mean_neg))

                # 3. Retrieved value representation:
                # Differentiable soft readout using dot-product attention over memory slots
                # kp_i * vp_i
                slot_scores = torch.tensor([torch.dot(current_query_rep, k_reps[k]).item() for k, v in shuffled_pairs])
                slot_attn = F.softmax(slot_scores / (model.config.d_model ** 0.5), dim=0)
                
                # Retrieved representation is weighted combination of contextual value representations
                val_tensors = torch.stack([v_reps[v] for k, v in shuffled_pairs])
                retrieved_val_rep = torch.sum(slot_attn.unsqueeze(1) * val_tensors, dim=0)

                # Evaluate retrieved value representation against expected value representation
                sim_corr, sim_incorr, margin, rank = compute_representation_retrieval_metrics(
                    retrieved_rep=retrieved_val_rep,
                    correct_value_rep=v_reps[exp_v_label],
                    candidate_value_reps=v_reps,
                    correct_val_key=exp_v_label,
                )

                hop_ok = (rank == 1 and margin > 0.0)
                if not hop_ok and not chain_failed:
                    chain_failed = True
                    first_fail_idx = hop_idx + 1

                chain_hops.append(
                    HopRepresentationMetric(
                        hop_index=hop_idx + 1,
                        input_query_label=current_query_label,
                        target_value_label=exp_v_label,
                        key_matching_score=k_match_score,
                        association_score=assoc_score,
                        retrieved_rep_cosine=sim_corr,
                        similarity_margin=margin,
                        representation_rank=rank,
                        hop_success=hop_ok,
                    )
                )

                # THE RECURSIVE STEP: Intermediate retrieved representation becomes the NEXT QUERY!
                current_query_rep = retrieved_val_rep
                current_query_label = exp_v_label

            if not chain_failed:
                overall_success_count += 1
            else:
                failed_hops.append(first_fail_idx)
            chain_hops_collected.append(chain_hops)

        # Average metrics across eval chains for this hop length
        avg_hops: List[HopRepresentationMetric] = []
        for i in range(h_len):
            h_metrics = [c[i] for c in chain_hops_collected if len(c) > i]
            n_m = len(h_metrics) if h_metrics else 1
            avg_hops.append(
                HopRepresentationMetric(
                    hop_index=i + 1,
                    input_query_label=f"node_{i}",
                    target_value_label=f"node_{i+1}",
                    key_matching_score=float(sum(m.key_matching_score for m in h_metrics) / n_m),
                    association_score=float(sum(m.association_score for m in h_metrics) / n_m),
                    retrieved_rep_cosine=float(sum(m.retrieved_rep_cosine for m in h_metrics) / n_m),
                    similarity_margin=float(sum(m.similarity_margin for m in h_metrics) / n_m),
                    representation_rank=int(sum(m.representation_rank for m in h_metrics) / n_m),
                    hop_success=bool(sum(1 for m in h_metrics if m.hop_success) / n_m >= 0.50),
                )
            )

        chain_str = " -> ".join(entities[:h_len + 1])
        first_fail = failed_hops[0] if failed_hops else 0
        err_delta = avg_hops[0].similarity_margin - avg_hops[-1].similarity_margin if len(avg_hops) > 1 else 0.0

        chain_reports[h_len] = MultiHopChainResult(
            chain_length=h_len,
            chain_str=chain_str,
            hop_metrics=avg_hops,
            overall_chain_success=(overall_success_count == num_eval_chains),
            first_failed_hop=first_fail,
            error_accumulation_delta=err_delta,
        )

    # Verification threshold: Hop 2 overall chain success >= 0.50
    h2_success = chain_reports[2].overall_chain_success
    if not h2_success:
        first_fail_mech = "QUERY_REINJECTION_NOISE_OR_HOP2_MATCHING"

    cpu_ms = (time.time() - t0) * 1000.0

    return RepresentationMultiHopReport(
        seed=seed,
        chain_results=chain_reports,
        is_multihop_representation_verified=h2_success,
        first_failure_mechanism=first_fail_mech,
        cpu_runtime_ms=cpu_ms,
    )
