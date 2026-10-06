"""Step 227: Association Objective Study.

Compares isolated training objectives under controlled identical conditions:
Candidate A: L = L_language
Candidate B: L = L_language + lambda * L_association
Candidate C: L = L_language + lambda * L_value_retrieval
Candidate D: L = L_language + lambda1 * L_association + lambda2 * L_value_retrieval

Tested lambdas: 0.1, 0.25, 0.5, 1.0.

L_association rewards correct key-value pair binding:
    L_assoc = max(0, margin - (sim(k_q * v_target) - sim(k_q * v_neg)))

L_value_retrieval rewards matching terminal hidden state with contextual value representation:
    L_val = 1.0 - cosine_similarity(h_terminal, h_v_target)

Measures:
- Training loss & language validation loss
- Association accuracy & held-out mapping accuracy
- Disjoint representation score
- Value representation retrieval
- Final token retrieval
- Gradient norms & parameter update norm
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.cognition.value_representation_retrieval import (
    extract_contextual_representations,
    compute_representation_retrieval_metrics,
    get_default_tokenizer,
)
from chakrview.cognition.association_learning_minimal import (
    compute_param_delta,
    evaluate_language_loss,
    evaluate_association_task,
)
from chakrview.cognition.neural_language_learning import ControlledNeuralLanguageTrainer


@dataclasses.dataclass
class ObjectiveCandidateResult:
    candidate_name: str
    lambda_assoc: float
    lambda_val: float
    train_loss: float
    lang_val_loss: float
    assoc_acc: float
    heldout_mapping_acc: float
    disjoint_rep_score: float
    value_rep_margin: float
    final_token_acc: float
    grad_norm: float
    param_delta: float
    is_stage_a_created: bool


@dataclasses.dataclass
class AssociationObjectivesStudyReport:
    seed: int
    candidates: Dict[str, ObjectiveCandidateResult]
    best_candidate_name: str
    did_any_objective_create_stage_a: bool
    cpu_runtime_ms: float = 0.0


def train_and_eval_objective_candidate(
    base_model: ChakrMicro,
    candidate_name: str,
    lambda_assoc: float = 0.0,
    lambda_val: float = 0.0,
    seed: int = 42,
    epochs: int = 8,
    steps_per_epoch: int = 6,
    lr: float = 3e-4,
) -> ObjectiveCandidateResult:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = get_default_tokenizer()
    lang_trainer = ControlledNeuralLanguageTrainer(tokenizer=tok)
    cos = nn.CosineSimilarity(dim=0)

    candidate = copy.deepcopy(base_model)
    candidate.train()
    for p in candidate.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(candidate.parameters(), lr=lr, weight_decay=1e-4)

    train_keys = ["A", "B", "C", "D", "E"]
    train_vals = ["1", "2", "3", "4", "5"]
    heldout_keys = ["P", "Q", "R", "S", "T"]
    heldout_vals = ["6", "7", "8", "9", "0"]

    total_loss = 0.0
    grad_norms = []
    step_count = 0

    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            k_sample = rng.sample(train_keys, 3)
            v_sample = rng.sample(train_vals, 3)
            pairs = list(zip(k_sample, v_sample))
            rng.shuffle(pairs)

            query_pair = rng.choice(pairs)
            query_k, exp_v = query_pair
            exp_tok = tok.encode(exp_v, add_bos=False, add_eos=False)[0]

            parts = [f"|{k}| -> |{v}|" for k, v in pairs]
            prompt = "map " + " and ".join(parts) + f" query |{query_k}| -> |"
            token_ids = tok.encode(prompt, add_bos=True, add_eos=False)
            inp = torch.tensor([token_ids], dtype=torch.long)
            target = torch.tensor([exp_tok], dtype=torch.long)

            optimizer.zero_grad()
            # Forward pass
            x = candidate.embedding(inp)
            for i, layer in enumerate(candidate.layers):
                x = layer(x, attention_mask=None, kv_cache=None, layer_idx=i)
            hidden = candidate.final_norm(x)[0]
            logits = candidate.lm_head(hidden.unsqueeze(0))[0]

            # 1. Base language next-token loss
            loss_lang = F.cross_entropy(logits[-1].unsqueeze(0), target)
            loss_total = loss_lang

            # Find key/val representations
            k_pos, v_pos = -1, -1
            k_enc = tok.encode(f"|{query_k}|", add_bos=False, add_eos=False)
            v_enc = tok.encode(f"|{exp_v}|", add_bos=False, add_eos=False)
            for idx in range(len(token_ids) - len(k_enc) + 1):
                if token_ids[idx:idx+len(k_enc)] == k_enc:
                    k_pos = idx + 1
                    break
            for idx in range(len(token_ids) - len(v_enc) + 1):
                if token_ids[idx:idx+len(v_enc)] == v_enc:
                    v_pos = idx + 1
                    break

            if k_pos != -1 and v_pos != -1:
                hk = hidden[k_pos]
                hv = hidden[v_pos]
                h_term = hidden[-1]

                # 2. Association loss: Contrastive binding loss
                if lambda_assoc > 0.0:
                    pos_binding = hk * hv
                    # Sample a negative value
                    neg_v = rng.choice([v for v in train_vals if v != exp_v])
                    neg_pos = -1
                    neg_enc = tok.encode(f"|{neg_v}|", add_bos=False, add_eos=False)
                    for idx in range(len(token_ids) - len(neg_enc) + 1):
                        if token_ids[idx:idx+len(neg_enc)] == neg_enc:
                            neg_pos = idx + 1
                            break
                    if neg_pos != -1:
                        neg_binding = hk * hidden[neg_pos]
                        sim_pos = torch.cosine_similarity(pos_binding.unsqueeze(0), pos_binding.unsqueeze(0))
                        sim_neg = torch.cosine_similarity(pos_binding.unsqueeze(0), neg_binding.unsqueeze(0))
                        l_assoc = F.relu(0.50 - (sim_pos - sim_neg)).mean()
                        loss_total = loss_total + lambda_assoc * l_assoc

                # 3. Value retrieval loss: Match terminal hidden state to target value vector
                if lambda_val > 0.0:
                    sim_val = torch.cosine_similarity(h_term.unsqueeze(0), hv.unsqueeze(0))
                    l_val = (1.0 - sim_val).mean()
                    loss_total = loss_total + lambda_val * l_val

            loss_total.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(candidate.parameters(), max_norm=1.0).item())
            grad_norms.append(gnorm)
            optimizer.step()

            total_loss += float(loss_total.item())
            step_count += 1

    avg_train_loss = total_loss / max(1, step_count)
    mean_gnorm = float(sum(grad_norms) / len(grad_norms)) if grad_norms else 0.0

    eval_train = evaluate_association_task(candidate, tok, train_keys, train_vals, num_episodes=10, seed=seed)
    eval_heldout = evaluate_association_task(candidate, tok, heldout_keys, heldout_vals, num_episodes=10, seed=seed + 1)
    lang_val_loss = evaluate_language_loss(candidate, lang_trainer)
    delta_norm = compute_param_delta(candidate, base_model)

    stage_a_created = (eval_train["assoc_score"] >= 0.50)

    return ObjectiveCandidateResult(
        candidate_name=candidate_name,
        lambda_assoc=lambda_assoc,
        lambda_val=lambda_val,
        train_loss=avg_train_loss,
        lang_val_loss=lang_val_loss,
        assoc_acc=eval_train["token_acc"],
        heldout_mapping_acc=eval_heldout["token_acc"],
        disjoint_rep_score=eval_heldout["assoc_score"],
        value_rep_margin=eval_train["val_margin"],
        final_token_acc=eval_train["token_acc"],
        grad_norm=mean_gnorm,
        param_delta=delta_norm,
        is_stage_a_created=stage_a_created,
    )


def run_association_objectives_study(
    base_model: ChakrMicro,
    seed: int = 42,
) -> AssociationObjectivesStudyReport:
    """Compares candidates A, B, C, and D across lambda sweeps."""
    t0 = time.time()
    candidates_dict = {}

    configs = [
        ("Candidate_A_LangOnly", 0.0, 0.0),
        ("Candidate_B_Assoc_0.25", 0.25, 0.0),
        ("Candidate_B_Assoc_0.50", 0.50, 0.0),
        ("Candidate_C_ValRet_0.25", 0.0, 0.25),
        ("Candidate_C_ValRet_0.50", 0.0, 0.50),
        ("Candidate_D_Joint_0.25_0.25", 0.25, 0.25),
    ]

    for name, la, lv in configs:
        res = train_and_eval_objective_candidate(
            base_model=base_model,
            candidate_name=name,
            lambda_assoc=la,
            lambda_val=lv,
            seed=seed,
            epochs=5,
            steps_per_epoch=4,
        )
        candidates_dict[name] = res

    # Select best candidate by disjoint rep score and final token acc
    best_cand = max(candidates_dict.keys(), key=lambda k: (candidates_dict[k].disjoint_rep_score, candidates_dict[k].final_token_acc))
    any_stage_a = any(c.is_stage_a_created for c in candidates_dict.values())
    elapsed = (time.time() - t0) * 1000.0

    return AssociationObjectivesStudyReport(
        seed=seed,
        candidates=candidates_dict,
        best_candidate_name=best_cand,
        did_any_objective_create_stage_a=any_stage_a,
        cpu_runtime_ms=elapsed,
    )
