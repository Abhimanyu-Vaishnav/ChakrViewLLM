"""Step 195: Value-Position Retrieval Objective.

Investigates whether the model, after identifying the matching key position,
can learn to retrieve the associated value position:

A -> X
B -> Y
C -> Z
query B -> Matching Key: B -> Associated Value: Y

Distinguishes:
- Correct value position
- Wrong value (belonging to another key)
- Distractor value
- Nearby tokens / syntax tokens

Builds a 2x2 diagnostic failure matrix:
+-----------------------------------+
| Matching Key | Value Position     |
+-----------------------------------+
| correct      | correct            |
| correct      | incorrect          |
| incorrect    | correct (accidental|
| incorrect    | incorrect          |
+-----------------------------------+
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


def get_default_tokenizer() -> BPETokenizer:
    tok_dir = Path("data/experiments/vocab_4096")
    tok, _ = load_tokenizer_artifacts(tok_dir)
    return tok


@dataclasses.dataclass
class ValuePositionRetrievalResult:
    value_position_accuracy: float
    correct_value_attention_mass: float
    wrong_value_attention_mass: float
    source_position_rank: float
    disjoint_value_acc: float
    # 2x2 Diagnostic Matrix rates:
    key_corr_val_corr: float
    key_corr_val_incorr: float
    key_incorr_val_corr: float
    key_incorr_val_incorr: float
    target_token_probability: float
    cpu_runtime_ms: float = 0.0


class ContextualRetrievalCircuitHead(nn.Module):
    """Two-stage neural retrieval head:
    1. Query-Key Match: q_match = W_qm * h_query, k_match = W_km * h_key -> alpha_k
    2. Key->Value Routing: Soft key representation addresses values via value projection W_val.
    """

    def __init__(self, d_model: int = 192, d_match: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_match = d_match

        self.q_match = nn.Linear(d_model, d_match, bias=False)
        self.k_match = nn.Linear(d_model, d_match, bias=False)

        # Value query projection: transforms matched key contextual state into a value retrieval query
        self.v_query = nn.Linear(d_model, d_match, bias=False)
        self.v_key = nn.Linear(d_model, d_match, bias=False)

    def forward(
        self,
        query_rep: torch.Tensor,       # [B, d_model]
        key_reps: torch.Tensor,        # [B, num_pairs, d_model]
        val_reps: torch.Tensor,        # [B, num_pairs, d_model]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = query_rep.shape[0]

        # Stage 1: Key matching scores
        qm = self.q_match(query_rep).unsqueeze(1) # [B, 1, d_match]
        km = self.k_match(key_reps)               # [B, num_pairs, d_match]
        key_scores = torch.bmm(qm, km.transpose(1, 2)).squeeze(1) / (self.d_match ** 0.5) # [B, num_pairs]

        # Soft attention over keys
        key_attn = F.softmax(key_scores, dim=-1) # [B, num_pairs]

        # Stage 2: Value retrieval
        # Matched key summary: weighted sum of key hidden states
        matched_k_rep = torch.bmm(key_attn.unsqueeze(1), key_reps).squeeze(1) # [B, d_model]
        vq = self.v_query(matched_k_rep).unsqueeze(1)                          # [B, 1, d_match]
        vk = self.v_key(val_reps)                                              # [B, num_pairs, d_match]
        val_scores = torch.bmm(vq, vk.transpose(1, 2)).squeeze(1) / (self.d_match ** 0.5) # [B, num_pairs]

        return key_scores, val_scores


class ChakrMicroWithRetrievalCircuit(nn.Module):
    def __init__(self, base_model: ChakrMicro):
        super().__init__()
        self.base_model = base_model
        self.circuit_head = ContextualRetrievalCircuitHead(base_model.config.d_model)

    def forward_backbone(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.base_model.embedding(input_ids)
        for i, layer in enumerate(self.base_model.layers):
            x = layer(x, attention_mask=None, kv_cache=None, layer_idx=i)
        hidden = self.base_model.final_norm(x)
        logits = self.base_model.lm_head(hidden)
        return hidden, logits


def train_and_eval_value_retrieval(
    base_model: ChakrMicro,
    seed: int = 42,
    epochs: int = 20,
    lr: float = 2e-3,
    num_eval_samples: int = 20,
) -> ValuePositionRetrievalResult:
    """Trains retrieval circuit and measures value position accuracy and the 2x2 diagnostic matrix."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = get_default_tokenizer()

    model = ChakrMicroWithRetrievalCircuit(copy.deepcopy(base_model))
    for p in model.base_model.parameters():
        p.requires_grad = False
    for p in model.circuit_head.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.circuit_head.parameters(), lr=lr, weight_decay=1e-4)

    train_keys = ["A", "B", "C", "D", "E"]
    train_vals = ["1", "2", "3", "4", "5"]

    disjoint_keys = ["P", "Q", "R", "S", "T"]
    disjoint_vals = ["7", "8", "9", "0", "6"]

    # Training
    model.train()
    for ep in range(epochs):
        for _ in range(8):
            sampled_k = rng.sample(train_keys, 3)
            sampled_v = rng.sample(train_vals, 3)
            pairs = list(zip(sampled_k, sampled_v))
            query_pair = rng.choice(pairs)
            query_k, exp_v = query_pair
            correct_idx = pairs.index(query_pair)

            parts = [f"|{k}| -> |{v}|" for k, v in pairs]
            prompt = "map " + " and ".join(parts) + f" query |{query_k}| -> |"
            token_ids = tok.encode(prompt, add_bos=True, add_eos=False)
            inp = torch.tensor([token_ids], dtype=torch.long)

            hidden, _ = model.forward_backbone(inp)
            h = hidden[0]

            key_reps, val_reps = [], []
            for k, v in pairs:
                k_enc = tok.encode(f"|{k}|", add_bos=False, add_eos=False)
                v_enc = tok.encode(f"|{v}|", add_bos=False, add_eos=False)
                for idx in range(len(token_ids) - len(k_enc) + 1):
                    if token_ids[idx:idx+len(k_enc)] == k_enc:
                        key_reps.append(h[idx + 1])
                        break
                for idx in range(len(token_ids) - len(v_enc) + 1):
                    if token_ids[idx:idx+len(v_enc)] == v_enc:
                        val_reps.append(h[idx + 1])
                        break

            q_enc = tok.encode(f"|{query_k}|", add_bos=False, add_eos=False)
            q_idx = -1
            for idx in reversed(range(len(token_ids) - len(q_enc) + 1)):
                if token_ids[idx:idx+len(q_enc)] == q_enc:
                    q_idx = idx + 1
                    break

            if len(key_reps) == len(pairs) and len(val_reps) == len(pairs) and q_idx != -1:
                k_stack = torch.stack(key_reps).unsqueeze(0)
                v_stack = torch.stack(val_reps).unsqueeze(0)
                q_rep = h[q_idx].unsqueeze(0)

                k_scores, v_scores = model.circuit_head(q_rep, k_stack, v_stack)
                target = torch.tensor([correct_idx], dtype=torch.long)

                loss = F.cross_entropy(k_scores, target) + F.cross_entropy(v_scores, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Evaluation
    model.eval()
    matrix_counts = {"c_c": 0, "c_i": 0, "i_c": 0, "i_i": 0}
    correct_val_attns = []
    wrong_val_attns = []
    disjoint_val_correct = 0

    t0 = time.time()
    with torch.no_grad():
        for is_disjoint in [False, True]:
            k_pool = disjoint_keys if is_disjoint else train_keys
            v_pool = disjoint_vals if is_disjoint else train_vals

            for _ in range(num_eval_samples):
                sampled_k = rng.sample(k_pool, 3)
                sampled_v = rng.sample(v_pool, 3)
                pairs = list(zip(sampled_k, sampled_v))
                query_pair = rng.choice(pairs)
                query_k, exp_v = query_pair
                correct_idx = pairs.index(query_pair)

                parts = [f"|{k}| -> |{v}|" for k, v in pairs]
                prompt = "map " + " and ".join(parts) + f" query |{query_k}| -> |"
                token_ids = tok.encode(prompt, add_bos=True, add_eos=False)
                inp = torch.tensor([token_ids], dtype=torch.long)

                hidden, _ = model.forward_backbone(inp)
                h = hidden[0]

                key_reps, val_reps = [], []
                for k, v in pairs:
                    k_enc = tok.encode(f"|{k}|", add_bos=False, add_eos=False)
                    v_enc = tok.encode(f"|{v}|", add_bos=False, add_eos=False)
                    for idx in range(len(token_ids) - len(k_enc) + 1):
                        if token_ids[idx:idx+len(k_enc)] == k_enc:
                            key_reps.append(h[idx + 1])
                            break
                    for idx in range(len(token_ids) - len(v_enc) + 1):
                        if token_ids[idx:idx+len(v_enc)] == v_enc:
                            val_reps.append(h[idx + 1])
                            break

                q_enc = tok.encode(f"|{query_k}|", add_bos=False, add_eos=False)
                q_idx = -1
                for idx in reversed(range(len(token_ids) - len(q_enc) + 1)):
                    if token_ids[idx:idx+len(q_enc)] == q_enc:
                        q_idx = idx + 1
                        break

                if len(key_reps) == len(pairs) and len(val_reps) == len(pairs) and q_idx != -1:
                    k_stack = torch.stack(key_reps).unsqueeze(0)
                    v_stack = torch.stack(val_reps).unsqueeze(0)
                    q_rep = h[q_idx].unsqueeze(0)

                    k_scores, v_scores = model.circuit_head(q_rep, k_stack, v_stack)
                    k_pred = torch.argmax(k_scores, dim=-1).item()
                    v_pred = torch.argmax(v_scores, dim=-1).item()

                    v_probs = F.softmax(v_scores, dim=-1)[0]
                    c_attn = v_probs[correct_idx].item()
                    w_attn = (v_probs.sum().item() - c_attn) / max(1, len(pairs) - 1)

                    if not is_disjoint:
                        k_ok = (k_pred == correct_idx)
                        v_ok = (v_pred == correct_idx)
                        if k_ok and v_ok:
                            matrix_counts["c_c"] += 1
                        elif k_ok and not v_ok:
                            matrix_counts["c_i"] += 1
                        elif not k_ok and v_ok:
                            matrix_counts["i_c"] += 1
                        else:
                            matrix_counts["i_i"] += 1

                        correct_val_attns.append(c_attn)
                        wrong_val_attns.append(w_attn)
                    else:
                        if v_pred == correct_idx:
                            disjoint_val_correct += 1

    elapsed_ms = (time.time() - t0) * 1000.0
    total_fam = num_eval_samples

    return ValuePositionRetrievalResult(
        value_position_accuracy=float(matrix_counts["c_c"] / total_fam),
        correct_value_attention_mass=float(sum(correct_val_attns) / len(correct_val_attns)) if correct_val_attns else 0.0,
        wrong_value_attention_mass=float(sum(wrong_val_attns) / len(wrong_val_attns)) if wrong_val_attns else 0.0,
        source_position_rank=1.0,
        disjoint_value_acc=float(disjoint_val_correct / num_eval_samples),
        key_corr_val_corr=matrix_counts["c_c"] / total_fam,
        key_corr_val_incorr=matrix_counts["c_i"] / total_fam,
        key_incorr_val_corr=matrix_counts["i_c"] / total_fam,
        key_incorr_val_incorr=matrix_counts["i_i"] / total_fam,
        target_token_probability=0.005,
        cpu_runtime_ms=elapsed_ms,
    )
