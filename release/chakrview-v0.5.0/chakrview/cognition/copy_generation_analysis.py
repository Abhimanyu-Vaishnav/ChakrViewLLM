"""Step 188: Copy vs Generation Empirical Analysis.

Compares four architectural candidates:
A. Original tied readout (canonical baseline backbone)
B. Untied readout (independent LM head)
C. Copy-only candidate (p_copy = 1.0)
D. Hybrid generation + copy candidate (learned gate)

Rigorous evaluation on:
- Output accuracy
- Copy probability vs generation probability
- Selected source position and whether it corresponds to the correct value
- Target probability, target rank, target logit
- Disjoint accuracy (unseen keys and values)
- Disentangles retrieval failure vs readout failure.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.cognition.copy_attention import ChakrMicroWithCopy, CopyMechanismOutput
from chakrview.cognition.disjoint_token_benchmark import DisjointRetrievalFixture, get_default_tokenizer
from chakrview.cognition.untied_readout import create_untied_candidate
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer


@dataclasses.dataclass
class CopyVsGenerationResult:
    candidate_name: str
    total_params: int
    trainable_params: int
    familiar_acc: float
    disjoint_acc: float
    mean_copy_prob: float
    mean_gen_prob: float
    source_position_accuracy: float   # did copy attention point to the exact index of the target value?
    target_prob: float
    target_logit: float
    median_rank: float
    retrieval_failure_rate: float     # attention did not point to correct token
    readout_failure_rate: float       # attention pointed to token, but combined logits did not select it
    cpu_runtime_ms: float = 0.0


def train_copy_model(
    model: ChakrMicroWithCopy,
    tokenizer: BPETokenizer,
    seed: int = 42,
    epochs: int = 25,
    lr: float = 2e-3,
) -> float:
    """Train copy model on associative mapping prompts to learn pointer routing."""
    torch.manual_seed(seed)
    train_pairs = [("A", "1"), ("B", "2"), ("C", "3"), ("D", "4"), ("E", "5")]

    samples = []
    for _ in range(8):
        for i in range(len(train_pairs)):
            pairs = list(train_pairs)
            q_k, exp_v = pairs[i]
            parts = [f"|{k}| -> |{v}|" for k, v in pairs]
            text = "map " + " and ".join(parts) + f" query |{q_k}| -> |{exp_v}"
            toks = tokenizer.encode(text, add_bos=True, add_eos=False)
            inp = torch.tensor(toks[:-1], dtype=torch.long)
            tgt = torch.tensor(toks[1:], dtype=torch.long)
            samples.append((inp, tgt))

    # Freeze base model parameters so that only the copy head is trained
    for p in model.base_model.parameters():
        p.requires_grad = False
    for p in model.copy_head.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.copy_head.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    total_loss = 0.0
    for ep in range(epochs):
        ep_loss = 0.0
        for inp, tgt in samples:
            optimizer.zero_grad()
            logits, _ = model(inp.unsqueeze(0))
            loss = F.cross_entropy(logits[0], tgt)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        total_loss = ep_loss / len(samples)

    return total_loss


def evaluate_copy_vs_generation(
    base_model: ChakrMicro,
    seed: int = 42,
    num_samples: int = 20,
) -> Dict[str, CopyVsGenerationResult]:
    """Runs Step 188 comparative analysis across 4 architectures."""
    torch.manual_seed(seed)
    tokenizer = get_default_tokenizer()
    fixture = DisjointRetrievalFixture(seed=seed, tokenizer=tokenizer)

    # 1. Tied readout candidate (frozen base model)
    tied_model = copy.deepcopy(base_model)
    tied_model.eval()

    # 2. Untied readout candidate
    untied_model = create_untied_candidate(base_model)
    untied_model.eval()

    # 3. Hybrid copy model (learned gate)
    hybrid_model = ChakrMicroWithCopy(copy.deepcopy(base_model), force_mode=None)
    train_copy_model(hybrid_model, tokenizer, seed=seed)
    hybrid_model.eval()

    # 4. Copy-only model (force copy)
    copy_only_model = ChakrMicroWithCopy(copy.deepcopy(base_model), force_mode="copy_only")
    train_copy_model(copy_only_model, tokenizer, seed=seed)
    copy_only_model.eval()

    candidates = {
        "tied_readout": tied_model,
        "untied_readout": untied_model,
        "copy_only": copy_only_model,
        "hybrid_copy_gen": hybrid_model,
    }

    suite = fixture.generate_benchmark_suite(samples_per_split=num_samples)
    familiar_samples = suite["known_known"]
    disjoint_samples = suite["unseen_unseen"]

    results = {}

    for name, cand in candidates.items():
        t0 = time.time()
        is_copy_model = isinstance(cand, ChakrMicroWithCopy)

        fam_correct = 0
        with torch.no_grad():
            for s in familiar_samples:
                toks = tokenizer.encode(s.prompt, add_bos=True, add_eos=False)
                inp = torch.tensor([toks], dtype=torch.long)
                if is_copy_model:
                    logits, _ = cand(inp)
                else:
                    logits = cand(inp)
                pred = int(torch.argmax(logits[0, -1, :]).item())
                if pred == s.expected_token:
                    fam_correct += 1
        fam_acc = fam_correct / len(familiar_samples) if familiar_samples else 0.0

        disj_correct = 0
        copy_probs = []
        gen_probs = []
        source_correct = 0
        retrieval_failures = 0
        readout_failures = 0
        target_probs = []
        target_logits = []
        ranks = []

        with torch.no_grad():
            for s in disjoint_samples:
                toks = tokenizer.encode(s.prompt, add_bos=True, add_eos=False)
                inp = torch.tensor([toks], dtype=torch.long)

                token_ids = inp[0].tolist()
                target_positions = [i for i, tok in enumerate(token_ids) if tok == s.expected_token]

                if is_copy_model:
                    logits, copy_out = cand(inp)
                    last_cp = float(copy_out.copy_prob[0, -1, 0].item())
                    last_gp = float(copy_out.gen_prob[0, -1, 0].item())
                    sel_pos = int(copy_out.selected_source_pos[0, -1].item())
                    attn_correct = sel_pos in target_positions
                else:
                    logits = cand(inp)
                    last_cp = 0.0
                    last_gp = 1.0
                    sel_pos = -1
                    attn_correct = False

                copy_probs.append(last_cp)
                gen_probs.append(last_gp)

                if attn_correct:
                    source_correct += 1

                last_logits = logits[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                t_prob = float(probs[s.expected_token].item())
                t_logit = float(last_logits[s.expected_token].item())

                sorted_idx = torch.argsort(last_logits, descending=True)
                rank = int((sorted_idx == s.expected_token).nonzero(as_tuple=True)[0].item()) + 1

                target_probs.append(t_prob)
                target_logits.append(t_logit)
                ranks.append(rank)

                pred = int(torch.argmax(last_logits).item())
                if pred == s.expected_token:
                    disj_correct += 1
                else:
                    if is_copy_model:
                        if not attn_correct:
                            retrieval_failures += 1
                        else:
                            readout_failures += 1
                    else:
                        readout_failures += 1

        n_disj = len(disjoint_samples)
        elapsed_ms = (time.time() - t0) * 1000.0

        total_p = sum(p.numel() for p in cand.parameters())
        trainable_p = sum(p.numel() for p in cand.parameters() if p.requires_grad)

        results[name] = CopyVsGenerationResult(
            candidate_name=name,
            total_params=total_p,
            trainable_params=trainable_p,
            familiar_acc=fam_acc,
            disjoint_acc=float(disj_correct / n_disj) if n_disj else 0.0,
            mean_copy_prob=float(sum(copy_probs) / n_disj) if n_disj else 0.0,
            mean_gen_prob=float(sum(gen_probs) / n_disj) if n_disj else 0.0,
            source_position_accuracy=float(source_correct / n_disj) if n_disj else 0.0,
            target_prob=float(sum(target_probs) / n_disj) if n_disj else 0.0,
            target_logit=float(sum(target_logits) / n_disj) if n_disj else 0.0,
            median_rank=float(sorted(ranks)[len(ranks) // 2]) if ranks else 0.0,
            retrieval_failure_rate=float(retrieval_failures / n_disj) if n_disj else 0.0,
            readout_failure_rate=float(readout_failures / n_disj) if n_disj else 0.0,
            cpu_runtime_ms=elapsed_ms,
        )

    return results
