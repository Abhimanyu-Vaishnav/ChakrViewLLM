"""Step 202: Relative Key -> Value Binding Investigation.

Investigates whether the association between key and value is learned
relative to the semantic relationship rather than relying on absolute positions
or fixed positional offsets:

Case A: A -> X, B -> Y, query B -> Y
Case B: X -> A, Y -> B, query Y -> B (Reversed direction)
Case C: B -> Y, A -> X, query B -> Y (Permuted pair order)
Case D: Y -> B, X -> A, query Y -> B (Reversed direction and permuted)

Measures:
- Directional invariance
- Positional attention bias
- Relative association accuracy vs absolute positional heuristic.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.value_position_retrieval import ChakrMicroWithRetrievalCircuit
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def get_default_tokenizer() -> BPETokenizer:
    tok_dir = Path("data/experiments/vocab_4096")
    tok, _ = load_tokenizer_artifacts(tok_dir)
    return tok


@dataclasses.dataclass
class RelativeBindingResult:
    case_a_forward_acc: float
    case_b_reversed_acc: float
    case_c_permuted_acc: float
    case_d_reversed_permuted_acc: float
    mean_directional_invariance: float
    positional_attention_bias: float
    is_relative_relationship_learned: bool
    cpu_runtime_ms: float = 0.0


def evaluate_relative_binding(
    model: ChakrMicro,
    seed: int = 42,
    num_samples: int = 15,
) -> RelativeBindingResult:
    """Evaluates binding across relative layout transformations."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = get_default_tokenizer()
    model.eval()

    keys = ["A", "B", "C"]
    vals = ["1", "2", "3"]

    results = {"A": 0, "B": 0, "C": 0, "D": 0}
    t0 = time.time()

    with torch.no_grad():
        for _ in range(num_samples):
            # Sample 2 pairs
            sampled_k = rng.sample(keys, 2)
            sampled_v = rng.sample(vals, 2)
            k1, k2 = sampled_k
            v1, v2 = sampled_v

            # Case A: Forward standard (A -> 1, B -> 2, query B -> 2)
            prompt_a = f"map |{k1}| -> |{v1}| and |{k2}| -> |{v2}| query |{k2}| -> |"
            exp_a = tok.encode(v2, add_bos=False, add_eos=False)[0]

            # Case B: Reversed direction (1 -> A, 2 -> B, query 2 -> B)
            prompt_b = f"map |{v1}| -> |{k1}| and |{v2}| -> |{k2}| query |{v2}| -> |"
            exp_b = tok.encode(k2, add_bos=False, add_eos=False)[0]

            # Case C: Permuted pair order (B -> 2, A -> 1, query B -> 2)
            prompt_c = f"map |{k2}| -> |{v2}| and |{k1}| -> |{v1}| query |{k2}| -> |"
            exp_c = tok.encode(v2, add_bos=False, add_eos=False)[0]

            # Case D: Reversed and permuted (2 -> B, 1 -> A, query 2 -> B)
            prompt_d = f"map |{v2}| -> |{k2}| and |{v1}| -> |{k1}| query |{v2}| -> |"
            exp_d = tok.encode(k2, add_bos=False, add_eos=False)[0]

            for case_key, p, exp_tok in [("A", prompt_a, exp_a), ("B", prompt_b, exp_b), ("C", prompt_c, exp_c), ("D", prompt_d, exp_d)]:
                token_ids = tok.encode(p, add_bos=True, add_eos=False)
                inp = torch.tensor([token_ids], dtype=torch.long)
                logits = model(inp)
                pred = torch.argmax(logits[0, -1, :]).item()
                if pred == exp_tok:
                    results[case_key] += 1

    elapsed = (time.time() - t0) * 1000.0
    acc_a = float(results["A"] / num_samples)
    acc_b = float(results["B"] / num_samples)
    acc_c = float(results["C"] / num_samples)
    acc_d = float(results["D"] / num_samples)

    # Directional invariance is consistency between forward and reversed cases
    invariance = 1.0 - abs(acc_a - acc_b)
    # Positional bias: difference between order A and order C
    pos_bias = abs(acc_a - acc_c)

    return RelativeBindingResult(
        case_a_forward_acc=acc_a,
        case_b_reversed_acc=acc_b,
        case_c_permuted_acc=acc_c,
        case_d_reversed_permuted_acc=acc_d,
        mean_directional_invariance=invariance,
        positional_attention_bias=pos_bias,
        is_relative_relationship_learned=(acc_a >= 0.50 and acc_b >= 0.50),
        cpu_runtime_ms=elapsed,
    )
