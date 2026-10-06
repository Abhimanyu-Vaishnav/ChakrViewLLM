"""Step 212: Association Update / Rebinding Investigation.

Scientific Question:
Can the contextual representation dynamically update an association in context?
B -> Y ... then B -> X ... query B -> Expected contextual output: X
Then reverse: B -> Y ... query B -> Expected: Y.

Evaluates:
Case A: Immediate rebinding (B -> Y and B -> X query B)
Case B: Delayed rebinding (with intervening distractors between old and new)
Case C: Rebinding with distractors
Case D: Repeated rebinding (B -> Y ... B -> X ... B -> Z query B -> Z)
Case E: Reverse rebinding (B -> Y ... B -> X ... B -> Y query B -> Y)

Measures:
- Overwrite ratio: Probability of new value vs old value
- Residual old-association interference
- Target rank of updated target vs obsolete target.
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
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def get_default_tokenizer() -> BPETokenizer:
    tok_dir = Path("data/experiments/vocab_4096")
    tok, _ = load_tokenizer_artifacts(tok_dir)
    return tok


@dataclasses.dataclass
class RebindingCaseMetric:
    case_name: str
    updated_target_prob: float
    old_target_prob: float
    overwrite_ratio: float  # (updated_prob / (updated_prob + old_prob))
    updated_target_rank: int
    old_target_rank: int
    is_updated_preferred: bool


@dataclasses.dataclass
class RebindingReport:
    seed: int
    cases: Dict[str, RebindingCaseMetric]
    mean_overwrite_ratio: float
    residual_interference_rate: float
    is_rebinding_supported: bool
    cpu_runtime_ms: float = 0.0


def evaluate_association_rebinding(
    model: ChakrMicro,
    seed: int = 42,
    num_samples: int = 8,
) -> RebindingReport:
    """Evaluates contextual rebinding across conditions A through E."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = get_default_tokenizer()
    model.eval()

    cases = {
        "Case_A_immediate": [],
        "Case_B_delayed": [],
        "Case_C_distractors": [],
        "Case_D_repeated": [],
        "Case_E_reverse": [],
    }

    t0 = time.time()
    for _ in range(num_samples):
        # Case A: B -> 2 and B -> 3 query B -> Target: 3 (Old: 2)
        p_a = "map |B| -> |2| and |B| -> |3| query |B| -> |"
        cases["Case_A_immediate"].append((p_a, "3", "2"))

        # Case B: B -> 2 and |J| -> |w| and B -> 3 query B -> Target: 3 (Old: 2)
        p_b = "map |B| -> |2| and |J| -> |w| and |B| -> |3| query |B| -> |"
        cases["Case_B_delayed"].append((p_b, "3", "2"))

        # Case C: B -> 2 and |J| -> |w| and |K| -> |x| and B -> 3 query B -> Target: 3 (Old: 2)
        p_c = "map |B| -> |2| and |J| -> |w| and |K| -> |x| and |B| -> |3| query |B| -> |"
        cases["Case_C_distractors"].append((p_c, "3", "2"))

        # Case D: B -> 1 and B -> 2 and B -> 3 query B -> Target: 3 (Old: 1)
        p_d = "map |B| -> |1| and |B| -> |2| and |B| -> |3| query |B| -> |"
        cases["Case_D_repeated"].append((p_d, "3", "1"))

        # Case E: B -> 1 and B -> 2 and B -> 1 query B -> Target: 1 (Old: 2)
        p_e = "map |B| -> |1| and |B| -> |2| and |B| -> |1| query |B| -> |"
        cases["Case_E_reverse"].append((p_e, "1", "2"))

    case_metrics = {}
    with torch.no_grad():
        for case_name, items in cases.items():
            new_probs = []
            old_probs = []
            new_ranks = []
            old_ranks = []

            for prompt, new_char, old_char in items:
                new_tok = tok.encode(new_char, add_bos=False, add_eos=False)[0]
                old_tok = tok.encode(old_char, add_bos=False, add_eos=False)[0]

                tokens = tok.encode(prompt, add_bos=True, add_eos=False)
                inp = torch.tensor([tokens], dtype=torch.long)
                logits = model(inp)
                last_logits = logits[0, -1, :]
                p = F.softmax(last_logits, dim=-1)

                p_new = float(p[new_tok].item())
                p_old = float(p[old_tok].item())

                sorted_idx = torch.argsort(last_logits, descending=True)
                r_new = int((sorted_idx == new_tok).nonzero(as_tuple=True)[0].item()) + 1
                r_old = int((sorted_idx == old_tok).nonzero(as_tuple=True)[0].item()) + 1

                new_probs.append(p_new)
                old_probs.append(p_old)
                new_ranks.append(r_new)
                old_ranks.append(r_old)

            mean_new_p = float(sum(new_probs) / len(new_probs))
            mean_old_p = float(sum(old_probs) / len(old_probs))
            ratio = mean_new_p / max(1e-9, (mean_new_p + mean_old_p))

            case_metrics[case_name] = RebindingCaseMetric(
                case_name=case_name,
                updated_target_prob=mean_new_p,
                old_target_prob=mean_old_p,
                overwrite_ratio=ratio,
                updated_target_rank=int(sorted(new_ranks)[len(new_ranks) // 2]),
                old_target_rank=int(sorted(old_ranks)[len(old_ranks) // 2]),
                is_updated_preferred=(mean_new_p > mean_old_p),
            )

    elapsed = (time.time() - t0) * 1000.0
    mean_ratio = sum(m.overwrite_ratio for m in case_metrics.values()) / len(case_metrics)
    residual_inf = 1.0 - mean_ratio

    return RebindingReport(
        seed=seed,
        cases=case_metrics,
        mean_overwrite_ratio=mean_ratio,
        residual_interference_rate=residual_inf,
        is_rebinding_supported=(mean_ratio >= 0.50),
        cpu_runtime_ms=elapsed,
    )
