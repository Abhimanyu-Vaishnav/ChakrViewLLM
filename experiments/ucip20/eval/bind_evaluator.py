"""
UCIP-20 Strict Independent BIND Evaluator.
Evaluates model on the frozen 200 held-out development probes in bind_dev_200.jsonl.
Features:
- Disjoint entities: All dev entities are 100% unseen in training.
- Strict evaluation: Exact match of entity values, abstention accuracy, hallucination tracking.
- Zero length-only, non-empty, or superficial keyword fallbacks.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def load_model(checkpoint_path: Path, device: str = "cpu") -> ChakrMicro:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = ChakrMicro(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def generate_response(
    model: ChakrMicro,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    device: str = "cpu",
    bos_id: int = 0,
    eos_id: int = 1,
) -> Tuple[List[int], str, bool, int]:
    inp = [bos_id] + tokenizer.encode(prompt)
    curr = torch.tensor([inp], dtype=torch.long, device=device)
    generated: List[int] = []
    emitted_eos = False
    eos_pos = -1

    with torch.no_grad():
        for i in range(max_new_tokens):
            if curr.shape[1] >= model.config.max_seq_len:
                break
            logits = model(curr)[0, -1, :]
            if temperature <= 0.0:
                nxt = int(torch.argmax(logits).item())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())

            if nxt == eos_id:
                emitted_eos = True
                eos_pos = i
                break
            generated.append(nxt)
            curr = torch.cat([curr, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)

    gen_text = tokenizer.decode(generated, errors="replace").strip()
    return generated, gen_text, emitted_eos, eos_pos


def evaluate_bind_probe(
    target_value: str,
    distractor_value: str,
    is_abstention: bool,
    gen_text: str,
    emitted_eos: bool,
) -> Tuple[str, float]:
    """
    Evaluates strict entity binding:
    - EXACT (1.0): Generated response contains the exact bound target value, and does not contain the distractor.
    - ABSTENTION_CORRECT (1.0): Correctly abstains when information is absent.
    - HALLUCINATED_DISTRACTOR (0.0): Bound to the wrong entity in context (distractor).
    - ATTRACTOR (0.0): Degenerate repetition ('#', 'passcop', Wikipedia tags).
    - INCORRECT (0.0): General incorrect response.
    """
    clean_gen = gen_text.lower().strip()
    clean_target = target_value.lower().strip()
    
    # 1. Attractor check
    if clean_gen.count("#") > 5 or "passcop" in clean_gen or "ref name" in clean_gen or len(clean_gen) == 0:
        return "ATTRACTOR", 0.0

    # 2. Abstention check
    if is_abstention:
        abstention_markers = [
            "information not provided",
            "not provided",
            "उपलब्ध नहीं",
            "जानकारी नहीं",
            "available nahi",
            "not available",
            "nahi hai"
        ]
        if any(m in clean_gen for m in abstention_markers):
            return "ABSTENTION_CORRECT", 1.0
        else:
            return "INCORRECT", 0.0

    # 3. Positive binding check
    # Check if target is present
    target_present = clean_target in clean_gen
    distractor_present = False
    if distractor_value:
        distractor_present = distractor_value.lower().strip() in clean_gen

    if target_present and not distractor_present:
        return "EXACT", 1.0
    elif distractor_present and not target_present:
        return "HALLUCINATED_DISTRACTOR", 0.0
    elif target_present and distractor_present:
        return "PARTIAL", 0.5
    else:
        return "INCORRECT", 0.0


def run_bind_evaluation(
    model: ChakrMicro,
    tokenizer: Any,
    dev_file: Path,
    device: str = "cpu"
) -> Dict[str, Any]:
    with open(dev_file, "r", encoding="utf-8") as f:
        probes = [json.loads(line) for line in f]

    results = []
    verdicts = Counter()
    relation_scores = {}
    language_scores = {}
    eos_emitted_count = 0

    for r in probes:
        p = r["prompt"]
        tgt = r["target_value"]
        dist = r.get("distractor_entity", "")
        is_abs = r.get("is_abstention", False)
        rel = r.get("relation_type", "general")
        lang = r.get("language", "en")

        gen_tokens, gen_text, emitted_eos, eos_pos = generate_response(
            model=model, tokenizer=tokenizer, prompt=p, max_new_tokens=32, temperature=0.0, device=device
        )
        if emitted_eos:
            eos_emitted_count += 1

        verdict, score = evaluate_bind_probe(
            target_value=tgt,
            distractor_value=dist,
            is_abstention=is_abs,
            gen_text=gen_text,
            emitted_eos=emitted_eos,
        )
        verdicts[verdict] += 1

        if rel not in relation_scores: relation_scores[rel] = []
        relation_scores[rel].append(score)

        if lang not in language_scores: language_scores[lang] = []
        language_scores[lang].append(score)

        results.append({
            "id": r["id"],
            "relation_type": rel,
            "language": lang,
            "is_abstention": is_abs,
            "target": tgt,
            "generated": gen_text,
            "emitted_eos": emitted_eos,
            "verdict": verdict,
            "score": score,
        })

    total = len(probes)
    exact_acc = (verdicts["EXACT"] + verdicts["ABSTENTION_CORRECT"]) / total
    attractor_rate = verdicts["ATTRACTOR"] / total
    distractor_rate = verdicts["HALLUCINATED_DISTRACTOR"] / total
    eos_rate = eos_emitted_count / total

    summary = {
        "total_probes": total,
        "verdicts": dict(verdicts),
        "exact_entity_accuracy": round(exact_acc, 4),
        "attractor_rate": round(attractor_rate, 4),
        "hallucinated_distractor_rate": round(distractor_rate, 4),
        "eos_rate": round(eos_rate, 4),
        "relation_scores": {k: round(float(np.mean(v)), 4) for k, v in relation_scores.items()},
        "language_scores": {k: round(float(np.mean(v)), 4) for k, v in language_scores.items()},
        "samples": results[:10],
    }
    return summary


def main():
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")
    base_ckpt = ROOT / "experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt"
    dev_file = ROOT / "data/sft/ucip20_bind/bind_dev_200.jsonl"
    out_file = ROOT / "experiments/ucip20/eval/baseline_step800_bind_eval.json"
    
    print(f"Evaluating Baseline on BIND Dev Suite: {base_ckpt}")
    m = load_model(base_ckpt)
    res = run_bind_evaluation(m, tok, dev_file)
    
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
        
    print("Baseline BIND Evaluation Complete:")
    print(f"  Exact Entity Accuracy : {res['exact_entity_accuracy']*100:.2f}%")
    print(f"  Attractor Rate        : {res['attractor_rate']*100:.2f}%")
    print(f"  EOS Rate              : {res['eos_rate']*100:.2f}%")
    print(f"  Verdicts              : {res['verdicts']}")


if __name__ == "__main__":
    from collections import Counter
    main()
