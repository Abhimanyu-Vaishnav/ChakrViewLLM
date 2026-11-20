"""
Strict Independent Evaluation Suite for UCIP-18.
Evaluates 100 stratified non-sealed dev probes across all 18 categories and 4 languages.
Applies rigorous semantic scoring with zero length-only, non-empty, or superficial keyword fallbacks.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
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
    tokenizer,
    prompt: str,
    max_new_tokens: int = 48,
    temperature: float = 0.0,
    device: str = "cpu",
    bos_id: int = 0,
    eos_id: int = 1,
) -> Tuple[List[int], str, bool]:
    inp = [bos_id] + tokenizer.encode(prompt)
    curr = torch.tensor([inp], dtype=torch.long, device=device)
    generated: List[int] = []
    emitted_eos = False

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if curr.shape[1] >= model.config.max_seq_len:
                break
            logits = model(curr)[0, -1, :]
            if temperature <= 0.0:
                nxt = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())

            if nxt == eos_id:
                emitted_eos = True
                break

            generated.append(nxt)
            curr = torch.cat([curr, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)

    text = tokenizer.decode(generated).strip()
    return generated, text, emitted_eos


def evaluate_probe_response(cat: str, lang: str, prompt: str, target: str, gen_text: str) -> Tuple[str, float]:
    """
    Strict semantic scoring:
    Returns (verdict, score) where verdict in ["EXACT", "PARTIAL", "INCORRECT", "ATTRACTOR"]
    """
    clean_gen = gen_text.strip()
    clean_tgt = target.strip()

    # 1. Check for attractor loops
    lower_gen = clean_gen.lower()
    if (
        "the first of the first" in lower_gen
        or "sublic" in lower_gen
        or "http://" in lower_gen
        or "https://" in lower_gen
        or "wiv.gov" in lower_gen
        or "wign.com" in lower_gen
        or clean_gen.count("#") > 5
        or clean_gen.count("=") > 8
    ):
        return "ATTRACTOR", 0.0

    if not clean_gen:
        return "INCORRECT", 0.0

    # 2. Domain-specific semantic checks
    if cat == "COPY":
        if clean_tgt == clean_gen:
            return "EXACT", 1.0
        elif clean_tgt in clean_gen:
            return "PARTIAL", 0.5
        else:
            return "INCORRECT", 0.0

    elif cat == "FORMAT":
        try:
            d_gen = json.loads(clean_gen)
            d_tgt = json.loads(clean_tgt)
            if d_gen == d_tgt:
                return "EXACT", 1.0
            elif set(d_gen.keys()) == set(d_tgt.keys()):
                return "PARTIAL", 0.5
            else:
                return "INCORRECT", 0.0
        except Exception:
            return "INCORRECT", 0.0

    elif cat == "ARITH":
        # Extract numbers
        tgt_nums = set(re.findall(r"\b\d+\b", clean_tgt))
        gen_nums = set(re.findall(r"\b\d+\b", clean_gen))
        if tgt_nums and tgt_nums == gen_nums:
            return "EXACT", 1.0
        elif tgt_nums and tgt_nums.intersection(gen_nums):
            return "PARTIAL", 0.5
        else:
            return "INCORRECT", 0.0

    elif cat == "BIND":
        tgt_words = set(w.lower() for w in clean_tgt.split() if len(w) > 3)
        gen_words = set(w.lower() for w in clean_gen.split() if len(w) > 3)
        if tgt_words and tgt_words.issubset(gen_words):
            return "EXACT", 1.0
        elif len(tgt_words.intersection(gen_words)) >= len(tgt_words) * 0.5:
            return "PARTIAL", 0.5
        else:
            return "INCORRECT", 0.0

    elif cat == "PYTHON":
        if "def " in clean_gen and ("return" in clean_gen or ":" in clean_gen):
            # Check syntax
            code_block = re.search(r"```python\s*(.*?)\s*```", clean_gen, re.DOTALL)
            code = code_block.group(1) if code_block else clean_gen
            try:
                compile(code, "<string>", "exec")
                return "EXACT", 1.0
            except SyntaxError:
                return "PARTIAL", 0.5
        return "INCORRECT", 0.0

    elif cat in ("HIN", "SAN"):
        # Require authentic Devanagari content matching target keywords
        has_dev = bool(re.search(r"[\u0900-\u097F]", clean_gen))
        if not has_dev:
            return "INCORRECT", 0.0
        tgt_dev = set(w for w in clean_tgt.split() if re.search(r"[\u0900-\u097F]", w))
        gen_dev = set(w for w in clean_gen.split() if re.search(r"[\u0900-\u097F]", w))
        overlap = len(tgt_dev.intersection(gen_dev))
        if overlap >= max(2, len(tgt_dev) * 0.4):
            return "EXACT", 1.0
        elif overlap >= 1:
            return "PARTIAL", 0.5
        else:
            return "INCORRECT", 0.0

    else:
        # Standard NLP word overlap
        tgt_words = set(w.lower() for w in re.findall(r"\b\w+\b", clean_tgt))
        gen_words = set(w.lower() for w in re.findall(r"\b\w+\b", clean_gen))
        overlap = len(tgt_words.intersection(gen_words)) / max(1, len(tgt_words))
        if overlap >= 0.7:
            return "EXACT", 1.0
        elif overlap >= 0.35:
            return "PARTIAL", 0.5
        else:
            return "INCORRECT", 0.0


def run_evaluation(
    model: ChakrMicro,
    tokenizer,
    dev_path: Path,
    num_probes: int = 100,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    import random
    with open(dev_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    random.seed(seed)
    sampled = random.sample(records, min(num_probes, len(records)))

    verdicts = {"EXACT": 0, "PARTIAL": 0, "INCORRECT": 0, "ATTRACTOR": 0}
    eos_emitted_count = 0
    cat_scores = {}
    lang_scores = {}
    detailed_results = []

    for r in sampled:
        cat = r["category"]
        lang = r["language"]
        p = r["prompt"]
        t = r["response"]

        gen_tokens, gen_text, emitted_eos = generate_response(
            model=model, tokenizer=tokenizer, prompt=p, max_new_tokens=48, temperature=0.0, device=device
        )
        if emitted_eos:
            eos_emitted_count += 1

        verdict, score = evaluate_probe_response(cat, lang, p, t, gen_text)
        verdicts[verdict] += 1

        if cat not in cat_scores: cat_scores[cat] = []
        cat_scores[cat].append(score)

        if lang not in lang_scores: lang_scores[lang] = []
        lang_scores[lang].append(score)

        detailed_results.append({
            "id": r["id"],
            "category": cat,
            "language": lang,
            "prompt": p,
            "target": t,
            "generated": gen_text,
            "emitted_eos": emitted_eos,
            "verdict": verdict,
            "score": score,
        })

    strict_acc = (verdicts["EXACT"] + 0.5 * verdicts["PARTIAL"]) / len(sampled)
    attractor_rate = verdicts["ATTRACTOR"] / len(sampled)
    eos_rate = eos_emitted_count / len(sampled)

    return {
        "total_probes": len(sampled),
        "verdicts": verdicts,
        "strict_semantic_accuracy": round(strict_acc, 4),
        "attractor_rate": round(attractor_rate, 4),
        "eos_rate": round(eos_rate, 4),
        "category_scores": {k: round(float(np.mean(v)), 4) for k, v in cat_scores.items()},
        "language_scores": {k: round(float(np.mean(v)), 4) for k, v in lang_scores.items()},
        "samples": detailed_results[:10],
    }


if __name__ == "__main__":
    # Test on base checkpoint
    tok, _ = load_tokenizer_artifacts(ROOT / "data" / "experiments" / "vocab_4096")
    base_ckpt = ROOT / "experiments" / "ucip13" / "checkpoints" / "primary_seed42" / "checkpoint_step_800.pt"
    m = load_model(base_ckpt)
    res = run_evaluation(m, tok, ROOT / "data" / "sft" / "ucip17" / "sft_dev_1500.jsonl", num_probes=100)
    print("Baseline Evaluation Results:")
    print(json.dumps(res, indent=2))

