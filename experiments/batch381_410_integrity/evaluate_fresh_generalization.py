"""
Batches 401-405: Fresh Generalization, Paraphrase Robustness & Contamination Audit.
Location: experiments/batch381_410_integrity/evaluate_fresh_generalization.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch

sys.path.insert(0, ".")

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
CKPT_CORRECTED = Path("experiments/batch381_410_integrity/checkpoints/candidate_step150_corrected_eos.pt")
CKPT_BASE = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
OUT_JSON = Path("experiments/batch381_410_integrity/reports/fresh_generalization_results.json")


def build_fresh_manifest() -> List[Dict[str, Any]]:
    return [
        # Direct vs Paraphrase Pair 1 (Math)
        {
            "id": "para_math_direct",
            "type": "direct",
            "domain": "math",
            "query": "What is 14 plus 28?",
            "prompt": "Question: What is 14 plus 28?\nAnswer:",
            "expected": "42",
        },
        {
            "id": "para_math_variant",
            "type": "paraphrase",
            "domain": "math",
            "query": "Compute the sum of fourteen and twenty eight.",
            "prompt": "Question: Compute the sum of fourteen and twenty eight.\nAnswer:",
            "expected": "42",
        },

        # Direct vs Paraphrase Pair 2 (Architecture)
        {
            "id": "para_arch_direct",
            "type": "direct",
            "domain": "architecture",
            "query": "What does DMA stand for in computer systems?",
            "prompt": "Question: What does DMA stand for in computer systems?\nAnswer:",
            "expected": "Direct Memory Access",
        },
        {
            "id": "para_arch_variant",
            "type": "paraphrase",
            "domain": "architecture",
            "query": "In microprocessor design, what does the abbreviation DMA represent?",
            "prompt": "Question: In microprocessor design, what does the abbreviation DMA represent?\nAnswer:",
            "expected": "Direct Memory Access",
        },

        # Direct vs Paraphrase Pair 3 (Hindi)
        {
            "id": "para_hindi_direct",
            "type": "direct",
            "domain": "hindi",
            "query": "भारत की राजधानी क्या है?",
            "prompt": "प्रश्न: भारत की राजधानी क्या है?\nउत्तर:",
            "expected": "नई दिल्ली",
        },
        {
            "id": "para_hindi_variant",
            "type": "paraphrase",
            "domain": "hindi",
            "query": "भारतीय गणराज्य की राष्ट्रीय राजधानी कौन सा शहर है?",
            "prompt": "प्रश्न: भारतीय गणराज्य की राष्ट्रीय राजधानी कौन सा शहर है?\nउत्तर:",
            "expected": "नई दिल्ली",
        },

        # Compositional Unseen Pair (Novel combination of learned concepts)
        {
            "id": "comp_unseen_1",
            "type": "compositional",
            "domain": "unseen_transfer",
            "query": "What circuit in a CPU combines arithmetic operations with logic gates?",
            "prompt": "Question: What circuit in a CPU combines arithmetic operations with logic gates?\nAnswer:",
            "expected": "Arithmetic Logic Unit (ALU)",
        },
        {
            "id": "comp_unseen_2",
            "type": "compositional",
            "domain": "unseen_transfer",
            "query": "What is 7 multiplied by 6 minus 10?",
            "prompt": "Question: What is 7 multiplied by 6 minus 10?\nAnswer:",
            "expected": "32",
        },
    ]


def run_evaluation():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    ck = torch.load(CKPT_CORRECTED, map_location="cpu", weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    items = build_fresh_manifest()
    results = []
    direct_exact = 0
    para_exact = 0
    comp_exact = 0

    for item in items:
        p_toks = [0] + tokenizer.encode(item["prompt"])
        gen = list(p_toks)
        eos_reached = False

        with torch.no_grad():
            for _ in range(25):
                x = torch.tensor([gen[-256:]], dtype=torch.long)
                logits = model(x)
                nxt = int(torch.argmax(logits[0, -1, :]).item())
                if nxt == 1:
                    eos_reached = True
                    break
                gen.append(nxt)

        decoded = tokenizer.decode(gen[len(p_toks):], errors="replace").strip()
        exp = item["expected"]
        is_exact = (decoded == exp) or (exp in decoded)

        if item["type"] == "direct" and is_exact:
            direct_exact += 1
        elif item["type"] == "paraphrase" and is_exact:
            para_exact += 1
        elif item["type"] == "compositional" and is_exact:
            comp_exact += 1

        results.append({
            "id": item["id"],
            "type": item["type"],
            "domain": item["domain"],
            "query": item["query"],
            "expected": exp,
            "generated": decoded,
            "exact_match": is_exact,
            "eos_reached": eos_reached,
        })

    summary = {
        "candidate": "candidate_step150_corrected_eos",
        "total_fresh_items": len(items),
        "direct_exact_match_rate": direct_exact / 3.0,
        "paraphrase_exact_match_rate": para_exact / 3.0,
        "compositional_exact_match_rate": comp_exact / 2.0,
        "clean_termination_rate": sum(1 for r in results if r["eos_reached"]) / len(items),
        "details": results,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Direct Match Rate: {summary['direct_exact_match_rate']*100:.1f}%")
    print(f"Paraphrase Match Rate: {summary['paraphrase_exact_match_rate']*100:.1f}%")
    print(f"Compositional Match Rate: {summary['compositional_exact_match_rate']*100:.1f}%")
    print(f"Report written to: {OUT_JSON}")


if __name__ == "__main__":
    run_evaluation()
